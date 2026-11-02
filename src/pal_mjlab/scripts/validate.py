"""Validate a trained velocity policy on scripted scenarios, many envs at once.

A headless sibling of ``mjlab``'s ``play`` script: the same checkpoint
loading (local file or W&B run) and the same env construction, but instead
of a viewer it runs one of a few fixed *scenarios* on ``num_envs`` parallel
copies of the robot, all starting from the model's ``INIT_STATE`` standing
pose, and records every reward and metric term while they run.

Every env sees the same command schedule; they diverge only through the
per-env randomness the play config already applies -- the startup domain
randomization (foot friction, encoder bias, base CoM, joint friction) and the
small initial-pose jitter on reset -- so the run measures how robust the
policy is to that spread rather than a single rollout. ``--seed`` fixes the
draw. ``--domain-rand False`` drops the startup DR so only the pose jitter
remains.

Any ``Mjlab-Velocity-*`` task with a ``twist`` velocity command works, so
policies trained for ``pal_kangaroo_full`` (tendon model) and
``pal_kangaroo_full_full`` (connect-linkage model) are both run by naming
their task id, exactly as for ``play``. A checkpoint can also be validated
end to end on another model's task with a matching observation/action
layout: a tendon-model policy trained with the "actuator" transmission on
full_full's "actuator" task, one trained with the "joint" transmission on
full_full's "lut" task. ``--train-task`` names the task it was trained on
so the summary records which kind of run it was. The scenario overrides the task's
terrain, so a ``-Flat-`` task id can be validated on stairs and a ``-Rough-``
one on the plane.

Scenarios (``--scenario``):

* ``flat_ramp``: flat plane. Stand still for ``stand-time``, then ramp the
  forward velocity command linearly to ``target-vel`` (3.0 m/s by default)
  over ``ramp-time``, then hold it. ``--flat-ramp.start-vel V`` instead
  steps the command to V after the stand and holds it until
  ``start-fraction`` (50% by default) of the envs still alive have reached
  ``start-reach-fraction`` of V (90% by default); then the
  ramp from V starts for every env at once, so all of them get the same
  forward command at every step (``start-wait-time`` extends the run for
  the wait). ``--flat-ramp.start-after S`` skips the wait: the ramp starts
  S seconds after the step to V for every env, whatever their speed. ``--flat-ramp.steer True`` adds the
  stairs scenario's line steering (yaw-rate command back onto the line
  through the env origin along +x).
* ``pebbles_ramp``: same schedule up to 1.0 m/s on the rough task's
  "pebbles" sub-terrain (dense 2-5 cm boxes), regenerated as one long tile
  so the robot has room to run (``--pebbles-ramp.steer`` likewise).
* ``stairs``: spawn in the centre of a pyramid staircase and walk forward at
  a constant command. ``--stairs.variant`` picks the shape, mirroring the
  training Stairs env's two families: ``pit`` (inverted pyramid, the robot
  climbs out; the default) or ``peak`` (the robot starts on the summit and
  walks down). Step height and width are the difficulty knobs. The command
  is held at ``--stairs.velocity`` from the first step (no stand, no ramp).
  The yaw-rate command steers each robot onto the straight line through its
  env origin along +x (``--stairs.steer``, on by default), so heading drift
  doesn't walk it off the staircase diagonal or into a corner. An env is done
  (a success, not a failure) once its base is ``--stairs.finish-distance``
  (1 m) past the outer edge of the staircase along +x; the run ends when
  every env is done or has terminated, or at ``--stairs.duration``.

Results go to one ``.npz`` (plus a ``.json`` summary next to it) with
per-step aggregates over the still-alive envs and per-env outcomes; see
``_save`` for the exact keys. Run::

    uv run validate <task-id> --checkpoint-file logs/.../model_5000.pt \\
        --scenario flat_ramp --num-envs 4096
    uv run validate <task-id> --wandb-run-path org/proj/run --scenario stairs \\
        --stairs.step-height 0.12 --stairs.step-width 0.3
    uv run validate <task-id> --checkpoint-file ... --scenario stairs \\
        --stairs.variant peak
    uv run validate <task-id> --agent zero --scenario pebbles_ramp --num-envs 8
    uv run validate <task-id> --checkpoint-file ... --scenario stairs \\
        --num-envs 16 --viewer viser --video True

``--viewer native|viser`` shows env 0 while the same recording runs (the
viewer stops itself at the end of the scenario, or as soon as every env has
finished or terminated); with viser, ``--video True`` also writes an .mp4 of
env 0 next to the .npz.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg, dc_motor_clip
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, illegal_contact
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.terrains.config import (
  pyramid_stairs,
  pyramid_stairs_inv,
  random_spread_boxes,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.debug_visualizer import DebugVisualizer
from pal_mjlab.robots.pal_kangaroo_full.actuator import LutTransmissionActuator

Scenario = Literal["flat_ramp", "pebbles_ramp", "stairs"]
Integrator = Literal["implicit", "implicitfast", "euler"]
StairsVariant = Literal["pit", "peak"]

# The rough training tile in configure_kangaroo_rough_env spreads 350 pebbles
# over a 3 m x 3 m tile; keep that density whatever tile size is asked for.
_TRAINING_PEBBLE_DENSITY = 350.0 / 9.0

# Flat frame around the validation staircase inside its tile [m].
_STAIRS_BORDER_WIDTH = 0.1

# Torso collision capsule every kangaroo model carries (on pelvis_2_link; see
# pal_kangaroo_full.spec_edits.add_collision_capsules).
_TORSO_COLLISION_GEOM = "pelvis_2_collision"

# Line-following target line in the viewer: hovers just above the treads.
_LINE_Z_OFFSET = 0.01
_LINE_RADIUS = 0.008
_LINE_RGBA = (1.0, 0.45, 0.0, 0.9)


##
# Command: a deterministic ramp instead of the uniform sampler.
##


class RampVelocityCommand(UniformVelocityCommand):
  """Forward velocity ramp: 0 for stand_time_s, then linear to target, then hold.

  ramp_time_s <= 0 makes it a step at stand_time_s (with stand_time_s = 0, a
  constant command for the whole run, as the stairs scenario uses).

  With start_lin_vel_x > 0 the command steps to start_lin_vel_x at
  stand_time_s and stays there until start_fraction of the envs still alive
  have reached start_reach_fraction of it (forward velocity low-pass filtered
  over start_filter_time_s; an env counts once it has got there, even if it
  slows down later). An env reset mid-run (it failed) leaves the count, both
  the reached envs and the alive ones it is a fraction of. At
  that step the ramp from start_lin_vel_x to the target begins for every env
  at once, so all of them get the same forward command (``ramp_start_t`` is
  shared, inf until then); if too few ever get there, nobody ramps.

  Reuses the uniform command's buffers, metrics and debug arrows so the rest
  of the task (observation term, tracking rewards, posture reward) sees the
  same ``twist`` interface it was trained against; only where the numbers
  come from changes. The lateral command is always zero; the yaw command is
  zero too unless line following is on (``line_heading_gain > 0``), in which
  case it steers the robot onto the line through its env origin along world +x.
  """

  cfg: RampVelocityCommandCfg

  def __init__(self, cfg: RampVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.ramp_start_t = torch.full(
      (self.num_envs,), self._initial_ramp_start(), device=self.device
    )
    self.filtered_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
    self.reached_start = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    # Envs reset mid-run: failed (or timed out), out of the start count.
    self.ended = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

  def value_at(self, t: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg
    start = cfg.start_lin_vel_x
    if cfg.ramp_time_s <= 0.0:
      frac = (t >= self.ramp_start_t).to(t.dtype)
    else:
      frac = ((t - self.ramp_start_t) / cfg.ramp_time_s).clamp(0.0, 1.0)
    value = start + frac * (cfg.target_lin_vel_x - start)
    return torch.where(t >= cfg.stand_time_s, value, torch.zeros_like(value))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # No sampling: every mode flag off so _update_command's parent logic
    # (heading control, standing envs, world-frame envs) never fires.
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_w[env_ids] = 0.0
    self.heading_target[env_ids] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    # Only called on reset (the resampling time is never reached). The ramp
    # start is shared, so only a reset of every env (a scenario restart)
    # waits for the start velocity again; a single env reset mid-run keeps it.
    if env_ids.numel() == self.num_envs:
      self.ended.fill_(False)
      self.ramp_start_t.fill_(self._initial_ramp_start())
    else:
      self.ended[env_ids] = True
    self.filtered_lin_vel_x[env_ids] = 0.0
    self.reached_start[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Pure function of each env's episode time and the shared ramp start, so
    # refreshing all envs is safe.
    del env_ids
    t = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
    if self.cfg.start_lin_vel_x > 0.0 and self.cfg.start_after_s is None:
      self._detect_ramp_start(t)
    self.vel_command_b[:, 0] = self.value_at(t)
    self.vel_command_b[:, 1:] = 0.0
    if self.cfg.line_heading_gain > 0.0:
      self.vel_command_b[:, 2] = self._line_following_yaw_rate()

  def _initial_ramp_start(self) -> float:
    """Ramp start until detected: stand_time_s from 0, stand_time_s +
    start_after_s for a fixed wait at start_lin_vel_x, inf when waiting for
    the envs to reach it."""
    cfg = self.cfg
    if cfg.start_lin_vel_x <= 0.0:
      return cfg.stand_time_s
    if cfg.start_after_s is not None:
      return cfg.stand_time_s + cfg.start_after_s
    return math.inf

  def _detect_ramp_start(self, t: torch.Tensor) -> None:
    """Start every env's ramp once start_fraction of the alive envs (not
    reset mid-run) have reached start_reach_fraction of start_lin_vel_x
    (filtered) while it was being commanded."""
    cfg = self.cfg
    alpha = min(1.0, self._env.step_dt / max(cfg.start_filter_time_s, 1e-6))
    vx = self.robot.data.root_link_lin_vel_b[:, 0]
    self.filtered_lin_vel_x += alpha * (vx - self.filtered_lin_vel_x)
    self.reached_start |= (t >= cfg.stand_time_s) & (
      self.filtered_lin_vel_x >= cfg.start_reach_fraction * cfg.start_lin_vel_x
    )
    alive = ~self.ended
    n_alive = int(alive.sum())
    if (
      self.ramp_start_t[0].isinf()
      and n_alive > 0
      and int((self.reached_start & alive).sum()) >= cfg.start_fraction * n_alive
    ):
      # Envs reset mid-run are behind on episode time; the rest share it.
      self.ramp_start_t.fill_(t.max())

  def _line_following_yaw_rate(self) -> torch.Tensor:
    """Stanley-style steering onto y = env_origin_y, heading along +x.

    The desired heading leans back toward the line by atan(k_lat * y_err), so
    it saturates at 90 deg far from the line and goes to 0 on it; the yaw-rate
    command is proportional to the remaining heading error. Walking backwards
    (negative command) flips the lean so the correction still closes the gap.
    """
    cfg = self.cfg
    y_err = self.robot.data.root_link_pos_w[:, 1] - self._env.scene.env_origins[:, 1]
    direction = 1.0 if cfg.target_lin_vel_x >= 0.0 else -1.0
    desired_heading = -direction * torch.atan(cfg.line_lateral_gain * y_err)
    heading_err = wrap_to_pi(desired_heading - self.robot.data.heading_w)
    return (cfg.line_heading_gain * heading_err).clamp(
      -cfg.max_yaw_rate, cfg.max_yaw_rate
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Velocity arrows, plus the line-following target line (draped over the
    terrain along line_profile) when line following is on."""
    super()._debug_vis_impl(visualizer)
    profile = self.cfg.line_profile
    if self.cfg.line_heading_gain <= 0.0 or len(profile) < 2:
      return
    env_indices = list(visualizer.get_env_indices(self.num_envs))
    if not env_indices:
      return
    origins = self._env.scene.env_origins.cpu().numpy()
    xz = np.asarray(profile, dtype=np.float64)
    for i in env_indices:
      ox, oy, oz = origins[i]
      points = np.stack(
        [ox + xz[:, 0], np.full(len(xz), oy), oz + xz[:, 1] + _LINE_Z_OFFSET], axis=1
      )
      for start, end in zip(points[:-1], points[1:], strict=True):
        visualizer.add_cylinder(start, end, _LINE_RADIUS, _LINE_RGBA)


@dataclass(kw_only=True)
class RampVelocityCommandCfg(UniformVelocityCommandCfg):
  target_lin_vel_x: float
  stand_time_s: float
  ramp_time_s: float
  start_lin_vel_x: float = 0.0
  """Command held after stand_time_s until start_fraction of the alive envs
  reach it, where every env's ramp then starts; 0 ramps from 0 right at
  stand_time_s."""
  start_fraction: float = 0.5
  """Fraction of the alive envs (not reset mid-run) that must reach
  start_lin_vel_x to start the ramp."""
  start_reach_fraction: float = 0.9
  """An env has reached start_lin_vel_x once its filtered forward velocity is
  at least this fraction of it."""
  start_filter_time_s: float = 0.5
  """Time constant of the low-pass filter on the forward velocity that is
  compared against start_lin_vel_x [s]; smooths out the gait's oscillation."""
  start_after_s: float | None = None
  """With start_lin_vel_x > 0: start every env's ramp this long after
  stand_time_s instead of waiting for start_fraction to reach it; None
  waits."""
  line_heading_gain: float = 0.0
  """Yaw-rate command per rad of heading error [1/s]; 0 disables line following."""
  line_lateral_gain: float = 1.0
  """Desired-heading lean per metre of lateral error, inside atan [1/m]."""
  max_yaw_rate: float = 1.0
  """Clamp on the line-following yaw-rate command [rad/s]."""
  line_profile: tuple[tuple[float, float], ...] = ()
  """(x, z) polyline of the target line relative to the env origin, for the
  viewer only; empty draws nothing."""

  def build(self, env: ManagerBasedRlEnv) -> RampVelocityCommand:
    return RampVelocityCommand(self, env)


##
# CLI config.
##


@dataclass(frozen=True)
class RampScenarioCfg:
  target_vel: float
  """Forward velocity the command ramps up to [m/s]."""
  stand_time: float = 2.0
  """Seconds of zero command before the ramp (or the start_vel step) starts."""
  start_vel: float = 0.0
  """Forward command after stand_time [m/s]. Above 0, it is held until
  start_fraction of the alive envs have reached it (filtered forward velocity), and
  then the ramp from start_vel starts for every env at once, so all of them
  get the same command; 0 ramps from 0 right at stand_time."""
  start_fraction: float = 0.5
  """Fraction of the envs still alive that must reach start_vel to start the
  ramp (failed envs count neither as reached nor in the total)."""
  start_reach_fraction: float = 0.9
  """An env has reached start_vel once its filtered forward velocity is at
  least this fraction of it (0.9: 90% of start_vel)."""
  start_wait_time: float = 10.0
  """Seconds budgeted for reaching start_vel (only when start_vel > 0); they
  extend the run, and if too few envs get there by the end nobody ramps.
  Unused with start_after."""
  start_after: float | None = None
  """Seconds the command is held at start_vel (only when start_vel > 0)
  before every env's ramp starts, without waiting for start_fraction of the
  envs to reach it; None (default) waits. Replaces start_wait_time in the
  run length."""
  start_filter_time: float = 0.5
  """Time constant of the low-pass filter on the forward velocity compared
  against start_vel [s]."""
  ramp_time: float = 10.0
  """Seconds the ramp takes to go from start_vel to target_vel."""
  hold_time: float = 5.0
  """Seconds the command is held at target_vel before the run ends."""
  steer: bool = False
  """Command yaw rate to keep the robot on the line through its env origin
  along +x, as the stairs scenario does; off holds the yaw command at zero."""
  steer_heading_gain: float = 1.5
  """Yaw-rate command per rad of heading error [1/s]."""
  steer_lateral_gain: float = 1.0
  """Desired-heading lean per metre off the line, inside atan [1/m]."""
  steer_max_yaw_rate: float = 1.0
  """Clamp on the steering yaw-rate command [rad/s]."""


@dataclass(frozen=True)
class PebblesScenarioCfg(RampScenarioCfg):
  target_vel: float = 1.0
  tile_size: tuple[float, float] = (24.0, 8.0)
  """Pebble tile (x, y) extent [m]. The robot spawns at its centre facing +x."""
  box_density: float = _TRAINING_PEBBLE_DENSITY
  """Pebbles per square metre (the training tile's density by default)."""
  difficulty: float = 1.0
  """Sub-terrain difficulty in [0, 1]: scales pebble count and height as in
  training; 1.0 is the hardest training row."""


@dataclass(frozen=True)
class StairsScenarioCfg:
  variant: StairsVariant = "pit"
  """"pit": inverted pyramid, spawn at the bottom and climb up. "peak": regular
  pyramid, spawn on the summit and walk down."""
  step_height: float = 0.10
  """Rise of every step [m]. The main difficulty knob."""
  step_width: float = 0.3
  """Run (depth) of every step [m]; raised to min_width_to_height * step_height
  when shallower than that."""
  min_width_to_height: float = 1.5
  """Lower bound on step width / step height, as in the training Stairs env."""
  velocity: float = 0.5
  """Forward velocity command, held constant for the whole run [m/s]."""
  duration: float = 60.0
  """Maximum run length [s]; the run ends earlier once every env has finished
  (see finish_distance) or terminated."""
  finish_distance: float | None = 1.0
  """An env finishes, and stops being recorded, once its base is this far past
  the staircase's outer edge along +x [m]. None runs every env for the full
  duration."""
  tile_size: float = 8.0
  """Side of the square staircase tile [m]; sets how many steps there are."""
  platform_width: float = 1.5
  """Minimum side of the flat square the robot spawns on (pit bottom or summit)
  [m]; whatever doesn't fit a whole tread pair is added to it."""
  steer: bool = True
  """Command yaw rate to keep the robot on the line through the tile centre
  along +x; off holds the yaw command at zero."""
  steer_heading_gain: float = 1.5
  """Yaw-rate command per rad of heading error [1/s]."""
  steer_lateral_gain: float = 1.0
  """Desired-heading lean per metre off the line, inside atan [1/m]."""
  steer_max_yaw_rate: float = 1.0
  """Clamp on the steering yaw-rate command [rad/s]."""


@dataclass(frozen=True)
class ValidateConfig:
  scenario: Scenario = "flat_ramp"
  flat_ramp: RampScenarioCfg = field(
    default_factory=lambda: RampScenarioCfg(target_vel=3.0)
  )
  pebbles_ramp: PebblesScenarioCfg = field(default_factory=PebblesScenarioCfg)
  stairs: StairsScenarioCfg = field(default_factory=StairsScenarioCfg)

  agent: Literal["zero", "random", "trained"] = "trained"
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  train_task: str | None = None
  """Task id the checkpoint was trained on, recorded in the summary. When it
  differs from the validated task the run is end to end (e.g. a tendon-model
  policy on the full_full model: "actuator" policies on its "actuator"
  transmission, "joint" ones on "lut"); None assumes the validated task."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs (and cached checkpoints) live."""

  num_envs: int = 4096
  seed: int = 0
  """Env seed: fixes the domain-randomization draw and the initial-pose jitter."""
  device: str | None = None
  domain_rand: bool = True
  """Keep the task's startup domain randomization (the main source of spread
  between envs). False removes it, leaving only the initial-pose jitter."""
  illegal_contacts_terminate: bool = False
  """Keep the task's illegal_contacts termination. False (default) removes it
  and only logs the contacts, so a knee or shin brushing a step no longer ends
  the run; falls are still caught by the other terminations."""
  torso_contact_terminate: bool = True
  """End an env (as a failure, termination "torso_contact") as soon as the
  torso collision capsule touches the terrain, independent of
  illegal_contacts_terminate."""
  init_pos_noise: float = 0.05
  """Half-range of the uniform x/y jitter on the spawn position [m]."""
  init_yaw_noise: float = 0.0
  """Half-range of the uniform yaw jitter on the spawn heading [rad]. Zero
  makes every robot face +x, which the pebble and stair tiles assume."""
  init_joint_noise: float = 0.0
  """Half-range of the uniform offset added to every joint's INIT_STATE
  position on reset [rad or m]."""
  integrator: Integrator = "implicit"
  """MuJoCo integrator. "implicit" (default) also differentiates the
  Coriolis/centrifugal forces in velocity, which training's "implicitfast"
  leaves out; "implicitfast" and "euler" keep mjlab's own options."""

  output: str | None = None
  """Where to write the .npz (a .json summary goes next to it). Default:
  <checkpoint dir>/validate/<scenario>_<timestamp>.npz, or
  logs/validate/... for dummy agents."""
  save_per_env_steps: bool = False
  """Also store every env's per-step rewards, metrics and base state
  ([num_envs, steps, terms]); ~1.5 GB at 4096 envs and 850 steps."""
  trace_envs: int = 16
  """Store every step's position, velocity and force, in actuator space and in
  joint space, for the first N envs ([N, steps, 3, actuators|joints], NaN once
  an env is done); plot-validate draws these. The base pose is traced too
  ([N, steps, 7]: position relative to the env origin, then the wxyz
  quaternion), so replay-validate can play the traced envs back. 0 disables."""
  trace_keep: Literal["all", "extremes"] = "all"
  """Traces to save: of every traced env, or only of the extremes among them
  (the env with the highest peak forward velocity, the longest surviving one,
  the fastest of them on a tie, and the first one to fail), so --trace-envs
  can cover every env without a large .npz."""
  log_every: int = 1
  """Subsampling stride for save_per_env_steps."""
  print_every_s: float = 1.0
  """Progress line interval in simulated seconds."""

  viewer: Literal["none", "native", "viser"] = "none"
  """Watch the run: "native" opens the MuJoCo window, "viser" serves a browser
  viewer. The viewer drives the same logged rollout (env 0 is shown) and
  stops itself when the scenario ends; "none" runs headless as fast as it can."""
  video: bool = False
  """Record env 0 to an .mp4 next to the .npz (needs --viewer viser)."""
  video_length: int | None = None
  """Frames to record, one per env step; None records the whole scenario."""
  video_height: int | None = None
  video_width: int | None = None


##
# Env config per scenario.
##


@dataclass
class _MujocoCfgWithImplicit(MujocoCfg):
  """MujocoCfg that also accepts the full "implicit" integrator, which
  mujoco_warp supports but mjlab's MujocoCfg doesn't offer."""

  integrator: Integrator = "implicit"  # type: ignore[assignment]

  def apply(self, model: mujoco.MjModel) -> None:
    if self.integrator != "implicit":
      super().apply(model)
      return
    MujocoCfg.apply(replace(self, integrator="implicitfast"), model)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT


def _set_integrator(env_cfg: ManagerBasedRlEnvCfg, integrator: Integrator) -> None:
  fields = asdict(env_cfg.sim.mujoco)
  fields["integrator"] = integrator
  env_cfg.sim.mujoco = _MujocoCfgWithImplicit(**fields)


def _rough_sim_settings(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Contact budget a Flat task lacks; same values configure_kangaroo_rough_env uses."""
  if env_cfg.sim.nconmax is None:
    env_cfg.sim.nconmax = 200
  if env_cfg.sim.njmax is not None:
    env_cfg.sim.njmax = max(env_cfg.sim.njmax, 700)
  env_cfg.sim.mujoco.ccd_iterations = max(env_cfg.sim.mujoco.ccd_iterations, 500)
  env_cfg.sim.contact_sensor_maxmatch = max(env_cfg.sim.contact_sensor_maxmatch, 500)


def _set_single_tile_terrain(
  env_cfg: ManagerBasedRlEnvCfg,
  sub_terrain: SubTerrainCfg,
  size: tuple[float, float],
  difficulty: float,
  seed: int,
) -> None:
  """One sub-terrain, one tile, all envs spawning at its centre."""
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_type = "generator"
  env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=seed,
    size=size,
    num_rows=1,
    num_cols=1,
    border_width=5.0,
    curriculum=False,
    difficulty_range=(difficulty, difficulty),
    sub_terrains={"validate": sub_terrain},
  )
  _rough_sim_settings(env_cfg)


def _set_plane_terrain(env_cfg: ManagerBasedRlEnvCfg) -> None:
  assert env_cfg.scene.terrain is not None
  env_cfg.scene.terrain.terrain_type = "plane"
  env_cfg.scene.terrain.terrain_generator = None


def _set_ramp_command(
  env_cfg: ManagerBasedRlEnvCfg,
  target_vel: float,
  stand_time: float,
  ramp_time: float,
  start_vel: float = 0.0,
  start_fraction: float = 0.5,
  start_reach_fraction: float = 0.9,
  start_filter_time: float = 0.5,
  start_after: float | None = None,
  line_heading_gain: float = 0.0,
  line_lateral_gain: float = 1.0,
  max_yaw_rate: float = 1.0,
  line_profile: tuple[tuple[float, float], ...] = (),
) -> None:
  twist = env_cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg), (
    "validate expects a velocity task with a 'twist' UniformVelocityCommandCfg"
  )
  # Wide enough that the posture term's walking/running thresholds and the
  # joystick sliders behave; the ramp never samples from these.
  hi = max(abs(target_vel), 1.0)
  env_cfg.commands["twist"] = RampVelocityCommandCfg(
    entity_name=twist.entity_name,
    resampling_time_range=(1e6, 1e6),
    heading_command=False,
    debug_vis=twist.debug_vis,
    viz=twist.viz,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-hi, hi),
      lin_vel_y=(-hi, hi),
      ang_vel_z=(-max(max_yaw_rate, 1.0), max(max_yaw_rate, 1.0)),
    ),
    target_lin_vel_x=target_vel,
    stand_time_s=stand_time,
    ramp_time_s=ramp_time,
    start_lin_vel_x=start_vel,
    start_fraction=start_fraction,
    start_reach_fraction=start_reach_fraction,
    start_filter_time_s=start_filter_time,
    start_after_s=start_after,
    line_heading_gain=line_heading_gain,
    line_lateral_gain=line_lateral_gain,
    max_yaw_rate=max_yaw_rate,
    line_profile=line_profile,
  )


def _start_wait(s: RampScenarioCfg) -> float:
  """Seconds between the step to start_vel and the end of the start phase."""
  if s.start_vel <= 0.0:
    return 0.0
  return s.start_after if s.start_after is not None else s.start_wait_time


def _ramp_scenario_duration(s: RampScenarioCfg) -> float:
  wait = _start_wait(s)
  return s.stand_time + wait + s.ramp_time + s.hold_time


def _set_ramp_scenario_command(env_cfg: ManagerBasedRlEnvCfg, s: RampScenarioCfg):
  # The line drawn in the viewer covers the distance the command asks for.
  start, target = abs(s.start_vel), abs(s.target_vel)
  wait = _start_wait(s)
  length = start * wait + (start + target) / 2 * s.ramp_time
  length += target * s.hold_time + 2.0
  _set_ramp_command(
    env_cfg,
    s.target_vel,
    s.stand_time,
    s.ramp_time,
    start_vel=s.start_vel,
    start_fraction=s.start_fraction,
    start_reach_fraction=s.start_reach_fraction,
    start_filter_time=s.start_filter_time,
    start_after=s.start_after,
    line_heading_gain=s.steer_heading_gain if s.steer else 0.0,
    line_lateral_gain=s.steer_lateral_gain,
    max_yaw_rate=s.steer_max_yaw_rate,
    line_profile=((0.0, 0.0), (length, 0.0)),
  )


def _stairs_line_profile(s: StairsScenarioCfg) -> tuple[tuple[float, float], ...]:
  """(x, z) tread profile along the staircase's centre line, origin-relative.

  Mirrors mjlab's (inverted) pyramid stairs: the env origin sits on the centre
  platform, every tread outward is one step_height lower (peak) or higher
  (pit), and the border frame plus the generator border sit at the outermost
  level. The polyline includes the risers, so it reads as a staircase.
  """
  inner = s.tile_size - 2 * _STAIRS_BORDER_WIDTH
  step_width = _stairs_step_width(s)
  num_steps = max(0, int((inner - s.platform_width) / (2 * step_width)))
  half_platform = 0.5 * inner - num_steps * step_width
  dz = -s.step_height if s.variant == "peak" else s.step_height
  # Past the tile the generator border is flat; show 1 m of it.
  x_end = 0.5 * s.tile_size + 1.0

  half = [(0.0, 0.0), (half_platform, 0.0)]
  for j in range(1, num_steps + 2):
    x = half_platform + (j - 1) * step_width
    x_next = x_end if j == num_steps + 1 else x + step_width
    half += [(x, j * dz), (x_next, j * dz)]
  mirrored = [(-x, z) for x, z in reversed(half[1:])]
  return tuple(mirrored + half)


def _stairs_step_width(s: StairsScenarioCfg) -> float:
  """s.step_width, raised to s.min_width_to_height * s.step_height if needed."""
  return max(s.step_width, s.min_width_to_height * s.step_height)


def _extreme_envs(
  vel_x: torch.Tensor, failed: torch.Tensor, survival_time: torch.Tensor
) -> torch.Tensor:
  """Indices (into the given envs) of the env with the highest peak forward
  velocity ([envs, steps], nan after an env ended), of the longest surviving
  one (the fastest of them when several lasted as long, e.g. to the end) and
  of the first one to fail (if any); each env once. plot-validate picks the
  same envs (``trace_extremes``)."""
  peak = vel_x.nan_to_num(float("-inf")).amax(1)
  ids = [int(peak.argmax())]
  longest = survival_time >= survival_time.max() - 1e-6
  ids.append(int(torch.where(longest, peak, -math.inf).argmax()))
  if failed.any():
    ids.append(int(torch.where(failed, survival_time, math.inf).argmin()))
  return torch.tensor(list(dict.fromkeys(ids)), device=vel_x.device)


def _stairs_finish_x(s: StairsScenarioCfg) -> float | None:
  """Origin-relative x the base must reach to finish: the outer edge of the
  last tread (the border frame starts there) plus s.finish_distance."""
  if s.finish_distance is None:
    return None
  return 0.5 * s.tile_size - _STAIRS_BORDER_WIDTH + s.finish_distance


def _configure_reset(env_cfg: ManagerBasedRlEnvCfg, cfg: ValidateConfig) -> None:
  reset_base = env_cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-cfg.init_pos_noise, cfg.init_pos_noise),
    "y": (-cfg.init_pos_noise, cfg.init_pos_noise),
    "z": (0.01, 0.02),
    "yaw": (-cfg.init_yaw_noise, cfg.init_yaw_noise),
  }
  reset_base.params["velocity_range"] = {}
  env_cfg.events["reset_robot_joints"].params["position_range"] = (
    -cfg.init_joint_noise,
    cfg.init_joint_noise,
  )
  env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
  # Play mode already drops push_robot; be explicit so a task that keeps it
  # doesn't shove the robot mid-ramp.
  env_cfg.events.pop("push_robot", None)
  if not cfg.domain_rand:
    # Encoder-bias events stay (kangaroo_full's biased joint observation term
    # looks its bias buffer up by event name) but draw a zero bias; every
    # other startup randomization goes.
    for name in list(env_cfg.events):
      term = env_cfg.events[name]
      if term.mode != "startup":
        continue
      if "bias_range" in term.params:
        term.params["bias_range"] = (0.0, 0.0)
      else:
        del env_cfg.events[name]


def build_scenario_env_cfg(
  task_id: str, cfg: ValidateConfig
) -> tuple[ManagerBasedRlEnvCfg, float, TerminationTermCfg | None]:
  """Play env cfg for task_id with the scenario's terrain, command and reset.

  Returns the cfg, the scenario duration in seconds, and the task's
  illegal_contacts termination term when it was taken out to be logged
  instead (None if it still terminates or the task has none).
  """
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  _set_integrator(env_cfg, cfg.integrator)
  _configure_reset(env_cfg, cfg)

  if cfg.scenario == "flat_ramp":
    s = cfg.flat_ramp
    _set_plane_terrain(env_cfg)
    _set_ramp_scenario_command(env_cfg, s)
    duration = _ramp_scenario_duration(s)
  elif cfg.scenario == "pebbles_ramp":
    s = cfg.pebbles_ramp
    num_boxes = int(round(s.box_density * s.tile_size[0] * s.tile_size[1]))
    _set_single_tile_terrain(
      env_cfg,
      random_spread_boxes(
        num_boxes=num_boxes,
        box_width_range=(0.02, 0.05),
        box_length_range=(0.02, 0.05),
        box_height_range=(0.02, 0.05),
        platform_width=0.5,
        border_width=0.0,
      ),
      size=s.tile_size,
      difficulty=s.difficulty,
      seed=cfg.seed,
    )
    _set_ramp_scenario_command(env_cfg, s)
    duration = _ramp_scenario_duration(s)
  elif cfg.scenario == "stairs":
    s = cfg.stairs
    if _stairs_step_width(s) > s.step_width:
      print(
        f"[INFO] step width {s.step_width:.3f} m raised to "
        f"{_stairs_step_width(s):.3f} m ({s.min_width_to_height}x the step height)"
      )
    make_stairs = pyramid_stairs_inv if s.variant == "pit" else pyramid_stairs
    _set_single_tile_terrain(
      env_cfg,
      make_stairs(
        step_height_range=(s.step_height, s.step_height),
        step_width=_stairs_step_width(s),
        platform_width=s.platform_width,
        border_width=_STAIRS_BORDER_WIDTH,
      ),
      size=(s.tile_size, s.tile_size),
      difficulty=1.0,
      seed=cfg.seed,
    )
    # No stand phase and no ramp: the command is s.velocity from t=0.
    _set_ramp_command(
      env_cfg,
      s.velocity,
      stand_time=0.0,
      ramp_time=0.0,
      line_heading_gain=s.steer_heading_gain if s.steer else 0.0,
      line_lateral_gain=s.steer_lateral_gain,
      max_yaw_rate=s.steer_max_yaw_rate,
      line_profile=_stairs_line_profile(s),
    )
    duration = s.duration
  else:
    raise ValueError(f"Unknown scenario: {cfg.scenario}")

  # The terrain curriculum is already gone in play mode; make sure nothing
  # else moves envs between tiles or reshapes the command mid-run.
  if env_cfg.curriculum is not None:
    env_cfg.curriculum.pop("terrain_levels", None)
    env_cfg.curriculum.pop("command_vel", None)
  illegal_contact_term = None
  if not cfg.illegal_contacts_terminate:
    illegal_contact_term = env_cfg.terminations.pop("illegal_contacts", None)
  if cfg.torso_contact_terminate:
    _add_torso_contact_termination(env_cfg)
  return env_cfg, duration, illegal_contact_term


def _add_torso_contact_termination(env_cfg: ManagerBasedRlEnvCfg) -> None:
  """Terminate on any contact between the torso capsule and the terrain."""
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="torso_ground_contact",
      primary=ContactMatch(
        mode="geom", pattern=rf"^{_TORSO_COLLISION_GEOM}$", entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  env_cfg.terminations["torso_contact"] = TerminationTermCfg(
    func=illegal_contact, params={"sensor_name": "torso_ground_contact"}
  )


##
# Policy loading (as in play).
##


def _resolve_checkpoint(cfg: ValidateConfig, agent_cfg) -> Path | None:
  """Local or W&B checkpoint path for a trained agent; None for dummy agents."""
  if cfg.agent in {"zero", "random"}:
    return None
  log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    return resume_path
  if cfg.wandb_run_path is None:
    raise ValueError(
      "`wandb_run_path` is required when `checkpoint_file` is not provided."
    )
  resume_path, was_cached = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  cached_str = "cached" if was_cached else "downloaded"
  print(
    f"[INFO]: Loading checkpoint: {resume_path.name} "
    f"(run: {resume_path.parent.name}, {cached_str})"
  )
  return resume_path


def _make_policy(
  task_id: str,
  cfg: ValidateConfig,
  env: RslRlVecEnvWrapper,
  agent_cfg,
  resume_path: Path | None,
  device: str,
):
  if resume_path is None:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      def policy(obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    else:

      def policy(obs) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    return policy

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return runner.get_inference_policy(device=device)


##
# Rollout.
##


def actuation_space(task_id: str) -> Literal["joint", "actuator", "lut"]:
  """Where the task's actuators act: on the robot's joints (the simple model
  and the "joint" transmission), on the actuator sliders and tendons driving
  them (the "actuator" transmission), or on the screws a LUT maps joint-space
  torques onto (the "lut" transmission, which has both)."""
  if "-Transmission-" not in task_id or "-Transmission-Joint-" in task_id:
    return "joint"
  if "-Transmission-Lut-" in task_id:
    return "lut"
  return "actuator"


@dataclass
class ActuationInfo:
  """Static actuation limits; bounds are [3 (position, velocity, force), n,
  2 (lo, hi)], NaN where unbounded, and DC motor parameters are [n, 3
  (effort_limit, saturation_effort, velocity_limit)], NaN without a
  torque-speed curve."""

  actuator_bounds: np.ndarray
  actuator_dc: np.ndarray
  joint_bounds: np.ndarray
  joint_dc: np.ndarray
  joint_actuated: np.ndarray
  """Joints an actuator drives directly (joint transmission)."""
  joint_actuator: np.ndarray
  """Index of the actuator driving each joint, -1 if none."""
  joint_gear: np.ndarray


def _actuation_info(raw: ManagerBasedRlEnv, robot) -> ActuationInfo:
  """Positions are bounded by the joint or tendon range behind each actuator,
  forces by the actuator forcerange (or the joint actuatorfrcrange),
  velocities by the DC motor velocity_limit (DcMotor actuators and the LUT
  transmission's DC motor screws). A joint driven by an actuator inherits the
  actuator's bounds through its gear."""
  m = raw.sim.mj_model
  act_ids = [m.actuator(a.name).id for a in robot.spec.actuators]
  jnt_ids = [
    m.joint(j.name).id
    for j in robot.spec.joints
    if j.type != mujoco.mjtJoint.mjJNT_FREE
  ]
  col_of_jnt = {j: c for c, j in enumerate(jnt_ids)}
  A, J = len(act_ids), len(jnt_ids)
  act = np.full((3, A, 2), np.nan)
  jnt = np.full((3, J, 2), np.nan)
  act_dc = np.full((A, 3), np.nan)
  jnt_dc = np.full((J, 3), np.nan)
  joint_actuator = np.full(J, -1)
  joint_gear = np.zeros(J)

  for a in robot.actuators:
    ids = a.ctrl_ids.cpu().numpy()
    if isinstance(a, LutTransmissionActuator):
      if a._velocity_limit_list is not None:
        cols = [robot.actuator_names.index(n) for n in a.screw_names]
        act_dc[cols] = np.stack(
          [
            a._screw_effort_limit_list,
            a._saturation_effort_list,
            a._velocity_limit_list,
          ],
          axis=-1,
        )
    elif isinstance(a.cfg, DcMotorActuatorCfg):
      act_dc[ids] = (a.cfg.effort_limit, a.cfg.saturation_effort, a.cfg.velocity_limit)

  for c, j in enumerate(jnt_ids):
    if m.jnt_limited[j]:
      jnt[0, c] = m.jnt_range[j]
    if m.jnt_actfrclimited[j]:
      jnt[2, c] = m.jnt_actfrcrange[j]
  for c, a in enumerate(act_ids):
    gear = float(m.actuator_gear[a, 0])
    if m.actuator_forcelimited[a]:
      act[2, c] = m.actuator_forcerange[a]
    act[1, c] = (-act_dc[c, 2], act_dc[c, 2])
    trn, tid = m.actuator_trntype[a], int(m.actuator_trnid[a, 0])
    if trn == mujoco.mjtTrn.mjTRN_JOINT:
      if m.jnt_limited[tid]:
        act[0, c] = np.sort(gear * m.jnt_range[tid])
      col = col_of_jnt.get(tid)
      if col is not None and gear != 0.0:
        joint_actuator[col] = c
        joint_gear[col] = gear
        g = abs(gear)
        jnt[1, col] = act[1, c] / g
        jnt_dc[col] = act_dc[c] * (g, g, 1 / g)
        if np.isnan(jnt[2, col, 0]):
          jnt[2, col] = act[2, c] * g
    elif trn == mujoco.mjtTrn.mjTRN_TENDON and m.tendon_limited[tid]:
      act[0, c] = np.sort(gear * m.tendon_range[tid])
  return ActuationInfo(
    actuator_bounds=act,
    actuator_dc=act_dc,
    joint_bounds=jnt,
    joint_dc=jnt_dc,
    joint_actuated=joint_actuator >= 0,
    joint_actuator=joint_actuator,
    joint_gear=joint_gear,
  )


class _LutProbe:
  """Keeps the intermediate values of a LutTransmissionActuator's last
  compute (the last physics substep of an env step): the joint-space PD
  torques and the positions/velocities they were taken at (map coordinates:
  the leg length is the actuator distance), the screw forces they map to
  before the motor clip, and the screw velocities fed to the DC motor
  curve."""

  def __init__(self, act: LutTransmissionActuator):
    self.act = act
    self.torque = self.pos = self.vel = None
    self.force = self.screw_vel = None
    joint_torques, screw_forces = act.joint_torques, act.screw_forces
    screw_velocities = act.screw_velocities

    def probe_joint_torques(cmd):
      out = joint_torques(cmd)
      self.torque, self.pos, self.vel, _ = out
      return out

    def probe_screw_forces(torque, pos, s):
      self.force = screw_forces(torque, pos, s)
      return self.force

    def probe_screw_velocities(pos, vel, s):
      self.screw_vel = screw_velocities(pos, vel, s)
      return self.screw_vel

    act.joint_torques = probe_joint_torques
    act.screw_forces = probe_screw_forces
    act.screw_velocities = probe_screw_velocities

  @property
  def joint_names(self) -> list[str]:
    leg = set(self.act._leg_length_cols_list)
    return [
      f"{n} (distance)" if i in leg else n for i, n in enumerate(self.act.servo_names)
    ]


class ScenarioRecorder:
  """Env wrapper that logs every step of the scenario.

  Sits on top of the RslRlVecEnvWrapper and exposes the same interface the
  mjlab viewers expect (``get_observations`` / ``step`` / ``reset`` /
  ``unwrapped``), so the headless loop and the viewers drive the very same
  bookkeeping: per-step aggregates over the envs still alive, and per-env
  outcomes at the first termination. A viewer-triggered ``reset`` restarts
  the recording from scratch.
  """

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    cfg: ValidateConfig,
    duration: float,
    illegal_contact_term: TerminationTermCfg | None = None,
  ):
    self.env = env
    self.illegal_contact_term = illegal_contact_term
    self.vcfg = cfg
    self.duration = duration
    raw = env.unwrapped
    self.robot = raw.scene["robot"]
    twist = raw.command_manager.get_term("twist")
    assert isinstance(twist, RampVelocityCommand)
    self.twist = twist
    self.step_dt = raw.step_dt
    self.num_steps = int(round(duration / self.step_dt))
    self.print_every = max(1, int(round(cfg.print_every_s / self.step_dt)))
    self.finish_x = _stairs_finish_x(cfg.stairs) if cfg.scenario == "stairs" else None

    self.reward_names = list(raw.reward_manager.active_terms)
    self.metric_names = list(raw.metrics_manager.active_terms)
    self.termination_names = list(raw.termination_manager.active_terms)
    self.state_names = [
      "base_lin_vel_x_b",
      "base_lin_vel_y_b",
      "base_ang_vel_z_b",
      "base_z_w",
      "lateral_error_y",
      "illegal_contact",
    ]
    # Torques: each actuator's scalar output (actuation space: N*m on a hinge,
    # N on a slider or tendon) and the actuators' force mapped onto every
    # joint DoF (qfrc_actuator; zero on passive joints the actuators don't
    # reach).
    self.actuator_names = list(self.robot.actuator_names)
    self.joint_names = list(self.robot.joint_names)
    # One DoF per joint (hinges and slides), so joint_pos, joint_vel and
    # qfrc_actuator columns line up with joint_names.
    assert self.robot.data.joint_vel.shape[-1] == len(self.joint_names)
    self.actuation = _actuation_info(raw, self.robot)
    self.lut_probes = [
      _LutProbe(a)
      for a in self.robot.actuators
      if isinstance(a, LutTransmissionActuator)
    ]
    self.lut_joint_names = [n for p in self.lut_probes for n in p.joint_names]
    N, T = raw.num_envs, self.num_steps
    R, M, S = len(self.reward_names), len(self.metric_names), len(self.state_names)
    A, J = len(self.actuator_names), len(self.joint_names)
    dev = raw.device

    # Per-step aggregates over the envs still alive at that step.
    self.times = torch.arange(T, device=dev, dtype=torch.float32) * self.step_dt
    # Forward command over the alive envs (the same for all of them on the
    # ramp scenarios; an env reset mid-run restarts its schedule, but it is
    # no longer alive by then).
    self.cmd_vel_x = torch.zeros(T, device=dev)
    self.cmd_vel_x_std = torch.zeros(T, device=dev)
    # The fastest forward command over the alive envs (nan once none is).
    self.cmd_vel_x_max = torch.zeros(T, device=dev)
    self.alive_count = torch.zeros(T, device=dev, dtype=torch.long)
    self.reward_mean = torch.zeros(T, R, device=dev)
    self.reward_std = torch.zeros(T, R, device=dev)
    self.metric_mean = torch.zeros(T, M, device=dev)
    self.metric_std = torch.zeros(T, M, device=dev)
    self.state_mean = torch.zeros(T, S, device=dev)
    self.state_std = torch.zeros(T, S, device=dev)
    # Signed lateral error averages out across envs; track its magnitude too.
    self.lateral_error_abs_mean = torch.zeros(T, device=dev)
    # Envs (still alive) touching the ground with an illegal body at each step;
    # only filled when illegal contacts are logged instead of terminating.
    self.illegal_contact_count = torch.zeros(T, device=dev, dtype=torch.long)
    # Torque mean/std and the largest magnitude over the alive envs.
    self.actuator_force_mean = torch.zeros(T, A, device=dev)
    self.actuator_force_std = torch.zeros(T, A, device=dev)
    self.actuator_force_abs_max = torch.zeros(T, A, device=dev)
    self.joint_torque_mean = torch.zeros(T, J, device=dev)
    self.joint_torque_std = torch.zeros(T, J, device=dev)
    self.joint_torque_abs_max = torch.zeros(T, J, device=dev)
    # Min and max over the alive envs of each position, velocity and force:
    # [steps, 2 (min, max), 3 (position, velocity, force), actuators|joints].
    self.actuator_envelope = torch.zeros(T, 2, 3, A, device=dev)
    self.joint_envelope = torch.zeros(T, 2, 3, J, device=dev)

    # Per-env outcomes.
    self.alive = torch.ones(N, device=dev, dtype=torch.bool)
    self.survival_time = torch.full((N,), duration, device=dev)
    self.vel_at_failure = torch.full((N,), float("nan"), device=dev)
    # When each env's ramp started (nan if it failed before the ramp).
    self.ramp_start_time = torch.full((N,), float("nan"), device=dev)
    # Each env's forward base velocity after every step it came through (nan
    # after it ended), to plot the tracking of the envs that didn't fail.
    self.vel_x_per_env = torch.full((N, T), float("nan"), device=dev)
    # Its lateral base velocity alongside, for the planar speed |v_xy|.
    self.vel_y_per_env = torch.full((N, T), float("nan"), device=dev)
    self.termination_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
    self.truncated_only = torch.zeros(N, device=dev, dtype=torch.bool)
    # Envs that reached finish_x (stairs only), and when (nan if never).
    self.finished_env = torch.zeros(N, device=dev, dtype=torch.bool)
    self.finish_time = torch.full((N,), float("nan"), device=dev)
    self.reward_sum = torch.zeros(N, R, device=dev)
    self.metric_sum = torch.zeros(N, M, device=dev)
    self.metric_max = torch.full((N, M), float("-inf"), device=dev)
    self.alive_steps = torch.zeros(N, device=dev, dtype=torch.long)
    # Lateral error over the steps each env survived (state is post-step).
    self.lateral_abs_sum = torch.zeros(N, device=dev)
    self.lateral_abs_max = torch.zeros(N, device=dev)
    self.lateral_final = torch.zeros(N, device=dev)
    self.lateral_steps = torch.zeros(N, device=dev, dtype=torch.long)
    # Illegal contacts per env: steps in contact, separate contact events
    # (onsets), and the time of the first one (nan if none).
    self.illegal_contact_steps = torch.zeros(N, device=dev, dtype=torch.long)
    self.illegal_contact_events = torch.zeros(N, device=dev, dtype=torch.long)
    self.illegal_contact_first = torch.full((N,), float("nan"), device=dev)
    self.illegal_contact_prev = torch.zeros(N, device=dev, dtype=torch.bool)
    # Torques per env over the steps it was alive: sum of squares (for the
    # RMS) and peak magnitude.
    self.actuator_force_sq_sum = torch.zeros(N, A, device=dev)
    self.actuator_force_abs_max_env = torch.zeros(N, A, device=dev)
    self.joint_torque_sq_sum = torch.zeros(N, J, device=dev)
    self.joint_torque_abs_max_env = torch.zeros(N, J, device=dev)

    self.per_env_steps: torch.Tensor | None = None
    if cfg.save_per_env_steps:
      n_logged = (T + cfg.log_every - 1) // cfg.log_every
      self.per_env_steps = torch.zeros(N, n_logged, S + R + M, device=dev)
    # Position, velocity, force traces of the first trace_envs envs. They
    # live in host memory: at --trace-envs 4096 they are [4096, T, ~480]
    # floats on full_full (~10 GB at 1250 steps), which on top of the
    # simulation overflows the GPU (the LUT variant first).
    K = min(max(cfg.trace_envs, 0), N)
    tdev = "cpu"
    self.trace_actuator = torch.zeros(K, T, 3, A, device=tdev)
    self.trace_joint = torch.zeros(K, T, 3, J, device=tdev)
    # Base pose: position relative to the env origin, then the wxyz quaternion.
    self.trace_root = torch.zeros(K, T, 7, device=tdev)
    # Force limits in effect at each step (lo, hi): the forcerange, narrowed
    # by the DC motor torque-speed curve at the step's velocity.
    self.trace_actuator_limit = torch.zeros(K, T, 2, A, device=tdev)
    self.trace_joint_limit = torch.zeros(K, T, 2, J, device=tdev)
    # LUT transmission only: the screw forces before the motor clip, and the
    # joint-space position, velocity and PD torque the LUT maps from.
    L = len(self.lut_joint_names)
    self.trace_actuator_request = torch.zeros(K, T, A, device=tdev)
    self.trace_lut_joint = torch.zeros(K, T, 3, L, device=tdev)
    info = self.actuation
    self._act_force_hi = torch.tensor(
      np.nan_to_num(info.actuator_bounds[2, :, 1], nan=np.inf),
      device=dev,
      dtype=torch.float32,
    )
    self._act_dc = torch.tensor(info.actuator_dc, device=dev, dtype=torch.float32)
    self._act_dc_cols = torch.tensor(
      np.flatnonzero(~np.isnan(info.actuator_dc[:, 0])), device=dev
    )
    self._jnt_force_bounds = torch.tensor(
      info.joint_bounds[2].T, device=dev, dtype=torch.float32
    )
    driven = np.flatnonzero(info.joint_actuated)
    self._jnt_driven = torch.tensor(driven, device=dev)
    self._jnt_driver = torch.tensor(info.joint_actuator[driven], device=dev)
    self._jnt_gear = torch.tensor(
      info.joint_gear[driven], device=dev, dtype=torch.float32
    )

    self._start()

  # EnvProtocol.

  @property
  def num_envs(self) -> int:
    return self.env.num_envs

  @property
  def device(self):
    return self.env.device

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.env.cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  def get_observations(self):
    return self.env.get_observations()

  def reset(self):
    out = self.env.reset()
    print("[INFO] env reset: restarting the scenario recording")
    self._start()
    return out

  def close(self) -> None:
    self.env.close()

  def step(self, actions: torch.Tensor):
    # Envs alive going into this step: their reward and termination at this
    # step are still on the scenario, even for the ones that fall now.
    mask = self.alive.clone()
    # The forward command these actions answer (a done env's command is
    # already its post-reset one after the step).
    cmd_vel_x = self.twist.vel_command_b[:, 0].clone()
    out = self.env.step(actions)
    if self.k < self.num_steps and not self.finished:
      self._record(self.k, mask, out[2].bool(), cmd_vel_x)
      self.k += 1
    return out

  # Recording.

  @property
  def finished(self) -> bool:
    return self.k >= self.num_steps or self._all_done

  @property
  def steps_recorded(self) -> int:
    return self.k

  def _start(self) -> None:
    self.k = 0
    self._all_done = False
    self.wall_start = time.time()
    for buf in (
      self.cmd_vel_x,
      self.cmd_vel_x_std,
      self.cmd_vel_x_max,
      self.alive_count,
      self.reward_mean,
      self.reward_std,
      self.metric_mean,
      self.metric_std,
      self.state_mean,
      self.state_std,
      self.lateral_error_abs_mean,
      self.reward_sum,
      self.metric_sum,
      self.alive_steps,
      self.lateral_abs_sum,
      self.lateral_abs_max,
      self.lateral_final,
      self.lateral_steps,
      self.illegal_contact_count,
      self.illegal_contact_steps,
      self.illegal_contact_events,
      self.actuator_force_mean,
      self.actuator_force_std,
      self.actuator_force_abs_max,
      self.joint_torque_mean,
      self.joint_torque_std,
      self.joint_torque_abs_max,
      self.actuator_envelope,
      self.joint_envelope,
      self.actuator_force_sq_sum,
      self.actuator_force_abs_max_env,
      self.joint_torque_sq_sum,
      self.joint_torque_abs_max_env,
    ):
      buf.zero_()
    self.illegal_contact_first.fill_(float("nan"))
    self.illegal_contact_prev.fill_(False)
    self.alive.fill_(True)
    self.survival_time.fill_(self.duration)
    self.vel_at_failure.fill_(float("nan"))
    self.ramp_start_time.fill_(float("nan"))
    self.vel_x_per_env.fill_(float("nan"))
    self.vel_y_per_env.fill_(float("nan"))
    self.termination_idx.fill_(-1)
    self.truncated_only.fill_(False)
    self.finished_env.fill_(False)
    self.finish_time.fill_(float("nan"))
    self.metric_max.fill_(float("-inf"))
    if self.per_env_steps is not None:
      self.per_env_steps.zero_()
    for buf in (
      self.trace_actuator,
      self.trace_joint,
      self.trace_root,
      self.trace_actuator_limit,
      self.trace_joint_limit,
      self.trace_actuator_request,
      self.trace_lut_joint,
    ):
      buf.fill_(float("nan"))
    # The wrapper has just reset: the robots stand at INIT_STATE and their
    # command schedules start counting from here.
    self.start_pos = self.robot.data.root_link_pos_w.clone()
    self.last_pos = self.start_pos.clone()

  def _state(self) -> torch.Tensor:
    data = self.robot.data
    return torch.stack(
      [
        data.root_link_lin_vel_b[:, 0],
        data.root_link_lin_vel_b[:, 1],
        data.root_link_ang_vel_b[:, 2],
        data.root_link_pos_w[:, 2],
        self._lateral_error(),
        self._illegal_contact().to(data.root_link_pos_w.dtype),
      ],
      dim=-1,
    )

  def _illegal_contact(self) -> torch.Tensor:
    """The task's illegal_contacts condition, evaluated without terminating;
    all False when it still terminates (or the task has none)."""
    term = self.illegal_contact_term
    raw = self.env.unwrapped
    if term is None:
      return torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)
    return term.func(raw, **term.params).bool()

  def _lateral_error(self) -> torch.Tensor:
    """Signed offset [m] from the line through the env origin along +x: the
    line the stairs steering follows (left of it is positive)."""
    raw = self.env.unwrapped
    return self.robot.data.root_link_pos_w[:, 1] - raw.scene.env_origins[:, 1]

  @staticmethod
  def _masked_stats(x: torch.Tensor, mask: torch.Tensor):
    n = mask.sum().clamp(min=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    mean = (x * m).sum(0) / n
    var = (((x - mean) ** 2) * m).sum(0) / n
    return mean, var.sqrt()

  @torch.no_grad()
  def _record(
    self, k: int, mask: torch.Tensor, dones: torch.Tensor, cmd_vel_x: torch.Tensor
  ) -> None:
    raw = self.env.unwrapped
    done_now = dones & mask
    step_rew = raw.reward_manager._step_reward
    step_met = raw.metrics_manager._step_values
    self.alive_count[k] = mask.sum()
    cmd_mean, cmd_std = self._masked_stats(cmd_vel_x.unsqueeze(-1), mask)
    self.cmd_vel_x[k], self.cmd_vel_x_std[k] = cmd_mean[0], cmd_std[0]
    self.cmd_vel_x_max[k] = cmd_vel_x[mask].max() if mask.any() else float("nan")
    # Only envs that came through the step: a done env's ramp start is
    # already the post-reset one.
    started = (
      mask
      & ~done_now
      & self.ramp_start_time.isnan()
      & (self.twist.ramp_start_t <= self.times[k] + self.step_dt)
    )
    self.ramp_start_time[started] = self.twist.ramp_start_t[started]
    self.reward_mean[k], self.reward_std[k] = self._masked_stats(step_rew, mask)
    self.metric_mean[k], self.metric_std[k] = self._masked_stats(step_met, mask)
    self.reward_sum[mask] += step_rew[mask] * self.step_dt
    self.metric_sum[mask] += step_met[mask]
    self.metric_max[mask] = torch.maximum(self.metric_max[mask], step_met[mask])
    self.alive_steps[mask] += 1
    # An env done at this step already holds its post-reset state, so the
    # actuation is logged only for the envs that came through the step.
    self._record_actuation(k, mask & ~done_now)

    if done_now.any():
      # Read the termination buffers before anything overwrites them: they
      # hold this step's values until the next termination compute.
      self.survival_time[done_now] = (k + 1) * self.step_dt
      self.vel_at_failure[done_now] = cmd_vel_x[done_now]
      terminated = raw.termination_manager.terminated
      for i, name in enumerate(self.termination_names):
        fired = raw.termination_manager.get_term(name) & done_now
        first = fired & (self.termination_idx < 0)
        self.termination_idx[first] = i
      self.truncated_only[done_now] = ~terminated[done_now]
      self.alive &= ~done_now

    # State after the step, for envs that survived it (a done env's state
    # is already its post-reset one).
    state = self._state()
    self.state_mean[k], self.state_std[k] = self._masked_stats(state, self.alive)
    vel_x = state[:, self.state_names.index("base_lin_vel_x_b")]
    self.vel_x_per_env[self.alive, k] = vel_x[self.alive]
    vel_y = state[:, self.state_names.index("base_lin_vel_y_b")]
    self.vel_y_per_env[self.alive, k] = vel_y[self.alive]
    self.last_pos[self.alive] = self.robot.data.root_link_pos_w[self.alive]
    lat = state[:, self.state_names.index("lateral_error_y")]
    lat_abs = lat.abs()
    self.lateral_error_abs_mean[k] = self._masked_stats(
      lat_abs.unsqueeze(-1), self.alive
    )[0].squeeze(-1)
    self.lateral_abs_sum[self.alive] += lat_abs[self.alive]
    self.lateral_abs_max[self.alive] = torch.maximum(
      self.lateral_abs_max[self.alive], lat_abs[self.alive]
    )
    self.lateral_final[self.alive] = lat[self.alive]
    self.lateral_steps[self.alive] += 1
    contact = state[:, self.state_names.index("illegal_contact")].bool() & self.alive
    onset = contact & ~self.illegal_contact_prev
    self.illegal_contact_count[k] = contact.sum()
    self.illegal_contact_steps += contact
    self.illegal_contact_events += onset
    first = onset & self.illegal_contact_first.isnan()
    self.illegal_contact_first[first] = (k + 1) * self.step_dt
    self.illegal_contact_prev = contact

    if self.finish_x is not None:
      x = self.robot.data.root_link_pos_w[:, 0] - raw.scene.env_origins[:, 0]
      finished_now = self.alive & (x >= self.finish_x)
      self.finished_env |= finished_now
      self.finish_time[finished_now] = (k + 1) * self.step_dt
      self.survival_time[finished_now] = (k + 1) * self.step_dt
      self.alive &= ~finished_now

    if self.per_env_steps is not None and k % self.vcfg.log_every == 0:
      row = torch.cat([state, step_rew, step_met], dim=-1)
      row[~mask] = float("nan")
      self.per_env_steps[:, k // self.vcfg.log_every] = row

    self._all_done = not bool(self.alive.any())
    if (k + 1) % self.print_every == 0 or k == self.num_steps - 1 or self._all_done:
      elapsed = time.time() - self.wall_start
      print(
        f"  sim t={self.times[k] + self.step_dt:5.1f}s "
        f"cmd_vx={self.cmd_vel_x[k]:.2f}±{self.cmd_vel_x_std[k]:.2f} "
        f"alive={int(self.alive.sum())}/{self.num_envs} "
        f"ramping={int((self.alive & ~self.ramp_start_time.isnan()).sum())} "
        f"vx={self.state_mean[k, 0]:.2f}±{self.state_std[k, 0]:.2f} "
        f"[wall-clock elapsed {elapsed:.0f}s, display only]"
      )
    if self._all_done:
      print("[INFO] every env has finished or terminated; recording stops here")

  def _record_actuation(self, k: int, mask: torch.Tensor) -> None:
    """Torques applied during step k (the last physics substep's values) over
    the envs in ``mask``, plus the traces of the first trace_envs envs."""
    data = self.robot.data
    K = self.trace_actuator.shape[0]
    if K:
      ctrl_ids = self.robot.indexing.ctrl_ids
      act = torch.stack(
        [
          data.data.actuator_length[:K, ctrl_ids],
          data.data.actuator_velocity[:K, ctrl_ids],
          data.actuator_force[:K],
        ],
        dim=1,
      )
      jnt = torch.stack(
        [data.joint_pos[:K], data.joint_vel[:K], data.qfrc_actuator[:K]], dim=1
      )
      root = torch.cat(
        [
          data.root_link_pos_w[:K] - self.env.unwrapped.scene.env_origins[:K],
          data.root_link_quat_w[:K],
        ],
        dim=-1,
      )
      keep = mask[:K].cpu()
      self.trace_actuator[keep, k] = act.cpu()[keep]
      self.trace_joint[keep, k] = jnt.cpu()[keep]
      self.trace_root[keep, k] = root.cpu()[keep]
      self._record_limits(k, keep, act[:, 1])
    ctrl_ids = self.robot.indexing.ctrl_ids
    act_all = (
      data.data.actuator_length[:, ctrl_ids],
      data.data.actuator_velocity[:, ctrl_ids],
      data.actuator_force,
    )
    jnt_all = (data.joint_pos, data.joint_vel, data.qfrc_actuator)
    for x, envelope in (
      (torch.stack(act_all, dim=1), self.actuator_envelope),
      (torch.stack(jnt_all, dim=1), self.joint_envelope),
    ):
      if mask.any():
        envelope[k, 0], envelope[k, 1] = x[mask].amin(0), x[mask].amax(0)
      else:
        envelope[k] = float("nan")
    for x, mean, std, peak, sq_sum, peak_env in (
      (
        data.actuator_force,
        self.actuator_force_mean,
        self.actuator_force_std,
        self.actuator_force_abs_max,
        self.actuator_force_sq_sum,
        self.actuator_force_abs_max_env,
      ),
      (
        data.qfrc_actuator,
        self.joint_torque_mean,
        self.joint_torque_std,
        self.joint_torque_abs_max,
        self.joint_torque_sq_sum,
        self.joint_torque_abs_max_env,
      ),
    ):
      mean[k], std[k] = self._masked_stats(x, mask)
      x_abs = x.abs()
      if mask.any():
        peak[k] = x_abs[mask].amax(0)
      else:
        peak[k] = float("nan")
      sq_sum[mask] += x[mask] ** 2
      peak_env[mask] = torch.maximum(peak_env[mask], x_abs[mask])

  def _record_limits(self, k: int, keep: torch.Tensor, act_vel: torch.Tensor):
    """Force limits in effect at step k for the traced envs, and the LUT
    transmission's intermediate values. ``keep`` is on the host, like the
    traces."""
    K, A = act_vel.shape
    dc_vel = act_vel.clone()
    request = torch.full_like(act_vel, float("nan"))
    lut_rows = []
    for p in self.lut_probes:
      ids = p.act.ctrl_ids
      request[:, ids] = p.force[:K]
      if p.screw_vel is not None:
        # The curve runs on the LUT's own screw velocity estimate.
        dc_vel[:, ids] = p.screw_vel[:K]
      lut_rows.append(torch.stack([p.pos[:K], p.vel[:K], p.torque[:K]], dim=1))

    hi = self._act_force_hi.expand(K, A).clone()
    lo = -hi
    cols = self._act_dc_cols
    if len(cols):
      F, sat, vl = self._act_dc[cols].unbind(-1)
      v = dc_vel[:, cols]
      inf = torch.full_like(v, float("inf"))
      hi[:, cols] = dc_motor_clip(inf, sat, vl, F, v)
      lo[:, cols] = dc_motor_clip(-inf, sat, vl, F, v)
    act_lim = torch.stack([lo, hi], dim=1)
    jnt_lim = self._jnt_force_bounds.expand(K, 2, -1).clone()
    if len(self._jnt_driven):
      g = self._jnt_gear
      lim = act_lim[:, :, self._jnt_driver] * g
      jnt_lim[:, :, self._jnt_driven] = torch.where(g < 0, lim.flip(1), lim)

    self.trace_actuator_limit[keep, k] = act_lim.cpu()[keep]
    self.trace_joint_limit[keep, k] = jnt_lim.cpu()[keep]
    self.trace_actuator_request[keep, k] = request.cpu()[keep]
    if lut_rows:
      self.trace_lut_joint[keep, k] = torch.cat(lut_rows, dim=-1).cpu()[keep]

  def results(self) -> dict[str, torch.Tensor]:
    n = self.k
    if n < self.num_steps and not self._all_done:
      print(
        f"[WARN] scenario stopped after {n}/{self.num_steps} steps "
        "(viewer closed?); saving the partial recording"
      )
    # Envs still alive survived exactly as long as was recorded.
    survival_time = torch.where(
      self.alive,
      torch.full_like(self.survival_time, n * self.step_dt),
      self.survival_time,
    )
    displacement = self.last_pos - self.start_pos
    out = {
      "time": self.times[:n],
      "cmd_vel_x": self.cmd_vel_x[:n],
      "cmd_vel_x_std": self.cmd_vel_x_std[:n],
      "cmd_vel_x_max": self.cmd_vel_x_max[:n],
      "vel_x_per_env": self.vel_x_per_env[:, :n],
      "vel_y_per_env": self.vel_y_per_env[:, :n],
      "ramp_start_time": self.ramp_start_time,
      "alive_count": self.alive_count[:n],
      "reward_mean": self.reward_mean[:n],
      "reward_std": self.reward_std[:n],
      "metric_mean": self.metric_mean[:n],
      "metric_std": self.metric_std[:n],
      "state_mean": self.state_mean[:n],
      "state_std": self.state_std[:n],
      "lateral_error_abs_mean": self.lateral_error_abs_mean[:n],
      "illegal_contact_count": self.illegal_contact_count[:n],
      "survival_time": survival_time,
      "vel_at_failure": self.vel_at_failure,
      "failed": (self.termination_idx >= 0) & ~self.truncated_only,
      "truncated": self.truncated_only,
      "finished": self.finished_env,
      "finish_time": self.finish_time,
      "termination_idx": self.termination_idx,
      "reward_sum": self.reward_sum,
      "metric_mean_per_env": self.metric_sum
      / self.alive_steps.clamp(min=1).unsqueeze(-1),
      "metric_max_per_env": self.metric_max,
      "alive_steps": self.alive_steps,
      "distance_xy": displacement[:, :2].norm(dim=-1),
      "distance_x": displacement[:, 0],
      "height_gain": displacement[:, 2],
      "lateral_error_abs_mean_per_env": self.lateral_abs_sum
      / self.lateral_steps.clamp(min=1),
      "lateral_error_abs_max_per_env": self.lateral_abs_max,
      "lateral_error_final_per_env": self.lateral_final,
      "illegal_contact_steps_per_env": self.illegal_contact_steps,
      "illegal_contact_events_per_env": self.illegal_contact_events,
      "illegal_contact_first_time": self.illegal_contact_first,
      "illegal_contact_logged": torch.tensor(self.illegal_contact_term is not None),
      "actuator_force_mean": self.actuator_force_mean[:n],
      "actuator_force_std": self.actuator_force_std[:n],
      "actuator_force_abs_max": self.actuator_force_abs_max[:n],
      "joint_torque_mean": self.joint_torque_mean[:n],
      "joint_torque_std": self.joint_torque_std[:n],
      "joint_torque_abs_max": self.joint_torque_abs_max[:n],
      "actuator_envelope": self.actuator_envelope[:n],
      "joint_envelope": self.joint_envelope[:n],
      "actuator_force_rms_per_env": (
        self.actuator_force_sq_sum / self.alive_steps.clamp(min=1).unsqueeze(-1)
      ).sqrt(),
      "actuator_force_abs_max_per_env": self.actuator_force_abs_max_env,
      "joint_torque_rms_per_env": (
        self.joint_torque_sq_sum / self.alive_steps.clamp(min=1).unsqueeze(-1)
      ).sqrt(),
      "joint_torque_abs_max_per_env": self.joint_torque_abs_max_env,
    }
    if self.per_env_steps is not None:
      n_logged = (n + self.vcfg.log_every - 1) // self.vcfg.log_every
      out["per_env_steps"] = self.per_env_steps[:, :n_logged]
    if self.trace_actuator.shape[0]:
      ids = torch.arange(self.trace_actuator.shape[0], device=self.alive.device)
      if self.vcfg.trace_keep == "extremes":
        ids = _extreme_envs(
          self.vel_x_per_env[ids, :n], out["failed"][ids], survival_time[ids]
        )
      out["trace_env_ids"] = ids
      ids = ids.cpu()
      out["trace_actuator"] = self.trace_actuator[ids, :n]
      out["trace_joint"] = self.trace_joint[ids, :n]
      out["trace_root"] = self.trace_root[ids, :n]
      out["trace_actuator_limit"] = self.trace_actuator_limit[ids, :n]
      out["trace_joint_limit"] = self.trace_joint_limit[ids, :n]
      if self.lut_probes:
        out["trace_actuator_request"] = self.trace_actuator_request[ids, :n]
        out["trace_lut_joint"] = self.trace_lut_joint[ids, :n]
    return out


class _StopWhenRecordingFinished:
  """Viewer mixin: stop as soon as the recorder has finished (every env done
  or the scenario over), not only after the viewer's own step budget; viser
  otherwise runs until killed."""

  env: ScenarioRecorder

  def is_running(self) -> bool:
    return super().is_running() and not self.env.finished  # type: ignore[misc]


class _NativeViewer(_StopWhenRecordingFinished, NativeMujocoViewer):
  pass


class _ViserViewer(_StopWhenRecordingFinished, ViserPlayViewer):
  pass


def run_validate(task_id: str, cfg: ValidateConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.video and cfg.viewer != "viser":
    raise ValueError("--video needs --viewer viser")
  if cfg.train_task is not None and cfg.train_task not in list_tasks():
    raise ValueError(f"--train-task {cfg.train_task} is not a registered task")
  if cfg.train_task not in (None, task_id):
    print(f"[INFO] end-to-end run: trained on {cfg.train_task}, validated on {task_id}")

  env_cfg, duration, illegal_contact_term = build_scenario_env_cfg(task_id, cfg)
  agent_cfg = load_rl_cfg(task_id)
  resume_path = _resolve_checkpoint(cfg, agent_cfg)
  out_path = _output_path(cfg, resume_path.parent if resume_path else None)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )
  num_steps = int(round(duration / env.step_dt))
  if cfg.video:
    # Same offscreen path play uses: env 0, one frame per env step, starting
    # at the first step; the file lands next to the .npz.
    env = VideoRecorder(
      env,
      video_folder=out_path.parent,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length or num_steps,
      name_prefix=out_path.stem,
      disable_logger=False,
    )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = _make_policy(task_id, cfg, env, agent_cfg, resume_path, device)
  rec = ScenarioRecorder(env, cfg, duration, illegal_contact_term)

  print(
    f"[INFO] scenario={cfg.scenario} envs={rec.num_envs} steps={rec.num_steps} "
    f"({duration:.1f} s @ {1 / rec.step_dt:.0f} Hz) device={device} "
    f"viewer={cfg.viewer}"
  )
  if cfg.viewer == "none":
    obs = rec.get_observations()
    with torch.inference_mode():
      while not rec.finished:
        obs, *_ = rec.step(policy(obs))
  elif cfg.viewer == "native":
    _NativeViewer(rec, policy).run(num_steps=rec.num_steps)
  elif cfg.viewer == "viser":
    _ViserViewer(rec, policy).run(num_steps=rec.num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {cfg.viewer}")

  results = rec.results()
  rec.close()  # Finalizes the video, if any.
  if cfg.video:
    recorded = out_path.parent / f"{out_path.stem}-step-0.mp4"
    if recorded.exists():
      recorded.replace(out_path.with_suffix(".mp4"))
      print(f"[INFO] video: {out_path.with_suffix('.mp4')}")
  _save(out_path, task_id, cfg, results, rec)
  return out_path


def _output_path(cfg: ValidateConfig, log_dir: Path | None) -> Path:
  if cfg.output is not None:
    return Path(cfg.output)
  root = (log_dir / "validate") if log_dir is not None else Path("logs/validate")
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = cfg.scenario
  if cfg.scenario == "stairs":
    name += f"_{cfg.stairs.variant}"
  return root / f"{name}_{stamp}.npz"


def _lut_joint_bounds(rec: ScenarioRecorder) -> np.ndarray:
  """[3, n, 2] bounds of the LUT joint-space signals: only the PD torque
  clamp; the map-coordinate positions and velocities are unbounded here."""
  limit = np.array([v for p in rec.lut_probes for v in p.act._effort_limit_list])
  bounds = np.full((3, len(limit), 2), np.nan)
  bounds[2] = np.stack([-limit, limit], axis=-1)
  return bounds


def _pooled_rms(
  rms_per_env: np.ndarray, steps: np.ndarray, names: list[str]
) -> dict[str, float]:
  w = steps.astype(np.float64)[:, None]
  pooled = np.sqrt((rms_per_env**2 * w).sum(0) / max(w.sum(), 1.0))
  return dict(zip(names, pooled.tolist(), strict=True))


def _save(
  out_path: Path,
  task_id: str,
  cfg: ValidateConfig,
  results: dict[str, torch.Tensor],
  rec: ScenarioRecorder,
) -> None:
  reward_names, metric_names = rec.reward_names, rec.metric_names
  termination_names, state_names = rec.termination_names, rec.state_names
  actuator_names, joint_names = rec.actuator_names, rec.joint_names
  step_dt = rec.step_dt
  out_path.parent.mkdir(parents=True, exist_ok=True)
  arrays = {k: v.detach().cpu().numpy() for k, v in results.items()}
  np.savez_compressed(
    out_path,
    reward_names=np.array(reward_names),
    metric_names=np.array(metric_names),
    termination_names=np.array(termination_names),
    state_names=np.array(state_names),
    actuator_names=np.array(actuator_names),
    joint_names=np.array(joint_names),
    actuation_space=np.array(actuation_space(task_id)),
    **asdict(rec.actuation),
    lut_joint_names=np.array(rec.lut_joint_names),
    lut_joint_bounds=_lut_joint_bounds(rec),
    **arrays,
  )

  failed = arrays["failed"]
  n = len(failed)
  summary = {
    "task_id": task_id,
    "train_task_id": cfg.train_task or task_id,
    "end_to_end": cfg.train_task not in (None, task_id),
    "config": asdict(cfg),
    "step_dt": step_dt,
    "num_envs": n,
    "num_steps": int(len(arrays["time"])),
    "failure_rate": float(failed.mean()),
    "truncation_rate": float(arrays["truncated"].mean()),
    "finish_rate": float(arrays["finished"].mean()),
    "finish_time_mean": (
      float(np.nanmean(arrays["finish_time"])) if arrays["finished"].any() else None
    ),
    "survival_time_mean": float(arrays["survival_time"].mean()),
    "survival_time_min": float(arrays["survival_time"].min()),
    "distance_x_mean": float(arrays["distance_x"].mean()),
    "height_gain_mean": float(arrays["height_gain"].mean()),
    "lateral_error_abs_mean": float(arrays["lateral_error_abs_mean_per_env"].mean()),
    "lateral_error_abs_max": float(arrays["lateral_error_abs_max_per_env"].max()),
    "illegal_contact_logged": bool(arrays["illegal_contact_logged"]),
    "illegal_contact_envs": int((arrays["illegal_contact_events_per_env"] > 0).sum()),
    "illegal_contact_events": int(arrays["illegal_contact_events_per_env"].sum()),
    "illegal_contact_time_mean": float(
      arrays["illegal_contact_steps_per_env"].mean() * step_dt
    ),
    "vel_at_failure_mean": (
      float(np.nanmean(arrays["vel_at_failure"])) if failed.any() else None
    ),
    "ramp_started_rate": float((~np.isnan(arrays["ramp_start_time"])).mean()),
    "ramp_start_time_mean": (
      float(np.nanmean(arrays["ramp_start_time"]))
      if (~np.isnan(arrays["ramp_start_time"])).any()
      else None
    ),
    "terminations": {
      name: int((arrays["termination_idx"] == i).sum())
      for i, name in enumerate(termination_names)
    },
    "reward_sum_mean": dict(
      zip(reward_names, arrays["reward_sum"].mean(0).tolist(), strict=True)
    ),
    "metric_mean": dict(
      zip(
        metric_names,
        np.nanmean(arrays["metric_mean_per_env"], axis=0).tolist(),
        strict=True,
      )
    ),
    # Torques over every env and step it was alive: the RMS pooled across
    # envs (weighted by steps alive) and the peak magnitude.
    "actuator_force_rms": _pooled_rms(
      arrays["actuator_force_rms_per_env"], arrays["alive_steps"], actuator_names
    ),
    "actuator_force_abs_max": dict(
      zip(
        actuator_names,
        arrays["actuator_force_abs_max_per_env"].max(0, initial=0.0).tolist(),
        strict=True,
      )
    ),
    "joint_torque_rms": _pooled_rms(
      arrays["joint_torque_rms_per_env"], arrays["alive_steps"], joint_names
    ),
    "joint_torque_abs_max": dict(
      zip(
        joint_names,
        arrays["joint_torque_abs_max_per_env"].max(0, initial=0.0).tolist(),
        strict=True,
      )
    ),
  }
  json_path = out_path.with_suffix(".json")
  json_path.write_text(json.dumps(summary, indent=2))

  print(f"[INFO] saved {out_path} and {json_path}")
  print(
    f"[RESULT] failed {int(failed.sum())}/{n} ({100 * failed.mean():.1f}%), "
    f"mean survival {summary['survival_time_mean']:.2f} s, "
    f"mean distance x {summary['distance_x_mean']:.2f} m, "
    f"lateral error mean |y| {summary['lateral_error_abs_mean']:.3f} m "
    f"(max {summary['lateral_error_abs_max']:.3f} m)"
  )
  if cfg.scenario == "stairs" and cfg.stairs.finish_distance is not None:
    finished = arrays["finished"]
    line = f"[RESULT] finished {int(finished.sum())}/{n} ({100 * finished.mean():.1f}%)"
    if finished.any():
      line += f", mean time {summary['finish_time_mean']:.2f} s"
    print(line)
  ramp = {"flat_ramp": cfg.flat_ramp, "pebbles_ramp": cfg.pebbles_ramp}.get(
    cfg.scenario
  )
  if ramp is not None and ramp.start_vel > 0.0:
    started = ~np.isnan(arrays["ramp_start_time"])
    if ramp.start_after is not None:
      line = (
        f"[RESULT] ramp start ({ramp.start_after:.1f} s at "
        f"{ramp.start_vel:.2f} m/s, no wait for the envs): "
      )
    else:
      line = (
        f"[RESULT] ramp start (once {100 * ramp.start_fraction:.0f}% of alive "
        f"envs reached {100 * ramp.start_reach_fraction:.0f}% of "
        f"{ramp.start_vel:.2f} m/s): "
      )
    if started.any():
      t0 = float(np.nanmin(arrays["ramp_start_time"]))
      line += f"at {t0:.2f} s, {int(started.sum())}/{n} envs alive and ramped"
    else:
      line += "never"
    print(line)
  if failed.any():
    v = arrays["vel_at_failure"][failed]
    print(
      f"[RESULT] command speed at failure: mean {v.mean():.2f} "
      f"min {v.min():.2f} max {v.max():.2f} m/s"
    )
  for name, count in summary["terminations"].items():
    if count:
      print(f"[RESULT]   {name}: {count}")
  if summary["illegal_contact_logged"]:
    print(
      f"[RESULT] illegal contacts (logged, not terminating): "
      f"{summary['illegal_contact_envs']}/{n} envs, "
      f"{summary['illegal_contact_events']} events, "
      f"{summary['illegal_contact_time_mean']:.2f} s in contact per env on average"
    )


def main():
  maybe_print_top_level_help("validate")

  # Import tasks to populate the registry, then pick the task from argv[1]
  # exactly as play does.
  import mjlab.tasks  # noqa: F401

  velocity_tasks = [t for t in list_tasks() if t.startswith("Mjlab-Velocity-")]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ValidateConfig,
    args=remaining_args,
    default=ValidateConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_validate(chosen_task, args)


if __name__ == "__main__":
  main()
