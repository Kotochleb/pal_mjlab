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
their task id, exactly as for ``play``. The scenario overrides the task's
terrain, so a ``-Flat-`` task id can be validated on stairs and a ``-Rough-``
one on the plane.

Scenarios (``--scenario``):

* ``flat_ramp``: flat plane. Stand still for ``stand-time``, then ramp the
  forward velocity command linearly to ``target-vel`` (3.0 m/s by default)
  over ``ramp-time``, then hold it.
* ``pebbles_ramp``: same schedule up to 1.0 m/s on the rough task's
  "pebbles" sub-terrain (dense 2-5 cm boxes), regenerated as one long tile
  so the robot has room to run.
* ``stairs``: spawn in the centre of an inverted-pyramid staircase (a pit,
  the robot climbs out) and walk forward at a constant command. Step height
  and width are the difficulty knobs.

Results go to one ``.npz`` (plus a ``.json`` summary next to it) with
per-step aggregates over the still-alive envs and per-env outcomes; see
``_save`` for the exact keys. Run::

    uv run validate <task-id> --checkpoint-file logs/.../model_5000.pt \\
        --scenario flat_ramp --num-envs 4096
    uv run validate <task-id> --wandb-run-path org/proj/run --scenario stairs \\
        --stairs.step-height 0.12 --stairs.step-width 0.3
    uv run validate <task-id> --agent zero --scenario pebbles_ramp --num-envs 8
    uv run validate <task-id> --checkpoint-file ... --scenario stairs \\
        --num-envs 16 --viewer viser --video True

``--viewer native|viser`` shows env 0 while the same recording runs (the
viewer stops itself at the end of the scenario); with viser, ``--video True``
also writes an .mp4 of env 0 next to the .npz.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.terrains.config import pyramid_stairs_inv, random_spread_boxes
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

Scenario = Literal["flat_ramp", "pebbles_ramp", "stairs"]

# The rough training tile in configure_kangaroo_rough_env spreads 350 pebbles
# over a 3 m x 3 m tile; keep that density whatever tile size is asked for.
_TRAINING_PEBBLE_DENSITY = 350.0 / 9.0


##
# Command: a deterministic ramp instead of the uniform sampler.
##


class RampVelocityCommand(UniformVelocityCommand):
  """Forward velocity ramp: 0 for stand_time_s, then linear to target, then hold.

  Reuses the uniform command's buffers, metrics and debug arrows so the rest
  of the task (observation term, tracking rewards, posture reward) sees the
  same ``twist`` interface it was trained against; only where the numbers
  come from changes. Lateral and yaw commands are always zero.
  """

  cfg: RampVelocityCommandCfg

  @staticmethod
  def value_at(cfg: RampVelocityCommandCfg, t: torch.Tensor) -> torch.Tensor:
    frac = (t - cfg.stand_time_s) / max(cfg.ramp_time_s, 1e-6)
    return frac.clamp(0.0, 1.0) * cfg.target_lin_vel_x

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

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Pure function of each env's episode time, so refreshing all envs is
    # safe (and an auto-reset env restarts its own ramp from zero).
    del env_ids
    t = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
    self.vel_command_b[:, 0] = self.value_at(self.cfg, t)
    self.vel_command_b[:, 1:] = 0.0


@dataclass(kw_only=True)
class RampVelocityCommandCfg(UniformVelocityCommandCfg):
  target_lin_vel_x: float
  stand_time_s: float
  ramp_time_s: float

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
  """Seconds of zero command before the ramp starts."""
  ramp_time: float = 10.0
  """Seconds the ramp takes to go from 0 to target_vel."""
  hold_time: float = 5.0
  """Seconds the command is held at target_vel before the run ends."""


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
  step_height: float = 0.10
  """Rise of every step [m]. The main difficulty knob."""
  step_width: float = 0.3
  """Run (depth) of every step [m]."""
  velocity: float = 0.5
  """Forward velocity command held while climbing [m/s]."""
  stand_time: float = 2.0
  """Seconds of zero command before walking starts."""
  ramp_time: float = 1.0
  """Seconds to ramp from 0 to velocity (avoids a step change in the command)."""
  duration: float = 20.0
  """Total run length [s] including stand and ramp."""
  tile_size: float = 8.0
  """Side of the square staircase tile [m]; sets how many steps there are."""
  platform_width: float = 0.5
  """Flat square at the bottom of the pit the robot spawns on [m]."""


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
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs (and cached checkpoints) live."""

  num_envs: int = 4096
  seed: int = 0
  """Env seed: fixes the domain-randomization draw and the initial-pose jitter."""
  device: str | None = None
  domain_rand: bool = True
  """Keep the task's startup domain randomization (the main source of spread
  between envs). False removes it, leaving only the initial-pose jitter."""
  init_pos_noise: float = 0.05
  """Half-range of the uniform x/y jitter on the spawn position [m]."""
  init_yaw_noise: float = 0.0
  """Half-range of the uniform yaw jitter on the spawn heading [rad]. Zero
  makes every robot face +x, which the pebble and stair tiles assume."""
  init_joint_noise: float = 0.0
  """Half-range of the uniform offset added to every joint's INIT_STATE
  position on reset [rad or m]."""

  output: str | None = None
  """Where to write the .npz (a .json summary goes next to it). Default:
  <checkpoint dir>/validate/<scenario>_<timestamp>.npz, or
  logs/validate/... for dummy agents."""
  save_per_env_steps: bool = False
  """Also store every env's per-step rewards, metrics and base state
  ([num_envs, steps, terms]); ~1.5 GB at 4096 envs and 850 steps."""
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
  env_cfg: ManagerBasedRlEnvCfg, target_vel: float, stand_time: float, ramp_time: float
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
      lin_vel_x=(-hi, hi), lin_vel_y=(-hi, hi), ang_vel_z=(-1.0, 1.0)
    ),
    target_lin_vel_x=target_vel,
    stand_time_s=stand_time,
    ramp_time_s=ramp_time,
  )


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
) -> tuple[ManagerBasedRlEnvCfg, float]:
  """Play env cfg for task_id with the scenario's terrain, command and reset.

  Returns the cfg and the scenario duration in seconds.
  """
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  _configure_reset(env_cfg, cfg)

  if cfg.scenario == "flat_ramp":
    s = cfg.flat_ramp
    _set_plane_terrain(env_cfg)
    _set_ramp_command(env_cfg, s.target_vel, s.stand_time, s.ramp_time)
    duration = s.stand_time + s.ramp_time + s.hold_time
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
    _set_ramp_command(env_cfg, s.target_vel, s.stand_time, s.ramp_time)
    duration = s.stand_time + s.ramp_time + s.hold_time
  elif cfg.scenario == "stairs":
    s = cfg.stairs
    _set_single_tile_terrain(
      env_cfg,
      pyramid_stairs_inv(
        step_height_range=(s.step_height, s.step_height),
        step_width=s.step_width,
        platform_width=s.platform_width,
        border_width=0.1,
      ),
      size=(s.tile_size, s.tile_size),
      difficulty=1.0,
      seed=cfg.seed,
    )
    _set_ramp_command(env_cfg, s.velocity, s.stand_time, s.ramp_time)
    duration = s.duration
  else:
    raise ValueError(f"Unknown scenario: {cfg.scenario}")

  # The terrain curriculum is already gone in play mode; make sure nothing
  # else moves envs between tiles or reshapes the command mid-run.
  if env_cfg.curriculum is not None:
    env_cfg.curriculum.pop("terrain_levels", None)
    env_cfg.curriculum.pop("command_vel", None)
  return env_cfg, duration


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


class ScenarioRecorder:
  """Env wrapper that logs every step of the scenario.

  Sits on top of the RslRlVecEnvWrapper and exposes the same interface the
  mjlab viewers expect (``get_observations`` / ``step`` / ``reset`` /
  ``unwrapped``), so the headless loop and the viewers drive the very same
  bookkeeping: per-step aggregates over the envs still alive, and per-env
  outcomes at the first termination. A viewer-triggered ``reset`` restarts
  the recording from scratch.
  """

  def __init__(self, env: RslRlVecEnvWrapper, cfg: ValidateConfig, duration: float):
    self.env = env
    self.vcfg = cfg
    self.duration = duration
    raw = env.unwrapped
    self.robot = raw.scene["robot"]
    twist_cfg = raw.command_manager.get_term_cfg("twist")
    assert isinstance(twist_cfg, RampVelocityCommandCfg)
    self.step_dt = raw.step_dt
    self.num_steps = int(round(duration / self.step_dt))
    self.print_every = max(1, int(round(cfg.print_every_s / self.step_dt)))

    self.reward_names = list(raw.reward_manager.active_terms)
    self.metric_names = list(raw.metrics_manager.active_terms)
    self.termination_names = list(raw.termination_manager.active_terms)
    self.state_names = [
      "base_lin_vel_x_b",
      "base_lin_vel_y_b",
      "base_ang_vel_z_b",
      "base_z_w",
    ]
    N, T = raw.num_envs, self.num_steps
    R, M, S = len(self.reward_names), len(self.metric_names), len(self.state_names)
    dev = raw.device

    # Per-step aggregates over the envs still alive at that step.
    self.times = torch.arange(T, device=dev, dtype=torch.float32) * self.step_dt
    self.cmd_vel_x = RampVelocityCommand.value_at(twist_cfg, self.times)
    self.alive_count = torch.zeros(T, device=dev, dtype=torch.long)
    self.reward_mean = torch.zeros(T, R, device=dev)
    self.reward_std = torch.zeros(T, R, device=dev)
    self.metric_mean = torch.zeros(T, M, device=dev)
    self.metric_std = torch.zeros(T, M, device=dev)
    self.state_mean = torch.zeros(T, S, device=dev)
    self.state_std = torch.zeros(T, S, device=dev)

    # Per-env outcomes.
    self.alive = torch.ones(N, device=dev, dtype=torch.bool)
    self.survival_time = torch.full((N,), duration, device=dev)
    self.vel_at_failure = torch.full((N,), float("nan"), device=dev)
    self.termination_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
    self.truncated_only = torch.zeros(N, device=dev, dtype=torch.bool)
    self.reward_sum = torch.zeros(N, R, device=dev)
    self.metric_sum = torch.zeros(N, M, device=dev)
    self.metric_max = torch.full((N, M), float("-inf"), device=dev)
    self.alive_steps = torch.zeros(N, device=dev, dtype=torch.long)

    self.per_env_steps: torch.Tensor | None = None
    if cfg.save_per_env_steps:
      n_logged = (T + cfg.log_every - 1) // cfg.log_every
      self.per_env_steps = torch.zeros(N, n_logged, S + R + M, device=dev)

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
    out = self.env.step(actions)
    if self.k < self.num_steps and not self.finished:
      self._record(self.k, mask, out[2].bool())
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
      self.alive_count,
      self.reward_mean,
      self.reward_std,
      self.metric_mean,
      self.metric_std,
      self.state_mean,
      self.state_std,
      self.reward_sum,
      self.metric_sum,
      self.alive_steps,
    ):
      buf.zero_()
    self.alive.fill_(True)
    self.survival_time.fill_(self.duration)
    self.vel_at_failure.fill_(float("nan"))
    self.termination_idx.fill_(-1)
    self.truncated_only.fill_(False)
    self.metric_max.fill_(float("-inf"))
    if self.per_env_steps is not None:
      self.per_env_steps.zero_()
    # The wrapper has just reset: the robots stand at INIT_STATE with zero
    # command and their ramps start counting from here.
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
      ],
      dim=-1,
    )

  @staticmethod
  def _masked_stats(x: torch.Tensor, mask: torch.Tensor):
    n = mask.sum().clamp(min=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    mean = (x * m).sum(0) / n
    var = (((x - mean) ** 2) * m).sum(0) / n
    return mean, var.sqrt()

  @torch.no_grad()
  def _record(self, k: int, mask: torch.Tensor, dones: torch.Tensor) -> None:
    raw = self.env.unwrapped
    done_now = dones & mask
    step_rew = raw.reward_manager._step_reward
    step_met = raw.metrics_manager._step_values
    self.alive_count[k] = mask.sum()
    self.reward_mean[k], self.reward_std[k] = self._masked_stats(step_rew, mask)
    self.metric_mean[k], self.metric_std[k] = self._masked_stats(step_met, mask)
    self.reward_sum[mask] += step_rew[mask] * self.step_dt
    self.metric_sum[mask] += step_met[mask]
    self.metric_max[mask] = torch.maximum(self.metric_max[mask], step_met[mask])
    self.alive_steps[mask] += 1

    if done_now.any():
      # Read the termination buffers before anything overwrites them: they
      # hold this step's values until the next termination compute.
      self.survival_time[done_now] = (k + 1) * self.step_dt
      self.vel_at_failure[done_now] = self.cmd_vel_x[k]
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
    self.last_pos[self.alive] = self.robot.data.root_link_pos_w[self.alive]

    if self.per_env_steps is not None and k % self.vcfg.log_every == 0:
      row = torch.cat([state, step_rew, step_met], dim=-1)
      row[~mask] = float("nan")
      self.per_env_steps[:, k // self.vcfg.log_every] = row

    self._all_done = not bool(self.alive.any())
    if (k + 1) % self.print_every == 0 or k == self.num_steps - 1 or self._all_done:
      elapsed = time.time() - self.wall_start
      print(
        f"  sim t={self.times[k] + self.step_dt:5.1f}s "
        f"cmd_vx={self.cmd_vel_x[k]:.2f} "
        f"alive={int(self.alive.sum())}/{self.num_envs} "
        f"vx={self.state_mean[k, 0]:.2f}±{self.state_std[k, 0]:.2f} "
        f"[wall-clock elapsed {elapsed:.0f}s, display only]"
      )
    if self._all_done:
      print("[INFO] every env has terminated; recording stops here")

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
      "alive_count": self.alive_count[:n],
      "reward_mean": self.reward_mean[:n],
      "reward_std": self.reward_std[:n],
      "metric_mean": self.metric_mean[:n],
      "metric_std": self.metric_std[:n],
      "state_mean": self.state_mean[:n],
      "state_std": self.state_std[:n],
      "survival_time": survival_time,
      "vel_at_failure": self.vel_at_failure,
      "failed": (self.termination_idx >= 0) & ~self.truncated_only,
      "truncated": self.truncated_only,
      "termination_idx": self.termination_idx,
      "reward_sum": self.reward_sum,
      "metric_mean_per_env": self.metric_sum
      / self.alive_steps.clamp(min=1).unsqueeze(-1),
      "metric_max_per_env": self.metric_max,
      "alive_steps": self.alive_steps,
      "distance_xy": displacement[:, :2].norm(dim=-1),
      "distance_x": displacement[:, 0],
      "height_gain": displacement[:, 2],
    }
    if self.per_env_steps is not None:
      n_logged = (n + self.vcfg.log_every - 1) // self.vcfg.log_every
      out["per_env_steps"] = self.per_env_steps[:, :n_logged]
    return out


def run_validate(task_id: str, cfg: ValidateConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.video and cfg.viewer != "viser":
    raise ValueError("--video needs --viewer viser")

  env_cfg, duration = build_scenario_env_cfg(task_id, cfg)
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
  rec = ScenarioRecorder(env, cfg, duration)

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
    NativeMujocoViewer(rec, policy).run(num_steps=rec.num_steps)
  elif cfg.viewer == "viser":
    ViserPlayViewer(rec, policy).run(num_steps=rec.num_steps)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {cfg.viewer}")

  results = rec.results()
  rec.close()  # Finalizes the video, if any.
  if cfg.video:
    recorded = out_path.parent / f"{out_path.stem}-step-0.mp4"
    if recorded.exists():
      recorded.replace(out_path.with_suffix(".mp4"))
      print(f"[INFO] video: {out_path.with_suffix('.mp4')}")
  _save(
    out_path,
    task_id,
    cfg,
    results,
    rec.reward_names,
    rec.metric_names,
    rec.termination_names,
    rec.state_names,
    rec.step_dt,
  )
  return out_path


def _output_path(cfg: ValidateConfig, log_dir: Path | None) -> Path:
  if cfg.output is not None:
    return Path(cfg.output)
  root = (log_dir / "validate") if log_dir is not None else Path("logs/validate")
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  return root / f"{cfg.scenario}_{stamp}.npz"


def _save(
  out_path: Path,
  task_id: str,
  cfg: ValidateConfig,
  results: dict[str, torch.Tensor],
  reward_names: list[str],
  metric_names: list[str],
  termination_names: list[str],
  state_names: list[str],
  step_dt: float,
) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  arrays = {k: v.detach().cpu().numpy() for k, v in results.items()}
  np.savez_compressed(
    out_path,
    reward_names=np.array(reward_names),
    metric_names=np.array(metric_names),
    termination_names=np.array(termination_names),
    state_names=np.array(state_names),
    **arrays,
  )

  failed = arrays["failed"]
  n = len(failed)
  summary = {
    "task_id": task_id,
    "config": asdict(cfg),
    "step_dt": step_dt,
    "num_envs": n,
    "num_steps": int(len(arrays["time"])),
    "failure_rate": float(failed.mean()),
    "truncation_rate": float(arrays["truncated"].mean()),
    "survival_time_mean": float(arrays["survival_time"].mean()),
    "survival_time_min": float(arrays["survival_time"].min()),
    "distance_x_mean": float(arrays["distance_x"].mean()),
    "height_gain_mean": float(arrays["height_gain"].mean()),
    "vel_at_failure_mean": (
      float(np.nanmean(arrays["vel_at_failure"])) if failed.any() else None
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
  }
  json_path = out_path.with_suffix(".json")
  json_path.write_text(json.dumps(summary, indent=2))

  print(f"[INFO] saved {out_path} and {json_path}")
  print(
    f"[RESULT] failed {int(failed.sum())}/{n} ({100 * failed.mean():.1f}%), "
    f"mean survival {summary['survival_time_mean']:.2f} s, "
    f"mean distance x {summary['distance_x_mean']:.2f} m"
  )
  if failed.any():
    v = arrays["vel_at_failure"][failed]
    print(
      f"[RESULT] command speed at failure: mean {v.mean():.2f} "
      f"min {v.min():.2f} max {v.max():.2f} m/s"
    )
  for name, count in summary["terminations"].items():
    if count:
      print(f"[RESULT]   {name}: {count}")


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
