"""Height commands and once-per-jump state, inspired by Olympus Lab.

https://github.com/ntnu-arl/olympus_lab/tree/main/envs/vertical_jump
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# SETTLED is the machine's last state: landed, and settled (see
# JumpHeightCommand.settled). It falls back to LANDED if settling breaks.
STANCE, FLIGHT, LANDED, SETTLED = range(4)


def match_joint_values(
  mapping: dict[str, float], names: list[str], key: str
) -> list[float]:
  """One positive finite value per joint name, from a {regex: value} mapping."""
  values = []
  for name in names:
    matches = [
      value for pattern, value in mapping.items() if re.fullmatch(pattern, name)
    ]
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0:
      raise ValueError(f"{key} must match {name!r} once with a positive finite std.")
    values.append(matches[0])
  return values


class JumpHeightCommand(CommandTerm):
  """One jump per episode; the scalar input becomes zero after touchdown.

  Heights (targets, apex, measured) are the base link's world-frame z; the
  ground plane is z = 0. State is
  refreshed before rewards/terminations, not by the command manager (which runs
  after rewards in mjlab). The step guard makes all consumers share one update.
  """

  cfg: JumpHeightCommandCfg

  def __init__(self, cfg: JumpHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.gravity = -float(env.sim.mj_model.opt.gravity[2])
    if self.gravity <= 0:
      raise ValueError("Vertical jumping requires downward gravity.")
    self.height_range = cfg.height_range
    self.target_height = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros((self.num_envs, 1), device=self.device)
    self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.peak_height = torch.zeros_like(self.target_height)
    self.air_time = torch.zeros_like(self.target_height)
    self.landed_time = torch.zeros_like(self.target_height)
    self.settle_joint_ids, names = self.robot.find_joints(
      cfg.settle_joint_names, preserve_order=True
    )
    if not names:
      raise ValueError("Settling requires at least one joint.")
    self.settle_pose_std = torch.tensor(
      match_joint_values(cfg.settle_pose_std, names, "settle_pose_std"),
      device=self.device,
    )
    # Horizontal whole-body COM at takeoff; the landing-drift cost reference.
    self.takeoff_com_xy = torch.zeros((self.num_envs, 2), device=self.device)
    self.seen_contact = torch.zeros_like(self.target_height, dtype=torch.bool)
    self.apex_reached = torch.zeros_like(self.seen_contact)
    self.just_apex = torch.zeros_like(self.seen_contact)
    self.just_takeoff = torch.zeros_like(self.seen_contact)
    self.just_touchdown = torch.zeros_like(self.seen_contact)
    self.reset_pos_xy = torch.zeros((self.num_envs, 2), device=self.device)
    self.reset_yaw = torch.zeros_like(self.target_height)
    self.episode_steps = torch.zeros_like(self.phase)
    self.idle_steps = torch.zeros_like(self.phase)
    self.landed_steps = torch.zeros_like(self.phase)
    # Consecutive landed steps spent settled at the default pose.
    self.settled_steps = torch.zeros_like(self.phase)
    # Consecutive SETTLED steps with exactly one foot off the ground.
    self.contact_lost_steps = torch.zeros_like(self.phase)
    self._last_step = torch.full_like(self.phase, -1)
    self._updated_step = -1
    self.metrics = {
      name: torch.zeros_like(self.target_height)
      for name in ("target_height", "peak_height", "height_error", "landed")
    }

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample(self, env_ids: torch.Tensor) -> None:
    # torch.uniform_ cannot sample infinite bounds.
    self.time_left[env_ids] = math.inf
    self._resample_command(env_ids)
    self.command_counter[env_ids] += 1

  def reset(
    self, env_ids: torch.Tensor | slice | None
  ) -> dict[str, float | torch.Tensor]:
    self._update_metrics()
    # One |apex - target| per finished episode that reached its apex, logged as
    # a tensor so the logger averages over jumps only; episodes without an
    # apex are left out instead of counting as zero or as a full shortfall.
    jumped = env_ids[self.apex_reached[env_ids]]
    apex_error = (self.peak_height[jumped] - self.target_height[jumped]).abs()
    extras: dict[str, float | torch.Tensor] = dict(super().reset(env_ids))
    if len(jumped):
      extras["apex_height_error"] = apex_error
    return extras

  @property
  def com_xy(self) -> torch.Tensor:
    """Whole-robot center of mass (root subtree), world-frame xy."""
    data = self.robot.data
    return data.data.subtree_com[:, data.indexing.root_body_id, :2]

  @property
  def height(self) -> torch.Tensor:
    """World-frame z of the base link; the apex is its maximum in flight."""
    return self.robot.data.root_link_pos_w[:, 2]

  def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(len(env_ids), device=self.device).uniform_(*self.height_range)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.target_height[env_ids] = self._sample_targets(env_ids)
    self.phase[env_ids] = STANCE
    for value in (
      self.peak_height,
      self.air_time,
      self.landed_time,
      self.seen_contact,
      self.apex_reached,
      self.just_apex,
      self.just_takeoff,
      self.just_touchdown,
      self.episode_steps,
      self.idle_steps,
      self.landed_steps,
      self.settled_steps,
      self.contact_lost_steps,
    ):
      value[env_ids] = 0
    self.takeoff_com_xy[env_ids] = 0
    # Reset observations must not count as a physics step or a takeoff.
    self._last_step[env_ids] = self._env.common_step_counter
    # Reset events have written qpos, but derived poses can still belong
    # to the previous episode until forward(). Anchor to the actual reset.
    data = self.robot.data
    root_pose = data.data.qpos[env_ids][:, data.indexing.free_joint_q_adr]
    self.reset_pos_xy[env_ids] = root_pose[:, :2]
    self.reset_yaw[env_ids] = euler_xyz_from_quat(root_pose[:, 3:7])[2]
    self._update_command(env_ids)

  def update_state(self) -> None:
    step = self._env.common_step_counter
    if self._updated_step == step:
      return
    self._updated_step = step
    fresh = self._last_step != step
    dt = self._env.step_dt
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    contact = (sensor.data.found > 0).any(dim=-1)
    both_feet = (sensor.data.found > 0).all(dim=-1)
    velocity = self.robot.data.root_link_lin_vel_w
    height = self.height
    self.just_apex[fresh] = False
    self.just_takeoff[fresh] = False
    self.just_touchdown[fresh] = False
    self.episode_steps[fresh] += 1
    self.seen_contact |= contact & fresh
    takeoff = (
      fresh
      & (self.phase != FLIGHT)
      & self.seen_contact
      & ~contact
      & (velocity[:, 2] > self.cfg.takeoff_velocity)
    )
    self.just_takeoff[takeoff] = True
    # Never restart the jump or its apex reward on a second takeoff.
    first_takeoff = takeoff & (self.phase == STANCE)
    self.phase[first_takeoff] = FLIGHT
    self.takeoff_com_xy[first_takeoff] = self.com_xy[first_takeoff]
    flying = fresh & (self.phase == FLIGHT)
    self.air_time[flying] += dt
    self.peak_height[flying] = torch.maximum(self.peak_height, height)[flying]
    # Landing starts once both feet are down; a single-foot touch stays in
    # flight. Settling (JumpHeightCommand.settled) then finishes the jump.
    touchdown = flying & both_feet
    self.just_touchdown[touchdown] = True
    apex = flying & ~self.apex_reached & ((velocity[:, 2] <= 0) | touchdown)
    self.just_apex[apex] = True
    self.apex_reached[apex] = True
    self.phase[touchdown] = LANDED
    landed = fresh & (self.phase >= LANDED)
    self.landed_steps[landed] += 1
    self.landed_steps[touchdown] = 0
    self.landed_time[landed] = self.landed_steps[landed] * dt
    # Deadzone: once SETTLED, one foot may lose contact for up to
    # settle_contact_grace while the rest of the settled state holds.
    lifted = landed & (self.phase == SETTLED) & contact & ~both_feet
    self.contact_lost_steps[lifted] += 1
    self.contact_lost_steps[landed & ~lifted] = 0
    grace = lifted & (
      self.contact_lost_steps <= round(self.cfg.settle_contact_grace / dt)
    )
    settled = landed & (self.settled | (grace & self.still))
    self.phase[settled] = SETTLED
    self.phase[landed & ~settled] = LANDED
    self.settled_steps[settled] += 1
    self.settled_steps[landed & ~settled] = 0
    # Olympus accumulates idle steps across pauses in stance (not necessarily
    # consecutive), and clears them once the robot leaves stance.
    stance = fresh & (self.phase == STANCE)
    self.idle_steps[stance & (velocity.norm(dim=-1) < self.cfg.idle_speed)] += 1
    self.idle_steps[fresh & (self.phase != STANCE)] = 0
    self._last_step[fresh] = step

  @property
  def pose_error(self) -> torch.Tensor:
    """mean((q - q_default)^2 / std^2) over the settle joints."""
    data = self.robot.data
    ids = self.settle_joint_ids
    error = (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]) / (
      self.settle_pose_std
    )
    return error.square().mean(dim=-1)

  @property
  def settled(self) -> torch.Tensor:
    """Both feet down, near the default pose, joints and base nearly still.

    The pose is close when exp(-pose_error), the posture reward's kernel, is
    at least exp(-1): an RMS error within one std. update_state additionally
    tolerates one foot off for settle_contact_grace once SETTLED.
    """
    sensor = self._env.scene[self.cfg.contact_sensor_name]
    return (sensor.data.found > 0).all(dim=-1) & self.still

  @property
  def still(self) -> torch.Tensor:
    """settled without the contact check: pose, joints and base only."""
    data = self.robot.data
    ids = self.settle_joint_ids
    return (
      (self.pose_error <= 1)
      & (data.joint_vel[:, ids].abs().amax(dim=-1) < self.cfg.settle_joint_speed)
      & (data.root_link_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed)
      & (data.root_link_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang_speed)
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._command[ids, 0] = torch.where(
      self.phase[ids] >= LANDED, 0.0, self.target_height[ids]
    )

  def _update_metrics(self) -> None:
    self.metrics["target_height"][:] = self.target_height
    self.metrics["peak_height"][:] = self.peak_height
    self.metrics["height_error"][:] = (self.peak_height - self.target_height).abs()
    self.metrics["landed"][:] = self.phase >= LANDED


@dataclass(kw_only=True)
class JumpHeightCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  # An episode owns its target; never change the goal during a jump.
  resampling_time_range: tuple[float, float] = (math.inf, math.inf)
  # World-frame base-link apex heights, in metres.
  height_range: tuple[float, float] = (1.0, 1.1)
  # Olympus Lab's jump state machine threshold.
  takeoff_velocity: float = 0.75
  min_air_time: float = 0.04
  # Lowest world-frame apex that counts as a jump (rewarded by jump_height).
  min_jump_height: float = 0.96
  # Landing is complete once settled: pose within settle_pose_std of default
  # (RMS over settle_joint_names), every such joint slower than
  # settle_joint_speed, and the base slower than settle_speed/settle_ang_speed.
  settle_joint_names: tuple[str, ...] = (".*",)
  settle_pose_std: dict[str, float] = field(default_factory=lambda: {".*": 0.05})
  settle_joint_speed: float = 0.1
  settle_speed: float = 0.05
  settle_ang_speed: float = 0.1
  # Once SETTLED, seconds one foot may be off the ground without breaking it.
  settle_contact_grace: float = 0.1
  idle_speed: float = 0.1

  def __post_init__(self):
    low, high = self.height_range
    if not (0 < low <= high < math.inf):
      raise ValueError("height_range must contain finite positive ordered heights.")
    if self.resampling_time_range != (math.inf, math.inf):
      raise ValueError("Jump targets can only be resampled on episode reset.")

  def build(self, env: ManagerBasedRlEnv) -> JumpHeightCommand:
    return JumpHeightCommand(self, env)
