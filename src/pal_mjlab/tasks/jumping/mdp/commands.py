"""Height commands and once-per-jump state, inspired by Olympus Lab.

https://github.com/ntnu-arl/olympus_lab/tree/main/envs/vertical_jump
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STANCE, FLIGHT, LANDED = range(3)


class JumpHeightCommand(CommandTerm):
  """One jump per episode; the scalar input becomes zero after touchdown.

  Heights are rises above the model's nominal standing root height. State is
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
    self.settled_time = torch.zeros_like(self.target_height)
    self.impact_deceleration = torch.zeros_like(self.target_height)
    self.previous_velocity = torch.zeros((self.num_envs, 3), device=self.device)
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

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    self._update_metrics()
    return super().reset(env_ids)

  @property
  def com_xy(self) -> torch.Tensor:
    """Whole-robot center of mass (root subtree), world-frame xy."""
    data = self.robot.data
    return data.data.subtree_com[:, data.indexing.root_body_id, :2]

  @property
  def height(self) -> torch.Tensor:
    return (
      self.robot.data.root_link_pos_w[:, 2]
      - self._env.scene.env_origins[:, 2]
      - self.robot.data.default_root_state[:, 2]
    )

  def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty(len(env_ids), device=self.device).uniform_(*self.height_range)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.target_height[env_ids] = self._sample_targets(env_ids)
    self.phase[env_ids] = STANCE
    for value in (
      self.peak_height,
      self.air_time,
      self.landed_time,
      self.settled_time,
      self.impact_deceleration,
      self.seen_contact,
      self.apex_reached,
      self.just_apex,
      self.just_takeoff,
      self.just_touchdown,
      self.episode_steps,
      self.idle_steps,
      self.landed_steps,
    ):
      value[env_ids] = 0
    self.previous_velocity[env_ids] = 0
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
    # A second takeoff is terminal; never restart the jump or its apex reward.
    first_takeoff = takeoff & (self.phase == STANCE)
    self.phase[first_takeoff] = FLIGHT
    self.takeoff_com_xy[first_takeoff] = self.com_xy[first_takeoff]
    flying = fresh & (self.phase == FLIGHT)
    self.air_time[flying] += dt
    self.peak_height[flying] = torch.maximum(self.peak_height, height)[flying]
    touchdown = flying & contact
    self.just_touchdown[touchdown] = True
    apex = flying & ~self.apex_reached & ((velocity[:, 2] <= 0) | touchdown)
    self.just_apex[apex] = True
    self.apex_reached[apex] = True
    self.phase[touchdown] = LANDED
    landed = fresh & (self.phase == LANDED)
    self.landed_steps[landed] += 1
    self.landed_steps[touchdown] = 0
    self.landed_time[landed] = self.landed_steps[landed] * dt
    stable = (
      both_feet
      & (velocity.norm(dim=-1) < self.cfg.settle_speed)
      & (self.robot.data.root_link_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang_speed)
    )
    self.settled_time[landed] = torch.where(stable, self.settled_time + dt, 0.0)[landed]
    # Olympus accumulates idle steps across pauses in stance (not necessarily
    # consecutive), and clears them once the robot leaves stance.
    stance = fresh & (self.phase == STANCE)
    self.idle_steps[stance & (velocity.norm(dim=-1) < self.cfg.idle_speed)] += 1
    self.idle_steps[fresh & (self.phase != STANCE)] = 0
    acceleration = (velocity - self.previous_velocity) / dt
    direction = torch.nn.functional.normalize(self.previous_velocity, dim=-1)
    self.impact_deceleration[fresh] = (
      -(acceleration * direction).sum(dim=-1)
    ).clamp_min(0)[fresh]
    self.previous_velocity[fresh] = velocity[fresh]
    self._last_step[fresh] = step

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._command[ids, 0] = torch.where(
      self.phase[ids] == LANDED, 0.0, self.target_height[ids]
    )

  def _update_metrics(self) -> None:
    self.metrics["target_height"][:] = self.target_height
    self.metrics["peak_height"][:] = self.peak_height
    self.metrics["height_error"][:] = (self.peak_height - self.target_height).abs()
    self.metrics["landed"][:] = self.phase == LANDED


@dataclass(kw_only=True)
class JumpHeightCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  # An episode owns its target; never change the goal during a jump.
  resampling_time_range: tuple[float, float] = (math.inf, math.inf)
  height_range: tuple[float, float] = (0.05, 0.15)
  # Olympus Lab's jump state machine threshold.
  takeoff_velocity: float = 0.75
  min_air_time: float = 0.04
  min_jump_height: float = 0.03
  settle_speed: float = 0.2
  settle_ang_speed: float = 0.5
  idle_speed: float = 0.1

  def __post_init__(self):
    low, high = self.height_range
    if not (0 < low <= high < math.inf):
      raise ValueError("height_range must contain finite positive ordered heights.")
    if self.resampling_time_range != (math.inf, math.inf):
      raise ValueError("Jump targets can only be resampled on episode reset.")

  def build(self, env: ManagerBasedRlEnv) -> JumpHeightCommand:
    return JumpHeightCommand(self, env)
