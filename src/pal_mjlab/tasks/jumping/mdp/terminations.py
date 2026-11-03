from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_error_magnitude

from .commands import LANDED, SETTLED, STANCE, JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _jump(env: ManagerBasedRlEnv, command_name: str) -> JumpHeightCommand:
  term: JumpHeightCommand = env.command_manager.get_term(command_name)
  term.update_state()
  return term


def jump_finished(
  env: ManagerBasedRlEnv, command_name: str, hold_time: float | None = None
) -> torch.Tensor:
  """Truncate once the machine has stayed in its last state, SETTLED, for hold_time.

  Settled is JumpHeightCommand.settled. hold_time defaults to the command's
  settle_hold_time, which the settle curriculum raises during training.
  """
  term = _jump(env, command_name)
  if hold_time is None:
    return term.jump_complete
  return (term.phase == SETTLED) & (term.settled_steps >= hold_time / env.step_dt)


def bad_touchdown(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_height_error: float = 0.3,
  min_base_height: float = 0.2,
) -> torch.Tensor:
  """Fail only at touchdown on an inaccurate apex or excessively low base.

  Both the apex and min_base_height are world-frame heights of the command's
  jump frame.
  """
  term = _jump(env, command_name)
  base_height = term.height
  height_error = (term.peak_height - term.target_height).abs()
  return term.just_touchdown & (
    (height_error > max_height_error) | (base_height < min_base_height)
  )


def jumped_again(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Fail on a second takeoff after landing (from LANDED or SETTLED): one jump
  per episode. The takeoff test is the command's (no foot in contact, base
  rising faster than takeoff_velocity), so a single lifted foot is no jump."""
  term = _jump(env, command_name)
  return term.just_takeoff & (term.phase >= LANDED)


def sideways_takeoff(
  env: ManagerBasedRlEnv, command_name: str, max_speed: float = 0.75
) -> torch.Tensor:
  """Fail if horizontal speed exceeds the limit at takeoff."""
  term = _jump(env, command_name)
  speed = term.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  return term.just_takeoff & (speed > max_speed)


def walking_before_jump(
  env: ManagerBasedRlEnv, command_name: str, max_distance: float = 0.2
) -> torch.Tensor:
  """Limit displacement from the sampled reset position while in stance."""
  term = _jump(env, command_name)
  displacement = term.robot.data.root_link_pos_w[:, :2] - term.reset_pos_xy
  return (term.phase == STANCE) & (displacement.norm(dim=-1) > max_distance)


def idle_before_jump(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_stance_time: float = 3.0,
  max_idle_time: float = 1.0,
) -> torch.Tensor:
  """Limit elapsed stance time and cumulative idle time, as in Olympus Lab."""
  term = _jump(env, command_name)
  return (term.phase == STANCE) & (
    (term.episode_steps > max_stance_time / env.step_dt)
    | (term.idle_steps > max_idle_time / env.step_dt)
  )


def excessive_attitude(
  env: ManagerBasedRlEnv, command_name: str, max_angle_deg: float = 25.0
) -> torch.Tensor:
  """Limit roll/pitch and wrapped yaw change from the randomized reset heading."""
  term = _jump(env, command_name)
  roll, pitch, yaw = euler_xyz_from_quat(term.robot.data.root_link_quat_w)
  yaw_error = yaw - term.reset_yaw
  yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos())
  limit = math.radians(max_angle_deg)
  return (roll.abs() > limit) | (pitch.abs() > limit) | (yaw_error.abs() > limit)


def horizontal_drift(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_distance: float = 0.2,
  max_landed_distance: float | None = None,
) -> torch.Tensor:
  """Limit the base's horizontal distance from the environment origin.

  max_distance applies before landing (stance and flight); once both feet
  have touched down (LANDED or SETTLED), max_landed_distance does (the same
  limit when None).
  """
  term = _jump(env, command_name)
  displacement = term.robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  landed_limit = max_distance if max_landed_distance is None else max_landed_distance
  limit = torch.where(term.phase >= LANDED, landed_limit, max_distance)
  return displacement.norm(dim=-1) > limit


def excessive_rotation(
  env: ManagerBasedRlEnv, command_name: str, max_angle_deg: float = 30.0
) -> torch.Tensor:
  """Fail once the base has turned more than max_angle_deg, about any axis,
  from the orientation the episode started in."""
  term = _jump(env, command_name)
  angle = quat_error_magnitude(term.robot.data.root_link_quat_w, term.reset_quat)
  return angle > math.radians(max_angle_deg)
