from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HighSpeedHoldVelocityCommand(UniformVelocityCommand):
  """``UniformVelocityCommand`` that holds fast forward commands longer.

  A resample whose ``|lin_vel_x|`` reaches ``hold_speed_threshold`` draws its
  hold time from ``hold_time_range`` instead of ``resampling_time_range`` and
  is never a standing env, so the robot keeps running at that speed.
  """

  cfg: HighSpeedHoldVelocityCommandCfg

  def _resample(self, env_ids: torch.Tensor) -> None:
    super()._resample(env_ids)
    if len(env_ids) == 0:
      return
    fast = self.vel_command_b[env_ids, 0].abs() >= self.cfg.hold_speed_threshold
    fast_ids = env_ids[fast]
    if len(fast_ids) > 0:
      self.is_standing_env[fast_ids] = False
      self.time_left[fast_ids] = self.time_left[fast_ids].uniform_(
        *self.cfg.hold_time_range
      )


@dataclass(kw_only=True)
class HighSpeedHoldVelocityCommandCfg(UniformVelocityCommandCfg):
  hold_speed_threshold: float
  """``|lin_vel_x|`` (m/s) at or above which a command is held longer."""
  hold_time_range: tuple[float, float]
  """Hold time range (s) for commands at or above ``hold_speed_threshold``."""

  @classmethod
  def from_uniform(
    cls,
    cfg: UniformVelocityCommandCfg,
    hold_speed_threshold: float,
    hold_time_range: tuple[float, float],
  ) -> HighSpeedHoldVelocityCommandCfg:
    return cls(
      **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
      hold_speed_threshold=hold_speed_threshold,
      hold_time_range=hold_time_range,
    )

  def build(self, env: ManagerBasedRlEnv) -> HighSpeedHoldVelocityCommand:
    return HighSpeedHoldVelocityCommand(self, env)
