"""Curriculum terms for kangaroo_full."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def top_speed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  step_interval: int,
  increment: float,
  start_speed: float,
  max_speed: float,
) -> dict[str, torch.Tensor]:
  """Step up the ``lin_vel_x`` command's upper bound as training progresses.

  The bound starts at ``start_speed`` and increases by ``increment`` every
  ``step_interval`` values of ``env.common_step_counter``, capped at
  ``max_speed``. Only the upper bound is touched -- the lower bound is left
  as whatever it already is, so a staged term like ``commands_vel`` can keep
  driving it (and the y/yaw ranges) independently. Register this term after
  that one so it overrides the upper bound each step.
  """
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  stages_elapsed = env.common_step_counter // step_interval
  target = min(start_speed + stages_elapsed * increment, max_speed)
  cfg.ranges.lin_vel_x = (cfg.ranges.lin_vel_x[0], target)
  return {"lin_vel_x_max": torch.tensor(target)}
