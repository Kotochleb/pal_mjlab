from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import LANDED, JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def jump_finished(
  env: ManagerBasedRlEnv, command_name: str, settle_time: float = 0.5
) -> torch.Tensor:
  """Finish after sustained stable contact; retain the normal fall terminations."""
  term: JumpHeightCommand = env.command_manager.get_term(command_name)
  term.update_state()
  return (term.phase == LANDED) & (term.settled_time >= settle_time)
