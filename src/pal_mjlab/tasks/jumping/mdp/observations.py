from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def base_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """World-frame base-link z, as the jump command measures targets and apex."""
  term: JumpHeightCommand = env.command_manager.get_term(command_name)
  return term.height.unsqueeze(-1)
