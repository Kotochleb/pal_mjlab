"""Expand the sampled height range at configured training steps."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def jump_height_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stages: tuple[tuple[int, float, float], ...],
) -> dict[str, float]:
  """Update only the sampling bounds; active targets stay fixed until reset.

  Each stage is (environment step, minimum rise, maximum rise), in metres.
  mjlab's runner checkpoints common_step_counter, preserving this schedule.
  """
  del env_ids
  if not stages or stages[0][0] != 0:
    raise ValueError("Height curriculum must start at step zero.")
  for i, (step, low, high) in enumerate(stages):
    if not (0 < low <= high < math.inf):
      raise ValueError("Curriculum heights must be finite, positive, and ordered.")
    if i and step <= stages[i - 1][0]:
      raise ValueError("Curriculum steps must be strictly increasing.")
  selected = stages[0]
  for stage in stages:
    if env.common_step_counter >= stage[0]:
      selected = stage
  term = env.command_manager.get_term(command_name)
  term.height_range = selected[1:]
  return {"min_height": selected[1], "max_height": selected[2]}
