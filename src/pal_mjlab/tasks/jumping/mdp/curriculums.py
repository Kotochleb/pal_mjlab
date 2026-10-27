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

  Each stage is (environment step, minimum apex, maximum apex): world-frame
  base-link heights in metres.
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


def settle_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stages: tuple[tuple[int, float, float], ...],
) -> dict[str, float]:
  """Set the settled joint speed limit and hold time from control points.

  Each stage is (environment step, joint speed limit in rad/s, hold time in
  s). Both are linearly interpolated between consecutive control points and
  held at the last one's values after it; they apply immediately, also to
  jumps in progress.
  """
  del env_ids
  if not stages or stages[0][0] != 0:
    raise ValueError("Settle curriculum must start at step zero.")
  for i, (step, speed, hold_time) in enumerate(stages):
    if not (0 < speed < math.inf):
      raise ValueError("Settle joint speeds must be finite and positive.")
    if not (0 <= hold_time < math.inf):
      raise ValueError("Settle hold times must be finite and non-negative.")
    if i and step <= stages[i - 1][0]:
      raise ValueError("Curriculum steps must be strictly increasing.")
  step = env.common_step_counter
  _, speed, hold_time = stages[-1]
  for (start, speed0, hold0), (end, speed1, hold1) in zip(stages, stages[1:]):
    if step < end:
      blend = (step - start) / (end - start)
      speed = speed0 + blend * (speed1 - speed0)
      hold_time = hold0 + blend * (hold1 - hold0)
      break
  term = env.command_manager.get_term(command_name)
  term.settle_joint_speed = speed
  term.settle_hold_time = hold_time
  return {"settle_joint_speed": speed, "settle_hold_time": hold_time}
