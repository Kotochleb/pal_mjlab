"""Training curricula of the vertical jump: target heights and settling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


class jump_height_curriculum:
  """Moves the sampled apex range from initial_range to final_range, gated by
  the jump success rate.

  The range is ``initial + progress * (final - initial)`` (both bounds), with
  ``progress`` in [0, 1]. It stays 0 for the first ``hold_steps`` environment
  steps. After that, every environment step changes progress by ``+rate``
  while the smoothed success rate is at least ``success_threshold`` and by
  ``-backoff_ratio * rate`` below it, where ``rate = 1 / (final_step -
  hold_steps)``: without a back-off the final range is reached at
  ``final_step``. While the success rate is high enough but the smoothed apex
  error exceeds ``max_apex_error`` (metres), progress holds still instead of
  rising. Only the bounds change; active targets stay fixed until reset.

  The success rate is the share of finished episodes that ended on
  ``termination_name``. Episodes restarted from a replayed state
  (jump_state_replay) start after takeoff and are left out. It is an
  exponential moving average with a per-episode weight of
  ``1 / success_window``, so it spans about ``success_window`` episodes.
  The apex error is ``|apex - target|`` of finished, non-replayed episodes
  that reached their apex (as the command's apex_height_error metric),
  averaged the same way per jump.

  Progress, the success average and the last step are not derived from
  common_step_counter: ``state_dict``/``load_state_dict`` carry them through
  checkpoints (JumpOnPolicyRunner).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.env = env
    self.command_name: str = params["command_name"]
    self.termination_name: str = params.get("termination_name", "jump_finished")
    self.initial_range = tuple(float(v) for v in params["initial_range"])
    self.final_range = tuple(float(v) for v in params["final_range"])
    self.hold_steps = int(params["hold_steps"])
    self.final_step = int(params["final_step"])
    self.success_threshold = float(params.get("success_threshold", 0.5))
    self.backoff_ratio = float(params.get("backoff_ratio", 0.5))
    self.success_window = float(params.get("success_window", 4096))
    self.max_apex_error = float(params.get("max_apex_error", math.inf))
    for low, high in (self.initial_range, self.final_range):
      if not (0 < low <= high < math.inf):
        raise ValueError("Height ranges must be finite, positive, and ordered.")
    if any(f < i for i, f in zip(self.initial_range, self.final_range, strict=True)):
      raise ValueError("The final range must not lie below the initial range.")
    if not 0 <= self.hold_steps < self.final_step:
      raise ValueError("hold_steps must be non-negative and before final_step.")
    if not 0 < self.success_threshold < 1:
      raise ValueError("success_threshold must be in (0, 1).")
    if self.backoff_ratio < 0 or self.success_window < 1:
      raise ValueError("backoff_ratio must be >= 0 and success_window >= 1.")
    if not self.max_apex_error > 0:
      raise ValueError("max_apex_error must be positive.")
    self.rate = 1.0 / (self.final_step - self.hold_steps)
    self.progress = 0.0
    # None until the first episode ends.
    self.success_rate: float | None = None
    # None until the first non-replayed episode with an apex ends.
    self.apex_error: float | None = None
    self.last_step = env.common_step_counter

  def state_dict(self) -> dict[str, float | int | None]:
    return {
      "progress": self.progress,
      "success_rate": self.success_rate,
      "apex_error": self.apex_error,
      "last_step": self.last_step,
    }

  def load_state_dict(self, state: dict[str, float | int | None]) -> None:
    self.progress = float(state["progress"])
    self.success_rate = state["success_rate"]
    # Checkpoints from before the apex-error gate lack it.
    self.apex_error = state.get("apex_error")
    self.last_step = int(state["last_step"])

  def height_range(self) -> tuple[float, float]:
    low0, high0 = self.initial_range
    low1, high1 = self.final_range
    p = self.progress
    return (low0 + p * (low1 - low0), high0 + p * (high1 - high0))

  def _smooth(self, average: float | None, batch: torch.Tensor) -> float:
    """EMA with a per-sample weight of 1 / success_window."""
    mean = batch.float().mean().item()
    if average is None:
      return mean
    weight = 1.0 - (1.0 - 1.0 / self.success_window) ** len(batch)
    return average + weight * (mean - average)

  def _update_averages(self, env_ids: torch.Tensor | slice) -> None:
    env = self.env
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command = env.command_manager.get_term(self.command_name)
    # Only episodes that actually ended this step (not the initial reset),
    # and that started standing.
    terminations = env.termination_manager
    ended = env_ids[terminations.dones[env_ids] & ~command.replayed[env_ids]]
    if len(ended) == 0:
      return
    success = terminations.get_term(self.termination_name)[ended]
    self.success_rate = self._smooth(self.success_rate, success)
    # Runs before the command's reset: the ended episodes' apex and target
    # are still in place.
    jumped = ended[command.apex_reached[ended]]
    if len(jumped):
      error = (command.peak_height[jumped] - command.target_height[jumped]).abs()
      self.apex_error = self._smooth(self.apex_error, error)

  def _apex_accurate(self) -> bool:
    return self.apex_error is None or self.apex_error <= self.max_apex_error

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice, **params
  ) -> dict[str, float]:
    del params
    self._update_averages(env_ids)
    step = env.common_step_counter
    # Steps since the last call, counting only those after the hold.
    elapsed = max(0, step - max(self.last_step, self.hold_steps))
    self.last_step = max(self.last_step, step)
    if elapsed and self.success_rate is not None:
      if self.success_rate < self.success_threshold:
        delta = -self.backoff_ratio * elapsed * self.rate
      elif self._apex_accurate():
        delta = elapsed * self.rate
      else:
        delta = 0.0
      self.progress = min(1.0, max(0.0, self.progress + delta))
    low, high = self.height_range()
    env.command_manager.get_term(self.command_name).height_range = (low, high)
    return {
      "min_height": low,
      "max_height": high,
      "progress": self.progress,
      "success_rate": -1.0 if self.success_rate is None else self.success_rate,
      "apex_error": -1.0 if self.apex_error is None else self.apex_error,
    }


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
