"""Training curricula of the vertical jump: target heights and settling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


class jump_height_curriculum:
  """Moves the sampled apex range up in stairs, gated by the jump success rate.

  The range is ``initial_range + offset`` (both bounds, so its width stays).
  ``offset`` starts at 0 and does not change for the first ``hold_steps``
  environment steps. From then on the curriculum decides once every
  ``stair_steps`` steps (the first decision at ``hold_steps``):

  - smoothed success rate below ``success_threshold``: back off by
    ``step_down`` (never below 0);
  - otherwise, smoothed apex error within ``max_apex_error`` (or no apex yet):
    climb by ``step_up``, without the upper bound passing ``max_height``;
  - otherwise (jumps succeed but miss their target): stay.

  Every move starts a new curriculum *stage* (``stage`` counts them; the
  replay buffer tags its states with it) and restarts both averages, so the
  next decision only sees episodes of the current range. Only the bounds
  change; active targets stay fixed until reset.

  The success rate is the share of finished episodes that ended on
  ``termination_name``. Episodes restarted from a replayed state
  (jump_state_replay) start after takeoff and are left out. It is an
  exponential moving average with a per-episode weight of
  ``1 / success_window``, so it spans about ``success_window`` episodes.
  The apex error is ``|apex - target|`` of finished, non-replayed episodes
  that reached their apex (as the command's apex_height_error metric),
  averaged the same way per jump.

  The state is not derived from common_step_counter: ``state_dict`` /
  ``load_state_dict`` carry it through checkpoints (JumpOnPolicyRunner).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.env = env
    self.command_name: str = params["command_name"]
    self.termination_name: str = params.get("termination_name", "jump_finished")
    self.initial_range = tuple(float(v) for v in params["initial_range"])
    self.max_height = float(params["max_height"])
    self.hold_steps = int(params["hold_steps"])
    self.stair_steps = int(params["stair_steps"])
    self.step_up = float(params.get("step_up", 0.05))
    self.step_down = float(params.get("step_down", 0.01))
    self.success_threshold = float(params.get("success_threshold", 0.5))
    self.success_window = float(params.get("success_window", 4096))
    self.max_apex_error = float(params.get("max_apex_error", math.inf))
    low, high = self.initial_range
    if not (0 < low <= high <= self.max_height < math.inf):
      raise ValueError(
        "Heights must be finite, positive, and ordered up to max_height."
      )
    if self.hold_steps < 0 or self.stair_steps < 1:
      raise ValueError("hold_steps must be >= 0 and stair_steps >= 1.")
    if not (self.step_up > 0 and self.step_down >= 0):
      raise ValueError("step_up must be positive and step_down non-negative.")
    if not 0 < self.success_threshold < 1:
      raise ValueError("success_threshold must be in (0, 1).")
    if self.success_window < 1:
      raise ValueError("success_window must be >= 1.")
    if not self.max_apex_error > 0:
      raise ValueError("max_apex_error must be positive.")
    self.max_offset = self.max_height - high
    self.offset = 0.0
    self.stage = 0
    # None until the first episode of the stage ends.
    self.success_rate: float | None = None
    # None until the stage's first non-replayed episode with an apex ends.
    self.apex_error: float | None = None
    # Environment step of the next decision.
    self.next_step = self.hold_steps

  @property
  def failing(self) -> bool:
    """Whether the current stage's jumps succeed less than success_threshold."""
    return self.success_rate is not None and self.success_rate < self.success_threshold

  def state_dict(self) -> dict[str, float | int | None]:
    return {
      "offset": self.offset,
      "stage": self.stage,
      "success_rate": self.success_rate,
      "apex_error": self.apex_error,
      "next_step": self.next_step,
    }

  def load_state_dict(self, state: dict[str, float | int | None]) -> None:
    self.offset = float(state["offset"])
    self.stage = int(state["stage"])
    self.success_rate = state["success_rate"]
    self.apex_error = state["apex_error"]
    self.next_step = int(state["next_step"])

  def height_range(self) -> tuple[float, float]:
    low, high = self.initial_range
    return (low + self.offset, high + self.offset)

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

  def _decide(self) -> None:
    if self.success_rate is None:
      return  # No episode of this stage has ended yet: decide next time.
    if self.failing:
      offset = max(0.0, self.offset - self.step_down)
    elif self._apex_accurate():
      offset = min(self.max_offset, self.offset + self.step_up)
    else:
      offset = self.offset
    if not math.isclose(offset, self.offset, abs_tol=1e-9):
      self.offset = offset
      self.stage += 1
      self.success_rate = None
      self.apex_error = None

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice, **params
  ) -> dict[str, float]:
    del params
    self._update_averages(env_ids)
    step = env.common_step_counter
    if step >= self.next_step:
      self._decide()
      # One decision per stair_steps, however far apart the calls are.
      periods = (step - self.next_step) // self.stair_steps + 1
      self.next_step += periods * self.stair_steps
    low, high = self.height_range()
    env.command_manager.get_term(self.command_name).height_range = (low, high)
    return {
      "min_height": low,
      "max_height": high,
      "offset": self.offset,
      "stage": float(self.stage),
      "success_rate": -1.0 if self.success_rate is None else self.success_rate,
      "apex_error": -1.0 if self.apex_error is None else self.apex_error,
    }
