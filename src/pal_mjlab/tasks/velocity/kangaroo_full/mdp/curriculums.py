"""Curriculum terms for kangaroo_full.

mjlab's ``reward_curriculum`` only steps a weight between stages; the terms
here ramp linearly instead, for schedules that hand something over gradually
rather than all at once -- a reward's weight, or the clamp on the policy's
action std.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from pal_mjlab.tasks.velocity.kangaroo_full.runner import POLICY_DISTRIBUTION_ATTR

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class reward_weight_linear_ramp:
  """Ramp a reward term's weight linearly between two values over a step range.

  The weight is ``start_weight`` until ``env.common_step_counter`` reaches
  ``start_step``, interpolates linearly to ``end_weight`` at ``end_step``, and
  stays there afterwards.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_weight_linear_ramp,
      params={
        "reward_name": "pose",
        "start_step": 0,
        "end_step": 80_000,
        "start_weight": 2.0,
        "end_weight": 1.0,
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    start_step: int = cfg.params["start_step"]
    end_step: int = cfg.params["end_step"]
    if end_step <= start_step:
      raise ValueError(
        f"Curriculum '{reward_name}' needs end_step > start_step,"
        f" got start_step={start_step}, end_step={end_step}."
      )
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    start_step: int,
    end_step: int,
    start_weight: float,
    end_weight: float,
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name
    progress = (env.common_step_counter - start_step) / (end_step - start_step)
    progress = min(max(progress, 0.0), 1.0)
    self._term_cfg.weight = start_weight + progress * (end_weight - start_weight)
    return {"weight": torch.tensor(self._term_cfg.weight)}


class policy_std_range_linear_ramp:
  """Ramp the actor's ``std_range`` clamp linearly between two ranges.

  rsl_rl's ``GaussianDistribution`` re-reads ``std_range`` on every forward
  pass, so moving it moves the clamp on the learned std immediately, without
  touching the std parameter itself. Both ends of the range are interpolated
  independently: ``start_range`` until ``env.common_step_counter`` reaches
  ``start_step``, linearly to ``end_range`` at ``end_step``, then held.

  The distribution is reached through the attribute
  :data:`~pal_mjlab.tasks.velocity.kangaroo_full.runner.POLICY_DISTRIBUTION_ATTR`
  that :class:`~pal_mjlab.tasks.velocity.kangaroo_full.runner.KangarooFullOnPolicyRunner`
  sets on the env. Until it is set -- the env's own first reset happens before
  the runner exists, and an env built without a runner never gets one -- the
  term still computes and logs the range but has nothing to apply it to, so
  the runner cfg's ``std_range`` should match ``start_range``.

  Example::

    CurriculumTermCfg(
      func=mdp.policy_std_range_linear_ramp,
      params={
        "start_step": 0,
        "end_step": 9600,
        "start_range": (0.5, 1e6),
        "end_range": (1e-6, 1e6),
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del env
    start_step: int = cfg.params["start_step"]
    end_step: int = cfg.params["end_step"]
    if end_step <= start_step:
      raise ValueError(
        "policy_std_range_linear_ramp needs end_step > start_step,"
        f" got start_step={start_step}, end_step={end_step}."
      )
    for key in ("start_range", "end_range"):
      lo, hi = cfg.params[key]
      if not 0.0 < lo <= hi:
        raise ValueError(f"{key} must satisfy 0 < min <= max, got {(lo, hi)}.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_step: int,
    end_step: int,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
  ) -> dict[str, torch.Tensor]:
    del env_ids
    progress = (env.common_step_counter - start_step) / (end_step - start_step)
    progress = min(max(progress, 0.0), 1.0)
    std_min = start_range[0] + progress * (end_range[0] - start_range[0])
    std_max = start_range[1] + progress * (end_range[1] - start_range[1])
    distribution = getattr(env, POLICY_DISTRIBUTION_ATTR, None)
    if distribution is not None:
      # Mirror GaussianDistribution.__init__: it keeps both the plain and the
      # log-space range and picks one by std_type, so update both.
      distribution.std_range = [max(std_min, 1e-6), std_max]
      distribution.log_std_range = [
        math.log(distribution.std_range[0]),
        math.log(distribution.std_range[1]),
      ]
    return {
      "std_min": torch.tensor(std_min),
      "std_max": torch.tensor(std_max),
    }
