"""Curriculum terms for kangaroo_full.

mjlab's ``reward_curriculum`` only steps a weight between stages; the term here
ramps it linearly instead, for a schedule that hands a reward's emphasis over
gradually rather than all at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
