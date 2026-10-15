"""LP-ACRL over jump-height bins, reusing the velocity task's teacher.

Each task is one equal-width interval of target rise. The shared
lp_acrl_curriculum term records the finished episode's return for its bin,
updates the teacher, and assigns a new bin before the command is resampled.
Only LpAcrlCfg's teacher fields apply; its velocity_bins and include_standing
have no meaning here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from pal_mjlab.tasks.jumping.mdp.commands import JumpHeightCommand, JumpHeightCommandCfg
from pal_mjlab.tasks.velocity.mdp.lp_acrl import LearningProgressSampler, LpAcrlCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LpAcrlJumpHeightCommand(JumpHeightCommand):
  """Sample each episode's target uniformly inside its teacher-selected bin."""

  cfg: LpAcrlJumpHeightCommandCfg

  def __init__(self, cfg: LpAcrlJumpHeightCommandCfg, env: ManagerBasedRlEnv):
    cfg.__post_init__()  # Also validate configs modified by CLI after construction.
    super().__init__(cfg, env)
    low, high = cfg.height_range
    width = (high - low) / cfg.height_bins
    self.height_lows = low + torch.arange(cfg.height_bins, device=self.device) * width
    self.height_highs = self.height_lows + width
    self.sampler = LearningProgressSampler(cfg.height_bins, cfg.lp_acrl, self.device)
    self.task_ids = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.task_space = (tuple(cfg.height_range), cfg.height_bins)
    self.metrics["height_bin"] = torch.zeros_like(self.target_height)

  def select_tasks(self, env_ids: torch.Tensor) -> None:
    self.task_ids[env_ids] = self.sampler.sample(len(env_ids))

  def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
    tasks = self.task_ids[env_ids]
    if (tasks < 0).any():
      raise RuntimeError("LP-ACRL requires the lp_acrl curriculum reset term")
    lows = self.height_lows[tasks]
    return lows + torch.rand_like(lows) * (self.height_highs[tasks] - lows)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["height_bin"][:] = self.task_ids.float()

  def curriculum_state_dict(self) -> dict:
    return {"task_space": self.task_space, "sampler": self.sampler.state_dict()}

  def load_curriculum_state_dict(self, state: dict) -> None:
    if not self.cfg.lp_acrl.adaptive:
      return  # Playback is uniform evaluation, independent of training weights.
    if state["task_space"] != self.task_space:
      raise ValueError("LP-ACRL checkpoint has a different task space")
    self.sampler.load_state_dict(state["sampler"])


@dataclass(kw_only=True)
class LpAcrlJumpHeightCommandCfg(JumpHeightCommandCfg):
  height_bins: int = 15
  """Equal-width bins over height_range, each one LP-ACRL task."""
  lp_acrl: LpAcrlCfg = field(default_factory=LpAcrlCfg)

  def __post_init__(self):
    super().__post_init__()
    self.lp_acrl.__post_init__()
    if not isinstance(self.height_bins, int) or self.height_bins < 1:
      raise ValueError("height_bins must be a positive integer.")

  def build(self, env: ManagerBasedRlEnv) -> LpAcrlJumpHeightCommand:
    return LpAcrlJumpHeightCommand(self, env)
