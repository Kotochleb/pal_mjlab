"""Jump-height command and motion-end termination for jump-motion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pal_mjlab.tasks.jumping.mdp.commands import (
  JumpHeightCommand,
  JumpHeightCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp import MotionCommand


class MotionApexJumpHeightCommand(JumpHeightCommand):
  """The vertical-jump task's command, with the tracked motion's apex as target.

  Each episode's target is the motion anchor's highest world-frame z plus
  U(-apex_noise, apex_noise). Phases, and the observed command (the target
  until both feet land, then zero), follow ``JumpHeightCommand``'s state
  machine exactly, so a policy observing it transfers to the jump task. With no
  jump reward or termination to drive that state machine, it also advances on
  the command manager's per-step update.
  """

  cfg: MotionApexJumpHeightCommandCfg

  def __init__(self, cfg: MotionApexJumpHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._apex_height: float | None = None

  @property
  def apex_height(self) -> float:
    """World-frame apex of the motion anchor over the whole motion."""
    if self._apex_height is None:
      motion: MotionCommand = self._env.command_manager.get_term(
        self.cfg.motion_command_name
      )
      anchor_z = motion.motion.body_pos_w[:, motion.motion_anchor_body_index, 2]
      self._apex_height = float(anchor_z.max())
    return self._apex_height

  def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
    noise = self.cfg.apex_noise
    return self.apex_height + torch.empty(len(env_ids), device=self.device).uniform_(
      -noise, noise
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.update_state()
    super()._update_command(env_ids)


@dataclass(kw_only=True)
class MotionApexJumpHeightCommandCfg(JumpHeightCommandCfg):
  motion_command_name: str = "motion"
  # Half-width, in metres, of the uniform noise on each episode's target apex.
  apex_noise: float = 0.1

  def __post_init__(self):
    super().__post_init__()
    if not 0 <= self.apex_noise < float("inf"):
      raise ValueError("apex_noise must be finite and non-negative.")

  def build(self, env: ManagerBasedRlEnv) -> MotionApexJumpHeightCommand:
    return MotionApexJumpHeightCommand(self, env)


def motion_finished(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Truncate on the motion's last frame: one jump per episode, never wrapped."""
  motion: MotionCommand = env.command_manager.get_term(command_name)
  return motion.time_steps >= motion.motion.time_step_total - 1
