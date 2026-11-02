"""Jump-height command and motion-end termination for jump-motion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from pal_mjlab.tasks.jumping.mdp.commands import (
  JumpHeightCommand,
  JumpHeightCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp import MotionCommand


class MotionApexHeight:
  """Highest world-frame z of one robot site (or the root body) over a motion.

  Replays the motion's root pose and joint positions through forward kinematics
  on a CPU copy of the model and reads the frame's own placement. Model ids and
  qpos addresses are resolved once, here.
  """

  def __init__(self, env: ManagerBasedRlEnv, robot: Entity, site_id: int | None):
    self.model = env.sim.mj_model
    self.data = mujoco.MjData(self.model)
    indexing = robot.indexing
    self.root_name = self.model.body(indexing.root_body_id).name.split("/")[-1]
    self.site_id = None if site_id is None else int(indexing.site_ids[site_id])
    self.root_body_id = indexing.root_body_id
    self.free_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    self.joint_q_adr = indexing.joint_q_adr.cpu().numpy()

  def __call__(self, motion: MotionCommand) -> float:
    if motion.cfg.anchor_body_name != self.root_name:
      raise ValueError(
        f"The motion anchor must be the root body {self.root_name!r} to replay it."
      )
    anchor = motion.motion_anchor_body_index
    root_pos = motion.motion.body_pos_w[:, anchor].cpu().numpy()
    root_quat = motion.motion.body_quat_w[:, anchor].cpu().numpy()
    joint_pos = motion.motion.joint_pos.cpu().numpy()
    data = self.data
    data.qpos[:] = self.model.qpos0
    heights = np.empty(len(joint_pos))
    for t in range(len(joint_pos)):
      data.qpos[self.free_q_adr[:3]] = root_pos[t]
      data.qpos[self.free_q_adr[3:7]] = root_quat[t]
      data.qpos[self.joint_q_adr] = joint_pos[t]
      mujoco.mj_kinematics(self.model, data)
      heights[t] = (
        data.xpos[self.root_body_id, 2]
        if self.site_id is None
        else data.site_xpos[self.site_id, 2]
      )
    return float(heights.max())


class MotionApexJumpHeightCommand(JumpHeightCommand):
  """The vertical-jump task's command, with the tracked motion's apex as target.

  Each episode's target is the motion's highest world-frame z of the jump frame
  (cfg.height_site_name, else the root body; see MotionApexHeight) plus
  U(-apex_noise, apex_noise). Phases, and the observed command (the target
  until both feet land, then zero), follow ``JumpHeightCommand``'s state
  machine exactly, so a policy observing it transfers to the jump task. With no
  jump reward or termination to drive that state machine, it also advances on
  the command manager's per-step update.
  """

  cfg: MotionApexJumpHeightCommandCfg

  def __init__(self, cfg: MotionApexJumpHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._apex = MotionApexHeight(env, self.robot, self.height_site_id)
    self._apex_height: float | None = None

  @property
  def apex_height(self) -> float:
    """World-frame apex of the jump frame over the whole motion, computed once
    on first use (after the command manager has built the motion term)."""
    if self._apex_height is None:
      self._apex_height = self._apex(
        self._env.command_manager.get_term(self.cfg.motion_command_name)
      )
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
