"""Learning-progress automatic curriculum (Li et al., arXiv:2601.17428).

Tasks are Cartesian products of signed velocity intervals and terrain cells.
Equations (5)--(7) use mean episodic returns, their signed differences, and
softmax(LP / beta). Stage aggregation and optional uniform exploration are
inspired by FIBO-Physical-AI/curriculum_training's teacher_guided.py.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

Curriculum = Literal["default", "lp_acrl"]


@dataclass(kw_only=True)
class LpAcrlCfg:
  velocity_bins: tuple[int, int, int] = (8, 1, 4)
  """Equal-width bins for signed vx, vy and yaw rate; fixed axes collapse to one."""
  beta: float = 1.0
  """Softmax temperature in units of total (dt-scaled) episodic reward."""
  update_interval_steps: int = 1200
  """Environment steps per stage (50 PPO iterations at the default rollout size)."""
  min_episodes_per_task: int = 4
  uniform_mix: float = 0.05
  """Exploration fraction, as in the reference code; zero gives Eq. (7) exactly."""
  include_standing: bool = True
  adaptive: bool = True
  """False samples tasks uniformly and never collects or updates teacher state."""

  def __post_init__(self) -> None:
    if len(self.velocity_bins) != 3 or any(
      not isinstance(n, int) or n < 1 for n in self.velocity_bins
    ):
      raise ValueError("velocity_bins must contain three positive integers")
    if not math.isfinite(self.beta) or self.beta <= 0:
      raise ValueError("beta must be finite and positive")
    for name in ("update_interval_steps", "min_episodes_per_task"):
      value = getattr(self, name)
      if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if not 0.0 <= self.uniform_mix <= 1.0:
      raise ValueError("uniform_mix must be in [0, 1]")


class LearningProgressSampler:
  """Device-local, episode-weighted stage statistics, independent of the simulator."""

  _state_names = (
    "probabilities",
    "mean_returns",
    "learning_progress",
    "seen",
    "return_sums",
    "episode_counts",
  )

  def __init__(self, num_tasks: int, cfg: LpAcrlCfg, device: str):
    cfg.__post_init__()  # Also validate configs modified by CLI after construction.
    if num_tasks < 1:
      raise ValueError("num_tasks must be positive")
    self.cfg = cfg
    self.probabilities = torch.full((num_tasks,), 1.0 / num_tasks, device=device)
    self.mean_returns = torch.zeros(num_tasks, device=device)
    self.learning_progress = torch.zeros_like(self.mean_returns)
    self.return_sums = torch.zeros_like(self.mean_returns)
    self.episode_counts = torch.zeros(num_tasks, dtype=torch.long, device=device)
    self.seen = torch.zeros(num_tasks, dtype=torch.bool, device=device)
    self.last_update_step = 0

  def record(self, task_ids: torch.Tensor, returns: torch.Tensor) -> None:
    if not self.cfg.adaptive:
      return
    valid = (task_ids >= 0) & torch.isfinite(returns)
    ids = task_ids[valid]
    self.return_sums.scatter_add_(0, ids, returns[valid])
    self.episode_counts.scatter_add_(0, ids, torch.ones_like(ids))

  def update(self, step: int) -> None:
    if not self.cfg.adaptive or (
      step - self.last_update_step < self.cfg.update_interval_steps
    ):
      return
    self.last_update_step = step
    ready = self.episode_counts >= self.cfg.min_episodes_per_task
    means = self.return_sums / self.episode_counts.clamp_min(1)
    # First observations establish a baseline, not fictitious progress from zero.
    # Unobserved tasks get neutral LP, never a synthetic zero-return observation.
    self.learning_progress.zero_()
    comparable = ready & self.seen
    self.learning_progress[comparable] = (
      means[comparable] - self.mean_returns[comparable]
    )
    self.mean_returns[ready] = means[ready]
    self.seen |= ready
    self.return_sums[ready] = 0
    self.episode_counts[ready] = 0
    # Center in float64 before scaling, including for very small temperatures.
    scores = self.learning_progress.double()
    scores = (scores - scores.max()) / self.cfg.beta
    probabilities = torch.softmax(scores, dim=0).to(self.probabilities.dtype)
    self.probabilities.copy_(
      (1 - self.cfg.uniform_mix) * probabilities
      + self.cfg.uniform_mix / probabilities.numel()
    )

  def sample(self, count: int) -> torch.Tensor:
    if count == 0:
      return torch.empty(0, dtype=torch.long, device=self.probabilities.device)
    return torch.multinomial(self.probabilities, count, replacement=True)

  def state_dict(self) -> dict:
    return {
      **{name: getattr(self, name).clone() for name in self._state_names},
      "last_update_step": self.last_update_step,
    }

  def load_state_dict(self, state: dict) -> None:
    for name in self._state_names:
      target = getattr(self, name)
      if state[name].shape != target.shape:
        raise ValueError("LP-ACRL checkpoint has a different task space")
    for name in self._state_names:
      getattr(self, name).copy_(state[name])
    self.last_update_step = int(state["last_update_step"])


@runtime_checkable
class LpAcrlCommand(Protocol):
  """Any command term whose episodes are drawn from a LearningProgressSampler."""

  sampler: LearningProgressSampler
  task_ids: torch.Tensor

  def select_tasks(self, env_ids: torch.Tensor) -> None: ...

  def curriculum_state_dict(self) -> dict: ...

  def load_curriculum_state_dict(self, state: dict) -> None: ...


class LpAcrlVelocityCommand(UniformVelocityCommand):
  """Keep each task bin for an episode; timer resamples stay inside that bin."""

  cfg: LpAcrlVelocityCommandCfg

  def __init__(self, cfg: LpAcrlVelocityCommandCfg, env: ManagerBasedRlEnv):
    cfg.__post_init__()
    super().__init__(cfg, env)
    ranges = (cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z)
    bins = tuple(
      count if lower != upper else 1
      for count, (lower, upper) in zip(cfg.lp_acrl.velocity_bins, ranges, strict=True)
    )
    grid = torch.cartesian_prod(*[torch.arange(n, device=self.device) for n in bins])
    lower = torch.tensor([r[0] for r in ranges], device=self.device)
    upper = torch.tensor([r[1] for r in ranges], device=self.device)
    widths = (upper - lower) / torch.tensor(bins, device=self.device)
    self.velocity_lows = lower + grid * widths
    self.velocity_highs = self.velocity_lows + widths
    self.moving_tasks = grid.shape[0]
    if cfg.lp_acrl.include_standing:
      zeros = torch.zeros(1, 3, device=self.device)
      self.velocity_lows = torch.cat((self.velocity_lows, zeros))
      self.velocity_highs = torch.cat((self.velocity_highs, zeros))
    self.num_velocity_tasks = len(self.velocity_lows)
    self.terrain = env.scene.terrain
    origins = self.terrain.terrain_origins if self.terrain is not None else None
    self.terrain_shape = tuple(origins.shape[:2]) if origins is not None else (1, 1)
    num_tasks = math.prod(self.terrain_shape) * self.num_velocity_tasks
    self.sampler = LearningProgressSampler(num_tasks, cfg.lp_acrl, self.device)
    self.task_ids = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    # Validate semantics, not just tensor sizes, when resuming a checkpoint.
    generator = self.terrain.cfg.terrain_generator if self.terrain is not None else None
    self.task_space = (
      ranges,
      bins,
      cfg.lp_acrl.include_standing,
      self.terrain_shape,
      tuple(generator.sub_terrains) if generator is not None else (),
    )

  def select_tasks(self, env_ids: torch.Tensor) -> None:
    tasks = self.sampler.sample(len(env_ids))
    self.task_ids[env_ids] = tasks
    if self.terrain is not None and self.terrain.terrain_origins is not None:
      cells = tasks // self.num_velocity_tasks
      levels = cells // self.terrain_shape[1]
      types = cells % self.terrain_shape[1]
      self.terrain.terrain_levels[env_ids] = levels
      self.terrain.terrain_types[env_ids] = types
      self.terrain.env_origins[env_ids] = self.terrain.terrain_origins[levels, types]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    tasks = self.task_ids[env_ids]
    if (tasks < 0).any():
      raise RuntimeError("LP-ACRL requires the lp_acrl curriculum reset term")
    velocity_ids = tasks % self.num_velocity_tasks
    lows = self.velocity_lows[velocity_ids]
    highs = self.velocity_highs[velocity_ids]
    self.vel_command_b[env_ids] = lows + torch.rand_like(lows) * (highs - lows)
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_standing_env[env_ids] = velocity_ids == self.moving_tasks
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def curriculum_state_dict(self) -> dict:
    return {"task_space": self.task_space, "sampler": self.sampler.state_dict()}

  def load_curriculum_state_dict(self, state: dict) -> None:
    if not self.cfg.lp_acrl.adaptive:
      return  # Playback is uniform evaluation, independent of training weights.
    if state["task_space"] != self.task_space:
      raise ValueError("LP-ACRL checkpoint has a different task space")
    self.sampler.load_state_dict(state["sampler"])


@dataclass(kw_only=True)
class LpAcrlVelocityCommandCfg(UniformVelocityCommandCfg):
  lp_acrl: LpAcrlCfg = field(default_factory=LpAcrlCfg)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.lp_acrl.__post_init__()
    if (
      self.heading_command
      or self.ranges.heading is not None
      or self.rel_world_envs
      or self.rel_forward_envs
      or self.rel_standing_envs
    ):
      raise ValueError(
        "LP-ACRL requires body-frame velocity bins without heading/forward/standing "
        "overrides; use lp_acrl.include_standing for a separate standing task"
      )
    for interval in (
      self.ranges.lin_vel_x,
      self.ranges.lin_vel_y,
      self.ranges.ang_vel_z,
    ):
      if not all(math.isfinite(v) for v in interval) or interval[0] > interval[1]:
        raise ValueError("Velocity ranges must be finite, ordered intervals")

  def build(self, env: ManagerBasedRlEnv) -> LpAcrlVelocityCommand:
    return LpAcrlVelocityCommand(self, env)


def lp_acrl_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
) -> dict[str, torch.Tensor]:
  """Run before reward reset and before reset events place robots on terrain."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, LpAcrlCommand)
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  if command.sampler.cfg.adaptive:
    completed = env_ids[env.episode_length_buf[env_ids] > 0]
    returns = torch.zeros(len(completed), device=env.device)
    # mjlab's curriculum manager runs before RewardManager.reset clears these.
    # Keep total episodic return: dividing by actual duration hides early falls.
    for episode_sum in env.reward_manager._episode_sums.values():
      returns += episode_sum[completed]
    command.sampler.record(command.task_ids[completed], returns)
    command.sampler.update(env.common_step_counter)
  command.select_tasks(env_ids)
  sampler = command.sampler
  p = sampler.probabilities
  return {
    "entropy": -(p * p.clamp_min(1e-30).log()).sum(),
    "max_probability": p.max(),
    "mean_progress": sampler.learning_progress.mean(),
    "evaluated_fraction": sampler.seen.float().mean(),
  }


def configure_lp_acrl(
  cfg: ManagerBasedRlEnvCfg, curriculum: Curriculum, *, play: bool
) -> ManagerBasedRlEnvCfg:
  """Apply after the terrain-specific configuration has finished."""
  if curriculum == "default":
    return cfg
  if curriculum != "lp_acrl":
    raise ValueError(f"Unknown curriculum: {curriculum!r}")
  old = cfg.commands["twist"]
  assert isinstance(old, UniformVelocityCommandCfg)
  values = {
    f.name: deepcopy(getattr(old, f.name)) for f in fields(UniformVelocityCommandCfg)
  }
  ranges = values["ranges"]
  # The flat task's final scheduled range becomes the teacher's full task space.
  # Use the same bounds in play, where the legacy speed schedule is already removed.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "plane":
    ranges.lin_vel_x = (-2.0, 3.0)
    ranges.ang_vel_z = (-0.7, 0.7)
  ranges.heading = None
  values.update(
    heading_command=False,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    rel_standing_envs=0.0,
  )
  cfg.commands["twist"] = LpAcrlVelocityCommandCfg(
    **values, lp_acrl=LpAcrlCfg(adaptive=not play)
  )
  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["lp_acrl"] = CurriculumTermCfg(
    func=lp_acrl_curriculum, params={"command_name": "twist"}
  )
  # This reset event would overwrite the jointly sampled terrain during play.
  cfg.events.pop("randomize_terrain", None)
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    # Generator curriculum mode lays out one column/type, with levels in rows;
    # the LP sampler, rather than terrain_levels_vel, chooses cells in this grid.
    cfg.scene.terrain.terrain_generator.curriculum = True
  return cfg


TOP_SPEED_LIN_VEL_X_MAX = 10.0
# Bin widths match the flat LP-ACRL space (~0.6 m/s in vx, 2 m/s in vy,
# 0.35 rad/s in yaw) after widening vx to (-2, 10) and doubling vy/yaw.
TOP_SPEED_VELOCITY_BINS = (20, 3, 8)


def configure_top_speed(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Widen a flat LP-ACRL task: vx up to 10 m/s, doubled vy and yaw rate."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, LpAcrlVelocityCommandCfg)
  ranges = twist.ranges
  ranges.lin_vel_x = (ranges.lin_vel_x[0], TOP_SPEED_LIN_VEL_X_MAX)
  ranges.lin_vel_y = (2 * ranges.lin_vel_y[0], 2 * ranges.lin_vel_y[1])
  ranges.ang_vel_z = (2 * ranges.ang_vel_z[0], 2 * ranges.ang_vel_z[1])
  twist.lp_acrl.velocity_bins = TOP_SPEED_VELOCITY_BINS
  twist.__post_init__()
  # "last" logs the value at the reset step: the current maximum over all envs.
  for axis, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
    cfg.metrics[f"max_abs_command_{name}"] = MetricsTermCfg(
      func=max_abs_velocity_command,
      params={"command_name": "twist", "axis": axis},
      reduce="last",
    )
  return cfg


def max_abs_velocity_command(
  env: ManagerBasedRlEnv, command_name: str, axis: int
) -> torch.Tensor:
  """Largest |command| on one axis over all envs, broadcast to every env."""
  command = env.command_manager.get_command(command_name)
  return command[:, axis].abs().max().expand(env.num_envs)
