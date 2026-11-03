"""Replay of apex, touchdown and landing states from the best finished jumps."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from .commands import FLIGHT, LANDED, SETTLED, STANCE, JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import EventTermCfg

# The jump's state transitions an episode's recording can follow, in order:
# takeoff (STANCE -> FLIGHT), the apex, touchdown (FLIGHT -> LANDED, both feet
# down) and settling (LANDED -> SETTLED).
TRANSITIONS = ("takeoff", "apex", "touchdown", "settled")

# (read rows, write rows) of one piece of per-env state.
_Field = tuple[
  Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], None]
]


class jump_state_replay:
  """Restarts some episodes from states of the best earlier finished jumps.

  Use with ``mode="step"``. Every env records at most one state per episode.
  On reset it draws one of TRANSITIONS uniformly and a uniform offset of 0 to
  ``max_offset`` seconds (in whole steps), and records the state that many
  steps after the first time that transition happens. The state is discarded
  when it is not full: an observation history still holds backfilled frames,
  or the episode ends on that step (a restart would end at once). An episode
  that ends before the recording, or never reaches its transition, records
  nothing. When the episode ends on ``termination_name`` (the jump finished),
  its recorded state joins the buffer, scored with the episode's return (the
  sum of every reward term's episode sum, this step's reward included); other
  endings, and episodes that recorded nothing, add nothing. Episodes that were themselves restarted from the
  buffer are not stored: their return only covers the part after the restart.
  The buffer holds up to ``capacity`` states; once full, a new state replaces
  the lowest-scoring one when it scores higher.

  With ``curriculum_name`` (the jump height curriculum), every state is tagged
  with the curriculum stage it was stored in. While that curriculum is not
  failing (its success rate is at least its threshold, or not known yet), at
  least ``keep_previous_fraction`` of the capacity stays with states of
  earlier stages, as long as there are that many: new states only replace
  earlier-stage ones above that share, otherwise current-stage ones. While it
  fails, new states may replace any state, as without a curriculum.

  On reset (``reset``, after every manager has reset), each env restarts with
  probability ``probability`` from a state drawn uniformly among the best
  ``top_fraction`` of the buffer by score, otherwise from the reset events'
  standing pose. A state holds everything the episode continues
  from: the simulator's qpos/qvel (root xy relative to the env origin), the
  action history, the jump command's state machine, the contact sensors' air
  and contact timers, and every observation term's history. It is captured
  before the step's observations are appended, so the post-reset append
  completes the history exactly as the original episode's did. The restored
  target gets U(-target_noise, target_noise), shifted into the history of the
  observed command (whose nonzero frames are that target) too.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.env = env
    self.command_name: str = params["command_name"]
    self.termination_name: str = params.get("termination_name", "jump_finished")
    self.probability: float = params["probability"]
    self.capacity: int = params["capacity"]
    self.top_fraction: float = params.get("top_fraction", 0.25)
    self.max_offset: float = params["max_offset"]
    self.target_noise: float = params["target_noise"]
    self.curriculum_name: str | None = params.get("curriculum_name")
    self.keep_previous_fraction: float = params.get("keep_previous_fraction", 0.5)
    if not 0 <= self.keep_previous_fraction <= 1:
      raise ValueError("keep_previous_fraction must be in [0, 1].")
    if not 0 <= self.probability <= 1:
      raise ValueError("probability must be in [0, 1].")
    if not 0 < self.top_fraction <= 1:
      raise ValueError("top_fraction must be in (0, 1].")
    if self.max_offset < 0:
      raise ValueError("max_offset must be non-negative.")
    if self.capacity < 1 or self.target_noise < 0:
      raise ValueError("capacity must be positive and target_noise non-negative.")
    # Managers are built after the event manager: resolved on first use.
    self._fields: dict[str, _Field] | None = None
    self._command_history: list[tuple[str, str]] = []
    self.size = 0

  def _setup(self) -> None:
    env = self.env
    self.command: JumpHeightCommand = env.command_manager.get_term(self.command_name)
    robot = self.command.robot
    data = env.sim.data
    root_xy = robot.indexing.free_joint_q_adr[:2]
    origins = env.scene.env_origins

    def read_qpos(ids: torch.Tensor) -> torch.Tensor:
      qpos = data.qpos[ids].clone()
      qpos[:, root_xy] -= origins[ids, :2]
      return qpos

    def write_qpos(ids: torch.Tensor, value: torch.Tensor) -> None:
      value = value.clone()
      value[:, root_xy] += origins[ids, :2]
      data.qpos[ids] = value

    fields: dict[str, _Field] = {"qpos": (read_qpos, write_qpos)}
    sim_fields = ["qvel", "act", "qacc_warmstart"]
    for name in sim_fields:
      tensor = getattr(data, name, None)
      if tensor is not None and tensor.shape[-1] > 0:
        fields[name] = self._tensor_field(tensor)
    actions = env.action_manager
    for name in ("_action", "_prev_action", "_prev_prev_action"):
      fields[f"actions/{name}"] = self._tensor_field(getattr(actions, name))
    for sensor_name, sensor in env.scene.sensors.items():
      timers = getattr(sensor, "_air_time_state", None)
      if timers is None:
        continue
      for name, value in vars(timers).items():
        fields[f"sensor/{sensor_name}/{name}"] = self._tensor_field(value)
    observations = env.observation_manager
    for group, delays in observations._group_obs_term_delay_buffer.items():
      if delays:
        raise NotImplementedError(f"Delayed observations ({group}) cannot be replayed.")
    for group, buffers in observations._group_obs_term_history_buffer.items():
      for term, buffer in buffers.items():
        fields[f"history/{group}/{term}"] = self._history_field(buffer)
        params = observations.get_term_cfg(group, term).params
        if term == "command" and params.get("command_name") == self.command_name:
          self._command_history.append((group, term))
    self._fields = fields
    self._pending: dict[str, torch.Tensor] = {}
    self._buffer: dict[str, torch.Tensor] = {}
    num_envs, device = env.num_envs, env.device
    self._has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._scores = torch.full((self.capacity,), -math.inf, device=device)
    # Curriculum stage each stored state was added in.
    self._stages = torch.zeros(self.capacity, dtype=torch.long, device=device)
    self._curriculum = (
      None
      if self.curriculum_name is None
      else env.curriculum_manager.get_term_cfg(self.curriculum_name).func
    )
    self._max_offset_steps = round(self.max_offset / env.step_dt)
    # Per env: the drawn transition and offset (steps), whether the transition
    # has happened and the steps still to wait after it, whether this episode's
    # recording step has passed, and the jump phase on the previous step.
    self._transition = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._offset = torch.zeros_like(self._transition)
    self._armed = torch.zeros_like(self._has_pending)
    self._wait = torch.zeros_like(self._transition)
    self._spent = torch.zeros_like(self._has_pending)
    self._previous_phase = self.command.phase.clone()
    self._histories = [
      buffer
      for buffers in observations._group_obs_term_history_buffer.values()
      for buffer in buffers.values()
    ]
    self._draw_capture(torch.arange(num_envs, device=device))

  def _draw_capture(self, env_ids: torch.Tensor) -> None:
    """Draws each env's transition and offset for its next episode."""
    n, device = len(env_ids), self.env.device
    self._transition[env_ids] = torch.randint(0, len(TRANSITIONS), (n,), device=device)
    self._offset[env_ids] = torch.randint(
      0, self._max_offset_steps + 1, (n,), device=device
    )
    self._armed[env_ids] = False
    self._spent[env_ids] = False
    self._has_pending[env_ids] = False

  @staticmethod
  def _tensor_field(tensor) -> _Field:
    def read(ids: torch.Tensor) -> torch.Tensor:
      return tensor[ids].clone()

    def write(ids: torch.Tensor, value: torch.Tensor) -> None:
      tensor[ids] = value

    return read, write

  @staticmethod
  def _history_field(buffer) -> _Field:
    """Frames of a CircularBuffer in chronological order (oldest first).

    Written so that every slot is filled (no backfill on the next append) and
    the next append drops the oldest frame, as in the original episode.
    """

    def order() -> torch.Tensor:
      start = (buffer._pointer + 1) % buffer._max_len
      return (torch.arange(buffer._max_len, device=buffer.device) + start) % (
        buffer._max_len
      )

    def read(ids: torch.Tensor) -> torch.Tensor:
      return buffer._buffer[:, ids][order()].transpose(0, 1).clone()

    def write(ids: torch.Tensor, value: torch.Tensor) -> None:
      raw = buffer._buffer
      raw[order().unsqueeze(1), ids.unsqueeze(0)] = value.transpose(0, 1)
      buffer._num_pushes[ids] = buffer._max_len

    return read, write

  def _read(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    assert self._fields is not None
    state = {name: read(ids) for name, (read, _) in self._fields.items()}
    for name, value in self.command.snapshot(ids).items():
      state[f"command/{name}"] = value
    return state

  def _write(self, ids: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
    assert self._fields is not None
    for name, (_, write) in self._fields.items():
      write(ids, state[name])
    command = {
      name.removeprefix("command/"): value
      for name, value in state.items()
      if name.startswith("command/")
    }
    noise = torch.empty(len(ids), device=self.env.device).uniform_(
      -self.target_noise, self.target_noise
    )
    command["target_height"] = command["target_height"] + noise
    for group, term in self._command_history:
      _, write = self._fields[f"history/{group}/{term}"]
      frames = state[f"history/{group}/{term}"]
      shift = noise.view(-1, *([1] * (frames.ndim - 1)))
      write(ids, torch.where(frames != 0, frames + shift, frames))
    self.command.restore(ids, command)

  def _store(self, target: dict[str, torch.Tensor], index, state, rows: int) -> None:
    for name, value in state.items():
      if name not in target:
        target[name] = torch.zeros(
          (rows, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
      target[name][index] = value

  def _stage(self) -> tuple[int, bool]:
    """(current curriculum stage, whether earlier-stage states are protected)."""
    if self._curriculum is None:
      return 0, False
    return int(self._curriculum.stage), not self._curriculum.failing

  def _insert(self, state: dict[str, torch.Tensor], scores: torch.Tensor) -> None:
    """Adds scored states: free slots first, then over lower-scoring ones,
    keeping the protected share of earlier-stage states (see the class)."""
    device = scores.device
    stage, protect = self._stage()
    free = min(len(scores), self.capacity - self.size)
    if free:
      slots = torch.arange(self.size, self.size + free, device=device)
      self._store(
        self._buffer, slots, {k: v[:free] for k, v in state.items()}, self.capacity
      )
      self._scores[slots] = scores[:free]
      self._stages[slots] = stage
      self.size += free
    rest = scores[free:]
    if len(rest) == 0:
      return
    # Keep the best `capacity` of the full buffer and the remaining candidates,
    # after reserving the best earlier-stage states when protected.
    total = self.capacity + len(rest)
    candidates = torch.cat([self._scores, rest])
    keep = torch.zeros(total, dtype=torch.bool, device=device)
    if protect:
      previous = (self._stages < stage).nonzero().squeeze(-1)
      reserved = min(
        len(previous), math.ceil(self.keep_previous_fraction * self.capacity)
      )
      if reserved:
        best = torch.topk(self._scores[previous], reserved).indices
        keep[previous[best]] = True
    others = (~keep).nonzero().squeeze(-1)
    chosen = torch.topk(candidates[others], self.capacity - int(keep.sum())).indices
    keep[others[chosen]] = True
    slots = (~keep[: self.capacity]).nonzero().squeeze(-1)
    incoming = keep[self.capacity :].nonzero().squeeze(-1)
    if len(incoming):
      rows = incoming + free
      self._store(
        self._buffer, slots, {k: v[rows] for k, v in state.items()}, self.capacity
      )
      self._scores[slots] = scores[rows]
      self._stages[slots] = stage

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params):
    del env_ids, params
    if self._fields is None:
      self._setup()
    command = self.command
    command.update_state()
    # Restarted episodes are never stored (see the class docstring).
    fresh = ~command.replayed
    phase, previous = command.phase, self._previous_phase
    events = torch.stack(
      (
        (previous == STANCE) & (phase == FLIGHT),
        command.just_apex,
        command.just_touchdown,
        (previous == LANDED) & (phase == SETTLED),
      ),
      dim=-1,
    )
    self._previous_phase = phase.clone()
    # The drawn transition starts the wait; an offset of 0 records on its step.
    reached = (
      fresh
      & ~self._armed
      & ~self._spent
      & events.gather(-1, self._transition.unsqueeze(-1)).squeeze(-1)
    )
    self._armed |= reached
    self._wait[reached] = self._offset[reached]
    due = self._armed & (self._wait == 0)
    self._wait[self._armed & ~due] -= 1
    self._armed &= ~due
    self._spent |= due
    full = ~env.termination_manager.dones
    for buffer in self._histories:
      full &= buffer.current_length >= buffer.max_length
    ids = (due & full).nonzero().squeeze(-1)
    if len(ids):
      self._store(self._pending, ids, self._read(ids), env.num_envs)
      self._has_pending[ids] = True
    done = env.termination_manager.get_term(self.termination_name) & fresh
    ids = (done & self._has_pending).nonzero().squeeze(-1)
    if len(ids) == 0:
      return
    episode_return = sum(env.reward_manager._episode_sums.values())
    state = {name: value[ids] for name, value in self._pending.items()}
    self._insert(state, episode_return[ids])

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._fields is None:
      return
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.env.num_envs, device=self.env.device)[
        slice(None) if env_ids is None else env_ids
      ]
    self._draw_capture(env_ids)
    self._previous_phase[env_ids] = self.command.phase[env_ids]
    if self.size == 0:
      return
    chosen = torch.rand(len(env_ids), device=self.env.device) < self.probability
    ids = env_ids[chosen]
    if len(ids) == 0:
      return
    top = max(1, math.ceil(self.top_fraction * self.size))
    best = torch.topk(self._scores[: self.size], top).indices
    slots = best[torch.randint(0, top, (len(ids),), device=self.env.device)]
    self._write(ids, {name: value[slots] for name, value in self._buffer.items()})
