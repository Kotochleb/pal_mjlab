"""Replay of falling and landed states from successful jumps as episode starts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from .commands import FLIGHT, LANDED, JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import EventTermCfg

# (read rows, write rows) of one piece of per-env state.
_Field = tuple[
  Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], None]
]


class jump_state_replay:
  """Restarts some episodes from states of earlier successful jumps.

  Use with ``mode="step"``. Every step, each env keeps one uniformly sampled
  state of its current episode (reservoir sampling) among those that are
  falling (in flight after the apex) or landed (both feet down, not yet
  settled). When the episode ends on ``termination_name`` (the jump finished),
  that state joins a buffer of up to ``capacity`` states; other endings drop it.

  On reset (``reset``, after every manager has reset), each env restarts from a
  random buffered state with probability ``probability``, otherwise from the
  reset events' standing pose. A state holds everything the episode continues
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
    self.target_noise: float = params["target_noise"]
    if not 0 <= self.probability <= 1:
      raise ValueError("probability must be in [0, 1].")
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
    self._eligible_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)

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

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params):
    del env_ids, params
    if self._fields is None:
      self._setup()
    command = self.command
    command.update_state()
    falling = (command.phase == FLIGHT) & command.apex_reached
    eligible = falling | (command.phase == LANDED)
    self._eligible_steps += eligible
    keep = eligible & (
      torch.rand(env.num_envs, device=env.device) * self._eligible_steps < 1
    )
    ids = keep.nonzero().squeeze(-1)
    if len(ids):
      self._store(self._pending, ids, self._read(ids), env.num_envs)
      self._has_pending[ids] = True
    done = env.termination_manager.get_term(self.termination_name)
    success = (done & self._has_pending).nonzero().squeeze(-1)
    if len(success):
      free = min(len(success), self.capacity - self.size)
      slots = torch.cat(
        [
          torch.arange(self.size, self.size + free, device=env.device),
          torch.randint(0, self.capacity, (len(success) - free,), device=env.device),
        ]
      )
      self.size += free
      state = {name: value[success] for name, value in self._pending.items()}
      self._store(self._buffer, slots, state, self.capacity)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._fields is None:
      return
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.env.num_envs, device=self.env.device)[
        slice(None) if env_ids is None else env_ids
      ]
    self._eligible_steps[env_ids] = 0
    self._has_pending[env_ids] = False
    if self.size == 0:
      return
    chosen = torch.rand(len(env_ids), device=self.env.device) < self.probability
    ids = env_ids[chosen]
    if len(ids) == 0:
      return
    slots = torch.randint(0, self.size, (len(ids),), device=self.env.device)
    self._write(ids, {name: value[slots] for name, value in self._buffer.items()})
