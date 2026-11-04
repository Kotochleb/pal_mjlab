"""Observation and domain randomization terms over the actuators.

An actuator is named as in ``pal_kangaroo_full.actuator_space.actuator_order``:
whichever of a tendon (``pal_kangaroo_full``'s leg screws) or a joint
(``pal_kangaroo_full_full``'s leg screws, the leg-length screw of both, every
pelvis and arm joint) carries that name on the entity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ActuatorReader:
  """A list of actuators, resolved once against an entity.

  ``reference_positions`` are the positions the relative readings are taken
  from; an actuator it does not name falls back to its joint's default, which
  a tendon does not have.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    names: Sequence[str],
    reference_positions: Mapping[str, float],
  ) -> None:
    asset: Entity = env.scene[asset_cfg.name]
    self.names = tuple(names)

    joint_cols: list[int] = []
    joint_ids: list[int] = []
    tendon_cols: list[int] = []
    tendon_ids: list[int] = []
    for col, name in enumerate(self.names):
      if name in asset.tendon_names:
        tendon_cols.append(col)
        tendon_ids.append(asset.tendon_names.index(name))
      elif name in asset.joint_names:
        joint_cols.append(col)
        joint_ids.append(asset.joint_names.index(name))
      else:
        raise ValueError(f"Entity '{asset_cfg.name}' has no actuator '{name}'")

    def _ids(values: list[int]) -> torch.Tensor:
      return torch.as_tensor(values, device=env.device, dtype=torch.long)

    self.joint_cols, self.joint_ids = _ids(joint_cols), _ids(joint_ids)
    self.tendon_cols, self.tendon_ids = _ids(tendon_cols), _ids(tendon_ids)

    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.reference = torch.zeros(env.num_envs, len(self.names), device=env.device)
    self.reference[:, self.joint_cols] = default_joint_pos[:, self.joint_ids]
    for col, name in enumerate(self.names):
      if name in reference_positions:
        self.reference[:, col] = reference_positions[name]
      elif name in asset.tendon_names:
        raise ValueError(f"Tendon '{name}' needs a reference position")

  def _gather(
    self, joint_values: torch.Tensor, tendon_values: torch.Tensor
  ) -> torch.Tensor:
    out = torch.empty(
      joint_values.shape[0],
      len(self.names),
      device=joint_values.device,
      dtype=joint_values.dtype,
    )
    out[:, self.joint_cols] = joint_values[:, self.joint_ids]
    out[:, self.tendon_cols] = tendon_values[:, self.tendon_ids]
    return out

  def position_rel(self, asset: Entity) -> torch.Tensor:
    position = self._gather(asset.data.joint_pos, asset.data.tendon_len)
    return position - self.reference

  def velocity(self, asset: Entity) -> torch.Tensor:
    return self._gather(asset.data.joint_vel, asset.data.tendon_vel)


class actuator_encoder_bias:
  """Persistent startup bias on each actuator's position reading.

  ``bias_half_ranges`` holds one ``U(-half, half)`` width per actuator. The
  joint-backed columns are also written to the entity's own ``encoder_bias``,
  which is what makes the stock joint action terms subtract them from their
  targets; ``ActuatorBiasedTendonLengthAction`` does the same for tendons.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.names = tuple(cfg.params["bias_half_ranges"])
    self.reader = ActuatorReader(
      env, asset_cfg, self.names, cfg.params["reference_positions"]
    )
    self.half_ranges = torch.tensor(
      [cfg.params["bias_half_ranges"][name] for name in self.names], device=env.device
    )
    self.bias = torch.zeros(env.num_envs, len(self.names), device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bias_half_ranges: Mapping[str, float],
    reference_positions: Mapping[str, float],
  ) -> None:
    del bias_half_ranges, reference_positions  # Resolved in __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    self.bias[env_ids] = sample_uniform(
      -self.half_ranges, self.half_ranges, (len(env_ids), len(self.names)), env.device
    )
    asset: Entity = env.scene[asset_cfg.name]
    asset.data.encoder_bias[env_ids[:, None], self.reader.joint_ids] = self.bias[
      env_ids[:, None], self.reader.joint_cols
    ]


class actuator_state:
  """``joint_pos_rel`` / ``joint_vel_rel`` over the actuators.

  With ``biased`` set, positions carry the ``encoder_bias`` event's bias.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.reader = ActuatorReader(
      env, params["asset_cfg"], params["actuator_names"], params["reference_positions"]
    )
    self.bias: torch.Tensor | None = None
    if params.get("biased", False):
      bias_event = env.event_manager.get_term_cfg("encoder_bias").func
      if not isinstance(bias_event, actuator_encoder_bias):
        raise TypeError("Biased actuator positions require actuator_encoder_bias")
      self.bias = bias_event.bias
      self.bias_cols = torch.as_tensor(
        [bias_event.names.index(name) for name in self.reader.names],
        device=env.device,
        dtype=torch.long,
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    actuator_names: Sequence[str],
    reference_positions: Mapping[str, float],
    mode: str = "pos",
    biased: bool = False,
  ) -> torch.Tensor:
    del actuator_names, reference_positions, biased  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    if mode == "vel":
      return self.reader.velocity(asset)
    position = self.reader.position_rel(asset)
    return position if self.bias is None else position + self.bias[:, self.bias_cols]
