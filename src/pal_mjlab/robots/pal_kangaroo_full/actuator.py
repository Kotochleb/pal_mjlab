"""A PD actuator whose joint-space torque passes through a position-dependent
mechanical transmission before being applied to the physical actuator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
from mjlab.actuator.actuator import ActuatorCmd, TransmissionType
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData


from mjlab.actuator.pd_actuator import (
  IdealPdActuator,
  IdealPdActuatorCfg,
  pd_torque,
)

TranssmitedIdealPdCfgT = TypeVar(
  "TranssmitedIdealPdCfgT", bound="TransmitedIdealPdActuatorCfg"
)


def load_transmission_table(csv_path: str | Path) -> torch.Tensor:
  data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
  table = torch.from_numpy(data)
  return table[torch.argsort(table[:, 0])]


def interpolate_transmission(pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
  xp = table[:, 0].contiguous()
  fp = table[:, 1].contiguous()

  pos_clamped = torch.clamp(pos, xp[0], xp[-1])
  idx = torch.clamp(torch.searchsorted(xp, pos_clamped), 1, xp.numel() - 1)

  x0, x1 = xp[idx - 1], xp[idx]
  f0, f1 = fp[idx - 1], fp[idx]
  t = (pos_clamped - x0) / (x1 - x0)
  return f0 + t * (f1 - f0)


@dataclass(kw_only=True)
class TransmitedIdealPdActuatorCfg(IdealPdActuatorCfg):
  """Configuration for a PD actuator behind a nonlinear mechanical transmission.

  The PD control law runs in joint space: `effort_limit` (inherited) bounds
  the joint torque. That torque is then transmitted to the physical actuator
  by multiplying it with a position-dependent Jacobian looked up from
  `transmission`, and the result is clamped to `actuator_effort_limit`.
  """

  joint_to_actuator_map: dict[str, str]
  """Map between joints servoed and actuators executing motion."""

  transmission: torch.Tensor
  """Transmission lookup table: (N, 2) tensor of (joint_pos, force_J) rows,
  sorted by joint_pos ascending. See `load_transmission_table`."""

  actuator_effort_limit: float = float("inf")
  """Maximum force/torque limit downstream of the transmission (actuator-side,
  as opposed to `effort_limit` which bounds the upstream joint torque)."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> TranssmitedIdealPdActuator:
    return TranssmitedIdealPdActuator(self, entity, target_ids, target_names)


class TranssmitedIdealPdActuator(IdealPdActuator, Generic[TranssmitedIdealPdCfgT]):
  """PD actuator whose joint torque is transmitted through a position-dependent
  Jacobian before being applied to the physical actuator."""

  param_names = ("stiffness", "damping", "force_limit", "actuator_force_limit")

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    force_limit = params["force_limit"]
    joint_torque = torch.clamp(torque, -force_limit, force_limit)

    J = interpolate_transmission(cmd.pos, params["transmission"])
    actuator_torque = joint_torque * J

    actuator_force_limit = params["actuator_force_limit"]
    return torch.clamp(actuator_torque, -actuator_force_limit, actuator_force_limit)

  def __init__(
    self,
    cfg: TranssmitedIdealPdCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.actuator_force_limit: torch.Tensor | None = None
    self.default_actuator_force_limit: torch.Tensor | None = None
    self.transmission: torch.Tensor | None = None

    servo_joint_ids, _ = entity.find_joints(target_names, preserve_order=True)
    assert set(target_names) == set(cfg.joint_to_actuator_map.keys()), (
      f"{type(self).__name__}: target_names {target_names} must match "
      f"joint_to_actuator_map keys {list(cfg.joint_to_actuator_map.keys())}."
    )
    self._servo_joint_ids_list = servo_joint_ids
    self._servo_joint_ids: torch.Tensor | None = None

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    params = {name: getattr(self, name) for name in self.param_names}
    params["transmission"] = self.transmission
    return type(self).control_law(params, cmd)

  def get_command(self, data: EntityData) -> ActuatorCmd:
    assert self.transmission_type == TransmissionType.JOINT, (
      f"{type(self).__name__} only supports JOINT transmission "
      f"(got {self.transmission_type})."
    )
    assert self._servo_joint_ids is not None
    ids = self._servo_joint_ids
    return ActuatorCmd(
      position_target=data.joint_pos_target[:, ids],
      velocity_target=data.joint_vel_target[:, ids],
      effort_target=data.joint_effort_target[:, ids],
      pos=data.joint_pos[:, ids],
      vel=data.joint_vel[:, ids],
    )

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for target_name in target_names:
      actuator_name = self.cfg.joint_to_actuator_map[target_name]
      actuator = create_motor_actuator(
        spec,
        actuator_name,
        effort_limit=self.cfg.actuator_effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    num_envs = data.nworld
    num_targets = len(self._target_names)
    self.actuator_force_limit = torch.full(
      (num_envs, num_targets),
      self.cfg.actuator_effort_limit,
      dtype=torch.float,
      device=device,
    )
    self.default_actuator_force_limit = self.actuator_force_limit.clone()

    self.transmission = self.cfg.transmission.to(device=device, dtype=torch.float)

    self._servo_joint_ids = torch.as_tensor(
      self._servo_joint_ids_list, dtype=torch.long, device=device
    )

  def set_actuator_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    assert self.actuator_force_limit is not None

    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.actuator_force_limit[env_ids] = effort_limit
