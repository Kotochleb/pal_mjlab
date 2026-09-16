"""Actuators whose commanded joint sits behind a position-dependent mechanical
transmission from the joint the hardware actually drives.

Both take a position target for the servo joint, turn it into a joint torque,
and push that torque through a lookup table of the transmission's Jacobian.
They differ in where the loop closes:

* :class:`TransmitedIdealPdActuatorCfg` -- full PD (kp and kd) on the servo
  joint; the transmitted force goes to a ``<motor>`` element as an opaque
  external force.
* :class:`TransmittedPositionActuatorCfg` -- P only on the servo joint; the
  transmitted force is turned back into a setpoint for a native ``<position>``
  element on the driven joint, which carries its own kp and kd.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
from mjlab.actuator.actuator import (
  Actuator,
  ActuatorCfg,
  ActuatorCmd,
  TransmissionType,
)
from mjlab.utils.spec import create_motor_actuator, create_position_actuator

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


def load_transmission_table(
  csv_path: str | Path,
  columns: tuple[int, int] = (0, 1),
  key_offset: float = 0.0,
) -> torch.Tensor:
  """Load a CSV as an (N, 2) ``(key, value)`` table sorted by key.

  ``columns`` picks which two columns of the file are the key and the value
  -- the default reads a ``pos,force_J`` transmission file as is; another
  choice reads e.g. ``knee_distance_map.csv``'s ``(knee_rad, distance_m)``.
  ``key_offset`` is added to the key column, for a file keyed by a
  coordinate that differs from the servoed one by a constant.
  """
  data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
  table = torch.from_numpy(data[:, list(columns)])
  table[:, 0] += key_offset
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


@dataclass
class TransmittedActuatorCmd(ActuatorCmd):
  """``ActuatorCmd`` plus the state of the joint actually being driven.

  The servo joint (``pos``, ``position_target``) and the driven joint
  (``actuator_pos``) are different joints here, so the inherited fields cannot
  carry both.
  """

  actuator_pos: torch.Tensor
  """Current position of the driven (transmission-output) joint."""


@dataclass(kw_only=True)
class TransmittedPositionActuatorCfg(ActuatorCfg):
  """Configuration for a position actuator behind a mechanical transmission.

  :class:`TransmitedIdealPdActuatorCfg` closes the whole PD loop in servo-joint
  space and hands MuJoCo a force. This one closes it on the *driven* joint
  instead: MuJoCo gets a native ``<position>`` element carrying ``stiffness``
  and ``damping``, and the only thing computed in Python is where to put its
  setpoint. Per control step, for each servo joint::

    tau  = joint_stiffness * (q_target - q)   # P only, no servo-joint velocity
    F    = tau * J(q)                         # J interpolated from `transmission`
    ctrl = x + F / stiffness                  # x = driven joint position

  The offset is divided by the driven joint's own ``stiffness`` because the
  ``<position>`` element multiplies by it again, so the force it applies at the
  instant of the write is exactly ``F`` -- less its own ``damping * dx/dt``
  term, which is what supplies the damping this law omits on the servo side.
  Unlike a Python-side PD, that force then keeps tracking as the driven joint
  moves through the control period, and the implicit integrators see the gains.

  By default ``effort_limit`` is the only limit in the chain: tau is not
  clamped on the servo side, and the force that reaches the mechanism is
  bounded where it is actually applied, by the element's ``forcerange``.
  ``joint_effort_limit`` adds an optional clamp on tau itself. The setpoint
  is deliberately unclamped (``create_position_actuator`` sets
  ``ctrllimited=False``), since it lives outside the driven joint's range
  whenever the transmission is asking for near-limit force.

  The servo coordinate ``q`` is the target joint's own position unless
  ``servo_map`` is given, in which case it is that table evaluated at the
  joint's position -- how the connect-linkage model servos its leg length:
  the target is ``leg_.*_knee_joint`` (the only DOF it has there), but the P
  law, the targets and ``transmission`` all run in the leg-length metres the
  simple model uses, through the swept knee-to-length map.
  """

  joint_to_actuator_map: dict[str, str]
  """Servo joint name -> name of the joint driven through the transmission."""

  transmission: torch.Tensor
  """Transmission lookup table: (N, 2) tensor of (servo pos, force_J) rows,
  sorted by position ascending, keyed by the servo coordinate (the joint
  position, or its ``servo_map`` image). See `load_transmission_table`."""

  joint_stiffness: float
  """Proportional gain of the servo joint. The derivative term lives on the
  driven joint as `damping`."""

  joint_effort_limit: float | None = None
  """Clamp on the servo-side torque before it is transmitted. None (the
  default) leaves the driven joint's `effort_limit` as the chain's only
  limit. Also what the action scale derives from when set, so a unit action
  means the same fraction of joint effort as on the plain joint actuator."""

  servo_map: torch.Tensor | None = None
  """Optional (N, 2) table of (joint pos, servo pos) rows, sorted by joint
  pos, mapping the target joint's position onto the coordinate the P law and
  `transmission` are keyed by. None servos the joint position itself."""

  stiffness: float
  """Proportional gain of the driven joint (the <position> element's kp)."""

  damping: float
  """Derivative gain of the driven joint (the <position> element's kv)."""

  effort_limit: float | None = None
  """Force limit of the driven joint, and the chain's only effort limit. None
  leaves it unbounded."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> TransmittedPositionActuator:
    return TransmittedPositionActuator(self, entity, target_ids, target_names)


class TransmittedPositionActuator(Actuator[TransmittedPositionActuatorCfg]):
  """Position actuator on the driven joint, commanded in servo joint space."""

  def __init__(
    self,
    cfg: TransmittedPositionActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    assert set(target_names) == set(cfg.joint_to_actuator_map.keys()), (
      f"{type(self).__name__}: target_names {target_names} must match "
      f"joint_to_actuator_map keys {list(cfg.joint_to_actuator_map.keys())}."
    )
    driven_names = [cfg.joint_to_actuator_map[name] for name in target_names]
    self._driven_joint_ids_list, _ = entity.find_joints(
      driven_names, preserve_order=True
    )
    self._driven_joint_ids: torch.Tensor | None = None
    self._transmission: torch.Tensor | None = None
    self._servo_map: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for target_name in target_names:
      actuator = create_position_actuator(
        spec,
        self.cfg.joint_to_actuator_map[target_name],
        stiffness=self.cfg.stiffness,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
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
    self._transmission = self.cfg.transmission.to(device=device, dtype=torch.float)
    if self.cfg.servo_map is not None:
      self._servo_map = self.cfg.servo_map.to(device=device, dtype=torch.float)
    self._driven_joint_ids = torch.as_tensor(
      self._driven_joint_ids_list, dtype=torch.long, device=device
    )

  def get_command(self, data: EntityData) -> TransmittedActuatorCmd:
    assert self.transmission_type == TransmissionType.JOINT, (
      f"{type(self).__name__} only supports JOINT transmission "
      f"(got {self.transmission_type})."
    )
    assert self._driven_joint_ids is not None
    ids = self.target_ids
    return TransmittedActuatorCmd(
      position_target=data.joint_pos_target[:, ids],
      velocity_target=data.joint_vel_target[:, ids],
      effort_target=data.joint_effort_target[:, ids],
      pos=data.joint_pos[:, ids],
      vel=data.joint_vel[:, ids],
      actuator_pos=data.joint_pos[:, self._driven_joint_ids],
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert isinstance(cmd, TransmittedActuatorCmd)
    assert self._transmission is not None
    pos = cmd.pos
    if self._servo_map is not None:
      pos = interpolate_transmission(pos, self._servo_map)
    joint_torque = self.cfg.joint_stiffness * (cmd.position_target - pos)
    if self.cfg.joint_effort_limit is not None:
      limit = self.cfg.joint_effort_limit
      joint_torque = torch.clamp(joint_torque, -limit, limit)
    force = joint_torque * interpolate_transmission(pos, self._transmission)
    return cmd.actuator_pos + force / self.cfg.stiffness
