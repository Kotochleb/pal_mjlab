"""Actuators for the connect-linkage model's (``pal_kangaroo_full_full``)
screws, commanded in the simple model's joint space through the lookup
tables of ``lut_maps``. They live here, with the other transmitted
actuators of ``pal_kangaroo_full.actuator`` and the tables in
``pal_kangaroo_full/transmission`` and ``lut_transmission``, because the
mechanisms and their maps are the robot's, not one MJCF's.

The law is ``pal_kangaroo_full.actuator.TransmitedIdealPdActuator``'s, per
mechanism: a full PD (kp, kd, feed-forward) on the simple-model joints,
clamped to the joint torque limit, pushed through the mechanism's Jacobian
to the screw forces, clamped to the screw force limit, and handed to a
``<motor>`` element on each screw as an external force. What differs per
mechanism is only the transmission step (``lut_maps`` conventions:
``J = d(actuators)/d(joints)``, ``tau = J^T F``):

* :class:`HipZLutPdActuatorCfg` -- hip yaw, ``leg_*_1_joint`` ->
  ``leg_*_1_actuator``: ``F = tau / J(q)``.
* :class:`LegLengthLutPdActuatorCfg` -- the knee is the joint, but the PD
  runs in the simple model's leg-length metres: the knee angle is mapped to
  the slider position and from there to the femur-ankle distance ``d``,
  ``F_d = PD(d)``, and ``F_s = F_d / J(s)`` with ``J = d(slider)/d(distance)``.
* :class:`HipXyLutPdActuatorCfg` -- hip pitch/roll, ``leg_*_(2|3)_joint`` ->
  ``leg_*_(2|3)_actuator``: ``F = J(q2, q3)^-T tau``.
* :class:`AnkleLutPdActuatorCfg` -- ankle pitch/roll, ``leg_*_(4|5)_joint``
  -> ``leg_*_(4|5)_actuator``, with the knee as context: ``s`` from the knee
  through the leg-length map, ``F = J(q4, q5, s)^-T tau`` (pseudo-inverse at
  the mechanism's dead point).

One instance drives **both legs**. The maps are built from the right leg
and the mechanisms are mirror images with a naming convention that makes the
tables identical for the left leg (hip pitch/roll, ankle, leg length --
checked against the per-leg sweep tables), except hip yaw, whose left joint
runs the other way (``x_left(q) = x_right(-q)``); ``joint_sign`` flips that
leg's joint coordinates going into the map, and the screw force comes out
unflipped. The map is therefore loaded once, and both legs' queries are
batched into one lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, TransmissionType
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.spec import create_motor_actuator
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import (
  AnkleMap,
  HipXyMap,
  HipZMap,
  LegLengthMap,
  reside,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData

LutPdCfgT = TypeVar("LutPdCfgT", bound="LutPdActuatorCfg")

GainSpec = float | dict[str, float]


@dataclass
class LutActuatorCmd(ActuatorCmd):
  """``ActuatorCmd`` on the servo joints, plus the context joints the map is
  keyed by beyond them (the ankle's knee), ``(num_envs, n_sides * n_context)``
  -- ``None`` when there are none."""

  context_pos: torch.Tensor | None = None
  context_vel: torch.Tensor | None = None


@dataclass(kw_only=True)
class LutPdActuatorCfg(ActuatorCfg):
  """Shared configuration of the LUT-transmitted PD actuators.

  ``target_names_expr`` must resolve to exactly the servo joints of every
  side in ``sides`` (the map's ``leg_right_*`` joints, re-sided). The
  per-joint gains may be one value or a dict keyed by regexes over the joint
  names -- the ``target_names_expr`` entries themselves are the natural
  keys, which is also what ``_build_action_scales`` reads them by.
  """

  joint_stiffness: GainSpec
  """PD proportional gain on the servo joints (N m/rad; N/m for the leg
  length, whose servo coordinate is metres)."""

  joint_damping: GainSpec
  """PD derivative gain on the servo joints."""

  joint_effort_limit: GainSpec
  """Clamp on the servo-joint PD output before it is transmitted. Also what
  the action scale derives from."""

  actuator_effort_limit: float = math.inf
  """Force limit of the screws' ``<motor>`` elements, and the clamp on the
  transmitted force (N)."""

  sides: tuple[str, ...] = ("left", "right")
  """Legs served by this one instance; the map's names are re-sided to each."""

  reference_side: str = "right"
  """The side the map files name their joints and actuators after."""

  joint_sign: dict[str, float] = field(default_factory=dict)
  """Per-side sign of the servo joint coordinates relative to the map's
  (``{"left": -1.0}`` for hip yaw); sides not listed are ``+1``."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type != TransmissionType.JOINT:
      raise ValueError(f"{type(self).__name__} only supports JOINT transmission")
    for side in self.joint_sign:
      if side not in self.sides:
        raise ValueError(f"joint_sign names {side!r}, not one of sides {self.sides}")
    if self.actuator_effort_limit <= 0:
      raise ValueError("actuator_effort_limit must be positive")

  # Subclasses describe the mechanism through the map they carry.
  def servo_joint_names(self) -> tuple[str, ...]:
    """The reference side's servo joints, in map order."""
    raise NotImplementedError

  def context_joint_names(self) -> tuple[str, ...]:
    """The reference side's context joints (keys beyond the servo joints)."""
    return ()

  def screw_names(self) -> tuple[str, ...]:
    """The reference side's screws (slider joints that get the ``<motor>``)."""
    raise NotImplementedError

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> LutPdActuator:
    raise NotImplementedError


class LutPdActuator(Actuator[LutPdCfgT], Generic[LutPdCfgT]):
  """Joint-space PD, transmitted through a LUT to ``<motor>``s on the screws,
  both legs batched. Subclasses implement :meth:`_transmit`."""

  def __init__(
    self,
    cfg: LutPdCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    ref = cfg.reference_side
    self.n_sides = len(cfg.sides)
    self.n_servo = len(cfg.servo_joint_names())
    self.n_context = len(cfg.context_joint_names())
    self.n_screws = len(cfg.screw_names())
    # Per side, in map order; flat lists are side-major.
    self.servo_names = [
      reside(name, side, ref) for side in cfg.sides for name in cfg.servo_joint_names()
    ]
    self.context_names = [
      reside(name, side, ref)
      for side in cfg.sides
      for name in cfg.context_joint_names()
    ]
    self.screw_names = [
      reside(name, side, ref) for side in cfg.sides for name in cfg.screw_names()
    ]
    if sorted(target_names) != sorted(self.servo_names):
      raise ValueError(
        f"{type(self).__name__}: target_names_expr {cfg.target_names_expr} resolved "
        f"to {target_names}, but the map serves {self.servo_names}"
      )
    self._servo_ids_list = self._find(entity, self.servo_names)
    self._context_ids_list = self._find(entity, self.context_names)
    self._screw_ids_list = self._find(entity, self.screw_names)

    self._stiffness_list = self._per_joint(cfg.joint_stiffness, "joint_stiffness")
    self._damping_list = self._per_joint(cfg.joint_damping, "joint_damping")
    self._effort_limit_list = self._per_joint(
      cfg.joint_effort_limit, "joint_effort_limit"
    )
    self._sign_list = [
      float(cfg.joint_sign.get(side, 1.0))
      for side in cfg.sides
      for _ in range(self.n_servo)
    ]

    self._servo_ids: torch.Tensor | None = None
    self._context_ids: torch.Tensor | None = None
    self._screw_ids: torch.Tensor | None = None
    self._sign: torch.Tensor | None = None
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None
    self.actuator_force_limit: torch.Tensor | None = None
    self.default_stiffness: torch.Tensor | None = None
    self.default_damping: torch.Tensor | None = None
    self.default_force_limit: torch.Tensor | None = None
    self.default_actuator_force_limit: torch.Tensor | None = None

  @staticmethod
  def _find(entity: Entity, names: list[str]) -> list[int]:
    if not names:
      return []
    ids, found = entity.find_joints(names, preserve_order=True)
    if found != names:
      raise ValueError(f"joints {names} resolved to {found}")
    return ids

  def _per_joint(self, value: GainSpec, what: str) -> list[float]:
    """``value`` as one float per servo joint, in ``servo_names`` order."""
    if not isinstance(value, dict):
      return [float(value)] * len(self.servo_names)
    idx, names, values = resolve_matching_names_values(value, self.servo_names)
    if sorted(names) != sorted(self.servo_names):
      missing = sorted(set(self.servo_names) - set(names))
      raise ValueError(f"{type(self).__name__}.{what} does not cover {missing}")
    out = [0.0] * len(self.servo_names)
    for i, v in zip(idx, values, strict=True):
      out[i] = float(v)
    return out

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    del target_names  # The screws, not the servo joints, get the elements.
    for screw_name in self.screw_names:
      actuator = create_motor_actuator(
        spec,
        screw_name,
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
    as_long = lambda ids: torch.as_tensor(ids, dtype=torch.long, device=device)  # noqa: E731
    self._servo_ids = as_long(self._servo_ids_list)
    self._context_ids = as_long(self._context_ids_list)
    self._screw_ids = as_long(self._screw_ids_list)
    self._sign = torch.tensor(self._sign_list, dtype=torch.float, device=device)

    def per_env(values: list[float]) -> torch.Tensor:
      row = torch.tensor(values, dtype=torch.float, device=device)
      return row.unsqueeze(0).repeat(num_envs, 1)

    self.stiffness = per_env(self._stiffness_list)
    self.damping = per_env(self._damping_list)
    self.force_limit = per_env(self._effort_limit_list)
    self.actuator_force_limit = torch.full(
      (num_envs, len(self.screw_names)),
      self.cfg.actuator_effort_limit,
      dtype=torch.float,
      device=device,
    )
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()
    self.default_actuator_force_limit = self.actuator_force_limit.clone()
    self._maps_to(device)

  def _maps_to(self, device: str) -> None:
    raise NotImplementedError

  # Same DR hooks as IdealPdActuator / TranssmitedIdealPdActuator.
  def set_gains(
    self,
    env_ids: torch.Tensor | slice,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
  ) -> None:
    assert self.stiffness is not None and self.damping is not None
    if kp is not None:
      self.stiffness[env_ids] = kp.unsqueeze(-1) if kp.ndim == 1 else kp
    if kd is not None:
      self.damping[env_ids] = kd.unsqueeze(-1) if kd.ndim == 1 else kd

  def set_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    assert self.force_limit is not None
    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.force_limit[env_ids] = effort_limit

  def set_actuator_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    assert self.actuator_force_limit is not None
    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.actuator_force_limit[env_ids] = effort_limit

  def get_command(self, data: EntityData) -> LutActuatorCmd:
    assert self._servo_ids is not None and self._context_ids is not None
    ids = self._servo_ids
    has_context = self.n_context > 0
    return LutActuatorCmd(
      position_target=data.joint_pos_target[:, ids],
      velocity_target=data.joint_vel_target[:, ids],
      effort_target=data.joint_effort_target[:, ids],
      pos=data.joint_pos[:, ids],
      vel=data.joint_vel[:, ids],
      context_pos=data.joint_pos[:, self._context_ids] if has_context else None,
      context_vel=data.joint_vel[:, self._context_ids] if has_context else None,
    )

  # -- The law, in three steps shared by every mechanism.

  def _batched(self, x: torch.Tensor | None, width: int) -> torch.Tensor | None:
    """``(num_envs, n_sides * width)`` -> ``(num_envs * n_sides, width)``."""
    if x is None:
      return None
    return x.reshape(x.shape[0] * self.n_sides, width)

  def _pd(
    self,
    pos: torch.Tensor,
    vel: torch.Tensor,
    position_target: torch.Tensor,
    velocity_target: torch.Tensor,
    effort_target: torch.Tensor,
  ) -> torch.Tensor:
    """Clamped PD on the servo coordinate, ``(num_envs, n_sides * n_servo)``."""
    assert self.stiffness is not None and self.damping is not None
    assert self.force_limit is not None
    torque = self.stiffness * (position_target - pos)
    torque = torque + self.damping * (velocity_target - vel)
    torque = torque + effort_target
    return torch.clamp(torque, -self.force_limit, self.force_limit)

  def _signed(self, cmd: LutActuatorCmd) -> LutActuatorCmd:
    """The command in the map's joint coordinates (``joint_sign`` applied)."""
    sign = self._sign
    assert sign is not None
    return LutActuatorCmd(
      position_target=cmd.position_target * sign,
      velocity_target=cmd.velocity_target * sign,
      effort_target=cmd.effort_target * sign,
      pos=cmd.pos * sign,
      vel=cmd.vel * sign,
      context_pos=cmd.context_pos,
      context_vel=cmd.context_vel,
    )

  def _transmit(
    self, pos: torch.Tensor, context_pos: torch.Tensor | None, torque: torch.Tensor
  ) -> torch.Tensor:
    """Screw forces ``(num_envs * n_sides, n_screws)`` from the servo joint
    torques and positions ``(num_envs * n_sides, n_servo)`` in map
    coordinates, and the context positions ``(..., n_context)``."""
    raise NotImplementedError

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert isinstance(cmd, LutActuatorCmd)
    assert self.actuator_force_limit is not None
    cmd = self._signed(cmd)
    torque = self._pd(
      cmd.pos, cmd.vel, cmd.position_target, cmd.velocity_target, cmd.effort_target
    )
    force = self._transmit(
      self._batched(cmd.pos, self.n_servo),
      self._batched(cmd.context_pos, self.n_context),
      self._batched(torque, self.n_servo),
    )
    force = force.reshape(-1, self.n_sides * self.n_screws)
    return torch.clamp(force, -self.actuator_force_limit, self.actuator_force_limit)


##
# Hip yaw.
##


@dataclass(kw_only=True)
class HipZLutPdActuatorCfg(LutPdActuatorCfg):
  """Hip yaw: PD on ``leg_*_1_joint``, force on ``leg_*_1_actuator`` through
  ``hip_z_map.npz`` (``F = tau / J(q)``). The left joint is the mirror of
  the right one, hence the default ``joint_sign``."""

  hip_z_map: HipZMap

  joint_sign: dict[str, float] = field(default_factory=lambda: {"left": -1.0})

  def servo_joint_names(self) -> tuple[str, ...]:
    return self.hip_z_map.joint_names

  def screw_names(self) -> tuple[str, ...]:
    return self.hip_z_map.actuator_names

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> HipZLutPdActuator:
    return HipZLutPdActuator(self, entity, target_ids, target_names)


class HipZLutPdActuator(LutPdActuator[HipZLutPdActuatorCfg]):
  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.map: HipZMap = cfg.hip_z_map

  def _maps_to(self, device: str) -> None:
    self.map = self.cfg.hip_z_map.to(device)

  def _transmit(self, pos, context_pos, torque):
    del context_pos
    return self.map.force(pos, torque)


##
# Leg length.
##


@dataclass(kw_only=True)
class LegLengthLutPdActuatorCfg(LutPdActuatorCfg):
  """Leg length: the target joint is ``leg_*_knee_joint`` (the only
  leg-length DOF this MJCF has) but the PD runs in the simple model's
  leg-length metres, the femur-ankle ``distance`` of ``leg_length_map.npz``
  -- the coordinate the mapped leg-length action term and the observations
  use. ``joint_stiffness`` / ``joint_damping`` / ``joint_effort_limit`` are
  therefore N/m, N s/m and N. Per leg::

    s     = slider_of_knee(knee)
    d     = distance(s)                       d_dot = knee_dot / (J(s) dknee_dslider(s))
    F_d   = clamp(kp (d_t - d) + kd (d_dot_t - d_dot) + F_ff, joint_effort_limit)
    F_s   = clamp(F_d / J(s), actuator_effort_limit)     # J = d(slider)/d(distance)
  """

  leg_length_map: LegLengthMap

  def servo_joint_names(self) -> tuple[str, ...]:
    return (self.leg_length_map.knee_name,)

  def screw_names(self) -> tuple[str, ...]:
    return (self.leg_length_map.slider_name,)

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> LegLengthLutPdActuator:
    return LegLengthLutPdActuator(self, entity, target_ids, target_names)


class LegLengthLutPdActuator(LutPdActuator[LegLengthLutPdActuatorCfg]):
  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.map: LegLengthMap = cfg.leg_length_map

  def _maps_to(self, device: str) -> None:
    self.map = self.cfg.leg_length_map.to(device)

  def leg_length(self, knee_pos: torch.Tensor, knee_vel: torch.Tensor):
    """``(slider, distance, distance rate)`` of the knee state."""
    s = self.map.slider_of_knee(knee_pos)
    return s, self.map.distance(s), self.map.distance_rate(s, knee_vel)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert isinstance(cmd, LutActuatorCmd)
    assert self.actuator_force_limit is not None
    cmd = self._signed(cmd)
    s, d, d_dot = self.leg_length(cmd.pos, cmd.vel)
    F_d = self._pd(
      d, d_dot, cmd.position_target, cmd.velocity_target, cmd.effort_target
    )
    F_s = self.map.actuator_force(s, F_d)
    return torch.clamp(F_s, -self.actuator_force_limit, self.actuator_force_limit)

  def _transmit(self, pos, context_pos, torque):
    raise NotImplementedError(
      "the leg length's PD is in mapped coordinates; see compute"
    )


##
# Hip pitch / roll.
##


@dataclass(kw_only=True)
class HipXyLutPdActuatorCfg(LutPdActuatorCfg):
  """Hip pitch/roll: PD on ``leg_*_2_joint``, ``leg_*_3_joint``, forces on
  ``leg_*_2_actuator``, ``leg_*_3_actuator`` through
  ``hip_xy_jacobian_map.npz`` (``F = J(q2, q3)^-T tau``)."""

  hip_xy_map: HipXyMap

  def servo_joint_names(self) -> tuple[str, ...]:
    return self.hip_xy_map.joint_names

  def screw_names(self) -> tuple[str, ...]:
    return self.hip_xy_map.actuator_names

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> HipXyLutPdActuator:
    return HipXyLutPdActuator(self, entity, target_ids, target_names)


class HipXyLutPdActuator(LutPdActuator[HipXyLutPdActuatorCfg]):
  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.map: HipXyMap = cfg.hip_xy_map

  def _maps_to(self, device: str) -> None:
    self.map = self.cfg.hip_xy_map.to(device)

  def _transmit(self, pos, context_pos, torque):
    del context_pos
    return self.map.forces(pos, torque)


##
# Ankle pitch / roll.
##


@dataclass(kw_only=True)
class AnkleLutPdActuatorCfg(LutPdActuatorCfg):
  """Ankle pitch/roll: PD on ``leg_*_4_joint``, ``leg_*_5_joint``, forces
  on ``leg_*_4_actuator``, ``leg_*_5_actuator`` through
  ``ankle_xy_jacobian_map.npz``, whose third axis is the leg-length slider
  position: the knee (context joint) is mapped to it through
  ``leg_length_map.npz``, then ``F = J(q4, q5, s)^-T tau``."""

  ankle_map: AnkleMap
  leg_length_map: LegLengthMap

  def servo_joint_names(self) -> tuple[str, ...]:
    return self.ankle_map.joint_names

  def context_joint_names(self) -> tuple[str, ...]:
    return (self.leg_length_map.knee_name,)

  def screw_names(self) -> tuple[str, ...]:
    return self.ankle_map.actuator_names

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> AnkleLutPdActuator:
    return AnkleLutPdActuator(self, entity, target_ids, target_names)


class AnkleLutPdActuator(LutPdActuator[AnkleLutPdActuatorCfg]):
  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.map: AnkleMap = cfg.ankle_map
    self.leg_length_map: LegLengthMap = cfg.leg_length_map

  def _maps_to(self, device: str) -> None:
    self.map = self.cfg.ankle_map.to(device)
    self.leg_length_map = self.cfg.leg_length_map.to(device)

  def _transmit(self, pos, context_pos, torque):
    assert context_pos is not None
    s = self.leg_length_map.slider_of_knee(context_pos[:, 0])
    return self.map.forces(pos, s, torque)
