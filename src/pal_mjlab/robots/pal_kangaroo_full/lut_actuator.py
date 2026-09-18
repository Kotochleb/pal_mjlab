"""The ``"lut"`` transmission of the full KANGAROO models: one actuator that
servos every leg joint of the simple model and drives every leg screw.

The law, for all twelve leg joints at once:

1. a PD (kp, kd, feed-forward) on the simple model's joints -- hip yaw, hip
   pitch/roll, ankle pitch/roll and the leg length -- clamped to each
   joint's torque limit;
2. the joint torques pushed through the mechanisms' Jacobians, read from
   the lookup tables of ``lut_maps`` (``J = d(actuators)/d(joints)``,
   ``tau = J^T F``), to the screw forces;
3. each screw force clamped to the screw's limit and handed to MuJoCo as
   the ctrl of a ``<motor>`` on that screw.

Per mechanism the transmission step is

* hip yaw, ``leg_*_1_joint`` -> ``leg_*_1_actuator``: ``F = tau / J(q)``;
* hip pitch/roll, ``leg_*_(2|3)_joint`` -> ``leg_*_(2|3)_actuator``:
  ``F = J(q2, q3)^-T tau``;
* ankle pitch/roll, ``leg_*_(4|5)_joint`` -> ``leg_*_(4|5)_actuator``:
  ``F = J(q4, q5, s)^-T tau`` with ``s`` the leg-length slider position
  (pseudo-inverse at the mechanism's dead point);
* leg length -> ``leg_*_length_actuator``: the PD runs in the simple
  model's leg-length metres, the femur-ankle ``distance`` of
  ``leg_length_map.npz``, and ``F_s = F_d / J(s)`` with ``J =
  d(slider)/d(distance)``. The servo joint is the prismatic
  ``leg_*_length_joint`` where the MJCF has one (it reads the distance up to
  ``length_joint_to_distance``) or else the knee, mapped onto the distance;
  ``s`` follows from either, and is also the ankle's third coordinate.

The maps are built from the right leg and the mechanisms are mirror images
whose naming makes the tables identical for the left leg (hip pitch/roll,
ankle, leg length), except hip yaw, whose left joint runs the other way
(``x_left(q) = x_right(-q)``): ``joint_sign`` flips that joint's coordinates
going into the map and the screw force comes out unflipped. One instance
therefore drives **both legs** through one set of maps, every query batched.

Which MJCF element is "the screw" is the :class:`ScrewElement` table: the
connect-linkage model (``pal_kangaroo_full_full``) has the real prismatic
sliders, the tendon model (``pal_kangaroo_full``) has the spatial tendons
standing in for them. The hip tendons' lengths move with the joints exactly
as the sliders do (checked by finite differences against the maps); the
ankle tendons are the decoupler-to-butterfly chords, a stand-in for the real
screw that the maps do not describe exactly, and are driven through the maps
as they are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, TransmissionType
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.spec import create_motor_actuator
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import TransmissionMaps, reside

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData

GainSpec = float | dict[str, float]
LegLengthServo = Literal["length_joint", "knee"]


@dataclass(frozen=True)
class ScrewElement:
  """The ``<motor>`` that applies one of the map's actuator forces in an MJCF."""

  name: str
  """The element's name on the reference side (``right_hip_z_slider`` for the
  tendon model's ``leg_right_1_actuator``); re-sided for the other leg."""

  effort_limit: float
  """Force limit of the ``<motor>`` (``forcerange``) and the clamp on the
  transmitted force, N."""

  transmission_type: TransmissionType = TransmissionType.JOINT
  """Whether the element is a slider joint or a tendon."""

  armature: float | None = None
  frictionloss: float | None = None
  viscous_damping: float | None = None
  """Element overrides, as on any ``ActuatorCfg``; None keeps the XML value."""

  def __post_init__(self) -> None:
    if self.effort_limit <= 0:
      raise ValueError(f"{self.name}: effort_limit must be positive")


@dataclass(kw_only=True)
class LutTransmissionActuatorCfg(ActuatorCfg):
  """Configuration of the whole-leg LUT transmission (see the module docstring).

  ``target_names_expr`` must resolve to exactly the servo joints of every
  side in ``sides``: the maps' hip and ankle joints plus the leg-length
  servo joint, re-sided. Its entries are the keys the per-joint gains are
  looked up by (a plain float applies to every joint), and what
  ``_build_action_scales`` reads the action scale from.
  """

  maps: TransmissionMaps
  """The four maps, shared by both legs."""

  screws: dict[str, ScrewElement]
  """The MJCF element of each map actuator, keyed by the map's reference-side
  actuator names (``maps.actuator_names``)."""

  joint_stiffness: GainSpec
  """PD proportional gain on the servo joints: N m/rad, and N/m for the leg
  length, whose servo coordinate is metres."""

  joint_damping: GainSpec
  """PD derivative gain on the servo joints."""

  joint_effort_limit: GainSpec
  """Clamp on each joint's PD output before it is transmitted (N m; N for the
  leg length), and what the action scale derives from."""

  leg_length_servo: LegLengthServo = "knee"
  """Which joint the leg-length PD reads: the prismatic ``length_joint_name``
  (its position is the distance minus ``length_joint_to_distance``) or the
  knee, mapped onto the distance through ``maps.leg_length``."""

  length_joint_name: str = "leg_right_length_joint"
  """The reference side's prismatic leg-length joint (``"length_joint"``)."""

  length_joint_to_distance: float = 0.0
  """``length_joint + length_joint_to_distance`` = the map's distance."""

  sides: tuple[str, ...] = ("left", "right")
  """Legs served by this one instance; the map's names are re-sided to each."""

  reference_side: str = "right"
  """The side the map files name their joints and actuators after."""

  joint_sign: dict[str, float] = field(
    default_factory=lambda: {"leg_left_1_joint": -1.0}
  )
  """Sign of a servo joint's coordinates relative to the map's, keyed by
  regexes over the re-sided joint names; unlisted joints are ``+1``. The
  default is the hip yaw mirror."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.transmission_type != TransmissionType.JOINT:
      raise ValueError(
        f"{type(self).__name__} servos joints; transmission_type is JOINT"
      )
    if (
      self.armature is not None
      or self.frictionloss is not None
      or self.viscous_damping is not None
    ):
      raise ValueError(
        f"{type(self).__name__}: armature / frictionloss / viscous_damping belong "
        "to the screws (ScrewElement), not the servo joints"
      )
    expected = set(self.maps.actuator_names)
    if set(self.screws) != expected:
      raise ValueError(
        f"screws must be keyed by the maps' actuators {sorted(expected)}, "
        f"got {sorted(self.screws)}"
      )
    if self.leg_length_servo not in ("length_joint", "knee"):
      raise ValueError(
        f"leg_length_servo must be 'length_joint' or 'knee', got {self.leg_length_servo!r}"
      )

  @property
  def leg_length_servo_expr(self) -> str:
    """The ``target_names_expr`` entry naming the leg-length servo joints --
    the knee's targets are in metres, which the action term must know."""
    name = (
      self.maps.leg_length.knee_name
      if self.leg_length_servo == "knee"
      else self.length_joint_name
    )
    matches = [
      expr
      for expr in self.target_names_expr
      if any(
        re.fullmatch(expr, reside(name, side, self.reference_side))
        for side in self.sides
      )
    ]
    if len(matches) != 1:
      raise ValueError(
        f"exactly one target_names_expr entry must name {name}, found {matches}"
      )
    return matches[0]

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> LutTransmissionActuator:
    return LutTransmissionActuator(self, entity, target_ids, target_names)


class LutTransmissionActuator(Actuator[LutTransmissionActuatorCfg]):
  """Joint-space PD on every leg joint, transmitted through the maps to
  ``<motor>``s on every leg screw, both legs batched."""

  # Per side, in map order: hip yaw, hip pitch, hip roll, ankle pitch, ankle
  # roll, leg length. Screws follow the same order.
  N_JOINTS = 6

  def __init__(
    self,
    cfg: LutTransmissionActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    maps = cfg.maps
    ref = cfg.reference_side
    self.n_sides = len(cfg.sides)
    leg_length_servo = (
      maps.leg_length.knee_name
      if cfg.leg_length_servo == "knee"
      else cfg.length_joint_name
    )
    ref_joints = maps.joint_names + (leg_length_servo,)
    assert (
      len(ref_joints) == self.N_JOINTS and len(maps.actuator_names) == self.N_JOINTS
    )
    # Flat lists are side-major.
    self.servo_names = [reside(n, side, ref) for side in cfg.sides for n in ref_joints]
    self._screws = [cfg.screws[n] for _ in cfg.sides for n in maps.actuator_names]
    self.screw_names = [
      reside(screw.name, side, ref)
      for side in cfg.sides
      for screw in (cfg.screws[n] for n in maps.actuator_names)
    ]
    if sorted(target_names) != sorted(self.servo_names):
      raise ValueError(
        f"{type(self).__name__}: target_names_expr {cfg.target_names_expr} resolved "
        f"to {target_names}, but the maps serve {self.servo_names}"
      )
    ids, found = entity.find_joints(self.servo_names, preserve_order=True)
    if found != self.servo_names:
      raise ValueError(f"joints {self.servo_names} resolved to {found}")
    self._servo_ids_list = ids
    # Column of each side's leg-length servo in the (num_envs, n_sides * 6) layout.
    self._leg_length_cols_list = [
      side * self.N_JOINTS + self.N_JOINTS - 1 for side in range(self.n_sides)
    ]

    self._stiffness_list = self._per_joint(cfg.joint_stiffness, "joint_stiffness")
    self._damping_list = self._per_joint(cfg.joint_damping, "joint_damping")
    self._effort_limit_list = self._per_joint(
      cfg.joint_effort_limit, "joint_effort_limit"
    )
    self._sign_list = [1.0] * len(self.servo_names)
    if cfg.joint_sign:
      idx, _, values = resolve_matching_names_values(cfg.joint_sign, self.servo_names)
      for i, v in zip(idx, values, strict=True):
        self._sign_list[i] = float(v)
    self._screw_limit_list = [screw.effort_limit for screw in self._screws]

    self._servo_ids: torch.Tensor | None = None
    self._leg_length_cols: torch.Tensor | None = None
    self._sign: torch.Tensor | None = None
    self.maps: TransmissionMaps = cfg.maps
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None
    self.actuator_force_limit: torch.Tensor | None = None
    self.default_stiffness: torch.Tensor | None = None
    self.default_damping: torch.Tensor | None = None
    self.default_force_limit: torch.Tensor | None = None
    self.default_actuator_force_limit: torch.Tensor | None = None

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
    for name, screw in zip(self.screw_names, self._screws, strict=True):
      self._mjs_actuators.append(
        create_motor_actuator(
          spec,
          name,
          effort_limit=screw.effort_limit,
          armature=screw.armature,
          frictionloss=screw.frictionloss,
          viscous_damping=screw.viscous_damping,
          transmission_type=screw.transmission_type,
        )
      )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    self._servo_ids = torch.as_tensor(
      self._servo_ids_list, dtype=torch.long, device=device
    )
    self._leg_length_cols = torch.as_tensor(
      self._leg_length_cols_list, dtype=torch.long, device=device
    )
    self._sign = torch.tensor(self._sign_list, dtype=torch.float, device=device)

    def per_env(values: list[float]) -> torch.Tensor:
      row = torch.tensor(values, dtype=torch.float, device=device)
      return row.unsqueeze(0).repeat(num_envs, 1)

    self.stiffness = per_env(self._stiffness_list)
    self.damping = per_env(self._damping_list)
    self.force_limit = per_env(self._effort_limit_list)
    self.actuator_force_limit = per_env(self._screw_limit_list)
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()
    self.default_actuator_force_limit = self.actuator_force_limit.clone()
    self.maps = self.cfg.maps.to(device)

  # Same DR hooks as IdealPdActuator.
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

  def get_command(self, data: EntityData) -> ActuatorCmd:
    assert self._servo_ids is not None
    ids = self._servo_ids
    return ActuatorCmd(
      position_target=data.joint_pos_target[:, ids],
      velocity_target=data.joint_vel_target[:, ids],
      effort_target=data.joint_effort_target[:, ids],
      pos=data.joint_pos[:, ids],
      vel=data.joint_vel[:, ids],
    )

  def leg_length_state(
    self, q: torch.Tensor, q_dot: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(slider, distance, distance rate)`` from the leg-length servo joint's
    position and velocity, whichever joint that is."""
    leg = self.maps.leg_length
    if self.cfg.leg_length_servo == "knee":
      s = leg.slider_of_knee(q)
      return s, leg.distance(s), leg.distance_rate(s, q_dot)
    s = leg.slider_of_distance(q + self.cfg.length_joint_to_distance)
    return s, q, q_dot

  def joint_torques(
    self, cmd: ActuatorCmd
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 1 of the law: the clamped PD torques ``(num_envs, n_sides * 6)``
    in map coordinates, with the joint positions they were taken at (the
    leg length replaced by its distance) and the leg-length slider positions
    ``(num_envs, n_sides)``."""
    assert self._sign is not None and self._leg_length_cols is not None
    assert self.stiffness is not None and self.damping is not None
    assert self.force_limit is not None
    sign = self._sign
    cols = self._leg_length_cols
    pos, vel = cmd.pos * sign, cmd.vel * sign
    s, d, d_dot = self.leg_length_state(pos[:, cols], vel[:, cols])
    pos[:, cols] = d
    vel[:, cols] = d_dot
    torque = self.stiffness * (cmd.position_target * sign - pos)
    torque = torque + self.damping * (cmd.velocity_target * sign - vel)
    torque = torque + cmd.effort_target * sign
    return torch.clamp(torque, -self.force_limit, self.force_limit), pos, s

  def screw_forces(
    self, torque: torch.Tensor, pos: torch.Tensor, s: torch.Tensor
  ) -> torch.Tensor:
    """Step 2: the screw forces ``(num_envs, n_sides * 6)`` for joint torques
    and positions in map coordinates (from :meth:`joint_torques`)."""
    n = self.N_JOINTS
    q = pos.reshape(-1, n)
    tau = torque.reshape(-1, n)
    s = s.reshape(-1)
    maps = self.maps
    F = torch.cat(
      (
        maps.hip_z.force(q[:, 0], tau[:, 0]).unsqueeze(-1),
        maps.hip_xy.forces(q[:, 1:3], tau[:, 1:3]),
        maps.ankle.forces(q[:, 3:5], s, tau[:, 3:5]),
        maps.leg_length.actuator_force(s, tau[:, 5]).unsqueeze(-1),
      ),
      dim=-1,
    )
    return F.reshape(torque.shape)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.actuator_force_limit is not None
    torque, pos, s = self.joint_torques(cmd)
    force = self.screw_forces(torque, pos, s)
    # Step 3: the clamped force is the <motor>'s ctrl.
    return torch.clamp(force, -self.actuator_force_limit, self.actuator_force_limit)


__all__ = [
  "GainSpec",
  "LegLengthServo",
  "LutTransmissionActuator",
  "LutTransmissionActuatorCfg",
  "ScrewElement",
]
