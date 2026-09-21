from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, TransmissionType
from mjlab.actuator.dc_actuator import dc_motor_clip
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.spec import create_motor_actuator
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import TransmissionMaps, reside

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.entity.data import EntityData


@dataclass(kw_only=True)
class JointParams:
  """One servo joint's PD gains -- N m/rad (N/m for the leg length, whose
  servo coordinate is metres) and N m (N for the leg length)."""

  stiffness: float
  damping: float

  effort_limit: float
  """Clamp on the PD output before it is transmitted, and what the action
  scale derives from (see ``_build_action_scales``)."""


@dataclass(kw_only=True)
class ScrewParams:
  """One screw's element overrides and force law, in the same fields as
  ``ActuatorCfg`` exposes for any other actuator."""

  effort_limit: float
  """Force limit of the screw's ``<motor>`` (``forcerange``) and the clamp
  on the transmitted force, N."""

  transmission_type: TransmissionType = TransmissionType.JOINT

  armature: float | None = None
  viscous_damping: float | None = None
  """Overrides of the screw's ``<motor>`` actuator element (``armature``,
  ``damping``) -- the motor's own rotor inertia and friction, additive to
  the target joint/tendon's own. None keeps the actuator's default (0)."""

  frictionloss: float | None = None
  """Override of the screw's own joint/tendon ``frictionloss``; None keeps
  the XML value. MuJoCo actuators have no per-actuator frictionloss, so
  unlike ``armature``/``viscous_damping`` this cannot live on the ``<motor>``
  element."""

  saturation_effort: float | None = None
  """Peak screw force at zero screw velocity (stall force), N. None (with
  ``velocity_limit``) skips the DC motor torque-speed curve: the screw force
  is then an ideal PD, clamped only to ``effort_limit``."""

  velocity_limit: float | None = None
  """Screw velocity at which the DC motor curve reaches zero force
  (no-load speed), m/s. Must be set together with ``saturation_effort``."""

  def __post_init__(self) -> None:
    if self.effort_limit <= 0:
      raise ValueError(f"{type(self).__name__}.effort_limit must be positive")
    if (self.saturation_effort is None) != (self.velocity_limit is None):
      raise ValueError(
        f"{type(self).__name__}: saturation_effort and velocity_limit must be "
        "set together (both None for an ideal PD, both given for the DC motor "
        "torque-speed curve)"
      )
    for name in ("saturation_effort", "velocity_limit"):
      value = getattr(self, name)
      if value is not None and not 0 < value < float("inf"):
        raise ValueError(f"{type(self).__name__}.{name} must be finite and positive")


@dataclass(kw_only=True)
class LutMechanismCfg:
  """One mechanism's explicit simulation names on the reference leg.

  Joint columns and actuator rows follow the corresponding LUT's order.
  Hip XY and ankle are coupled 2-by-2 mappings. The leg-length source is
  the measured knee; its policy command and PD coordinate are virtual leg
  length in metres. The ankle lists that same knee as a context joint.
  """

  source_joint_names: tuple[str, ...]
  target_actuator_names: tuple[str, ...]
  context_joint_names: tuple[str, ...] = ()


@dataclass(kw_only=True)
class LutTransmissionActuatorCfg(ActuatorCfg):
  """Configuration of the whole-leg LUT transmission (see the module docstring).

  ``target_names_expr`` must resolve to exactly the servo joints of every
  side in ``sides``: ``mechanisms``' source joints, re-sided. Its entries
  are the keys ``joint_params`` is looked up by and what
  ``_build_action_scales`` reads the action scale from. Screw parameters
  instead resolve against the explicit target actuator names.
  """

  maps: TransmissionMaps
  """The four maps, shared by both legs."""

  mechanisms: dict[str, LutMechanismCfg]
  """Explicit bindings for ``hip_z``, ``hip_xy``, ``ankle`` and
  ``leg_length``. Names are given for ``reference_side``; the ankle's
  context joint must be the leg-length mechanism's knee source."""

  joint_params: dict[str, JointParams]
  """Each servo joint's PD gains, keyed by its ``target_names_expr`` entry
  (one per mechanism, shared by both legs -- e.g.
  ``pal_kangaroo_full.kangaroo_full_constants.lut_actuator``'s
  ``_calc_lut_joint_params``, a wrapped ``_calc_leg_params``)."""

  screw_params: dict[str, ScrewParams]
  """Each screw's element overrides and force law, keyed by expressions
  over the target actuator names, e.g. ``leg_(left|right)_2_actuator``."""

  sides: tuple[str, ...] = ("left", "right")
  """Legs served by this one instance; the configured names are re-sided."""

  reference_side: str = "right"
  """The side used for the names in ``mechanisms``."""

  _MECHANISM_SIZES: ClassVar[tuple[tuple[str, int], ...]] = (
    ("hip_z", 1),
    ("hip_xy", 2),
    ("ankle", 2),
    ("leg_length", 1),
  )

  joint_sign: dict[str, float] = field(
    default_factory=lambda: {"leg_left_1_joint": -1.0}
  )
  """Sign of a servo joint's coordinates relative to the map's, keyed by
  regexes over the re-sided joint names; unlisted joints are ``+1``. The
  default is the hip yaw mirror. A knee sign flips only the measured knee
  angle/rate before lookup: its targets already describe the virtual leg
  length in metres, metres/second and newtons, not the knee angle/torque."""

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
        "to the screws (screw_params[...].armature / .frictionloss / "
        ".viscous_damping), not the (virtual) servo joints"
      )
    if set(self.mechanisms) != {name for name, _ in self._MECHANISM_SIZES}:
      raise ValueError("mechanisms must contain hip_z, hip_xy, ankle and leg_length")
    for name, size in self._MECHANISM_SIZES:
      mechanism = self.mechanisms[name]
      if (
        len(mechanism.source_joint_names) != size
        or len(mechanism.target_actuator_names) != size
      ):
        raise ValueError(f"{name} must bind {size} source joints to {size} actuators")
      expected_context = (
        self.mechanisms["leg_length"].source_joint_names if name == "ankle" else ()
      )
      if mechanism.context_joint_names != expected_context:
        raise ValueError(f"{name}.context_joint_names must be {expected_context}")
    for label, names in (
      ("source joints", self.reference_joint_names),
      ("target actuators", self.reference_actuator_names),
    ):
      if len(set(names)) != len(names):
        raise ValueError(f"mechanisms must have distinct {label}")

  @property
  def reference_joint_names(self) -> tuple[str, ...]:
    return tuple(
      joint
      for name, _ in self._MECHANISM_SIZES
      for joint in self.mechanisms[name].source_joint_names
    )

  @property
  def reference_actuator_names(self) -> tuple[str, ...]:
    return tuple(
      actuator
      for name, _ in self._MECHANISM_SIZES
      for actuator in self.mechanisms[name].target_actuator_names
    )

  def mechanisms_for_side(self, side: str) -> dict[str, LutMechanismCfg]:
    def names(values: tuple[str, ...]) -> tuple[str, ...]:
      return tuple(reside(n, side, self.reference_side) for n in values)

    return {
      name: LutMechanismCfg(
        source_joint_names=names(self.mechanisms[name].source_joint_names),
        target_actuator_names=names(self.mechanisms[name].target_actuator_names),
        context_joint_names=names(self.mechanisms[name].context_joint_names),
      )
      for name, _ in self._MECHANISM_SIZES
    }

  @property
  def leg_length_servo_expr(self) -> str:
    """The ``target_names_expr`` entry naming the leg-length servo joints --
    the knee's targets are in metres, which the action term must know."""
    (name,) = self.mechanisms["leg_length"].source_joint_names
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
    ref = cfg.reference_side
    self.n_sides = len(cfg.sides)
    ref_joints = cfg.reference_joint_names
    # Side-major, then mechanism order. Both names and LUT query order come
    # from the explicit configuration, independently of NPZ name metadata.
    self.servo_names = [reside(n, side, ref) for side in cfg.sides for n in ref_joints]
    self.screw_names = [
      reside(n, side, ref) for side in cfg.sides for n in cfg.reference_actuator_names
    ]
    if sorted(target_names) != sorted(self.servo_names):
      raise ValueError(
        f"{type(self).__name__}: target_names_expr {cfg.target_names_expr} resolved "
        f"to {target_names}, but the configured mechanisms serve {self.servo_names}"
      )
    ids, found = entity.find_joints(self.servo_names, preserve_order=True)
    if found != self.servo_names:
      raise ValueError(f"joints {self.servo_names} resolved to {found}")
    self._servo_ids_list = ids
    # Column of each side's leg-length servo in the (num_envs, n_sides * 6) layout.
    self._leg_length_cols_list = [
      side * self.N_JOINTS + self.N_JOINTS - 1 for side in range(self.n_sides)
    ]

    joint_params_list = self._resolve_params(
      cfg.joint_params, "joint_params", self.servo_names
    )
    self._stiffness_list = [p.stiffness for p in joint_params_list]
    self._damping_list = [p.damping for p in joint_params_list]
    self._effort_limit_list = [p.effort_limit for p in joint_params_list]
    self._sign_list = [1.0] * len(self.servo_names)
    if cfg.joint_sign:
      idx, _, values = resolve_matching_names_values(cfg.joint_sign, self.servo_names)
      for i, v in zip(idx, values, strict=True):
        if v not in (-1.0, 1.0):
          raise ValueError("joint_sign values must be +1 or -1")
        self._sign_list[i] = float(v)

    screw_params_list = self._resolve_params(
      cfg.screw_params, "screw_params", self.screw_names
    )
    self._screw_effort_limit_list = [p.effort_limit for p in screw_params_list]
    self._screw_transmission_type_list = [
      p.transmission_type for p in screw_params_list
    ]
    self._screw_armature_list = [p.armature for p in screw_params_list]
    self._screw_frictionloss_list = [p.frictionloss for p in screw_params_list]
    self._screw_viscous_damping_list = [p.viscous_damping for p in screw_params_list]
    has_curve = [p.saturation_effort is not None for p in screw_params_list]
    if any(has_curve) and not all(has_curve):
      raise ValueError(
        f"{type(self).__name__}.screw_params: saturation_effort/velocity_limit "
        "must be set for every screw or none"
      )
    self._saturation_effort_list = (
      [p.saturation_effort for p in screw_params_list] if any(has_curve) else None
    )
    self._velocity_limit_list = (
      [p.velocity_limit for p in screw_params_list] if any(has_curve) else None
    )

    self._servo_ids: torch.Tensor | None = None
    self._leg_length_cols: torch.Tensor | None = None
    self._sign: torch.Tensor | None = None
    self._target_sign: torch.Tensor | None = None
    self.maps: TransmissionMaps = cfg.maps
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None
    self.actuator_force_limit: torch.Tensor | None = None
    self.saturation_effort: torch.Tensor | None = None
    self.velocity_limit: torch.Tensor | None = None
    self.default_stiffness: torch.Tensor | None = None
    self.default_damping: torch.Tensor | None = None
    self.default_force_limit: torch.Tensor | None = None
    self.default_actuator_force_limit: torch.Tensor | None = None

  def _resolve_params(self, value: dict, what: str, target_names: list[str]) -> list:
    idx, names, values = resolve_matching_names_values(value, target_names)
    if sorted(names) != sorted(target_names):
      missing = sorted(set(target_names) - set(names))
      raise ValueError(f"{type(self).__name__}.{what} does not cover {missing}")
    out: list = [None] * len(target_names)
    for i, v in zip(idx, values, strict=True):
      out[i] = v
    return out

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    del target_names  # The screws, not the servo joints, get the elements.
    for name, effort_limit, armature, frictionloss, viscous_damping, trn_type in zip(
      self.screw_names,
      self._screw_effort_limit_list,
      self._screw_armature_list,
      self._screw_frictionloss_list,
      self._screw_viscous_damping_list,
      self._screw_transmission_type_list,
      strict=True,
    ):
      # armature/viscous_damping are the motor's own properties, so they go
      # on the <motor> element itself (MuJoCo's per-actuator armature/damping)
      # rather than through create_motor_actuator's joint/tendon overrides.
      # frictionloss has no actuator-level counterpart, so it still goes
      # through to the screw's own joint/tendon.
      actuator = create_motor_actuator(
        spec,
        name,
        effort_limit=effort_limit,
        frictionloss=frictionloss,
        transmission_type=trn_type,
      )
      if armature is not None:
        actuator.armature = armature
      if viscous_damping is not None:
        actuator.damping[0] = viscous_damping
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
    self._servo_ids = torch.as_tensor(
      self._servo_ids_list, dtype=torch.long, device=device
    )
    self._leg_length_cols = torch.as_tensor(
      self._leg_length_cols_list, dtype=torch.long, device=device
    )
    self._sign = torch.tensor(self._sign_list, dtype=torch.float, device=device)
    self._target_sign = self._sign.clone()
    self._target_sign[self._leg_length_cols] = 1.0

    def per_env(values: list[float]) -> torch.Tensor:
      row = torch.tensor(values, dtype=torch.float, device=device)
      return row.unsqueeze(0).repeat(num_envs, 1)

    self.stiffness = per_env(self._stiffness_list)
    self.damping = per_env(self._damping_list)
    self.force_limit = per_env(self._effort_limit_list)
    self.actuator_force_limit = per_env(self._screw_effort_limit_list)
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()
    self.default_actuator_force_limit = self.actuator_force_limit.clone()
    if self._saturation_effort_list is not None:
      assert self._velocity_limit_list is not None
      self.saturation_effort = per_env(self._saturation_effort_list)
      self.velocity_limit = per_env(self._velocity_limit_list)
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
    """``(slider, distance, distance rate)`` from the knee's position and
    velocity."""
    leg = self.maps.leg_length
    s = leg.slider_of_knee(q)
    return s, leg.distance(s), leg.distance_rate(s, q_dot)

  def joint_torques(
    self, cmd: ActuatorCmd
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step 1 of the law: the clamped PD torques ``(num_envs, n_sides * 6)``
    in map coordinates, with the joint positions/velocities they were taken
    at (the leg length replaced by its distance/distance rate) and the
    leg-length slider positions ``(num_envs, n_sides)``."""
    assert self._sign is not None and self._leg_length_cols is not None
    assert self._target_sign is not None
    assert self.stiffness is not None and self.damping is not None
    assert self.force_limit is not None
    sign = self._sign
    cols = self._leg_length_cols
    pos, vel = cmd.pos * sign, cmd.vel * sign
    s, d, d_dot = self.leg_length_state(pos[:, cols], vel[:, cols])
    pos[:, cols] = d
    vel[:, cols] = d_dot
    torque = self.stiffness * (cmd.position_target * self._target_sign - pos)
    torque = torque + self.damping * (cmd.velocity_target * self._target_sign - vel)
    torque = torque + cmd.effort_target * self._target_sign
    torque = torch.clamp(torque, -self.force_limit, self.force_limit)
    return torque, pos, vel, s

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

  def screw_velocities(
    self, pos: torch.Tensor, vel: torch.Tensor, s: torch.Tensor
  ) -> torch.Tensor:
    """The screws' own velocities ``(num_envs, n_sides * 6)``, forward-mapped
    through the same Jacobians :meth:`screw_forces` inverts (``dx/dt = J(q)
    dq/dt``), at the joint positions/velocities in map coordinates (from
    :meth:`joint_torques`). Feeds the DC motor torque-speed curve."""
    n = self.N_JOINTS
    q = pos.reshape(-1, n)
    qd = vel.reshape(-1, n)
    s = s.reshape(-1)
    maps = self.maps
    V = torch.cat(
      (
        (maps.hip_z.jacobian(q[:, 0]) * qd[:, 0]).unsqueeze(-1),
        (maps.hip_xy.jacobian(q[:, 1:3]) @ qd[:, 1:3].unsqueeze(-1)).squeeze(-1),
        (maps.ankle.jacobian(q[:, 3:5], s) @ qd[:, 3:5].unsqueeze(-1)).squeeze(-1),
        (maps.leg_length.jacobian(s) * qd[:, 5]).unsqueeze(-1),
      ),
      dim=-1,
    )
    return V.reshape(vel.shape)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.actuator_force_limit is not None
    torque, pos, vel, s = self.joint_torques(cmd)
    force = self.screw_forces(torque, pos, s)
    if self.saturation_effort is None:
      # Ideal PD: joint_torques already clamped the torque, so the screw's
      # own effort limit is the only remaining clamp.
      return torch.clamp(force, -self.actuator_force_limit, self.actuator_force_limit)
    assert self.velocity_limit is not None
    screw_vel = self.screw_velocities(pos, vel, s)
    return dc_motor_clip(
      force,
      self.saturation_effort,
      self.velocity_limit,
      self.actuator_force_limit,
      screw_vel,
    )


__all__ = [
  "JointParams",
  "LutMechanismCfg",
  "LutTransmissionActuator",
  "LutTransmissionActuatorCfg",
  "ScrewParams",
]
