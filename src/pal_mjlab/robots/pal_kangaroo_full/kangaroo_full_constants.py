"""Pal Robotics KANGAROO FULL constants.

There is exactly one MJCF for the full model, ``xmls/kangaroo.xml``, and it
carries *every* mechanism the robot can have:

* the ``(left|right)_hip_z_slider`` spatial tendon (hip yaw screw),
* the four ``(left|right)_hip_xy_(l|r)_slider`` spatial tendons (hip pitch/roll
  parallel pair),
* the ``leg_(left|right)_length_actuator`` prismatic screw, closed onto the
  femur by the ``(left|right)_knee_rods`` equality tendon.

Variants are produced by *editing the spec*: each axis either keeps its
mechanism and actuates it, or deletes it and actuates the plain revolute /
prismatic joint underneath. That keeps a single geometry source of truth --
previously each combination was a hand-maintained copy of the same XML, which
drifted (rod lengths, ``solref``, inertias) between copies.

The three axes are independent, giving eight variants:

===========  ==========================  ===================================
axis         value                       actuation
===========  ==========================  ===================================
``hip_z``    ``"tendon"``                ``(left|right)_hip_z_slider`` tendon
             ``"joint"``                 ``leg_.*_1_joint`` revolute motor
``hip_xy``   ``"tendon"``                ``..._hip_xy_(l|r)_slider`` tendons
             ``"joint"``                 ``leg_.*_2_joint``/``leg_.*_3_joint``
``leg_len``  ``"actuator"``              ``leg_.*_length_actuator`` prismatic
             ``"joint"``                 ``leg_.*_length_joint`` directly
===========  ==========================  ===================================

With ``leg_length="joint"`` the compiled model has exactly the same joints as
the simple ``pal_kangaroo`` model; with ``"actuator"`` it adds only the two
``leg_.*_length_actuator`` screws. Tasks therefore observe and reward the
simple model's joint set in every variant -- see
:data:`REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.string import resolve_expr
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  FULL_COLLISION,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  _calc_leg_params,
)

##
# Joint name patterns.
##

# The full model differs from the simple pal_kangaroo model by at most the two
# leg_(left|right)_length_actuator screws. Excluding them yields the simple
# model's joint set exactly -- which is what every task observes and rewards,
# so that variants differ in *actuation* only.
REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = r"^(?!leg_.*_length_actuator$).*$"
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_length_actuator$).*$"
)

##
# MJCF.
##

KANGAROO_FULL_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls"
KANGAROO_FULL_XML = KANGAROO_FULL_PATH / "kangaroo.xml"

assert KANGAROO_FULL_XML.exists(), f"Missing: {KANGAROO_FULL_XML}"

HIP_Z_TENDON_NAMES = ("left_hip_z_slider", "right_hip_z_slider")
# Ordered left-outer, left-inner, right-inner, right-outer, matching the body
# tree so the action vector layout is stable.
HIP_XY_TENDON_NAMES = (
  "left_hip_xy_l_slider",
  "left_hip_xy_r_slider",
  "right_hip_xy_r_slider",
  "right_hip_xy_l_slider",
)
_KNEE_ROD_TENDON_NAMES = ("left_knee_rods", "right_knee_rods")
_LEG_LENGTH_BODY_NAMES = ("left_femur_slider", "right_femur_slider")

# Rest length of the knee rod, i.e. the length the *_knee_rods equality tendon
# holds between the leg_.*_length_actuator screw and the knee link. Applied as
# a reset event (see mdp.dr.tendon.enforce_tendon_lengths) because the value is
# a physical rod length, not the tendon's length at qpos0.
KANGAROO_TENDON_LENGTHS: dict[str, float] = {r"(left|right)_knee_rods": 0.215}

HipZActuation = Literal["tendon", "joint"]
HipXyActuation = Literal["tendon", "joint"]
LegLengthActuation = Literal["actuator", "joint"]


def _delete_tendons(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete spatial tendons and any equality constraint that references them."""
  targets = set(names)
  for eq in list(spec.equalities):
    if eq.type == mujoco.mjtEq.mjEQ_TENDON and eq.name1 in targets:
      spec.delete(eq)
  for tendon in list(spec.tendons):
    if tendon.name in targets:
      spec.delete(tendon)


def get_kangaroo_full_spec(
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
) -> mujoco.MjSpec:
  """Load ``kangaroo.xml`` and strip the mechanisms this variant doesn't use."""
  spec = mujoco.MjSpec.from_file(str(KANGAROO_FULL_XML))
  if hip_z == "joint":
    _delete_tendons(spec, HIP_Z_TENDON_NAMES)
  if hip_xy == "joint":
    _delete_tendons(spec, HIP_XY_TENDON_NAMES)
  if leg_length == "joint":
    # The knee rod tendon anchors on a site inside the slider body, so it must
    # go before the body it hangs off of.
    _delete_tendons(spec, _KNEE_ROD_TENDON_NAMES)
    for body_name in _LEG_LENGTH_BODY_NAMES:
      spec.delete(spec.body(body_name))
  return spec


##
# Actuator configs.
##

_HIP_Z_ACTUATORS: dict[HipZActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  "tendon": (
    BuiltinPositionActuatorCfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"(left|right)_hip_z_slider$",),
      **_calc_leg_params(2500.0, 2000.0),
    ),
  ),
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_1_joint",), **_calc_leg_params(100.0, 80.0)
    ),
  ),
}

_HIP_XY_ACTUATORS: dict[HipXyActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  "tendon": (
    BuiltinPositionActuatorCfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"(left|right)_hip_xy_(l|r)_slider$",),
      **_calc_leg_params(2500.0, 2000.0),
    ),
  ),
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_2_joint",), **_calc_leg_params(100.0, 230.0)
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_3_joint",), **_calc_leg_params(100.0, 139.0)
    ),
  ),
}

_ANKLE_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
  ),
)

_LEG_LENGTH_ACTUATORS: dict[
  LegLengthActuation, tuple[BuiltinPositionActuatorCfg, ...]
] = {
  "actuator": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_length_actuator$",),
      **_calc_leg_params(6000.0, 5000.0),
    ),
  ),
  # Same gains the simple pal_kangaroo model uses for this joint.
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_length_joint",), **_calc_leg_params(1600.0, 1100.0)
    ),
  ),
}

_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)

##
# Initial state.
##

# leg_.*_length_actuator is absent from the leg_length="joint" variants; a
# pattern that matches no joint is simply ignored by resolve_expr, so one
# init state covers all eight variants.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.95),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "leg_left_1_joint": -0.012,
    "leg_right_1_joint": 0.012,
    "leg_.*_2_joint": 0.054,
    "leg_left_3_joint": 0.04,
    "leg_right_3_joint": -0.04,
    "leg_.*_length_joint": 0.6,
    "leg_.*_length_actuator": 0.0284,
    "leg_.*_4_joint": -0.053,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": 0.9,
    "leg_.*_knee_joint": 1.8,
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


def _compute_tendon_lengths_at_init_state(
  spec: mujoco.MjSpec, tendon_names: tuple[str, ...]
) -> dict[str, float]:
  """Tendon lengths with the model posed at :data:`INIT_STATE`.

  This is the TENDON-transmission analogue of what ``use_default_offset=True``
  gives JOINT actuators for free: ``JointPositionAction`` reads the joint's own
  value at the default pose so a raw action of 0 holds that pose exactly.
  ``TendonLengthActionCfg`` has no ``use_default_offset``, so the equivalent
  offset is solved for here instead of being hand-maintained -- a hardcoded
  constant silently drifts out of sync with the model (it previously did, by
  ~0.3-0.5 mm).
  """
  # The raw XML references a `terrain` body for foot-collision excludes that
  # only resolves once attached into a full scene; add a placeholder so this
  # compiles standalone, matching what Scene assembly would provide.
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  data = mujoco.MjData(model)

  joint_names = tuple(
    model.joint(j).name
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  )
  for name, value in zip(
    joint_names, resolve_expr(INIT_STATE.joint_pos, joint_names, 0.0), strict=True
  ):
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  mujoco.mj_forward(model, data)

  return {name: float(data.ten_length[model.tendon(name).id]) for name in tendon_names}


##
# Variants.
##


def _build_action_scales(
  articulation: EntityArticulationInfoCfg, transmission_type: TransmissionType
) -> tuple[dict[str, float], tuple[str, ...]]:
  """Action scale dict and target names for one transmission type.

  The scale is a quarter of each actuator's torque-to-stiffness ratio, i.e. the
  position offset a quarter-effort command corresponds to.
  """
  scales: dict[str, float] = {}
  names: list[str] = []
  for actuator in articulation.actuators:
    if actuator.transmission_type != transmission_type:
      continue
    for name in actuator.target_names_expr:
      efforts = (
        actuator.effort_limit
        if isinstance(actuator.effort_limit, dict)
        else {name: actuator.effort_limit}
      )
      stiffnesses = (
        actuator.stiffness
        if isinstance(actuator.stiffness, dict)
        else {name: actuator.stiffness}
      )
      if name in efforts and stiffnesses.get(name):
        scales[name] = 0.25 * efforts[name] / stiffnesses[name]
        names.append(name)
  return scales, tuple(names)


@dataclass(frozen=True)
class TendonAction:
  """The three fields a ``TendonLengthActionCfg`` needs for one mechanism."""

  actuator_names: tuple[str, ...]
  """Tendon names, in the order they should occupy in the action vector."""
  scale: dict[str, float]
  offset: dict[str, float]


@dataclass(frozen=True)
class KangarooFullModel:
  """One actuation variant of the full KANGAROO model."""

  hip_z: HipZActuation
  hip_xy: HipXyActuation
  leg_length: LegLengthActuation

  articulation: EntityArticulationInfoCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]
  hip_z_tendon_action: TendonAction | None
  hip_xy_tendon_action: TendonAction | None

  @property
  def has_knee_rod_tendons(self) -> bool:
    """Whether the ``*_knee_rods`` equality tendons exist in this variant."""
    return self.leg_length == "actuator"

  def make_spec(self) -> mujoco.MjSpec:
    return get_kangaroo_full_spec(
      hip_z=self.hip_z, hip_xy=self.hip_xy, leg_length=self.leg_length
    )

  def make_robot_cfg(self) -> EntityCfg:
    return EntityCfg(
      init_state=INIT_STATE,
      collisions=(FULL_COLLISION,),
      spec_fn=self.make_spec,
      articulation=self.articulation,
    )


@lru_cache(maxsize=None)
def get_kangaroo_full_model(
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
) -> KangarooFullModel:
  """Assemble the actuators, action scales and tendon offsets for one variant.

  Cached because the tendon offsets require compiling the model, and every task
  registration asks for the same handful of variants.
  """
  articulation = EntityArticulationInfoCfg(
    # Ordered like the simple pal_kangaroo model's actuators (hip yaw, hip
    # pitch/roll, ankle, leg length, then upper body) so the JOINT action
    # vector reads the same way in every variant.
    actuators=(
      _HIP_Z_ACTUATORS[hip_z]
      + _HIP_XY_ACTUATORS[hip_xy]
      + _ANKLE_ACTUATORS
      + _LEG_LENGTH_ACTUATORS[leg_length]
      + _UPPER_BODY_ACTUATORS
    ),
    soft_joint_pos_limit_factor=0.99,
  )

  joint_action_scale, joint_actuator_names = _build_action_scales(
    articulation, TransmissionType.JOINT
  )
  tendon_scale, _ = _build_action_scales(articulation, TransmissionType.TENDON)

  tendon_names = (HIP_Z_TENDON_NAMES if hip_z == "tendon" else ()) + (
    HIP_XY_TENDON_NAMES if hip_xy == "tendon" else ()
  )
  offsets = (
    _compute_tendon_lengths_at_init_state(
      get_kangaroo_full_spec(hip_z=hip_z, hip_xy=hip_xy, leg_length=leg_length),
      tendon_names,
    )
    if tendon_names
    else {}
  )

  def _tendon_action(names: tuple[str, ...], key: str) -> TendonAction:
    # Each action term's scale/offset may only carry keys matching its own
    # targets: resolve_matching_names_values errors on a key that matches none.
    return TendonAction(
      actuator_names=tuple(f"{name}$" for name in names),
      scale={k: v for k, v in tendon_scale.items() if key in k},
      offset={name: offsets[name] for name in names},
    )

  return KangarooFullModel(
    hip_z=hip_z,
    hip_xy=hip_xy,
    leg_length=leg_length,
    articulation=articulation,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
    hip_z_tendon_action=(
      _tendon_action(HIP_Z_TENDON_NAMES, "hip_z") if hip_z == "tendon" else None
    ),
    hip_xy_tendon_action=(
      _tendon_action(HIP_XY_TENDON_NAMES, "hip_xy") if hip_xy == "tendon" else None
    ),
  )


def _pin_equality_tendon_lengths(model: mujoco.MjModel) -> None:
  """Set each equality tendon's rest length to its physical rod length.

  Standalone mirror of the ``enforce_tendon_lengths`` reset event the tasks
  run, down to reading the tendon's length at qpos0 from ``tendon_length0``
  rather than measuring it at the current pose, so the viewer shows the same
  mechanism the policy is trained against.
  """
  for i in range(model.neq):
    eq = model.eq(i)
    if eq.type != mujoco.mjtEq.mjEQ_TENDON:
      continue
    tendon_id = int(eq.obj1id.item())
    name = model.tendon(tendon_id).name
    for pattern, length in KANGAROO_TENDON_LENGTHS.items():
      if re.fullmatch(pattern, name):
        eq.data[0] = length - model.tendon_length0[tendon_id]


def main(
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the full KANGAROO model.

  Args:
    hip_z: Drive hip yaw through its spatial tendon, or through the plain
      leg_.*_1_joint revolute motor.
    hip_xy: Drive hip pitch/roll through the parallel tendon pair, or through
      the plain leg_.*_2_joint / leg_.*_3_joint revolute motors.
    leg_length: Drive leg length through the leg_.*_length_actuator screw and
      its knee rod equality tendon, or directly through leg_.*_length_joint.
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_model(hip_z=hip_z, hip_xy=hip_xy, leg_length=leg_length)

  # Go through Entity rather than compiling make_spec() directly: kangaroo.xml
  # declares no <actuator> elements at all, so the raw spec compiles to nu=0.
  # The actuators (and the collision setup, and an "init_state" keyframe) are
  # what the articulation config adds on top -- which is exactly the layer this
  # command exists to inspect.
  entity = Entity(model_cfg.make_robot_cfg())
  spec = entity.spec
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  _pin_equality_tendon_lengths(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)

  print(f"hip_z={hip_z} hip_xy={hip_xy} leg_length={leg_length}")
  # Called out because it is silent otherwise and changes what you are looking
  # at completely: with the base freejoint commented out in the MJCF, mjlab
  # classifies the robot as fixed base and bolts it to a mocap body, so it
  # hangs in the air instead of standing on the terrain.
  if entity.is_fixed_base:
    print("  base:    FIXED (no freejoint in the MJCF -- pinned to a mocap body)")
  else:
    print("  base:    floating")
  print(f"  joints:  {model.njnt} ({model.nv} dof)")
  print(f"  tendons: {model.ntendon}")
  print(f"  equalities: {model.neq}")
  print(f"  actuators ({model.nu}):")
  for i in range(model.nu):
    actuator = model.actuator(i)
    kind = "tendon" if actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON else "joint"
    print(f"    {actuator.name:34s} ({kind})  kp={actuator.gainprm[0]:.1f}")
  print(f"  joint action targets ({len(model_cfg.joint_actuator_names)}):")
  for name in model_cfg.joint_actuator_names:
    print(f"    {name}  scale={model_cfg.joint_action_scale[name]:.4f}")
  for label, tendon_action in (
    ("hip_z", model_cfg.hip_z_tendon_action),
    ("hip_xy", model_cfg.hip_xy_tendon_action),
  ):
    if tendon_action is None:
      print(f"  {label} tendon action: none (driven as a joint)")
      continue
    print(f"  {label} tendon action targets ({len(tendon_action.actuator_names)}):")
    for name, offset in tendon_action.offset.items():
      print(f"    {name}  offset={offset:.9f}")

  if launch_viewer:
    # Imported here, not at module scope, so importing these constants doesn't
    # drag in the viewer's GL dependencies.
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
