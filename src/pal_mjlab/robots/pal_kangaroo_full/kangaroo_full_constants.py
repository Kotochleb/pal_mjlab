"""Pal Robotics KANGAROO FULL constants.

There is exactly one MJCF for the full model,
``xmls/kangaroo_full_tendons.xml``, and it carries *every* mechanism the robot
can have:

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

The first three axes are actuation choices and are independent, giving sixteen
variants; the fourth picks how the femur closes and constrains the third:

===========  ===========================  ==================================
axis         value                        actuation
===========  ===========================  ==================================
``hip_z``    ``"tendon"``                 ``(left|right)_hip_z_slider``
             ``"joint"``                  ``leg_.*_1_joint`` revolute motor
``hip_xy``   ``"tendon"``                 ``..._hip_xy_(l|r)_slider`` tendons
             ``"joint"``                  ``leg_.*_2_joint``/``leg_.*_3_joint``
``leg_len``  ``"actuator"``               ``leg_.*_length_actuator`` prismatic
             ``"semi_serial"``            ``leg_.*_length_joint`` PD, torque
                                          pushed through the screw's LUT onto
                                          ``leg_.*_length_actuator``
             ``"semi_serial_actuator_pd"``  the same, but P only on the joint
                                          and the PD closes on the screw
             ``"joint"``                  ``leg_.*_length_joint`` directly
``femur``    ``"prismatic"``              -- (a geometry choice, not an
             ``"linkage"``                actuation one; see below)
===========  ===========================  ==================================

``femur_closure`` says which of two redundant descriptions of the femur the
compiled model keeps. ``"prismatic"`` keeps the straight-line stand-in: the
``leg_.*_length_joint`` slider pinned to the knee by the
``leg_.*_length_connect`` ``<connect>``, with the ``(left|right)_hip_xy_link``
and ``(left|right)_femur_rod`` tendons deleted and the now-idle
``(left|right)_femur_triangle`` crank's joint deleted (welding it to the femur
at qpos 0). ``"linkage"`` keeps the real
four-bar those two tendons form and deletes the slider and its ``<connect>``
instead -- which also removes the DOF three of the four ``leg_length`` values
actuate, so ``"linkage"`` only combines with ``leg_length="actuator"``.

Every value keeps the two ``leg_.*_length_actuator`` screws in the model;
``"joint"`` simply locks them at 0 rather than driving them, so the joint set
the tasks observe is the simple ``pal_kangaroo`` model's in every case -- see
:data:`REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY`. The two ``semi_serial`` values compile the same
model as ``"actuator"`` but command it differently: the policy servos
``leg_.*_length_joint`` (the simple model's DOF) and the resulting joint torque
is mapped onto the screw through the measured transmission Jacobian in
``transmission/leg_length.csv``. ``"semi_serial"`` then applies that force
directly; ``"semi_serial_actuator_pd"`` turns it back into a setpoint for a
native ``<position>`` element on the screw, so the derivative term is taken on
the screw's velocity rather than the joint's. Tasks therefore observe and reward the
simple model's joint set in every variant -- see
:data:`REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import mujoco
from mjlab.actuator import ActuatorCfg, BuiltinPositionActuatorCfg
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
from pal_mjlab.robots.pal_kangaroo_full.actuator import (
  TransmitedIdealPdActuatorCfg,
  TransmittedPositionActuatorCfg,
  load_transmission_table,
)

##
# Joint name patterns.
##

# The full model carries ten joints the simple pal_kangaroo model does not: the
# two leg_(left|right)_length_actuator screws, and the eight mechanism DOFs of
# the femur triangles and butterflies. Every one of the simple model's joints
# is named pelvis_*, arm_* or leg_*, and of the extras only the screws are, so
# that prefix plus one exclusion selects the simple model's joint set exactly
# -- which is what every task observes and rewards, so that variants differ in
# *actuation* only.
REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = (
  r"^(?!leg_.*_length_actuator$)(pelvis|arm|leg)_.*$"
)
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_length_actuator$)(pelvis|arm|leg)_.*$"
)

# The same joint set as an ordered list: the order the *simple* model compiles
# them in, which the full model does not share (it interleaves the mechanism
# DOFs, and puts leg_.*_femur_joint ahead of leg_.*_4_joint rather than after
# leg_.*_5_joint). Observation terms select on this with preserve_order=True so
# the policy reads the same vector layout on either model.
SIMPLE_MODEL_JOINT_ORDER: tuple[str, ...] = (
  "pelvis_1_joint",
  "pelvis_2_joint",
  "arm_left_1_joint",
  "arm_left_2_joint",
  "arm_left_3_joint",
  "arm_left_4_joint",
  "arm_right_1_joint",
  "arm_right_2_joint",
  "arm_right_3_joint",
  "arm_right_4_joint",
  "leg_left_1_joint",
  "leg_left_2_joint",
  "leg_left_3_joint",
  "leg_left_length_joint",
  "leg_left_4_joint",
  "leg_left_5_joint",
  "leg_left_femur_joint",
  "leg_left_knee_joint",
  "leg_right_1_joint",
  "leg_right_2_joint",
  "leg_right_3_joint",
  "leg_right_length_joint",
  "leg_right_4_joint",
  "leg_right_5_joint",
  "leg_right_femur_joint",
  "leg_right_knee_joint",
)

##
# MJCF.
##

KANGAROO_FULL_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls"
KANGAROO_FULL_XML = KANGAROO_FULL_PATH / "kangaroo_full_tendons.xml"
# Same geometry and actuation axes as KANGAROO_FULL_XML -- every hip_z / hip_xy
# / leg_length / femur_closure / ankle flag above applies identically -- but
# with extra closed-loop constraints layered on top of the ones the nominal
# MJCF already carries, for redundancy/over-constraint experiments. The file
# on disk keeps its typo'd name (kangaroo_full_tendons_over_constarined.xml);
# only the Python-facing MjcfVariant value below is spelled correctly.
KANGAROO_FULL_XML_OVER_CONSTRAINED = (
  KANGAROO_FULL_PATH / "kangaroo_full_tendons_over_constarined.xml"
)

MjcfVariant = Literal["tendons", "tendons_over_constrained"]
_MJCF_XML_PATHS: dict[MjcfVariant, Path] = {
  "tendons": KANGAROO_FULL_XML,
  "tendons_over_constrained": KANGAROO_FULL_XML_OVER_CONSTRAINED,
}

# Measured (leg_length_joint -> leg_length_actuator) transmission Jacobian of
# the knee screw, used by the leg_length="semi_serial" actuator to push a joint
# torque through the mechanism. Rows are (joint_pos, dF_actuator/dtau_joint).
LEG_LENGTH_TRANSMISSION_CSV = (
  KANGAROO_FULL_PATH.parent / "transmission" / "leg_length.csv"
)

# Swept over this same MJCF: where leg_.*_length_connect_b sits relative to the
# leg_.*_femur_joint anchor as the knee folds. Rows are (knee_rad, knee_deg,
# displacement_x_m, displacement_z_m, distance_m), world frame.
KNEE_DISTANCE_MAP_CSV = (
  KANGAROO_FULL_PATH.parent / "transmission" / "knee_distance_map.csv"
)

# Which knee stands in for which leg length joint, for the variants that have
# no leg length joint to observe (see mdp.observations).
LEG_LENGTH_FROM_KNEE_JOINTS: tuple[tuple[str, str], ...] = tuple(
  (f"leg_{side}_length_joint", f"leg_{side}_knee_joint") for side in ("left", "right")
)

# The triples that map reads: (femur joint, knee joint, connect site) per leg.
KNEE_DISTANCE_MAP_LEGS: tuple[tuple[str, str, str], ...] = tuple(
  (
    f"leg_{side}_femur_joint",
    f"leg_{side}_knee_joint",
    f"leg_{side}_length_connect_b",
  )
  for side in ("left", "right")
)

for _path in (
  KANGAROO_FULL_XML,
  KANGAROO_FULL_XML_OVER_CONSTRAINED,
  LEG_LENGTH_TRANSMISSION_CSV,
  KNEE_DISTANCE_MAP_CSV,
):
  assert _path.exists(), f"Missing: {_path}"

HIP_Z_TENDON_NAMES = ("left_hip_z_slider", "right_hip_z_slider")
# Ordered left-outer, left-inner, right-inner, right-outer, matching the body
# tree so the action vector layout is stable.
HIP_XY_TENDON_NAMES = (
  "left_hip_xy_l_slider",
  "left_hip_xy_r_slider",
  "right_hip_xy_r_slider",
  "right_hip_xy_l_slider",
)
# The ankle "virtual motor" tendon pair: each spans from the butterfly
# decoupler straight to one butterfly, a chord-length stand-in for actuating
# that butterfly joint directly (the real crank/motor mechanism the meshes
# suggest -- (left|right)_ankle_crank_(l|r) -- has its joint welded off in the
# MJCF and isn't modeled). Ordered left_l, left_r, right_r, right_l, matching
# HIP_XY_TENDON_NAMES's left-outer/left-inner/right-inner/right-outer pattern.
ANKLE_TENDON_NAMES = (
  "left_ankle_l_slider",
  "left_ankle_r_slider",
  "right_ankle_r_slider",
  "right_ankle_l_slider",
)
_KNEE_ROD_TENDON_NAMES = ("left_knee_rods", "right_knee_rods")
_LEG_LENGTH_ACTUATOR_JOINT_NAMES = (
  "leg_left_length_actuator",
  "leg_right_length_actuator",
)

# The femur four-bar: the (left|right)_hip_xy_link tendon ties the hip to the
# (left|right)_femur_triangle crank, and (left|right)_femur_rod ties that crank
# on towards the knee. Together they are what makes the femur fold; the
# "prismatic" closure replaces both with the straight-line slider below.
_FEMUR_ROD_TENDON_NAMES = ("left_femur_rod", "right_femur_rod")
_FEMUR_LINKAGE_TENDON_NAMES = (
  "left_hip_xy_link",
  "right_hip_xy_link",
) + _FEMUR_ROD_TENDON_NAMES
_FEMUR_TRIANGLE_JOINT_NAMES = ("left_femur_triangle", "right_femur_triangle")

# The ankle chain: each butterfly swings one ankle bar, and the decoupler they
# both hang off is geared to half the knee angle so that folding the knee does
# not drag the foot with it.
_BUTTERFLY_JOINT_NAMES = tuple(
  f"{side}_butterfly_{suffix}"
  for side in ("left", "right")
  for suffix in ("l", "r", "decoupler_joint")
)
_BUTTERFLY_DECOUPLER_EQ_NAMES = (
  "left_butterfly_decoupler_coupling",
  "right_butterfly_decoupler_coupling",
)
# The bars each butterfly swings the ankle through -- redundant once the
# butterflies themselves are deleted for a joint-actuated ankle, with nothing
# left to drive them.
_ANKLE_TIBIA_BAR_TENDON_NAMES = tuple(
  f"{side}_ankle_tibia_bar_{suffix}"
  for side in ("left", "right")
  for suffix in ("l", "r")
)
# The straight-line stand-in for that four-bar: a prismatic joint whose free
# end is pinned to the knee link by a <connect>, i.e. the simple model's leg
# length DOF.
_LEG_LENGTH_JOINT_NAMES = ("leg_left_length_joint", "leg_right_length_joint")
_LEG_LENGTH_CONNECT_EQ_NAMES = (
  "leg_left_length_connect",
  "leg_right_length_connect",
)

# Rest length of the knee rod, i.e. the length the *_knee_rods equality tendon
# holds between the leg_.*_length_actuator screw and the knee link. Applied as
# a reset event (see mdp.dr.tendon.enforce_tendon_lengths) because the value is
# a physical rod length, not the tendon's length at qpos0.
KANGAROO_TENDON_LENGTHS: dict[str, float] = {r"(left|right)_knee_rods": 0.215}

HipZActuation = Literal["tendon", "joint"]
HipXyActuation = Literal["tendon", "joint"]
LegLengthActuation = Literal[
  "actuator", "semi_serial", "semi_serial_actuator_pd", "joint"
]
FemurClosure = Literal["linkage", "prismatic"]
AnkleActuation = Literal["butterfly", "joint", "tendon"]


def _delete_tendons(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete spatial tendons and any equality constraint that references them."""
  targets = set(names)
  for eq in list(spec.equalities):
    if eq.type == mujoco.mjtEq.mjEQ_TENDON and eq.name1 in targets:
      spec.delete(eq)
  for tendon in list(spec.tendons):
    if tendon.name in targets:
      spec.delete(tendon)


def _delete_equalities(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete equality constraints by name."""
  targets = set(names)
  for eq in list(spec.equalities):
    if eq.name in targets:
      spec.delete(eq)


def _delete_joints(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete joints by name, welding their body to its parent."""
  targets = set(names)
  for joint in list(spec.joints):
    if joint.name in targets:
      spec.delete(joint)


##
# Collision geometry.
#
# Both MJCFs bake the foot and ankle capsules straight into the model (the
# ``foot_capsule`` / ``collision`` default classes at the top of the file), so
# those need no help here. What they don't carry is the rest of the capsule
# set the hand-written ``pal_kangaroo`` model uses: the pelvis, forearms,
# femurs and tibias, plus the inboard hip-xy motor of each pair (the pair
# sits 7 cm apart, so one capsule cannot reach both, and the inboard one is
# what the opposite leg can run into). ``_add_collision_capsules`` fills
# those in, lifted verbatim from ``pal_kangaroo``'s ``kangaroo.xml`` so both
# robots present the same contact geometry to a policy.
##


class _Capsule(NamedTuple):
  """One collision capsule, in the local frame of ``body``.

  Given either by its two endpoints (``fromto``) or, for the capsules that
  came out of a mesh-fitting tool, by ``pos``/``quat``/``half_length``.
  """

  body: str
  name: str
  radius: float
  fromto: tuple[float, ...] | None = None
  pos: tuple[float, float, float] | None = None
  quat: tuple[float, float, float, float] | None = None
  half_length: float | None = None


def _leg_capsules(side: str) -> tuple[_Capsule, ...]:
  """The femur/tibia/hip-xy-motor capsules for one leg."""
  motor = f"{side}_hip_xy_motor_{'r' if side == 'left' else 'l'}"
  return (
    _Capsule(
      motor, f"{motor}_collision", 0.04, fromto=(-0.01, 0.0, -0.015, -0.01, 0.0, 0.0)
    ),
    _Capsule(
      f"leg_{side}_femur_link",
      f"leg_{side}_femur_collision",
      0.08,
      pos=(0.03, 0.2, 0.0),
      quat=(0.0308436, 0.0308436, 0.7064338, 0.7064338),
      half_length=0.2,
    ),
    _Capsule(
      f"leg_{side}_knee_link",
      f"leg_{side}_knee_collision",
      0.04,
      fromto=(0.03896, 0.02628, 0.0, -0.16904, 0.24934, 0.0),
    ),
    _Capsule(
      f"leg_{side}_knee_link",
      f"leg_{side}_knee_bar_collision",
      0.05,
      fromto=(-0.063, 0.0, 0.0, -0.184, 0.23889, 0.0),
    ),
  )


_CAPSULES: tuple[_Capsule, ...] = (
  _Capsule(
    "pelvis_2_link",
    "pelvis_2_collision",
    0.163724,
    pos=(4.77798e-07, -1.67441e-06, 0.234507),
    quat=(1.0, -3.97075e-06, -1.19191e-05, 0.0),
    half_length=0.162985,
  ),
  _Capsule(
    "arm_left_4_link",
    "arm_left_4_collision",
    0.054849,
    pos=(0.210246, -0.0167497, -0.0171982),
    quat=(0.733112, 0.0532001, 0.678025, -2.2842e-07),
    half_length=0.219217,
  ),
  _Capsule(
    "arm_right_4_link",
    "arm_right_4_collision",
    0.0559132,
    pos=(0.210649, -0.0172373, -0.0265895),
    quat=(0.717524, 0.0513886, 0.694636, -1.38234e-06),
    half_length=0.220143,
  ),
  *_leg_capsules("left"),
  *_leg_capsules("right"),
)


def _require_body(spec: mujoco.MjSpec, name: str, wanted_by: str) -> mujoco.MjsBody:
  body = spec.body(name)
  if body is None:
    raise ValueError(f"MJCF has no body '{name}' to hang '{wanted_by}' off")
  return body


def _add_collision_capsules(spec: mujoco.MjSpec) -> None:
  """Add the pelvis/arm/femur/tibia/hip-xy-motor capsules ``_CAPSULES`` lists.

  Capsules get ``density=0``: every body they hang off declares an explicit
  ``<inertial>``, so geom-derived mass would be ignored anyway, and a
  collision proxy has no business changing the robot's dynamics if that ever
  stops being true.
  """
  for capsule in _CAPSULES:
    body = _require_body(spec, capsule.body, capsule.name)
    shape = (
      {"fromto": capsule.fromto, "size": [capsule.radius, 0.0, 0.0]}
      if capsule.fromto is not None
      else {
        "pos": capsule.pos,
        "quat": capsule.quat,
        "size": [capsule.radius, capsule.half_length, 0.0],
      }
    )
    body.add_geom(
      name=capsule.name,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      group=3,
      contype=1,
      conaffinity=1,
      density=0.0,
      material="bright_orange",
      **shape,
    )


def get_kangaroo_full_spec(
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
  femur_closure: FemurClosure = "prismatic",
  ankle: AnkleActuation = "joint",
  mjcf: MjcfVariant = "tendons",
) -> mujoco.MjSpec:
  """Load the MJCF and strip the mechanisms this variant doesn't use."""
  if femur_closure == "linkage" and leg_length != "actuator":
    # Caught here rather than downstream, where it surfaces as an opaque
    # "No joints matched expressions" from the actuator config.
    raise ValueError(
      f'femur_closure="linkage" deletes leg_.*_length_joint, but '
      f'leg_length="{leg_length}" actuates it. Use leg_length="actuator" '
      'to drive the screw directly, or femur_closure="prismatic".'
    )
  spec = mujoco.MjSpec.from_file(str(_MJCF_XML_PATHS[mjcf]))
  _add_collision_capsules(spec)
  if hip_z == "joint":
    _delete_tendons(spec, HIP_Z_TENDON_NAMES)
  if hip_xy == "joint":
    _delete_tendons(spec, HIP_XY_TENDON_NAMES)
  if leg_length == "joint":
    # Servoing leg_.*_length_joint directly makes the screw redundant. The
    # slider body stays -- its mass and inertia are still on the femur -- but
    # its joint is deleted (welding it to the femur at qpos 0) and the knee
    # rod that closed it onto the femur goes, so nothing drives it and
    # nothing hangs off it.
    _delete_tendons(spec, _KNEE_ROD_TENDON_NAMES)
    _delete_joints(spec, _LEG_LENGTH_ACTUATOR_JOINT_NAMES)
  if femur_closure == "prismatic":
    _delete_tendons(spec, _FEMUR_LINKAGE_TENDON_NAMES)
    _delete_joints(spec, _FEMUR_TRIANGLE_JOINT_NAMES)
  else:
    _delete_equalities(spec, _LEG_LENGTH_CONNECT_EQ_NAMES)
    _delete_joints(spec, _LEG_LENGTH_JOINT_NAMES)
  if ankle != "tendon":
    # The virtual-motor tendon pair is only meaningful as an actuator
    # transmission; with nothing commanding it, leave it out of the model.
    _delete_tendons(spec, ANKLE_TENDON_NAMES)
  if ankle == "joint":
    # Driving leg_.*_4_joint / leg_.*_5_joint directly makes the butterfly
    # chain redundant, so it is frozen rather than left to swing: the
    # decoupler's gearing to the knee goes, every butterfly joint is deleted
    # (welding butterfly_l/r to the decoupler, and the decoupler to the femur,
    # each at qpos 0), and the ankle_tibia_bar tendons they swung go with
    # them -- with the butterflies gone, nothing drives those bars either.
    # *_femur_rod stays out of this: under "linkage" it is one of the two
    # tendons actually closing the femur four-bar (with *_hip_xy_link), and
    # deleting it regardless of the ankle axis leaves leg_.*_femur_joint with
    # nothing holding it, so it swings to its limit under gravity. "prismatic"
    # already deleted it above, as part of the four-bar it replaces.
    _delete_equalities(spec, _BUTTERFLY_DECOUPLER_EQ_NAMES)
    _delete_joints(spec, _BUTTERFLY_JOINT_NAMES)
    _delete_tendons(spec, _ANKLE_TIBIA_BAR_TENDON_NAMES)
    if femur_closure != "linkage":
      _delete_tendons(spec, _FEMUR_ROD_TENDON_NAMES)
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

_ANKLE_ACTUATORS: dict[AnkleActuation, tuple[ActuatorCfg, ...]] = {
  # The hardware topology: the two butterflies per leg swing the ankle through
  # the *_ankle_tibia_bar equality tendons, leaving leg_.*_4_joint and
  # leg_.*_5_joint passive. Four targets per pair, as before, so the action
  # vector keeps its width.
  "butterfly": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"(left|right)_butterfly_l$",),
      **_calc_leg_params(100.0, 30.0),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"(left|right)_butterfly_r$",),
      **_calc_leg_params(100.0, 30.0),
    ),
  ),
  # The simple model's topology: servo the ankle pitch and roll joints
  # themselves, with the butterfly chain frozen out of the way by the spec
  # edits in get_kangaroo_full_spec.
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
    ),
  ),
  # Same hardware topology as "butterfly", but each butterfly is driven
  # through its virtual-motor tendon (ANKLE_TENDON_NAMES) rather than a joint
  # actuator on the butterfly itself -- tendon-space gains, so it takes the
  # hip tendons' stiffness/effort rather than the butterfly joint's.
  "tendon": (
    BuiltinPositionActuatorCfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"(left|right)_ankle_(l|r)_slider$",),
      **_calc_leg_params(2500.0, 2000.0),
    ),
  ),
}

_LEG_LENGTH_ACTUATORS: dict[LegLengthActuation, tuple[ActuatorCfg, ...]] = {
  "actuator": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_length_actuator$",),
      **_calc_leg_params(6000.0, 5000.0),
    ),
  ),
  # The screw is present, as in "actuator", but the PD law runs on the joint
  # the simple model actuates, at the simple model's gains; only the resulting
  # torque is transmitted to the screw. So the action means the same thing here
  # as in "joint", while the mechanism underneath is the full one.
  "semi_serial": (
    TransmitedIdealPdActuatorCfg(
      target_names_expr=(r"leg_(left|right)_length_joint$",),
      joint_to_actuator_map={
        "leg_left_length_joint": "leg_left_length_actuator",
        "leg_right_length_joint": "leg_right_length_actuator",
      },
      transmission=load_transmission_table(LEG_LENGTH_TRANSMISSION_CSV),
      actuator_effort_limit=5000.0,
      **_calc_leg_params(900.0, 1100.0),
    ),
  ),
  # Same three-stage command as "semi_serial", but only the P term is taken on
  # the joint; the transmitted force is handed back to the screw as a setpoint
  # offset (hence the division by the screw's own kp), so the D term is taken
  # on the screw's velocity by a native <position> element at the screw's own
  # gains -- the same gains, and the same element, the "actuator" variant uses.
  # The screw carries the only effort limit in the chain, so this variant also
  # takes the "actuator" variant's action scale (0.25 * 5000/6000), even though
  # the action it scales is a leg_.*_length_joint position, as in "semi_serial".
  "semi_serial_actuator_pd": (
    TransmittedPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_length_joint$",),
      joint_to_actuator_map={
        "leg_left_length_joint": "leg_left_length_actuator",
        "leg_right_length_joint": "leg_right_length_actuator",
      },
      transmission=load_transmission_table(LEG_LENGTH_TRANSMISSION_CSV),
      joint_stiffness=900.0,
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
# base init state covers every hip_z / hip_xy / leg_length / femur_closure
# variant. leg_.*_4_joint and the butterflies are the exception: the two
# MJCFs describe different ankle geometry, so those two keys rest at a
# different pose per :data:`MjcfVariant` and are layered on top by
# get_kangaroo_full_model rather than fixed here -- see
# :data:`_MJCF_ANKLE_INIT_STATE`.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.90),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "leg_left_1_joint": -0.012,
    "leg_right_1_joint": 0.012,
    "leg_.*_2_joint": 0.0522,
    "leg_left_3_joint": 0.04,
    "leg_right_3_joint": -0.04,
    "leg_.*_length_joint": -0.125,
    "leg_.*_length_actuator": 0.02766,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": -0.29636,
    "leg_.*_knee_joint": 0.5978,
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

# leg_.*_4_joint / butterfly rest pose, layered onto INIT_STATE.joint_pos by
# get_kangaroo_full_model -- see the note on INIT_STATE above.
_MJCF_ANKLE_INIT_STATE: dict[MjcfVariant, dict[str, float]] = {
  "tendons": {
    "leg_.*_4_joint": -0.2953,
    ".*_butterfly_(r|l)": 0.0,
  },
  "tendons_over_constrained": {
    "leg_.*_4_joint": 0.2953,
    ".*_butterfly_(r|l)": 0.543,
  },
}


def _compute_tendon_lengths_at_init_state(
  spec: mujoco.MjSpec, tendon_names: tuple[str, ...], joint_pos: dict[str, float]
) -> dict[str, float]:
  """Tendon lengths with the model posed at ``joint_pos``.

  This is the TENDON-transmission analogue of what ``use_default_offset=True``
  gives JOINT actuators for free: ``JointPositionAction`` reads the joint's own
  value at the default pose so a raw action of 0 holds that pose exactly.
  ``TendonLengthActionCfg`` has no ``use_default_offset``, so the equivalent
  offset is solved for here instead of being hand-maintained -- a hardcoded
  constant silently drifts out of sync with the model (it previously did, by
  ~0.3-0.5 mm). ``joint_pos`` is a caller's fully-assembled init state (see
  :data:`INIT_STATE` and :data:`_MJCF_ANKLE_INIT_STATE`), not read off a
  module-level default, since the rest pose differs by :data:`MjcfVariant`.
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
    joint_names, resolve_expr(joint_pos, joint_names, 0.0), strict=True
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
  femur_closure: FemurClosure
  ankle: AnkleActuation
  mjcf: MjcfVariant

  articulation: EntityArticulationInfoCfg
  init_state: EntityCfg.InitialStateCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]
  hip_z_tendon_action: TendonAction | None
  hip_xy_tendon_action: TendonAction | None
  ankle_tendon_action: TendonAction | None

  @property
  def has_knee_rod_tendons(self) -> bool:
    """Whether the ``*_knee_rods`` equality tendons exist in this variant."""
    return self.leg_length != "joint"

  @property
  def has_femur_linkage_tendons(self) -> bool:
    """Whether the ``*_hip_xy_link`` and ``*_femur_rod`` tendons exist.

    The two of them are the real femur four-bar and only ever exist or don't
    together -- one without the other leaves ``leg_.*_femur_joint`` connected
    to a linkage that doesn't close, or ``left_femur_triangle`` driven by a
    rod with nothing on its other end. A single property, checked once,
    rather than one per tendon, keeps that true by construction instead of by
    two definitions that happen to agree.

    The complement of :attr:`has_leg_length_joint`: a variant closes the femur
    one way or the other, never both.
    """
    return self.femur_closure == "linkage"

  @property
  def has_butterfly_decoupler_coupling(self) -> bool:
    """Whether the decoupler is still geared to the knee in this variant.

    Both "butterfly" and "tendon" keep the butterflies -- they only differ in
    whether a butterfly is driven by a joint actuator or by its virtual-motor
    tendon -- so both need that gearing; servoing the ankle joints directly
    deletes it and locks the butterflies instead.
    """
    return self.ankle != "joint"

  @property
  def has_ankle_tibia_bar_tendons(self) -> bool:
    """Whether the ``*_ankle_tibia_bar_(l|r)`` equality tendons exist.

    Same axis as :attr:`has_butterfly_decoupler_coupling`: these are the bars
    the butterflies swing the ankle through, so a joint-actuated ankle deletes
    them along with the butterflies that would have driven them.
    """
    return self.ankle != "joint"

  @property
  def has_joint_equalities(self) -> bool:
    """Whether any ``<joint>`` equality (a geared joint pair) is in the MJCF.

    Read off the spec rather than the variant, because unlike the tendons and
    the screw this is not something the variant axes add or remove -- it is
    whatever ``kangaroo_full_tendons.xml`` currently declares.
    """
    return any(eq.type == mujoco.mjtEq.mjEQ_JOINT for eq in self.make_spec().equalities)

  @property
  def has_leg_length_joint(self) -> bool:
    """Whether the ``leg_.*_length_joint`` prismatic DOF exists in this variant.

    Only the "prismatic" femur closure keeps it; "linkage" folds the femur
    through the ``*_hip_xy_link`` / ``*_femur_rod`` tendons instead and the
    straight-line slider goes away with its ``<connect>``.
    """
    return self.femur_closure == "prismatic"

  def make_spec(self) -> mujoco.MjSpec:
    return get_kangaroo_full_spec(
      hip_z=self.hip_z,
      hip_xy=self.hip_xy,
      leg_length=self.leg_length,
      femur_closure=self.femur_closure,
      ankle=self.ankle,
      mjcf=self.mjcf,
    )

  def make_robot_cfg(self) -> EntityCfg:
    return EntityCfg(
      init_state=self.init_state,
      collisions=(FULL_COLLISION,),
      spec_fn=self.make_spec,
      articulation=self.articulation,
    )


@lru_cache(maxsize=None)
def get_kangaroo_full_model(
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
  femur_closure: FemurClosure = "prismatic",
  ankle: AnkleActuation = "joint",
  mjcf: MjcfVariant = "tendons",
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
      + _ANKLE_ACTUATORS[ankle]
      + _LEG_LENGTH_ACTUATORS[leg_length]
      + _UPPER_BODY_ACTUATORS
    ),
    soft_joint_pos_limit_factor=0.99,
  )

  init_state = EntityCfg.InitialStateCfg(
    pos=INIT_STATE.pos,
    rot=INIT_STATE.rot,
    joint_pos={**INIT_STATE.joint_pos, **_MJCF_ANKLE_INIT_STATE[mjcf]},
    joint_vel=INIT_STATE.joint_vel,
  )

  joint_action_scale, joint_actuator_names = _build_action_scales(
    articulation, TransmissionType.JOINT
  )
  tendon_scale, _ = _build_action_scales(articulation, TransmissionType.TENDON)

  tendon_names = (
    (HIP_Z_TENDON_NAMES if hip_z == "tendon" else ())
    + (HIP_XY_TENDON_NAMES if hip_xy == "tendon" else ())
    + (ANKLE_TENDON_NAMES if ankle == "tendon" else ())
  )
  offsets = (
    _compute_tendon_lengths_at_init_state(
      get_kangaroo_full_spec(
        hip_z=hip_z,
        hip_xy=hip_xy,
        leg_length=leg_length,
        femur_closure=femur_closure,
        ankle=ankle,
        mjcf=mjcf,
      ),
      tendon_names,
      init_state.joint_pos,
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
    femur_closure=femur_closure,
    ankle=ankle,
    mjcf=mjcf,
    articulation=articulation,
    init_state=init_state,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
    hip_z_tendon_action=(
      _tendon_action(HIP_Z_TENDON_NAMES, "hip_z") if hip_z == "tendon" else None
    ),
    hip_xy_tendon_action=(
      _tendon_action(HIP_XY_TENDON_NAMES, "hip_xy") if hip_xy == "tendon" else None
    ),
    ankle_tendon_action=(
      _tendon_action(ANKLE_TENDON_NAMES, "ankle") if ankle == "tendon" else None
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
  femur_closure: FemurClosure = "prismatic",
  ankle: AnkleActuation = "joint",
  mjcf: MjcfVariant = "tendons",
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the full KANGAROO model.

  Args:
    hip_z: Drive hip yaw through its spatial tendon, or through the plain
      leg_.*_1_joint revolute motor.
    hip_xy: Drive hip pitch/roll through the parallel tendon pair, or through
      the plain leg_.*_2_joint / leg_.*_3_joint revolute motors.
    leg_length: Drive leg length through the leg_.*_length_actuator screw and
      its knee rod equality tendon, through that same screw but commanded in
      leg_.*_length_joint space via the transmission LUT (the two "semi_serial"
      values, differing in whether the PD closes on the joint or on the screw),
      or directly through leg_.*_length_joint.
    femur_closure: Fold the femur through the real four-bar -- the
      (left|right)_hip_xy_link and (left|right)_femur_rod equality tendons,
      with leg_.*_length_joint and its <connect> deleted -- or through the
      straight-line leg_.*_length_joint slider pinned to the knee by
      leg_.*_length_connect, with the four-bar tendons deleted and
      (left|right)_femur_triangle's joint deleted (welded at qpos 0).
    ankle: Swing the ankle from the (left|right)_butterfly_(l|r) joints, as the
      hardware does, or servo leg_.*_4_joint / leg_.*_5_joint directly with the
      butterfly chain frozen (femur rods and the decoupler gearing deleted,
      every butterfly joint deleted/welded at qpos 0).
    mjcf: Which MJCF to compile the variant from -- "tendons"
      (kangaroo_full_tendons.xml) or "tendons_over_constrained"
      (kangaroo_full_tendons_over_constarined.xml, same geometry and
      actuation axes, with extra closed-loop constraints layered on top).
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_model(
    hip_z=hip_z,
    hip_xy=hip_xy,
    leg_length=leg_length,
    femur_closure=femur_closure,
    ankle=ankle,
    mjcf=mjcf,
  )

  # Go through Entity rather than compiling make_spec() directly:
  # kangaroo_full_tendons.xml declares no <actuator> elements at all, so the
  # raw spec compiles to nu=0.
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

  print(
    f"hip_z={hip_z} hip_xy={hip_xy} leg_length={leg_length} "
    f"femur_closure={femur_closure} ankle={ankle} mjcf={mjcf}"
  )
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
    ("ankle", model_cfg.ankle_tendon_action),
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
