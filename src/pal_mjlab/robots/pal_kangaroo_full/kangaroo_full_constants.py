"""Pal Robotics KANGAROO FULL constants.

There is exactly one MJCF for the full model,
``xmls/kangaroo_full_tendons.xml``, and it carries *every* mechanism the robot
can have:

* the ``(left|right)_hip_z_slider`` spatial tendon (hip yaw screw),
* the four ``(left|right)_hip_xy_(l|r)_slider`` spatial tendons (hip pitch/roll
  parallel pair),
* the four ``(left|right)_ankle_(l|r)_slider`` spatial tendons (the ankle
  screws' virtual-motor chords, decoupler to butterfly),
* the ``leg_(left|right)_length_actuator`` prismatic screw, closed onto the
  femur by the ``(left|right)_knee_rods`` equality tendon.

Variants are produced by *editing the spec*: a variant either keeps the
mechanisms and actuates them, or deletes them and actuates the plain revolute
/ prismatic joints underneath. That keeps a single geometry source of truth --
previously each combination was a hand-maintained copy of the same XML, which
drifted (rod lengths, ``solref``, inertias) between copies.

One axis, ``transmission``, says how the whole robot's legs are driven:

===============  ==============================================================
``transmission``  actuation
===============  ==============================================================
``"joint"``       a PD on the simple ``pal_kangaroo`` model's joints:
                  ``leg_.*_(1|2|3|4|5)_joint`` and the leg length, i.e.
                  ``leg_.*_length_joint`` (prismatic femur closure) or the knee
                  (linkage closure). The tendons and the knee screw are deleted
                  and the butterfly chain frozen: nothing is left to drive them.
``"actuator"``    a PD on the actuators: the hip yaw, hip pitch/roll and ankle
                  tendons and the ``leg_.*_length_actuator`` screw.
``"lut"``         a PD on the simple model's joints, as in ``"joint"``, whose
                  torques are pushed through the mechanisms' Jacobians -- read
                  from the spline maps in ``lut_transmission/`` -- onto
                  ``<motor>``s on the same elements ``"actuator"`` drives
                  (``pal_kangaroo_full.lut_actuator``; one instance for all
                  twelve leg joints and both legs).
===============  ==============================================================

``femur_closure`` says which of two redundant descriptions of the femur the
compiled model keeps. ``"prismatic"`` keeps the straight-line stand-in: the
``leg_.*_length_joint`` slider pinned to the knee by the
``leg_.*_length_connect`` ``<connect>``, with the ``(left|right)_hip_xy_link``
and ``(left|right)_femur_rod`` tendons deleted and the now-idle
``(left|right)_femur_triangle`` crank's joint deleted (welding it to the femur
at qpos 0). ``"linkage"`` keeps the real four-bar those two tendons form and
deletes the slider and its ``<connect>`` instead, so the knee is the only
leg-length DOF: the ``"joint"`` transmission then servos the knee, and the
``"lut"`` one servos it in the leg-length metres ``leg_length_map.npz`` maps
it onto, the same coordinate the observations present as the missing joint.

``mjcf`` picks the XML (nominal or over-constrained) and ``lower_body``
deletes both arms -- everything from ``arm_(left|right)_base_link`` down --
and servos only the waist. The joint set the tasks observe is the simple
``pal_kangaroo`` model's in every case -- see
:data:`REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import mujoco
import torch
from mjlab.actuator import ActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.string import resolve_expr
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DAMPING_RATIO,
  FULL_COLLISION,
  KANGAROO_PELVIS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  NATURAL_FREQ,
  _calc_leg_params,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_actuator import (
  LegLengthServo,
  LutTransmissionActuatorCfg,
  ScrewElement,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import TransmissionMaps

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

# SIMPLE_MODEL_JOINT_ORDER with the arm joints dropped, for the lower_body=True
# variants: those have no arm_* joints at all (the whole arm subtree is
# deleted -- see _ARM_BASE_LINK_NAMES / _delete_subtrees), so the policy's
# joint vector layout drops them too rather than reading zeros for a limb that
# does not exist.
LOWER_BODY_JOINT_ORDER: tuple[str, ...] = tuple(
  name for name in SIMPLE_MODEL_JOINT_ORDER if not name.startswith("arm_")
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

# The spline-interpolated transmission maps of the leg mechanisms
# (hip_z_map.npz, leg_length_map.npz, hip_xy_jacobian_map.npz,
# ankle_xy_jacobian_map.npz), read through pal_kangaroo_full.lut_maps by the
# "lut" transmission of pal_kangaroo_full.lut_actuator -- on this MJCF and on
# pal_kangaroo_full_full's. Each map is built from the right leg and serves
# both legs.
LUT_TRANSMISSION_DIR = KANGAROO_FULL_PATH.parent / "lut_transmission"


@lru_cache(maxsize=None)
def load_transmission_maps() -> TransmissionMaps:
  """The four maps, read once per process and shared by every "lut"
  variant of both full models."""
  return TransmissionMaps.load(LUT_TRANSMISSION_DIR)


# leg_.*_length_joint + this = the femur-ankle distance leg_length_map.npz
# (and knee_distance_map.csv) are in, for the "lut" transmission on the
# "prismatic" femur closure. Exact and constant: the slide axis passes
# through the femur joint anchor, so the distance is the joint value plus the
# site's offset along the axis at qpos 0 -- measured at INIT_STATE, where the
# joint reads -0.125030 and the connect site sits 0.599916 m from the anchor.
# tests/test_kangaroo_lut_transmission.py checks both facts against the
# compiled model.
LEG_LENGTH_JOINT_TO_DISTANCE = 0.724946


# Swept over this same MJCF: where leg_.*_length_connect_b sits relative to the
# leg_.*_femur_joint anchor as the knee folds. Rows are (knee_rad, knee_deg,
# displacement_x_m, displacement_z_m, distance_m), world frame. Not a
# transmission: it is what the observation, reward and metric terms
# reconstruct the missing leg_.*_length_joint from (its distance_m agrees
# with leg_length_map.npz's distance to 1e-5 m).
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
  KNEE_DISTANCE_MAP_CSV,
):
  assert _path.exists(), f"Missing: {_path}"
assert TransmissionMaps.available(LUT_TRANSMISSION_DIR), (
  f"Missing transmission maps in {LUT_TRANSMISSION_DIR}"
)

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

# Each arm hangs off torso_link as one self-contained subtree -- no tendon,
# equality, or sensor elsewhere in the MJCF reaches into it -- so deleting
# these two bodies removes the whole arm (links, joints, geoms) in one call.
_ARM_BASE_LINK_NAMES = ("arm_left_base_link", "arm_right_base_link")

# Rest lengths of the rigid rods each equality tendon stands in for -- hip_xy
# link, knee rod, femur rod, ankle tibia bar -- keyed by physical rod length,
# not the tendon's length at qpos0. Applied as a reset event (see
# mdp.dr.tendon.enforce_tendon_lengths) since MuJoCo has no way to specify an
# equality tendon's rest length directly in the XML.
KANGAROO_TENDON_LENGTHS: dict[str, float] = {
  r"(left|right)_hip_xy_link": 0.09,
  r"(left|right)_knee_rods": 0.215,
  r"(left|right)_femur_rod": 0.40427,
  r"(left|right)_ankle_(femur|tibia)_bar_(l|r)": 0.38,
}

Transmission = Literal["joint", "actuator", "lut"]
FemurClosure = Literal["linkage", "prismatic"]
# Not an actuation choice like the axes above -- whether the arms exist at
# all -- but still a `Literal` alongside them rather than a bare `bool` so it
# reads the same way at every call site and export.
LowerBody = Literal[True, False]

# The transmission map's name of each mechanism's actuator -> the tendon that
# stands in for that screw in this MJCF (pinned to the same bodies; the "l"/"r"
# in a tendon's name is the slider body's, not the leg's, on both sides). The
# hip tendons read the screws' Jacobians exactly; the ankle tendons are the
# decoupler-to-butterfly chords, which the maps only approximate -- kept as
# is, unvalidated, by decision.
TENDON_OF_MAP_ACTUATOR: dict[str, str] = {
  "leg_right_1_actuator": "right_hip_z_slider",
  "leg_right_2_actuator": "right_hip_xy_l_slider",
  "leg_right_3_actuator": "right_hip_xy_r_slider",
  "leg_right_4_actuator": "right_ankle_l_slider",
  "leg_right_5_actuator": "right_ankle_r_slider",
}


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


def _delete_subtrees(spec: mujoco.MjSpec, body_names: tuple[str, ...]) -> None:
  """Delete each named body and everything hanging off it.

  ``spec.delete`` on a body removes the whole subtree under it -- descendant
  bodies, their joints, geoms and sites included -- in one call, so this is
  the right tool for dropping an entire limb rather than enumerating its
  joints/tendons the way the per-mechanism ``_delete_*`` helpers above do.
  A same-named site elsewhere in the model (e.g. a mounting-point marker) is
  untouched: bodies and sites are separate MuJoCo namespaces.
  """
  for name in body_names:
    body = spec.body(name)
    if body is None:
      raise ValueError(f"MJCF has no body '{name}' to delete")
    spec.delete(body)


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
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
) -> mujoco.MjSpec:
  """Load the MJCF and strip the mechanisms this variant doesn't use."""
  spec = mujoco.MjSpec.from_file(str(_MJCF_XML_PATHS[mjcf]))
  _add_collision_capsules(spec)
  if lower_body:
    # Whole-arm removal, not one of the leg mechanism axes below: delete
    # before those run so nothing downstream has to know the arms are gone.
    _delete_subtrees(spec, _ARM_BASE_LINK_NAMES)
  if femur_closure == "prismatic":
    _delete_tendons(spec, _FEMUR_LINKAGE_TENDON_NAMES)
    _delete_joints(spec, _FEMUR_TRIANGLE_JOINT_NAMES)
  else:
    _delete_equalities(spec, _LEG_LENGTH_CONNECT_EQ_NAMES)
    _delete_joints(spec, _LEG_LENGTH_JOINT_NAMES)
  if transmission == "joint":
    # Every simple-model joint is servoed directly, so nothing commands the
    # screws: the hip tendons go; the knee screw's slider body stays (its
    # mass and inertia are still on the femur) but its joint is deleted
    # (welding it to the femur at qpos 0) along with the knee rod that
    # closed it onto the femur, so nothing drives it and nothing hangs off
    # it; and the ankle chain is frozen rather than left to swing -- the
    # decoupler's gearing to the knee goes, every butterfly joint is deleted
    # (welding butterfly_l/r to the decoupler, and the decoupler to the
    # femur, each at qpos 0), and the ankle tendons and tibia bars they
    # swung go with them. *_femur_rod stays out of this under "linkage": it
    # is one of the two tendons actually closing the femur four-bar (with
    # *_hip_xy_link), and deleting it leaves leg_.*_femur_joint with nothing
    # holding it. "prismatic" already deleted it above, as part of the
    # four-bar it replaces.
    _delete_tendons(spec, HIP_Z_TENDON_NAMES)
    _delete_tendons(spec, HIP_XY_TENDON_NAMES)
    _delete_tendons(spec, _KNEE_ROD_TENDON_NAMES)
    _delete_joints(spec, _LEG_LENGTH_ACTUATOR_JOINT_NAMES)
    _delete_tendons(spec, ANKLE_TENDON_NAMES)
    _delete_equalities(spec, _BUTTERFLY_DECOUPLER_EQ_NAMES)
    _delete_joints(spec, _BUTTERFLY_JOINT_NAMES)
    _delete_tendons(spec, _ANKLE_TIBIA_BAR_TENDON_NAMES)
    if femur_closure != "linkage":
      _delete_tendons(spec, _FEMUR_ROD_TENDON_NAMES)
  return spec


##
# Actuator configs.
#
# Three tables, shared with pal_kangaroo_full_full, describe the legs: the
# simple model's PD on each joint (the "joint" transmission, and the joint
# side of the "lut" one), the PD on each screw (the "actuator" transmission)
# and the screws' <motor> limits (the "lut" transmission's elements, the
# "actuator" variants' effort limits and armature).
##


def _calc_linear_leg_params(
  stiffness: float,
  effort: float,
  armature: float,
) -> dict:
  """Calculate leg actuator parameters."""
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
    "viscous_damping": 0.01,
  }


# The simple pal_kangaroo model's (stiffness, effort limit) on each leg joint,
# keyed by the target expression the actuator configs and the action scale
# use for it. N m/rad and N m.
LEG_JOINT_PD: dict[str, tuple[float, float]] = {
  r"leg_(left|right)_1_joint": (100.0, 80.0),
  r"leg_(left|right)_2_joint": (100.0, 230.0),
  r"leg_(left|right)_3_joint": (100.0, 139.0),
  r"leg_(left|right)_4_joint": (30.0, 140.0),
  r"leg_(left|right)_5_joint": (30.0, 82.0),
}
LEG_JOINT_PD_ARMATURE = 0.01
# The leg length in the simple model's metres (N/m, N): the "joint"
# transmission's PD on leg_.*_length_joint, and the "lut" one's PD in the
# mapped distance. The same gains on purpose: a unit action then commands the
# same leg-length change *and* the same force, so a checkpoint trained on
# "joint" plays on "lut" (the old semi_serial variant's softer 900 N/m kept
# the force per unit action but not the stiffness, and a "joint" checkpoint
# fell over on it within a few seconds).
LEG_LENGTH_JOINT_PD = (1600.0, 1100.0)
LEG_LENGTH_LUT_PD = LEG_LENGTH_JOINT_PD
# The servo joint of the leg length, by which joint the MJCF has for it.
LEG_LENGTH_JOINT_EXPR = r"leg_(left|right)_length_joint"
KNEE_JOINT_EXPR = r"leg_(left|right)_knee_joint"

# Each screw's <position> stiffness (N/m), force limit (N) and armature (kg),
# keyed by the map actuator it drives. Hip pitch/roll and the ankle are pairs
# with identical screws.
LEG_SCREW_PD: dict[str, tuple[float, float, float]] = {
  # saturation_effort=4334.0, velocity_limit=0.314
  # Sum of linear inertia of the screw and inertia of nut plus motor rotor
  # armature=0.155 + 0.00004559 * (2.0 * math.pi / 0.005) ** 2,
  "leg_right_1_actuator": (2500.0, 2000.0, 0.1),
  # armature=0.178 + 0.00004559 * (2.0 * math.pi / 0.005) ** 2,
  "leg_right_2_actuator": (750.0, 2000.0, 0.1),
  "leg_right_3_actuator": (750.0, 2000.0, 0.1),
  # armature=0.155 + 0.00004559 * (2.0 * math.pi / 0.005) ** 2,
  "leg_right_4_actuator": (1500.0, 2000.0, 0.1),
  "leg_right_5_actuator": (1500.0, 2000.0, 0.1),
  # saturation_effort=10443.0, velocity_limit=0.288
  # Assuming nut is a cylinder of mass 0.26 Kg, hollow shaft of 10 mm and
  # external diameter of 40 mm. Inertia of a screw is still captured by the
  # model; second value is inertia of motor rotor. Everything multiplied by
  # pitch to make it a linear inertia:
  # armature=(0.000221 + 0.000098) * (2.0 * math.pi / 0.01) ** 2,
  "leg_right_length_actuator": (6000.0, 5000.0, 1.0),
}


def knee_pd_params(knee: float) -> dict:
  """The leg-length PD :data:`LEG_LENGTH_JOINT_PD` re-expressed on the knee,
  for the MJCFs whose only leg-length DOF is the knee: the same stiffness
  and effort in the leg-length metres, pulled back through the slope
  ``dd/dknee`` of ``leg_length_map.npz`` at the resting knee angle
  (``kp_knee = kp dd/dknee^2``, ``tau_max = F_max |dd/dknee|``), so a unit
  action commands the same leg-length change as on the prismatic joint."""
  leg = load_transmission_maps().leg_length
  s = leg.slider_of_knee(torch.tensor([knee], dtype=leg.dtype))
  slope = abs(float(1.0 / (leg.jacobian(s) * leg.dknee_dslider(s))))
  stiffness, effort = LEG_LENGTH_JOINT_PD
  return _calc_leg_params(
    stiffness * slope * slope, effort * slope, LEG_JOINT_PD_ARMATURE, None, None
  )


def joint_pd_actuators(
  leg_length_servo: LegLengthServo, knee: float
) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """The "joint" transmission: the simple model's PD on every leg joint,
  ordered like its actuators (hip yaw, hip pitch/roll, ankle, leg length)."""
  leg_length = (
    BuiltinPositionActuatorCfg(
      target_names_expr=(KNEE_JOINT_EXPR,), **knee_pd_params(knee)
    )
    if leg_length_servo == "knee"
    else BuiltinPositionActuatorCfg(
      target_names_expr=(LEG_LENGTH_JOINT_EXPR,),
      **_calc_leg_params(*LEG_LENGTH_JOINT_PD, LEG_JOINT_PD_ARMATURE, None, None),
    )
  )
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(expr,),
      **_calc_leg_params(stiffness, effort, LEG_JOINT_PD_ARMATURE, None, None),
    )
    for expr, (stiffness, effort) in LEG_JOINT_PD.items()
  ) + (leg_length,)


def screw_pd_params(map_actuator: str) -> dict:
  """The "actuator" transmission's <position> parameters of one screw."""
  stiffness, effort, armature = LEG_SCREW_PD[map_actuator]
  return _calc_linear_leg_params(stiffness=stiffness, effort=effort, armature=armature)


def screw_element(
  map_actuator: str,
  name: str | None = None,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> ScrewElement:
  """The "lut" transmission's <motor> on one screw: the "actuator"
  transmission's force limit, armature and viscous damping (the kp/kv belong
  to the <position> element that isn't there), on the element ``name``
  (default: the map actuator's own name, the connect-linkage model's slider)."""
  params = screw_pd_params(map_actuator)
  viscous_damping = params["viscous_damping"]
  frictionloss = None
  if map_actuator == "leg_right_length_actuator":
    armature = 50.72
    viscous_damping = 589.4603
    frictionloss = 11.7333
  elif map_actuator in ("leg_right_1_actuator", "leg_right_2_actuator", "leg_right_3_actuator"):
    armature = 72.07
  else:
    armature = 72.09
  return ScrewElement(
    name=map_actuator if name is None else name,
    transmission_type=transmission_type,
    effort_limit=params["effort_limit"],
    armature=armature,
    viscous_damping=viscous_damping,
    frictionloss=frictionloss,
  )


def lut_actuator(
  screws: dict[str, ScrewElement],
  leg_length_servo: LegLengthServo,
  length_joint_to_distance: float = 0.0,
) -> LutTransmissionActuatorCfg:
  """The "lut" transmission: :data:`LEG_JOINT_PD` on the hip and ankle
  joints and :data:`LEG_LENGTH_LUT_PD` on the leg length, on ``screws``."""
  leg_length_expr = (
    KNEE_JOINT_EXPR if leg_length_servo == "knee" else LEG_LENGTH_JOINT_EXPR
  )
  gains = dict(LEG_JOINT_PD)
  gains[leg_length_expr] = LEG_LENGTH_LUT_PD
  pd = {expr: _calc_leg_params(k, f, 0.0, None, None) for expr, (k, f) in gains.items()}
  return LutTransmissionActuatorCfg(
    target_names_expr=tuple(gains),
    maps=load_transmission_maps(),
    screws=screws,
    joint_stiffness={expr: p["stiffness"] for expr, p in pd.items()},
    joint_damping={expr: p["damping"] for expr, p in pd.items()},
    joint_effort_limit={expr: p["effort_limit"] for expr, p in pd.items()},
    leg_length_servo=leg_length_servo,
    length_joint_to_distance=length_joint_to_distance,
  )


def _tendon_screws() -> dict[str, ScrewElement]:
  """This MJCF's elements for the "lut" transmission: the hip and ankle
  tendons, and the knee screw's slider joint."""
  screws = {
    map_actuator: screw_element(map_actuator, tendon, TransmissionType.TENDON)
    for map_actuator, tendon in TENDON_OF_MAP_ACTUATOR.items()
  }
  screws["leg_right_length_actuator"] = screw_element("leg_right_length_actuator")
  return screws


def _tendon_pd_actuator(
  target_names_expr: tuple[str, ...], map_actuator: str, armature: float
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    transmission_type=TransmissionType.TENDON,
    target_names_expr=target_names_expr,
    **{**screw_pd_params(map_actuator), "armature": armature},
  )


# The "actuator" transmission: tendon-space PD on the hip yaw, hip pitch/roll
# and ankle screws (their tendons), and the knee screw's prismatic joint.
_ACTUATOR_TRANSMISSION_ACTUATORS: tuple[ActuatorCfg, ...] = (
  _tendon_pd_actuator((r"(left|right)_hip_z_slider$",), "leg_right_1_actuator", 72.07),
  _tendon_pd_actuator(
    (r"(left|right)_hip_xy_(l|r)_slider$",), "leg_right_2_actuator", 72.07
  ),
  _tendon_pd_actuator((r"(left|right)_ankle_(l|r)_slider$",), "leg_right_4_actuator", 72.09),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    **{
      **screw_pd_params("leg_right_length_actuator"),
      "armature": 50.72,
      "damping": 589.4603,
      "frictionloss": 11.7333,
    },
  ),
)


def _leg_actuators(
  transmission: Transmission, femur_closure: FemurClosure
) -> tuple[ActuatorCfg, ...]:
  """The leg actuators of one variant, ordered like the simple pal_kangaroo
  model's (hip yaw, hip pitch/roll, ankle, leg length) so the action vector
  reads the same way in every variant."""
  leg_length_servo: LegLengthServo = (
    "knee" if femur_closure == "linkage" else "length_joint"
  )
  if transmission == "joint":
    return joint_pd_actuators(
      leg_length_servo, INIT_STATE.joint_pos["leg_.*_knee_joint"]
    )
  if transmission == "actuator":
    return _ACTUATOR_TRANSMISSION_ACTUATORS
  if transmission == "lut":
    return (
      lut_actuator(_tendon_screws(), leg_length_servo, LEG_LENGTH_JOINT_TO_DISTANCE),
    )
  raise ValueError(f"unknown transmission {transmission!r}")


_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
# lower_body=True has no arm_* joints for KANGAROO_S_PLUS_ACTUATOR_CFG /
# KANGAROO_S_MINUS_ACTUATOR_CFG to target -- the latter's expression matches
# arm joints only, so it would find zero and raise. Servo the waist alone,
# exactly as pal_kangaroo's own KANGAROO_LOWER_BODY_ACTUATORS does.
_LOWER_BODY_UPPER_BODY_ACTUATORS = (KANGAROO_PELVIS_ACTUATOR_CFG,)

##
# Initial state.
##

# leg_.*_length_actuator is absent from the transmission="joint" variants
# and leg_.*_length_joint from the "linkage" ones; a pattern that matches no
# joint is simply ignored by resolve_expr, so one init state covers every
# :data:`MjcfVariant` and every transmission / femur_closure variant.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.91),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "leg_left_1_joint": -0.012074,
    "leg_right_1_joint": 0.012072,
    "leg_.*_2_joint": 0.052192,
    "leg_left_3_joint": -0.039992,
    "leg_right_3_joint": 0.040002,
    "leg_.*_length_joint": -0.125030,
    "leg_.*_4_joint": -0.352785,
    "leg_.*_5_joint": 0.000001,
    "leg_.*_femur_joint": -0.296368,
    "leg_.*_knee_joint": 0.597794,
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
  spec: mujoco.MjSpec, tendon_names: tuple[str, ...], joint_pos: dict[str, float]
) -> dict[str, float]:
  """Tendon lengths with the model posed at ``joint_pos``.

  This is the TENDON-transmission analogue of what ``use_default_offset=True``
  gives JOINT actuators for free: ``JointPositionAction`` reads the joint's own
  value at the default pose so a raw action of 0 holds that pose exactly.
  ``TendonLengthActionCfg`` has no ``use_default_offset``, so the equivalent
  offset is solved for here instead of being hand-maintained -- a hardcoded
  constant silently drifts out of sync with the model (it previously did, by
  ~0.3-0.5 mm). ``joint_pos`` is the caller's init state (see
  :data:`INIT_STATE`) rather than a module-level default read here, so the
  offset always follows whatever pose the caller actually spawns at.
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

ARM_ACTION_SCALE_FACTOR = 0.25
"""Fraction of an upper-body actuator's effort limit a unit action commands.

Applies to every actuator that isn't part of a leg mechanism -- the arms and
the pelvis alike -- and matches the simple pal_kangaroo model.
"""
LEG_ACTION_SCALE_FACTOR = 0.25
"""Fraction of a leg actuator's effort limit a unit action commands.

Applies to hip yaw, hip pitch/roll, ankle and leg length, whether the mechanism
is driven by a joint motor or by its tendon.
"""


def _build_action_scales(
  actuators: tuple[ActuatorCfg, ...],
  transmission_type: TransmissionType,
  action_scale_factor: float,
) -> tuple[dict[str, float], tuple[str, ...]]:
  """Action scale dict and target names for one transmission type.

  The scale is ``action_scale_factor`` times each actuator's torque-to-stiffness
  ratio, i.e. the position offset that fraction of full effort corresponds to.
  For the "lut" transmission that is the servo joint's (``joint_effort_limit``
  over ``joint_stiffness``), since that is the coordinate the action commands;
  otherwise it is the element's own, as for any other actuator.
  """
  scales: dict[str, float] = {}
  names: list[str] = []
  for actuator in actuators:
    if actuator.transmission_type != transmission_type:
      continue
    if isinstance(actuator, LutTransmissionActuatorCfg):
      effort_limit, stiffness = actuator.joint_effort_limit, actuator.joint_stiffness
    else:
      effort_limit, stiffness = actuator.effort_limit, actuator.stiffness
    for name in actuator.target_names_expr:
      efforts = effort_limit if isinstance(effort_limit, dict) else {name: effort_limit}
      stiffnesses = stiffness if isinstance(stiffness, dict) else {name: stiffness}
      if name in efforts and stiffnesses.get(name):
        scales[name] = action_scale_factor * efforts[name] / stiffnesses[name]
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
class LegLengthAction:
  """What the mapped leg-length action term needs when the "lut" transmission
  servos the knee: the knee joints it targets and the scale, in metres, a
  unit action commands. The offset is the map's value at the default knee
  angle, which the term reads for itself."""

  actuator_names: tuple[str, ...]
  scale: dict[str, float]


@dataclass(frozen=True)
class KangarooFullModel:
  """One actuation variant of the full KANGAROO model."""

  transmission: Transmission
  femur_closure: FemurClosure
  mjcf: MjcfVariant
  lower_body: LowerBody
  arm_action_scale_factor: float
  leg_action_scale_factor: float

  articulation: EntityArticulationInfoCfg
  init_state: EntityCfg.InitialStateCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]
  """Targets of the catch-all joint term: every JOINT-transmission actuator
  except the knee of a "lut" transmission (its own term, in metres)."""
  hip_z_tendon_action: TendonAction | None
  hip_xy_tendon_action: TendonAction | None
  ankle_tendon_action: TendonAction | None
  leg_length_action: LegLengthAction | None

  @property
  def has_knee_rod_tendons(self) -> bool:
    """Whether the ``*_knee_rods`` equality tendons exist in this variant.

    They close the knee screw onto the femur; the "joint" transmission
    deletes the screw and them with it.
    """
    return self.transmission != "joint"

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

    The "actuator" and "lut" transmissions keep the butterflies (driven
    through their virtual-motor tendons), so both need that gearing;
    servoing the ankle joints directly deletes it and locks the butterflies
    instead.
    """
    return self.transmission != "joint"

  @property
  def has_ankle_tibia_bar_tendons(self) -> bool:
    """Whether the ``*_ankle_tibia_bar_(l|r)`` equality tendons exist.

    Same axis as :attr:`has_butterfly_decoupler_coupling`: these are the bars
    the butterflies swing the ankle through, so a joint-actuated ankle deletes
    them along with the butterflies that would have driven them.
    """
    return self.transmission != "joint"

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
      transmission=self.transmission,
      femur_closure=self.femur_closure,
      mjcf=self.mjcf,
      lower_body=self.lower_body,
    )

  def make_robot_cfg(self) -> EntityCfg:
    return EntityCfg(
      init_state=self.init_state,
      collisions=(FULL_COLLISION,),
      spec_fn=self.make_spec,
      articulation=self.articulation,
    )


def split_joint_action_scales(
  leg_actuators: tuple[ActuatorCfg, ...],
  upper_body_actuators: tuple[ActuatorCfg, ...],
  leg_action_scale_factor: float,
  arm_action_scale_factor: float,
  carve_out: frozenset[str] = frozenset(),
) -> tuple[dict[str, float], tuple[str, ...], dict[str, float], tuple[str, ...]]:
  """The JOINT action scales of a variant, split into the catch-all joint
  term and the targets ``carve_out`` names (given their own terms).

  Legs and upper body get their own factor, so the term is built in two
  halves and concatenated in the same leg-then-upper-body order. Returns
  ``(joint_scale, joint_names, carved_scale, carved_names)``.
  """
  joint_scale: dict[str, float] = {}
  joint_names: tuple[str, ...] = ()
  carved_scale: dict[str, float] = {}
  carved_names: tuple[str, ...] = ()
  for actuators, factor in (
    (leg_actuators, leg_action_scale_factor),
    (upper_body_actuators, arm_action_scale_factor),
  ):
    scales, names = _build_action_scales(actuators, TransmissionType.JOINT, factor)
    for name in names:
      if name in carve_out:
        carved_scale[name] = scales[name]
        carved_names += (name,)
      else:
        joint_scale[name] = scales[name]
        joint_names += (name,)
  return joint_scale, joint_names, carved_scale, carved_names


def simple_model_action_names(
  joint_order: tuple[str, ...],
  joint_actuator_names: tuple[str, ...],
  leg_length_action: LegLengthAction,
) -> tuple[str, ...]:
  """The catch-all joint term's targets and the mapped leg-length term's
  knees merged into one explicit, ordered target list.

  The order is ``joint_order`` -- the simple model's joint order, the one the
  observations already use -- with each knee standing in the slot of the
  leg-length joint it serves (LEG_LENGTH_FROM_KNEE_JOINTS). That is exactly
  where mjlab's natural-order JOINT term puts ``leg_.*_length_joint`` on the
  MJCFs that have it (and where the "joint" transmission puts the knee on
  the ones that don't), so a "lut" policy's action vector is laid out like a
  "joint" one's and a checkpoint transfers between them slot for slot. With
  two terms instead (joints, then knees) the knees would trail the vector
  and every slot after the left hip would be permuted.
  """
  knee_of_length = dict(LEG_LENGTH_FROM_KNEE_JOINTS)
  exprs = tuple(joint_actuator_names)
  knee_exprs = tuple(leg_length_action.actuator_names)
  names: tuple[str, ...] = ()
  matched: set[str] = set()
  for name in joint_order:
    for expr in exprs:
      if re.fullmatch(expr, name):
        names += (name,)
        matched.add(expr)
        break
    else:
      knee = knee_of_length.get(name)
      if knee is None:
        continue
      for expr in knee_exprs:
        if re.fullmatch(expr, knee):
          names += (knee,)
          matched.add(expr)
          break
  unmatched = sorted(set(exprs + knee_exprs) - matched)
  if unmatched:
    raise ValueError(
      f"action targets {unmatched} name no joint of joint_order {joint_order}"
    )
  return names


def lut_leg_length_action(
  leg_actuators: tuple[ActuatorCfg, ...],
) -> frozenset[str]:
  """The catch-all term's keys to carve out into the mapped leg-length term:
  the "lut" transmission's knee servo, whose targets are in metres."""
  for actuator in leg_actuators:
    if (
      isinstance(actuator, LutTransmissionActuatorCfg)
      and actuator.leg_length_servo == "knee"
    ):
      return frozenset((actuator.leg_length_servo_expr,))
  return frozenset()


@lru_cache(maxsize=None)
def get_kangaroo_full_model(
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> KangarooFullModel:
  """Assemble the actuators, action scales and tendon offsets for one variant.

  The two ``*_action_scale_factor`` values set what fraction of an actuator's
  effort limit a unit action commands, expressed as a position offset through
  its stiffness: ``leg_action_scale_factor`` for every leg mechanism (joint and
  tendon terms alike), ``arm_action_scale_factor`` for the upper body. For the
  "lut" transmission that is the servo joint's effort and stiffness, so a unit
  action means the same thing as on the "joint" one.

  The action vector is one catch-all joint term plus, for the "actuator"
  transmission, one order-preserved tendon term per mechanism (see
  HIP_Z_TENDON_NAMES and friends) and, for the "lut" transmission on the
  "linkage" closure, the knee's leg-length term in metres.

  Cached because the tendon offsets require compiling the model, and every task
  registration asks for the same handful of variants.
  """
  leg_actuators = _leg_actuators(transmission, femur_closure)
  upper_body_actuators = (
    _LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS
  )
  articulation = EntityArticulationInfoCfg(
    actuators=leg_actuators + upper_body_actuators,
    soft_joint_pos_limit_factor=0.99,
  )

  joint_action_scale, joint_actuator_names, leg_length_scale, leg_length_names = (
    split_joint_action_scales(
      leg_actuators,
      upper_body_actuators,
      leg_action_scale_factor,
      arm_action_scale_factor,
      carve_out=lut_leg_length_action(leg_actuators),
    )
  )
  # Only the leg mechanisms are ever tendon driven.
  tendon_scale, _ = _build_action_scales(
    leg_actuators, TransmissionType.TENDON, leg_action_scale_factor
  )

  tendon_names = (
    HIP_Z_TENDON_NAMES + HIP_XY_TENDON_NAMES + ANKLE_TENDON_NAMES
    if transmission == "actuator"
    else ()
  )
  offsets = (
    _compute_tendon_lengths_at_init_state(
      get_kangaroo_full_spec(
        transmission=transmission,
        femur_closure=femur_closure,
        mjcf=mjcf,
        lower_body=lower_body,
      ),
      tendon_names,
      INIT_STATE.joint_pos,
    )
    if tendon_names
    else {}
  )

  def _tendon_action(names: tuple[str, ...], key: str) -> TendonAction | None:
    if transmission != "actuator":
      return None
    # Each action term's scale/offset may only carry keys matching its own
    # targets: resolve_matching_names_values errors on a key that matches none.
    return TendonAction(
      actuator_names=tuple(f"{name}$" for name in names),
      scale={k: v for k, v in tendon_scale.items() if key in k},
      offset={name: offsets[name] for name in names},
    )

  return KangarooFullModel(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
    articulation=articulation,
    init_state=INIT_STATE,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
    hip_z_tendon_action=_tendon_action(HIP_Z_TENDON_NAMES, "hip_z"),
    hip_xy_tendon_action=_tendon_action(HIP_XY_TENDON_NAMES, "hip_xy"),
    ankle_tendon_action=_tendon_action(ANKLE_TENDON_NAMES, "ankle"),
    leg_length_action=(
      LegLengthAction(actuator_names=leg_length_names, scale=leg_length_scale)
      if leg_length_names
      else None
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


def print_actuators(model: mujoco.MjModel) -> None:
  """One line per actuator: its transmission and whether it is a <position>
  element (with kp) or a <motor> (with its force range)."""
  print(f"  actuators ({model.nu}):")
  for i in range(model.nu):
    actuator = model.actuator(i)
    kind = "tendon" if actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON else "joint"
    if actuator.biasprm[1] != 0.0:  # <position>: gainprm[0] is kp
      print(f"    {actuator.name:34s} ({kind})  kp={actuator.gainprm[0]:.1f}")
    else:  # <motor>: an external force within forcerange
      lo, hi = actuator.forcerange
      print(f"    {actuator.name:34s} ({kind})  motor, force in [{lo:.0f}, {hi:.0f}]")


def print_leg_length_action(action: LegLengthAction | None) -> None:
  if action is None:
    print("  leg length action: none (in the joint term, or on the screw)")
    return
  print(f"  leg length action targets ({len(action.actuator_names)}), metres:")
  for name in action.actuator_names:
    print(f"    {name}  scale={action.scale[name]:.4f}")


def main(
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the full KANGAROO model.

  Args:
    transmission: Drive the simple model's joints directly ("joint": PD on
      leg_.*_(1|2|3|4|5)_joint and leg_.*_length_joint or the knee, tendons
      deleted, butterfly chain frozen), the actuators directly ("actuator":
      PD on the hip and ankle tendons and the leg_.*_length_actuator screw),
      or the actuators commanded on the simple model's joints ("lut": the
      "joint" PD, pushed through the transmission maps onto <motor>s on the
      "actuator" elements).
    femur_closure: Fold the femur through the real four-bar -- the
      (left|right)_hip_xy_link and (left|right)_femur_rod equality tendons,
      with leg_.*_length_joint and its <connect> deleted, the knee being the
      leg-length servo -- or through the straight-line leg_.*_length_joint
      slider pinned to the knee by leg_.*_length_connect, with the four-bar
      tendons deleted and (left|right)_femur_triangle's joint deleted (welded
      at qpos 0).
    mjcf: Which MJCF to compile the variant from -- "tendons"
      (kangaroo_full_tendons.xml) or "tendons_over_constrained"
      (kangaroo_full_tendons_over_constarined.xml, same geometry and
      actuation axes, with extra closed-loop constraints layered on top).
    lower_body: Delete both arms (everything from arm_(left|right)_base_link
      down) and servo only the waist where the full model would otherwise
      also drive the arms.
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_model(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
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
    f"transmission={transmission} femur_closure={femur_closure} mjcf={mjcf} "
    f"lower_body={lower_body}"
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
  print_actuators(model)
  print(f"  joint action targets ({len(model_cfg.joint_actuator_names)}):")
  for name in model_cfg.joint_actuator_names:
    print(f"    {name}  scale={model_cfg.joint_action_scale[name]:.4f}")
  for label, tendon_action in (
    ("hip_z", model_cfg.hip_z_tendon_action),
    ("hip_xy", model_cfg.hip_xy_tendon_action),
    ("ankle", model_cfg.ankle_tendon_action),
  ):
    if tendon_action is None:
      print(f"  {label} tendon action: none (commanded on joints)")
      continue
    print(f"  {label} tendon action targets ({len(tendon_action.actuator_names)}):")
    for name, offset in tendon_action.offset.items():
      print(f"    {name}  offset={offset:.9f}")
  print_leg_length_action(model_cfg.leg_length_action)

  if launch_viewer:
    # Imported here, not at module scope, so importing these constants doesn't
    # drag in the viewer's GL dependencies.
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
