"""Pal Robotics KANGAROO FULL FULL constants.

Same physical robot as ``pal_kangaroo_full``, but built from a different
source MJCF: ``xmls/kangaroo_full.xml`` models every hip/ankle screw as a
real, closed-loop rigid-body linkage -- a rotating housing plus a prismatic
``leg_.*_(1|2|3|4|5)_actuator`` slider, pinned to the output link by a
``<connect>`` equality -- instead of ``pal_kangaroo_full``'s spatial-tendon
approximation of the same screw. There isn't a single ``<tendon>`` in this
file; every mechanism that ``pal_kangaroo_full`` models with a spatial tendon
is a real body/joint/``<connect>`` chain here.

Each closed loop has exactly one physical DOF, shared between two redundant
generalized coordinates: the slider (the real linear actuator) and the
revolute joint it drives (``leg_.*_1_joint`` for hip yaw, etc). Nothing needs
to be deleted from the spec to choose between them -- the actuator config
simply picks which of the two coordinates gets driven, and the ``<connect>``
equality's constraint solver keeps the other one consistent. That is the same
trick ``leg_length="semi_serial"`` already uses in ``pal_kangaroo_full`` (the
simple joint's PD law drives the real screw through a measured transmission
LUT), except here the "transmission" is an actual mechanism rather than a
hand-fit curve, so no LUT is needed and the mapping is exact.

Two axes ``pal_kangaroo_full`` has don't exist here:

* ``femur_closure``: this MJCF only has the ``"linkage"`` four-bar (femur
  triangle + femur rod, closed through the ``(left|right)_hip_xy_link`` /
  ``(left|right)_femur_rod`` equalities) -- there is no straight-line
  ``leg_.*_length_joint`` stand-in to fall back to. Consequently none of
  ``pal_kangaroo_full``'s ``leg_length`` values other than ``"actuator"``
  apply (they all target that missing joint); the ``leg_length`` axis here
  is the ``"actuator"`` / ``"transmission"`` pair described below.
* ``mjcf``: there is one XML here, not a base/over-constrained pair.

What's left are four real actuation choices. ``hip_z`` and ``hip_xy`` each
pick between driving the linear actuator screw directly (``"slider"`` --
``pal_kangaroo_full``'s tendon axis, replaced by the real actuator it stood
in for) or driving the output joint directly (``"joint"``). ``ankle`` keeps a
third option, ``"butterfly"``, which targets the real butterfly hinges
instead of either screw-adjacent coordinate. ``lower_body`` is unchanged from
``pal_kangaroo_full``: it deletes both arms and servos only the waist.

Every screw axis, and ``leg_length``, also has a ``"transmission"`` value:
the real screw is driven, as in ``"slider"``, but *commanded* on the simple
model's joint, as in ``"joint"`` -- a full PD (kp and kd) runs on the joint,
the torque it asks for is pushed onto the screw through the mechanism's
Jacobian, and the resulting force is handed to a ``<motor>`` on the screw.
That is ``pal_kangaroo_full``'s ``leg_length="semi_serial"`` law
generalised, with the transmission read from spline-interpolated maps of
the mechanism geometry (one file per mechanism, built from the right leg and
shared by both legs -- hip yaw through a sign flip): see
``pal_kangaroo_full.lut_actuator`` and ``pal_kangaroo_full.lut_maps``, and
:data:`LUT_TRANSMISSION_DIR` for the files. ``leg_length`` is a two-valued
axis for this reason alone (``"actuator"`` drives the screw directly, as
before); its transmitted variant servos the knee in the leg-length metres
``knee_distance_map.csv`` maps it onto, the same coordinate the observations
present as the missing ``leg_.*_length_joint``.

:data:`INIT_STATE` is this MJCF's own, not ``pal_kangaroo_full``'s: every
mechanism-only coordinate (the motor housings, the sliders, the femur
triangle/rod, the knee rods, the butterflies and their tibia bars) rests at
the value consistent with its closed loop at the pose the simple joints rest
at, and the ankle is posed for flat feet -- see the note on INIT_STATE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  FULL_COLLISION,
  KANGAROO_PELVIS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  _calc_leg_params,
)
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  _ARM_BASE_LINK_NAMES,
  ARM_ACTION_SCALE_FACTOR,
  LEG_ACTION_SCALE_FACTOR,
  LUT_TRANSMISSION_DIR,
  _add_collision_capsules,
  _build_action_scales,
  _calc_linear_leg_params,
  _delete_subtrees,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_actuator import (
  AnkleLutPdActuatorCfg,
  GainSpec,
  HipXyLutPdActuatorCfg,
  HipZLutPdActuatorCfg,
  LegLengthLutPdActuatorCfg,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import (
  AnkleMap,
  HipXyMap,
  HipZMap,
  LegLengthMap,
)

##
# Joint name patterns.
##

# Every joint this MJCF carries that the simple pal_kangaroo model does not:
# the ten mechanism DOFs pal_kangaroo_full also has (the femur triangle/rod,
# knee rods, and butterfly chain), *and* the five extra
# leg_.*_(1|2|3|4|5)_actuator sliders this MJCF adds in place of
# pal_kangaroo_full's spatial tendons. Every one of the simple model's joints
# is named pelvis_*, arm_* or leg_*, and of the extras only the six screws
# (leg_.*_length_actuator plus the five new leg_.*_(1..5)_actuator sliders)
# share that prefix, so excluding those two patterns selects the simple
# model's joint set exactly -- what every task observes and rewards, so
# variants differ in *actuation* only. This is pal_kangaroo_full's
# REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY / _ACTUATED_JOINTS_ONLY with the
# five extra sliders folded into the exclusion; reusing those regexes as-is
# would wrongly let a policy observe and get rewarded on this MJCF's slider
# mechanism DOFs.
REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = (
  r"^(?!leg_.*_(?:[1-5]|length)_actuator$)(pelvis|arm|leg)_.*$"
)
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_(?:[1-5]|length)_actuator$)(pelvis|arm|leg)_.*$"

##
# MJCF.
##

KANGAROO_FULL_FULL_PATH = (
  PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full_full" / "xmls"
)
KANGAROO_FULL_FULL_XML = KANGAROO_FULL_FULL_PATH / "kangaroo_full.xml"
assert KANGAROO_FULL_FULL_XML.exists(), f"Missing: {KANGAROO_FULL_FULL_XML}"
# The transmission maps of this MJCF's screw mechanisms, read by the
# "transmission" actuation variants below, live with the actuator classes in
# pal_kangaroo_full: LUT_TRANSMISSION_DIR, imported above.

# The passive hinges of each leg's ankle linkage: the two motor cranks, the
# rods and cranks of the two femur four-bars, the two butterflies, and the
# tibia bars' two-hinge universal joints (whose *_1 intermediate body is a
# 1e-5 kg dummy). The MJCF gives none of them armature, and the parts they
# move weigh 0.07-0.25 kg, so under the 2000 N the ankle screws can apply the
# linkage is numerically near-massless: an untrained policy's first random
# targets drove it to 1e3-1e7 rad/s and NaN within a dozen 2 ms substeps.
# Reflected screw armature only regularizes the driven slider, not these
# hinges -- see get_kangaroo_full_full_spec.
REGEX_ANKLE_LINKAGE_JOINTS = (
  r"^(left|right)_(ankle_(motor|crank|femur_bar)_(l|r)|"
  r"ankle_tibia_bar_(l|r)[12]|butterfly_(l|r))$"
)
# On the order of the linkage's own inertia (butterfly ~1.7e-4 kg m^2, femur
# bar ~1.5e-3 about its hinge). On CPU with the ankle screws saturated, 0
# diverges, 1e-4 still peaks at ~1e6 rad/s on some random commands, 1e-3
# stays below ~250 rad/s, and 1e-2 would outweigh every part it sits on.
ANKLE_LINKAGE_ARMATURE = 1e-3
# The intermediate body of each tibia bar's universal joint is a 1e-5 kg
# placeholder in the MJCF (no mesh, no real part). Its two hinges are covered
# by ANKLE_LINKAGE_ARMATURE; the body itself is also brought up to a small but
# non-vanishing mass so the linear part of its inertia isn't at float32 noise
# level either. Its diaginertia (1e-5, already larger than the mass warrants)
# is left as is.
REGEX_ANKLE_LINKAGE_DUMMY_BODIES = r"^(left|right)_ankle_tibia_bar_(l|r)1$"
ANKLE_LINKAGE_DUMMY_BODY_MASS = 2e-3

HipZActuation = Literal["slider", "joint", "transmission"]
HipXyActuation = Literal["slider", "joint", "transmission"]
AnkleActuation = Literal["butterfly", "joint", "slider", "transmission"]
LegLengthActuation = Literal["actuator", "transmission"]
# Not an actuation choice like the axes above -- whether the arms exist at
# all -- but still a `Literal` alongside them rather than a bare `bool` so it
# reads the same way at every call site, mirroring pal_kangaroo_full.
LowerBody = Literal[True, False]

# The slider joint each of pal_kangaroo_full's virtual-motor tendons stands
# for, listed in that tendon list's order (HIP_Z_TENDON_NAMES,
# HIP_XY_TENDON_NAMES, ANKLE_TENDON_NAMES: left-outer, left-inner,
# right-inner, right-outer), so a policy trained on the tendon variant reads
# the same mechanism from the same action slot here. The pairing is by
# parent body: (left|right)_hip_z_slider holds leg_*_1_actuator,
# (left|right)_hip_xy_(l|r)_slider hold leg_*_(2|3)_actuator and
# (left|right)_ankle_(l|r)_slider hold leg_*_(4|5)_actuator -- the "l"/"r"
# in the tendon name is the slider body's, not the leg's, on both sides. The
# tree order in kangaroo_full.xml already lists the right hip_xy pair as
# 3-then-2 but the right ankle pair as 4-then-5, so these are consumed with
# preserve_order rather than as a regex.
HIP_Z_SLIDER_JOINT_NAMES = ("leg_left_1_actuator", "leg_right_1_actuator")
HIP_XY_SLIDER_JOINT_NAMES = (
  "leg_left_2_actuator",
  "leg_left_3_actuator",
  "leg_right_3_actuator",
  "leg_right_2_actuator",
)
ANKLE_SLIDER_JOINT_NAMES = (
  "leg_left_4_actuator",
  "leg_left_5_actuator",
  "leg_right_5_actuator",
  "leg_right_4_actuator",
)

# Every closed loop rests at a mutually consistent pose here (the <connect>
# residual at the keyframe is ~1e-8 m), so there is no pop at reset. The
# ankle chain was solved for flat feet -- sole normal parallel to base z in
# both pitch and roll, so leg_.*_5_joint takes up the 0.04 rad hip roll --
# by holding every simple-model joint with the "joint" actuation variant's
# PD in zero-g and Newton-iterating the ankle pitch/roll targets. Keys are
# paired left_*_l|right_*_r because the two legs are mirror images: the
# "l"/"r" of a slider body is its own, not the leg's.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.91),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
    "left_hip_z_motor": -0.0003058666,
    "right_hip_z_motor": 0.0003058174,
    "leg_.*_1_actuator": 0.0004805971,
    "leg_.*_1_joint": -0.0121160711,
    ".*_hip_xy_bracket_l": -0.0002925489,
    "(left_hip_xy_motor_l|right_hip_xy_motor_r)": -0.0023228059,
    "(leg_left_2_actuator|leg_right_3_actuator)": 0.0016356406,
    ".*_hip_xy_bracket_r": 0.0002925520,
    "(left_hip_xy_motor_r|right_hip_xy_motor_l)": 0.0074219151,
    "(leg_left_3_actuator|leg_right_2_actuator)": 0.0043910614,
    "leg_.*_2_joint": 0.0511187509,
    "leg_left_3_joint": -0.0399909712,
    "leg_right_3_joint": 0.0400007786,
    ".*_hip_xy_cross_(l|r)": 0.0000000000,
    ".*_ankle_motor_(l|r)": 0.0002736707,
    "leg_.*_4_actuator": 0.0017477202,
    "leg_.*_5_actuator": 0.0017566428,
    ".*_ankle_crank_(l|r)": -0.0501039268,
    ".*_ankle_femur_bar_l": 0.2453164250,
    ".*_ankle_femur_bar_r": -0.2453190225,
    ".*_hip_xy_link": 0.0827196495,
    "leg_.*_femur_joint": -0.2951322032,
    ".*_femur_triangle": -0.5978237834,
    ".*_femur_rod": -0.5978342493,
    ".*_butterfly_(l|r)": 0.2446112550,
    ".*_ankle_tibia_bar_l1": 0.3532615783,
    ".*_ankle_tibia_bar_(l|r)2": 0.0000000005,
    ".*_ankle_tibia_bar_r1": -0.3532606887,
    "leg_.*_length_actuator": 0.0276621618,
    ".*_knee_rods": -0.2080560298,
    "leg_.*_knee_joint": 0.5977924588,
    "leg_.*_4_joint": -0.3527847918,
    "leg_.*_5_joint": 0.0000007871,
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
  },
  joint_vel={".*": 0.0},
)


def _regularize_ankle_linkage(spec: mujoco.MjSpec) -> None:
  """Give every ankle linkage hinge ANKLE_LINKAGE_ARMATURE and the universal
  joints' placeholder bodies ANKLE_LINKAGE_DUMMY_BODY_MASS.

  Applied for every ``ankle`` actuation, not just ``"slider"``: the linkage is
  in the model either way (the ``"joint"`` variant leaves it free-wheeling
  along with the ankle) and the same near-massless hinges are what any
  contact or actuator impulse reaches.
  """
  pattern = re.compile(REGEX_ANKLE_LINKAGE_JOINTS)
  joints = [jnt for jnt in spec.joints if pattern.match(jnt.name)]
  assert len(joints) == 24, [jnt.name for jnt in joints]
  for jnt in joints:
    jnt.armature = ANKLE_LINKAGE_ARMATURE

  pattern = re.compile(REGEX_ANKLE_LINKAGE_DUMMY_BODIES)
  bodies = [body for body in spec.bodies if pattern.match(body.name)]
  assert len(bodies) == 4, [body.name for body in bodies]
  for body in bodies:
    body.mass = ANKLE_LINKAGE_DUMMY_BODY_MASS


def get_kangaroo_full_full_spec(lower_body: LowerBody = False) -> mujoco.MjSpec:
  """Load the MJCF and add what the raw file doesn't carry.

  That is the collision capsules, armature on the ankle linkage's passive
  hinges and mass on its placeholder bodies (see REGEX_ANKLE_LINKAGE_JOINTS
  and REGEX_ANKLE_LINKAGE_DUMMY_BODIES). Unlike ``pal_kangaroo_full``'s
  ``get_kangaroo_full_spec``, none of hip_z / hip_xy / ankle need
  variant-specific spec edits here: every mechanism they could pick between
  is a real ``<connect>``-closed loop with exactly one DOF, present
  unconditionally in the MJCF, and the actuator config alone decides which of
  its two redundant coordinates gets driven -- see
  :func:`get_kangaroo_full_full_model`. Only ``lower_body`` changes the spec
  itself.
  """
  spec = mujoco.MjSpec.from_file(str(KANGAROO_FULL_FULL_XML))
  _add_collision_capsules(spec)
  _regularize_ankle_linkage(spec)
  if lower_body:
    # Whole-arm removal: delete before returning so nothing downstream has to
    # know the arms are gone. Reused verbatim from pal_kangaroo_full -- same
    # two body names, same "no cross-tendon/equality/sensor reaches into the
    # arm subtree" property.
    _delete_subtrees(spec, _ARM_BASE_LINK_NAMES)
  return spec


##
# Actuator configs.
##

_HIP_Z_ACTUATORS: dict[HipZActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  "slider": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_1_actuator$",),
      **_calc_linear_leg_params(stiffness=2500.0, effort=2000.0, armature=0.1),
    ),
  ),
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_1_joint",),
      **_calc_leg_params(100.0, 80.0, 0.01, None, None),
    ),
  ),
}

_HIP_XY_ACTUATORS: dict[HipXyActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  "slider": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_[23]_actuator$",),
      **_calc_linear_leg_params(stiffness=750.0, effort=2000.0, armature=0.1),
    ),
  ),
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_2_joint",),
      **_calc_leg_params(100.0, 230.0, 0.01, None, None),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_3_joint",),
      **_calc_leg_params(100.0, 139.0, 0.01, None, None),
    ),
  ),
}

_ANKLE_ACTUATORS: dict[AnkleActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  # The real hardware topology: the two butterflies per leg swing the ankle
  # through the *_ankle_tibia_bar <connect> equalities, leaving leg_.*_4_joint
  # and leg_.*_5_joint passive.
  "butterfly": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"(left|right)_butterfly_l$",),
      **_calc_leg_params(100.0, 30.0, 0.01, None, None),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"(left|right)_butterfly_r$",),
      **_calc_leg_params(100.0, 30.0, 0.01, None, None),
    ),
  ),
  # The simple model's topology: servo the ankle pitch and roll joints
  # directly. The butterfly/slider chain stays in the model (nothing to
  # delete -- see get_kangaroo_full_full_spec) and simply free-wheels along
  # with whatever the <connect> equalities force it to.
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_4_joint",),
      **_calc_leg_params(30.0, 140.0, 0.01, None, None),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_5_joint",),
      **_calc_leg_params(30.0, 82.0, 0.01, None, None),
    ),
  ),
  # Same hardware topology as "butterfly", but driven at the linear actuator
  # screw (leg_.*_(4|5)_actuator) instead of the butterfly hinge itself --
  # the replacement for pal_kangaroo_full's ANKLE_TENDON_NAMES virtual-motor
  # tendon, now a real slider rather than a tendon standing in for one.
  "slider": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_[45]_actuator$",),
      **_calc_linear_leg_params(stiffness=1500.0, effort=2000.0, armature=0.1),
    ),
  ),
}

# The target_names_expr of the three "slider" configs above, i.e. the keys
# _build_action_scales hands back for them; get_kangaroo_full_full_model
# routes these into per-mechanism SliderActions instead of the catch-all term.
_SLIDER_ACTUATOR_KEYS = frozenset(
  name
  for actuators in (_HIP_Z_ACTUATORS, _HIP_XY_ACTUATORS, _ANKLE_ACTUATORS)
  for actuator in actuators["slider"]
  for name in actuator.target_names_expr
)

# Not a variant axis (see the module docstring): this MJCF's only femur
# closure is the four-bar linkage, which only ever pairs with driving the
# screw directly, so there is exactly one leg-length actuator config.
_LEG_LENGTH_ACTUATOR: tuple[BuiltinPositionActuatorCfg, ...] = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    **_calc_linear_leg_params(stiffness=6000.0, effort=5000.0, armature=1.0),
  ),
)

_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
# lower_body=True has no arm_* joints for KANGAROO_S_PLUS_ACTUATOR_CFG /
# KANGAROO_S_MINUS_ACTUATOR_CFG to target, so servo the waist alone instead --
# same substitution pal_kangaroo_full's lower_body axis makes.
_LOWER_BODY_UPPER_BODY_ACTUATORS = (KANGAROO_PELVIS_ACTUATOR_CFG,)


##
# Transmitted actuator configs.
#
# Built on demand rather than at import: each one loads its map from
# LUT_TRANSMISSION_DIR, so a checkout without the maps still imports this
# module and builds every other variant; get_kangaroo_full_full_model is
# cached, so each map is read once per process. One instance per mechanism
# serves both legs (the maps are built from the right leg and hold for the
# left one, hip yaw through a sign flip -- see lut_actuator), so each map is
# shared. Joint-side gains are the "joint" variants' (the leg length
# pal_kangaroo_full's "semi_serial" gains, in metres); the <motor> on each
# screw takes the "slider" / "actuator" variants' force limit, armature and
# viscous damping.
##

_SIDES = ("left", "right")

# The target_names_expr of the transmitted leg-length config, i.e. the keys
# _build_action_scales hands back for it; get_kangaroo_full_full_model
# routes these into the mapped leg-length action term rather than the
# catch-all, since their targets are in metres, not knee radians.
_LEG_LENGTH_TRANSMISSION_KEYS = frozenset(f"leg_{side}_knee_joint" for side in _SIDES)

TRANSMISSION_MAP_NAMES = (
  "hip_z_map.npz",
  "leg_length_map.npz",
  "hip_xy_jacobian_map.npz",
  "ankle_xy_jacobian_map.npz",
)
"""Every map the "transmission" variants read, relative to LUT_TRANSMISSION_DIR."""


def transmission_maps_available() -> bool:
  """Whether every "transmission" variant can be built from this checkout."""
  return all((LUT_TRANSMISSION_DIR / name).exists() for name in TRANSMISSION_MAP_NAMES)


def _transmission_file(name: str) -> Path:
  path = LUT_TRANSMISSION_DIR / name
  if not path.exists():
    raise FileNotFoundError(
      f'{path} is missing: the "transmission" actuation variants need the maps '
      "(hip_z_map.npz, leg_length_map.npz, hip_xy_jacobian_map.npz, "
      "ankle_xy_jacobian_map.npz) that ship in pal_kangaroo_full/lut_transmission/."
    )
  return path


@lru_cache(maxsize=None)
def _transmission_leg_length_map() -> LegLengthMap:
  # Shared by the leg-length and the ankle actuators (the ankle map's third
  # axis is the leg-length slider, reached from the knee through this map).
  return LegLengthMap.load(_transmission_file("leg_length_map.npz"))


def _screw_motor_params(stiffness: float, effort: float, armature: float) -> dict:
  """The "slider" variants' screw element, as the <motor> parameters of a
  transmitted actuator: its force limit, armature and viscous damping (the kp/kv
  belong to the <position> element that isn't there)."""
  params = _calc_linear_leg_params(
    stiffness=stiffness, effort=effort, armature=armature
  )
  return {
    "actuator_effort_limit": params["effort_limit"],
    "armature": params["armature"],
    "viscous_damping": params["viscous_damping"],
  }


def _joint_pd_params(stiffness: float, effort: GainSpec) -> dict:
  """The "joint" variants' PD, as the servo-side gains of a transmitted actuator."""
  # Only the damping rule (2 zeta kp / omega) is wanted from _calc_leg_params;
  # the effort may be per joint, which that helper doesn't take.
  joint = _calc_leg_params(stiffness, 0.0, 0.01, None, None)
  return {
    "joint_stiffness": joint["stiffness"],
    "joint_damping": joint["damping"],
    "joint_effort_limit": effort,
  }


def _hip_z_transmission_actuators() -> tuple[HipZLutPdActuatorCfg, ...]:
  return (
    HipZLutPdActuatorCfg(
      target_names_expr=(r"leg_(left|right)_1_joint",),
      hip_z_map=HipZMap.load(_transmission_file("hip_z_map.npz")),
      **_joint_pd_params(100.0, 80.0),
      **_screw_motor_params(stiffness=2500.0, effort=2000.0, armature=0.1),
    ),
  )


def _hip_xy_transmission_actuators() -> tuple[HipXyLutPdActuatorCfg, ...]:
  return (
    HipXyLutPdActuatorCfg(
      target_names_expr=(r"leg_(left|right)_2_joint", r"leg_(left|right)_3_joint"),
      hip_xy_map=HipXyMap.load(_transmission_file("hip_xy_jacobian_map.npz")),
      **_joint_pd_params(
        100.0, {r"leg_(left|right)_2_joint": 230.0, r"leg_(left|right)_3_joint": 139.0}
      ),
      **_screw_motor_params(stiffness=750.0, effort=2000.0, armature=0.1),
    ),
  )


def _ankle_transmission_actuators() -> tuple[AnkleLutPdActuatorCfg, ...]:
  return (
    AnkleLutPdActuatorCfg(
      target_names_expr=(r"leg_(left|right)_4_joint", r"leg_(left|right)_5_joint"),
      ankle_map=AnkleMap.load(_transmission_file("ankle_xy_jacobian_map.npz")),
      leg_length_map=_transmission_leg_length_map(),
      **_joint_pd_params(
        30.0, {r"leg_(left|right)_4_joint": 140.0, r"leg_(left|right)_5_joint": 82.0}
      ),
      **_screw_motor_params(stiffness=1500.0, effort=2000.0, armature=0.1),
    ),
  )


def _leg_length_transmission_actuators() -> tuple[LegLengthLutPdActuatorCfg, ...]:
  # Explicit knee names rather than a regex: _build_action_scales hands
  # these back as the keys, and _LEG_LENGTH_TRANSMISSION_KEYS routes them
  # into the mapped (metres) leg-length action term.
  return (
    LegLengthLutPdActuatorCfg(
      target_names_expr=tuple(f"leg_{side}_knee_joint" for side in _SIDES),
      leg_length_map=_transmission_leg_length_map(),
      **_joint_pd_params(900.0, 1100.0),
      **_screw_motor_params(stiffness=6000.0, effort=5000.0, armature=1.0),
    ),
  )


##
# Variants.
##


@dataclass(frozen=True)
class LegLengthAction:
  """What the mapped leg-length action term needs for the transmitted
  leg-length variant: the knee joints it targets and the scale, in metres,
  a unit action commands. The offset is the map's value at the default knee
  angle, which the term reads for itself."""

  actuator_names: tuple[str, ...]
  scale: dict[str, float]


@dataclass(frozen=True)
class SliderAction:
  """What an order-preserved joint action term needs for one screw mechanism.

  The counterpart of pal_kangaroo_full's ``TendonAction``: the same explicit,
  ordered target list, but naming slider joints rather than tendons, and no
  offset -- a joint term reads its rest position from INIT_STATE through
  ``use_default_offset``.
  """

  actuator_names: tuple[str, ...]
  """Slider joint names, in the order they should occupy in the action vector."""
  scale: dict[str, float]


@dataclass(frozen=True)
class KangarooFullFullModel:
  """One actuation variant of the connect-linkage KANGAROO full model."""

  hip_z: HipZActuation
  hip_xy: HipXyActuation
  ankle: AnkleActuation
  leg_length: LegLengthActuation
  lower_body: LowerBody
  arm_action_scale_factor: float
  leg_action_scale_factor: float

  articulation: EntityArticulationInfoCfg
  init_state: EntityCfg.InitialStateCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]
  """Targets of the catch-all joint term: everything that is not a screw
  driven in a "slider" variant (those get their own ordered terms below)
  and not the knee of a transmitted leg length (its own term too)."""
  hip_z_slider_action: SliderAction | None
  hip_xy_slider_action: SliderAction | None
  ankle_slider_action: SliderAction | None
  leg_length_action: LegLengthAction | None

  def make_spec(self) -> mujoco.MjSpec:
    return get_kangaroo_full_full_spec(lower_body=self.lower_body)

  def make_robot_cfg(self) -> EntityCfg:
    return EntityCfg(
      init_state=self.init_state,
      collisions=(FULL_COLLISION,),
      spec_fn=self.make_spec,
      articulation=self.articulation,
    )


@lru_cache(maxsize=None)
def get_kangaroo_full_full_model(
  hip_z: HipZActuation = "slider",
  hip_xy: HipXyActuation = "slider",
  ankle: AnkleActuation = "joint",
  leg_length: LegLengthActuation = "actuator",
  lower_body: LowerBody = False,
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> KangarooFullFullModel:
  """Assemble the actuators and action scale for one variant.

  Every actuator here is JOINT-transmission (there is no tendon left to
  target), so unlike pal_kangaroo_full's get_kangaroo_full_model this needs
  no tendon-length-at-init offset computation. The action vector still
  splits the same way, though: one catch-all joint term, plus one
  order-preserved term per mechanism driven at its screw, laid out exactly
  like the tendon term it replaces (see HIP_Z_SLIDER_JOINT_NAMES and
  friends) so a checkpoint transfers between the two models slot for slot.
  A "transmission" mechanism is commanded on its simple-model joints and so
  sits in the catch-all term like a "joint" one -- except the transmitted
  leg length, whose knee target is in metres and gets its own term.

  The two ``*_action_scale_factor`` values mean the same as pal_kangaroo_full's:
  what fraction of an actuator's effort limit a unit action commands, expressed
  as a position offset through its stiffness -- ``leg_action_scale_factor`` for
  every leg mechanism, ``arm_action_scale_factor`` for the upper body. For a
  transmitted mechanism that is the servo joint's effort and stiffness, so
  a unit action means the same thing as on the "joint" variant.
  """
  # Ordered like the simple pal_kangaroo model's actuators (hip yaw, hip
  # pitch/roll, ankle, leg length, then upper body) so the action vector reads
  # the same way in every variant, and the same way as pal_kangaroo_full's.
  leg_actuators = (
    (
      _hip_z_transmission_actuators()
      if hip_z == "transmission"
      else _HIP_Z_ACTUATORS[hip_z]
    )
    + (
      _hip_xy_transmission_actuators()
      if hip_xy == "transmission"
      else _HIP_XY_ACTUATORS[hip_xy]
    )
    + (
      _ankle_transmission_actuators()
      if ankle == "transmission"
      else _ANKLE_ACTUATORS[ankle]
    )
    + (
      _leg_length_transmission_actuators()
      if leg_length == "transmission"
      else _LEG_LENGTH_ACTUATOR
    )
  )
  upper_body_actuators = (
    _LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS
  )
  articulation = EntityArticulationInfoCfg(
    actuators=leg_actuators + upper_body_actuators,
    soft_joint_pos_limit_factor=0.99,
  )

  # Legs and upper body get their own factor, so build the action term in two
  # halves and concatenate them in the same leg-then-upper-body order. The
  # screws of a "slider" variant are carved out into their own terms.
  joint_action_scale: dict[str, float] = {}
  joint_actuator_names: tuple[str, ...] = ()
  slider_scale: dict[str, float] = {}
  slider_names: tuple[str, ...] = ()
  leg_length_scale: dict[str, float] = {}
  leg_length_names: tuple[str, ...] = ()
  for actuators, factor in (
    (leg_actuators, leg_action_scale_factor),
    (upper_body_actuators, arm_action_scale_factor),
  ):
    scales, names = _build_action_scales(actuators, TransmissionType.JOINT, factor)
    for name in names:
      if name in _SLIDER_ACTUATOR_KEYS:
        slider_scale[name] = scales[name]
        slider_names += (name,)
      elif name in _LEG_LENGTH_TRANSMISSION_KEYS:
        leg_length_scale[name] = scales[name]
        leg_length_names += (name,)
      else:
        joint_action_scale[name] = scales[name]
        joint_actuator_names += (name,)

  def _slider_action(names: tuple[str, ...], key: str) -> SliderAction:
    # Each term's scale may only carry keys matching its own targets, so
    # hand each mechanism the one regex that names its sliders.
    return SliderAction(
      actuator_names=tuple(f"{name}$" for name in names),
      scale={k: v for k, v in slider_scale.items() if key in k},
    )

  return KangarooFullFullModel(
    hip_z=hip_z,
    hip_xy=hip_xy,
    ankle=ankle,
    leg_length=leg_length,
    lower_body=lower_body,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
    articulation=articulation,
    init_state=INIT_STATE,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
    hip_z_slider_action=(
      _slider_action(HIP_Z_SLIDER_JOINT_NAMES, "_1_") if hip_z == "slider" else None
    ),
    hip_xy_slider_action=(
      _slider_action(HIP_XY_SLIDER_JOINT_NAMES, "[23]") if hip_xy == "slider" else None
    ),
    ankle_slider_action=(
      _slider_action(ANKLE_SLIDER_JOINT_NAMES, "[45]") if ankle == "slider" else None
    ),
    leg_length_action=(
      LegLengthAction(actuator_names=leg_length_names, scale=leg_length_scale)
      if leg_length == "transmission"
      else None
    ),
  )


def main(
  hip_z: HipZActuation = "slider",
  hip_xy: HipXyActuation = "slider",
  ankle: AnkleActuation = "joint",
  leg_length: LegLengthActuation = "actuator",
  lower_body: LowerBody = False,
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the connect-linkage KANGAROO model.

  Args:
    hip_z: Drive hip yaw through its linear actuator screw
      (leg_.*_1_actuator), through the plain leg_.*_1_joint revolute motor,
      or through the screw commanded on leg_.*_1_joint via the transmission
      map (PD on the joint, <motor> on the screw). Either way the physical
      mechanism -- housing, slider, <connect> -- stays in the model; only
      which coordinate is driven changes.
    hip_xy: Drive hip pitch/roll through the parallel screw pair
      (leg_.*_(2|3)_actuator), through the plain leg_.*_2_joint /
      leg_.*_3_joint revolute motors, or through the pair commanded on those
      joints via the 2-D transmission map.
    ankle: Swing the ankle from the (left|right)_butterfly_(l|r) joints, as
      the hardware does; servo leg_.*_4_joint / leg_.*_5_joint directly;
      drive the ankle screw pair (leg_.*_(4|5)_actuator); or drive the pair
      commanded on leg_.*_(4|5)_joint via the 3-D (pitch, roll, leg length)
      transmission map.
    leg_length: Drive the knee screw (leg_.*_length_actuator) directly, or
      commanded in leg-length metres on the knee via the transmission map.
    lower_body: Delete both arms (everything from arm_(left|right)_base_link
      down) and servo only the waist where the full model would otherwise
      also drive the arms.
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_full_model(
    hip_z=hip_z,
    hip_xy=hip_xy,
    ankle=ankle,
    leg_length=leg_length,
    lower_body=lower_body,
  )

  # Go through Entity rather than compiling make_spec() directly:
  # kangaroo_full.xml declares no <actuator> elements at all, so the raw spec
  # compiles to nu=0. The actuators (and the collision setup, and an
  # "init_state" keyframe) are what the articulation config adds on top --
  # which is exactly the layer this command exists to inspect.
  entity = Entity(model_cfg.make_robot_cfg())
  spec = entity.spec
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)

  print(
    f"hip_z={hip_z} hip_xy={hip_xy} ankle={ankle} leg_length={leg_length} "
    f"lower_body={lower_body}"
  )
  if entity.is_fixed_base:
    print("  base:    FIXED (no freejoint in the MJCF -- pinned to a mocap body)")
  else:
    print("  base:    floating")
  print(f"  joints:  {model.njnt} ({model.nv} dof)")
  print(f"  equalities: {model.neq}")
  print(f"  actuators ({model.nu}):")
  for i in range(model.nu):
    actuator = model.actuator(i)
    if actuator.biasprm[1] != 0.0:  # <position>: gainprm[0] is kp
      print(f"    {actuator.name:34s} (joint)  kp={actuator.gainprm[0]:.1f}")
    else:  # <motor>: an external force within forcerange
      lo, hi = actuator.forcerange
      print(f"    {actuator.name:34s} (joint)  motor, force in [{lo:.0f}, {hi:.0f}]")
  print(f"  joint action targets ({len(model_cfg.joint_actuator_names)}):")
  for name in model_cfg.joint_actuator_names:
    print(f"    {name}  scale={model_cfg.joint_action_scale[name]:.4f}")
  for label, slider_action in (
    ("hip_z", model_cfg.hip_z_slider_action),
    ("hip_xy", model_cfg.hip_xy_slider_action),
    ("ankle", model_cfg.ankle_slider_action),
  ):
    if slider_action is None:
      print(f"  {label} slider action: none (commanded on joints)")
      continue
    print(f"  {label} slider action targets ({len(slider_action.actuator_names)}):")
    for name in slider_action.actuator_names:
      print(f"    {name}")
  if model_cfg.leg_length_action is None:
    print("  leg length action: none (screw driven directly)")
  else:
    action = model_cfg.leg_length_action
    print(f"  leg length action targets ({len(action.actuator_names)}), metres:")
    for name in action.actuator_names:
      print(f"    {name}  scale={action.scale[name]:.4f}")

  if launch_viewer:
    # Imported here, not at module scope, so importing these constants
    # doesn't drag in the viewer's GL dependencies.
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
