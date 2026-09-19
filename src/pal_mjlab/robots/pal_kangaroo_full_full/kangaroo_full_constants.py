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
equality's constraint solver keeps the other one consistent. The femur is
closed by the ``"linkage"`` four-bar only (femur triangle + femur rod, closed
through the ``(left|right)_hip_xy_link`` / ``(left|right)_femur_rod``
equalities); there is no straight-line ``leg_.*_length_joint`` stand-in, so
the knee is the only leg-length DOF. And there is one XML, not
``pal_kangaroo_full``'s base/over-constrained pair.

One axis, ``transmission``, says how the legs are driven -- the same three
values as ``pal_kangaroo_full``, on this MJCF's elements:

===============  ==============================================================
``transmission``  actuation
===============  ==============================================================
``"joint"``       a PD on the simple ``pal_kangaroo`` model's joints,
                  ``leg_.*_(1|2|3|4|5)_joint``, and on the knee for the leg
                  length. The linkages stay in the model and free-wheel along
                  with whatever the ``<connect>`` equalities force them to.
``"actuator"``    a PD on the six screws per leg: the five sliders and
                  ``leg_.*_length_actuator``.
``"lut"``         a PD on the simple model's joints, as in ``"joint"`` -- the
                  leg length in the metres ``leg_length_map.npz`` maps the
                  knee onto, the coordinate the observations present as the
                  missing ``leg_.*_length_joint`` -- whose torques are pushed
                  through the mechanisms' Jacobians, read from the spline
                  maps in ``pal_kangaroo_full/lut_transmission/``, onto
                  ``<motor>``s on the six screws (``pal_kangaroo_full.
                  lut_actuator``; one instance for all twelve leg joints and
                  both legs).
===============  ==============================================================

``lower_body`` is unchanged from ``pal_kangaroo_full``: it deletes both arms
and servos only the waist.

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

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  FULL_COLLISION,
  KANGAROO_PELVIS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
)
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  _ARM_BASE_LINK_NAMES,
  ARM_ACTION_SCALE_FACTOR,
  LEG_ACTION_SCALE_FACTOR,
  LEG_SCREW_PD,
  ActuatorModel,
  LegLengthAction,
  LowerBody,
  Transmission,
  _add_collision_capsules,
  _delete_subtrees,
  joint_pd_actuators,
  lut_actuator,
  lut_leg_length_action,
  print_actuators,
  print_leg_length_action,
  screw_actuator_cfg,
  screw_element,
  split_joint_action_scales,
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

# The only actuation axis (Transmission) and lower_body come from
# pal_kangaroo_full; see the module docstring for what each value drives here.

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
  ``get_kangaroo_full_spec``, the transmission needs no spec edits here:
  every mechanism is a real ``<connect>``-closed loop with exactly one DOF,
  present unconditionally in the MJCF, and the actuator config alone decides
  which of its two redundant coordinates gets driven -- see
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
#
# The gains are pal_kangaroo_full's tables (LEG_JOINT_PD, LEG_SCREW_PD):
# a screw here is the slider joint that carries the map actuator's own name,
# where the tendon model has a tendon standing in for it.
##


# The hip/ankle slider actuators' (target_names_expr, map_actuator) pairs, in
# the simple model's order; the leg length screw is appended separately below
# since its target isn't one of _SLIDER_ACTUATOR_KEYS.
_SLIDER_ACTUATOR_TARGETS: tuple[tuple[str, str], ...] = (
  (r"leg_(left|right)_1_actuator$", "leg_right_1_actuator"),
  (r"leg_(left|right)_[23]_actuator$", "leg_right_2_actuator"),
  (r"leg_(left|right)_[45]_actuator$", "leg_right_4_actuator"),
)

# The target_names_expr of the three hip/ankle slider configs, i.e. the keys
# _build_action_scales hands back for them; get_kangaroo_full_full_model
# routes these into per-mechanism SliderActions instead of the catch-all term.
_SLIDER_ACTUATOR_KEYS = frozenset(expr for expr, _ in _SLIDER_ACTUATOR_TARGETS)


def _actuator_transmission_actuators(
  actuator_model: ActuatorModel,
) -> tuple[ActuatorCfg, ...]:
  """The "actuator" transmission: a PD on every screw, ordered like the
  simple model's actuators (hip yaw, hip pitch/roll, ankle, leg length) --
  the native <position> element ("builtin") or DcMotorActuatorCfg's
  torque-speed curve ("dc_motor")."""
  return tuple(
    screw_actuator_cfg((expr,), map_actuator, actuator_model)
    for expr, map_actuator in _SLIDER_ACTUATOR_TARGETS
  ) + (
    screw_actuator_cfg(
      (r"leg_(left|right)_length_actuator$",),
      "leg_right_length_actuator",
      actuator_model,
    ),
  )


def _leg_actuators(
  transmission: Transmission, actuator_model: ActuatorModel
) -> tuple[ActuatorCfg, ...]:
  """The leg actuators of one variant, in the simple model's actuator order."""
  if transmission == "joint":
    return joint_pd_actuators("knee", INIT_STATE.joint_pos["leg_.*_knee_joint"])
  if transmission == "actuator":
    return _actuator_transmission_actuators(actuator_model)
  if transmission == "lut":
    # The "lut" transmission's <motor>s sit on the sliders that carry the
    # map actuators' own names.
    return (
      lut_actuator(
        {name: screw_element(name) for name in LEG_SCREW_PD}, leg_length_servo="knee"
      ),
    )
  raise ValueError(f"unknown transmission {transmission!r}")


_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
# lower_body=True has no arm_* joints for KANGAROO_S_PLUS_ACTUATOR_CFG /
# KANGAROO_S_MINUS_ACTUATOR_CFG to target, so servo the waist alone instead --
# same substitution pal_kangaroo_full's lower_body axis makes.
_LOWER_BODY_UPPER_BODY_ACTUATORS = (KANGAROO_PELVIS_ACTUATOR_CFG,)


##
# Variants.
##


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

  transmission: Transmission
  lower_body: LowerBody
  actuator_model: ActuatorModel
  arm_action_scale_factor: float
  leg_action_scale_factor: float

  articulation: EntityArticulationInfoCfg
  init_state: EntityCfg.InitialStateCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]
  """Targets of the catch-all joint term: everything that is not a hip or
  ankle screw of the "actuator" transmission (those get their own ordered
  terms below) and not the knee of the "lut" one (its own term too)."""
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
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> KangarooFullFullModel:
  """Assemble the actuators and action scale for one variant.

  Every actuator here is JOINT-transmission (there is no tendon to target),
  so unlike pal_kangaroo_full's get_kangaroo_full_model this needs no
  tendon-length-at-init offset computation. The action vector still splits
  the same way, though: one catch-all joint term, plus -- for the "actuator"
  transmission -- one order-preserved term per hip/ankle mechanism, laid out
  exactly like the tendon term it replaces (see HIP_Z_SLIDER_JOINT_NAMES and
  friends) so a checkpoint transfers between the two models slot for slot.
  The "lut" transmission is commanded on the simple-model joints and so sits
  in the catch-all term like the "joint" one -- except the knee, whose
  targets are leg lengths in metres and get their own term.

  The two ``*_action_scale_factor`` values mean the same as pal_kangaroo_full's:
  what fraction of an actuator's effort limit a unit action commands, expressed
  as a position offset through its stiffness -- ``leg_action_scale_factor`` for
  every leg mechanism, ``arm_action_scale_factor`` for the upper body. For the
  "lut" transmission that is the servo joint's effort and stiffness, so a unit
  action means the same thing as on the "joint" one.
  """
  leg_actuators = _leg_actuators(transmission, actuator_model)
  upper_body_actuators = (
    _LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS
  )
  articulation = EntityArticulationInfoCfg(
    actuators=leg_actuators + upper_body_actuators,
    soft_joint_pos_limit_factor=0.99,
  )

  # The hip/ankle screws of the "actuator" transmission and the knee of the
  # "lut" one are carved out of the catch-all term into their own.
  leg_length_keys = lut_leg_length_action(leg_actuators)
  joint_action_scale, joint_actuator_names, carved_scale, carved_names = (
    split_joint_action_scales(
      leg_actuators,
      upper_body_actuators,
      leg_action_scale_factor,
      arm_action_scale_factor,
      carve_out=_SLIDER_ACTUATOR_KEYS | leg_length_keys,
    )
  )
  slider_scale = {k: v for k, v in carved_scale.items() if k in _SLIDER_ACTUATOR_KEYS}
  leg_length_scale = {k: v for k, v in carved_scale.items() if k in leg_length_keys}
  leg_length_names = tuple(n for n in carved_names if n in leg_length_keys)

  def _slider_action(names: tuple[str, ...], key: str) -> SliderAction | None:
    if transmission != "actuator":
      return None
    # Each term's scale may only carry keys matching its own targets, so
    # hand each mechanism the one regex that names its sliders.
    return SliderAction(
      actuator_names=tuple(f"{name}$" for name in names),
      scale={k: v for k, v in slider_scale.items() if key in k},
    )

  return KangarooFullFullModel(
    transmission=transmission,
    lower_body=lower_body,
    actuator_model=actuator_model,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
    articulation=articulation,
    init_state=INIT_STATE,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
    hip_z_slider_action=_slider_action(HIP_Z_SLIDER_JOINT_NAMES, "_1_"),
    hip_xy_slider_action=_slider_action(HIP_XY_SLIDER_JOINT_NAMES, "[23]"),
    ankle_slider_action=_slider_action(ANKLE_SLIDER_JOINT_NAMES, "[45]"),
    leg_length_action=(
      LegLengthAction(actuator_names=leg_length_names, scale=leg_length_scale)
      if leg_length_names
      else None
    ),
  )


def main(
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the connect-linkage KANGAROO model.

  Args:
    transmission: Drive the simple model's joints directly ("joint": PD on
      leg_.*_(1|2|3|4|5)_joint and the knee), the screws directly
      ("actuator": PD on the five leg_.*_(1..5)_actuator sliders and
      leg_.*_length_actuator), or the screws commanded on the simple model's
      joints ("lut": the "joint" PD, the knee in leg-length metres, pushed
      through the transmission maps onto <motor>s on the screws). Either way
      every physical mechanism -- housing, slider, <connect> -- stays in the
      model; only which coordinate is driven changes.
    lower_body: Delete both arms (everything from arm_(left|right)_base_link
      down) and servo only the waist where the full model would otherwise
      also drive the arms.
    actuator_model: Only matters for transmission="actuator": the native
      <position> element ("builtin") or DcMotorActuatorCfg's torque-speed
      curve, saturating at the screw's stall torque and dropping to zero at
      its no-load speed ("dc_motor").
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_full_model(
    transmission=transmission, lower_body=lower_body, actuator_model=actuator_model
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
    f"transmission={transmission} lower_body={lower_body} "
    f"actuator_model={actuator_model}"
  )
  if entity.is_fixed_base:
    print("  base:    FIXED (no freejoint in the MJCF -- pinned to a mocap body)")
  else:
    print("  base:    floating")
  print(f"  joints:  {model.njnt} ({model.nv} dof)")
  print(f"  equalities: {model.neq}")
  print_actuators(model)
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
  print_leg_length_action(model_cfg.leg_length_action)

  if launch_viewer:
    # Imported here, not at module scope, so importing these constants
    # doesn't drag in the viewer's GL dependencies.
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
