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
  ``leg_.*_length_joint`` stand-in to fall back to. Consequently ``leg_length``
  is not a variant axis either: every ``pal_kangaroo_full`` value other than
  ``"actuator"`` targets that missing joint, and ``get_kangaroo_full_spec``
  already restricts ``femur_closure="linkage"`` to pair only with
  ``leg_length="actuator"`` -- the one combination this model can produce, so
  the screw (``leg_.*_length_actuator``) is simply always driven directly.
* ``mjcf``: there is one XML here, not a base/over-constrained pair.

What's left are three real actuation choices. ``hip_z`` and ``hip_xy`` each
pick between driving the linear actuator screw directly (``"slider"`` --
``pal_kangaroo_full``'s tendon axis, replaced by the real actuator it stood
in for) or driving the output joint directly (``"joint"``). ``ankle`` keeps a
third option, ``"butterfly"``, which targets the real butterfly hinges
instead of either screw-adjacent coordinate. ``lower_body`` is unchanged from
``pal_kangaroo_full``: it deletes both arms and servos only the waist.

Caveat: :data:`INIT_STATE` (reused from ``pal_kangaroo_full``) sets a rest
value for every joint the *simple* model has, which now leaves every
mechanism-only coordinate this MJCF adds (the motor housings, the sliders,
the femur triangle/rod, the knee rods, the butterflies and their tibia bars)
at qpos 0 -- not the value consistent with its own closed loop at the pose
the simple joints rest at. The `<connect>` equalities are soft
(``solref``/``solimp``), so the solver pulls each loop consistent over the
first few physics steps after a reset rather than leaving a permanent error,
but this has not been checked for a visible pop at reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
  INIT_STATE as _SIMPLE_MJCF_INIT_STATE,
)
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  _add_collision_capsules,
  _ARM_BASE_LINK_NAMES,
  _build_action_scales,
  _calc_linear_leg_params,
  _delete_subtrees,
  _MJCF_ANKLE_INIT_STATE,
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
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_(?:[1-5]|length)_actuator$)(pelvis|arm|leg)_.*$"
)

##
# MJCF.
##

KANGAROO_FULL_FULL_PATH = (
  PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full_full" / "xmls"
)
KANGAROO_FULL_FULL_XML = KANGAROO_FULL_FULL_PATH / "kangaroo_full.xml"
assert KANGAROO_FULL_FULL_XML.exists(), f"Missing: {KANGAROO_FULL_FULL_XML}"

HipZActuation = Literal["slider", "joint"]
HipXyActuation = Literal["slider", "joint"]
AnkleActuation = Literal["butterfly", "joint", "slider"]
# Not an actuation choice like the axes above -- whether the arms exist at
# all -- but still a `Literal` alongside them rather than a bare `bool` so it
# reads the same way at every call site, mirroring pal_kangaroo_full.
LowerBody = Literal[True, False]

# leg_.*_4_joint / butterfly rest pose. pal_kangaroo_full layers a different
# one of these onto INIT_STATE per MjcfVariant (the geometry differs between
# its two XMLs); this MJCF's ankle geometry matches its "tendons" variant, so
# that is the one reused below.
_ANKLE_INIT_STATE = _MJCF_ANKLE_INIT_STATE["tendons"]

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=_SIMPLE_MJCF_INIT_STATE.pos,
  rot=_SIMPLE_MJCF_INIT_STATE.rot,
  joint_pos={**_SIMPLE_MJCF_INIT_STATE.joint_pos, **_ANKLE_INIT_STATE},
  joint_vel=_SIMPLE_MJCF_INIT_STATE.joint_vel,
)


def get_kangaroo_full_full_spec(lower_body: LowerBody = False) -> mujoco.MjSpec:
  """Load the MJCF and add the collision capsules the raw file doesn't carry.

  Unlike ``pal_kangaroo_full``'s ``get_kangaroo_full_spec``, none of hip_z /
  hip_xy / ankle need spec edits here: every mechanism they could pick
  between is a real ``<connect>``-closed loop with exactly one DOF, present
  unconditionally in the MJCF, and the actuator config alone decides which of
  its two redundant coordinates gets driven -- see
  :func:`get_kangaroo_full_full_model`. Only ``lower_body`` changes the spec
  itself.
  """
  spec = mujoco.MjSpec.from_file(str(KANGAROO_FULL_FULL_XML))
  _add_collision_capsules(spec)
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
      target_names_expr=("leg_.*_1_joint",), **_calc_leg_params(100.0, 80.0)
    ),
  ),
}

_HIP_XY_ACTUATORS: dict[HipXyActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  "slider": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_[23]_actuator$",),
      **_calc_linear_leg_params(stiffness=2500.0, effort=2000.0, armature=0.1),
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

_ANKLE_ACTUATORS: dict[AnkleActuation, tuple[BuiltinPositionActuatorCfg, ...]] = {
  # The real hardware topology: the two butterflies per leg swing the ankle
  # through the *_ankle_tibia_bar <connect> equalities, leaving leg_.*_4_joint
  # and leg_.*_5_joint passive.
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
  # directly. The butterfly/slider chain stays in the model (nothing to
  # delete -- see get_kangaroo_full_full_spec) and simply free-wheels along
  # with whatever the <connect> equalities force it to.
  "joint": (
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
    ),
  ),
  # Same hardware topology as "butterfly", but driven at the linear actuator
  # screw (leg_.*_(4|5)_actuator) instead of the butterfly hinge itself --
  # the replacement for pal_kangaroo_full's ANKLE_TENDON_NAMES virtual-motor
  # tendon, now a real slider rather than a tendon standing in for one.
  "slider": (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_[45]_actuator$",),
      **_calc_linear_leg_params(stiffness=2500.0, effort=2000.0, armature=0.1),
    ),
  ),
}

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
# Variants.
##


@dataclass(frozen=True)
class KangarooFullFullModel:
  """One actuation variant of the connect-linkage KANGAROO full model."""

  hip_z: HipZActuation
  hip_xy: HipXyActuation
  ankle: AnkleActuation
  lower_body: LowerBody

  articulation: EntityArticulationInfoCfg
  init_state: EntityCfg.InitialStateCfg
  joint_action_scale: dict[str, float]
  joint_actuator_names: tuple[str, ...]

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
  lower_body: LowerBody = False,
) -> KangarooFullFullModel:
  """Assemble the actuators and action scale for one variant.

  Every actuator here is JOINT-transmission (there is no tendon left to
  target), so unlike pal_kangaroo_full's get_kangaroo_full_model this needs
  no TENDON-side action term or tendon-length-at-init offset computation --
  the whole action vector is one JointPositionActionCfg.
  """
  articulation = EntityArticulationInfoCfg(
    # Ordered like the simple pal_kangaroo model's actuators (hip yaw, hip
    # pitch/roll, ankle, leg length, then upper body) so the action vector
    # reads the same way in every variant, and the same way as
    # pal_kangaroo_full's.
    actuators=(
      _HIP_Z_ACTUATORS[hip_z]
      + _HIP_XY_ACTUATORS[hip_xy]
      + _ANKLE_ACTUATORS[ankle]
      + _LEG_LENGTH_ACTUATOR
      + (_LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS)
    ),
    soft_joint_pos_limit_factor=0.99,
  )

  joint_action_scale, joint_actuator_names = _build_action_scales(
    articulation, TransmissionType.JOINT
  )

  return KangarooFullFullModel(
    hip_z=hip_z,
    hip_xy=hip_xy,
    ankle=ankle,
    lower_body=lower_body,
    articulation=articulation,
    init_state=INIT_STATE,
    joint_action_scale=joint_action_scale,
    joint_actuator_names=joint_actuator_names,
  )


def main(
  hip_z: HipZActuation = "slider",
  hip_xy: HipXyActuation = "slider",
  ankle: AnkleActuation = "joint",
  lower_body: LowerBody = False,
  launch_viewer: bool = True,
) -> None:
  """Inspect one actuation variant of the connect-linkage KANGAROO model.

  Args:
    hip_z: Drive hip yaw through its linear actuator screw
      (leg_.*_1_actuator), or through the plain leg_.*_1_joint revolute
      motor. Either way the physical mechanism -- housing, slider, <connect>
      -- stays in the model; only which coordinate is driven changes.
    hip_xy: Drive hip pitch/roll through the parallel screw pair
      (leg_.*_(2|3)_actuator), or through the plain leg_.*_2_joint /
      leg_.*_3_joint revolute motors.
    ankle: Swing the ankle from the (left|right)_butterfly_(l|r) joints, as
      the hardware does; servo leg_.*_4_joint / leg_.*_5_joint directly; or
      drive the ankle screw pair (leg_.*_(4|5)_actuator).
    lower_body: Delete both arms (everything from arm_(left|right)_base_link
      down) and servo only the waist where the full model would otherwise
      also drive the arms.
    launch_viewer: Open the MuJoCo viewer. Pass False for the summary only.
  """
  model_cfg = get_kangaroo_full_full_model(
    hip_z=hip_z, hip_xy=hip_xy, ankle=ankle, lower_body=lower_body
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

  print(f"hip_z={hip_z} hip_xy={hip_xy} ankle={ankle} lower_body={lower_body}")
  if entity.is_fixed_base:
    print("  base:    FIXED (no freejoint in the MJCF -- pinned to a mocap body)")
  else:
    print("  base:    floating")
  print(f"  joints:  {model.njnt} ({model.nv} dof)")
  print(f"  equalities: {model.neq}")
  print(f"  actuators ({model.nu}):")
  for i in range(model.nu):
    actuator = model.actuator(i)
    print(f"    {actuator.name:34s} (joint)  kp={actuator.gainprm[0]:.1f}")
  print(f"  joint action targets ({len(model_cfg.joint_actuator_names)}):")
  for name in model_cfg.joint_actuator_names:
    print(f"    {name}  scale={model_cfg.joint_action_scale[name]:.4f}")

  if launch_viewer:
    # Imported here, not at module scope, so importing these constants
    # doesn't drag in the viewer's GL dependencies.
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
