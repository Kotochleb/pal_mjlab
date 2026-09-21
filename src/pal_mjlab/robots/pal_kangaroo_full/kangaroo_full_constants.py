from __future__ import annotations

import re
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.actuator import ActuatorCfg, BuiltinPositionActuatorCfg, DcMotorActuatorCfg
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
from pal_mjlab.robots.pal_kangaroo_full.actuator import (
  JointParams,
  LutMechanismCfg,
  LutTransmissionActuatorCfg,
  ScrewParams,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import TransmissionMaps
from pal_mjlab.robots.pal_kangaroo_full.spec_edits import (
  add_collision_capsules,
  delete_equalities,
  delete_joints,
  delete_subtrees,
  delete_tendons,
)

REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = (
  r"^(?!leg_.*_length_actuator$)(pelvis|arm|leg)_.*$"
)
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_length_actuator$)(pelvis|arm|leg)_.*$"
)

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

LOWER_BODY_JOINT_ORDER: tuple[str, ...] = tuple(
  name for name in SIMPLE_MODEL_JOINT_ORDER if not name.startswith("arm_")
)

KANGAROO_FULL_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls"
KANGAROO_FULL_XML = KANGAROO_FULL_PATH / "kangaroo_full_tendons.xml"
KANGAROO_FULL_XML_OVER_CONSTRAINED = (
  KANGAROO_FULL_PATH / "kangaroo_full_tendons_over_constarined.xml"
)

MjcfVariant = Literal["tendons", "tendons_over_constrained"]
_MJCF_XML_PATHS: dict[MjcfVariant, Path] = {
  "tendons": KANGAROO_FULL_XML,
  "tendons_over_constrained": KANGAROO_FULL_XML_OVER_CONSTRAINED,
}

LUT_TRANSMISSION_DIR = KANGAROO_FULL_PATH.parent / "transmission"


@lru_cache(maxsize=None)
def load_transmission_maps() -> TransmissionMaps:
  return TransmissionMaps.load(LUT_TRANSMISSION_DIR)


for _path in (
  KANGAROO_FULL_XML,
  KANGAROO_FULL_XML_OVER_CONSTRAINED,
):
  assert _path.exists(), f"Missing: {_path}"
assert TransmissionMaps.available(LUT_TRANSMISSION_DIR), (
  f"Missing transmission maps in {LUT_TRANSMISSION_DIR}"
)

HIP_Z_TENDON_NAMES = ("leg_left_1_actuator", "leg_right_1_actuator")
HIP_XY_TENDON_NAMES = (
  "leg_left_2_actuator",
  "leg_left_3_actuator",
  "leg_right_3_actuator",
  "leg_right_2_actuator",
)
ANKLE_TENDON_NAMES = (
  "leg_left_4_actuator",
  "leg_left_5_actuator",
  "leg_right_5_actuator",
  "leg_right_4_actuator",
)
_KNEE_ROD_TENDON_NAMES = ("left_knee_rods", "right_knee_rods")
_LEG_LENGTH_ACTUATOR_JOINT_NAMES = (
  "leg_left_length_actuator",
  "leg_right_length_actuator",
)

_FEMUR_ROD_TENDON_NAMES = ("left_femur_rod", "right_femur_rod")
_FEMUR_LINKAGE_TENDON_NAMES = (
  "left_hip_xy_link",
  "right_hip_xy_link",
) + _FEMUR_ROD_TENDON_NAMES
_FEMUR_TRIANGLE_JOINT_NAMES = ("left_femur_triangle", "right_femur_triangle")

_BUTTERFLY_JOINT_NAMES = tuple(
  f"{side}_butterfly_{suffix}"
  for side in ("left", "right")
  for suffix in ("l", "r", "decoupler_joint")
)
_BUTTERFLY_DECOUPLER_EQ_NAMES = (
  "left_butterfly_decoupler_coupling",
  "right_butterfly_decoupler_coupling",
)
_ANKLE_TIBIA_BAR_TENDON_NAMES = tuple(
  f"{side}_ankle_tibia_bar_{suffix}"
  for side in ("left", "right")
  for suffix in ("l", "r")
)
_LEG_LENGTH_JOINT_NAMES = ("leg_left_length_joint", "leg_right_length_joint")
_LEG_LENGTH_CONNECT_EQ_NAMES = (
  "leg_left_length_connect",
  "leg_right_length_connect",
)

_ARM_BASE_LINK_NAMES = ("arm_left_base_link", "arm_right_base_link")

KANGAROO_TENDON_LENGTHS: dict[str, float] = {
  r"(left|right)_hip_xy_link": 0.09,
  r"(left|right)_knee_rods": 0.215,
  r"(left|right)_femur_rod": 0.40427,
  r"(left|right)_ankle_(femur|tibia)_bar_(l|r)": 0.38,
}

Transmission = Literal["joint", "actuator", "lut"]
FemurClosure = Literal["linkage", "prismatic"]
ActuatorModel = Literal["builtin", "dc_motor"]
LowerBody = Literal[True, False]


def get_kangaroo_full_spec(
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_MJCF_XML_PATHS[mjcf]))
  add_collision_capsules(spec)
  if lower_body:
    delete_subtrees(spec, _ARM_BASE_LINK_NAMES)
  if femur_closure == "prismatic":
    delete_tendons(spec, _FEMUR_LINKAGE_TENDON_NAMES)
    delete_joints(spec, _FEMUR_TRIANGLE_JOINT_NAMES)
  else:
    delete_equalities(spec, _LEG_LENGTH_CONNECT_EQ_NAMES)
    delete_joints(spec, _LEG_LENGTH_JOINT_NAMES)
  if transmission == "joint":
    if femur_closure == "linkage":
      raise ValueError(
        'transmission="joint" is not supported for femur_closure="linkage" '
        '(no leg_.*_length_joint to servo there) -- use "actuator" or "lut".'
      )
    delete_tendons(spec, HIP_Z_TENDON_NAMES)
    delete_tendons(spec, HIP_XY_TENDON_NAMES)
    delete_tendons(spec, _KNEE_ROD_TENDON_NAMES)
    delete_joints(spec, _LEG_LENGTH_ACTUATOR_JOINT_NAMES)
    delete_tendons(spec, ANKLE_TENDON_NAMES)
    delete_equalities(spec, _BUTTERFLY_DECOUPLER_EQ_NAMES)
    delete_joints(spec, _BUTTERFLY_JOINT_NAMES)
    delete_tendons(spec, _ANKLE_TIBIA_BAR_TENDON_NAMES)
    delete_tendons(spec, _FEMUR_ROD_TENDON_NAMES)
  return spec


def _calc_linear_leg_params(
  stiffness: float,
  effort: float,
  armature: float,
  frictionloss: float,
  viscous_damping: float,
  saturation_effort: float,
  velocity_limit: float,
) -> dict:
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
    "frictionloss": frictionloss,
    "viscous_damping": viscous_damping,
    "saturation_effort": saturation_effort,
    "velocity_limit": velocity_limit,
  }


def _calc_builtin_leg_params(**kwargs) -> dict:
  params = _calc_linear_leg_params(**kwargs)
  del params["saturation_effort"]
  del params["velocity_limit"]
  return params


def _calc_lut_joint_params(
  stiffness: float,
  effort: float,
  armature: float,
  frictionloss: float,
  viscous_damping: float,
) -> JointParams:
  params = _calc_leg_params(
    stiffness=stiffness,
    effort=effort,
    armature=armature,
    frictionloss=frictionloss,
    viscous_damping=viscous_damping,
  )
  return JointParams(
    stiffness=params["stiffness"],
    damping=params["damping"],
    effort_limit=params["effort_limit"],
  )


def _calc_lut_screw_params(
  transmission_type: TransmissionType,
  actuator_model: ActuatorModel,
  *,
  stiffness: float,
  effort: float,
  armature: float,
  frictionloss: float,
  viscous_damping: float,
  saturation_effort: float | None,
  velocity_limit: float | None,
) -> ScrewParams:
  del stiffness
  return ScrewParams(
    effort_limit=effort,
    transmission_type=transmission_type,
    armature=armature,
    frictionloss=frictionloss,
    viscous_damping=viscous_damping,
    saturation_effort=saturation_effort if actuator_model == "dc_motor" else None,
    velocity_limit=velocity_limit if actuator_model == "dc_motor" else None,
  )


LEG_LENGTH_JOINT_EXPR = r"leg_(left|right)_length_joint"
KNEE_JOINT_EXPR = r"leg_(left|right)_knee_joint"
LEG_JOINT_PD: dict[str, tuple[float, float]] = {
  r"leg_(left|right)_1_joint": (100.0, 80.0),
  r"leg_(left|right)_2_joint": (100.0, 230.0),
  r"leg_(left|right)_3_joint": (100.0, 139.0),
  r"leg_(left|right)_4_joint": (30.0, 140.0),
  r"leg_(left|right)_5_joint": (30.0, 82.0),
  LEG_LENGTH_JOINT_EXPR: (1600.0, 1100.0),
}

LegFrictionParam = tuple[float, tuple[float, float]]

LEG_JOINT_VISCOUS_DAMPING: dict[str, LegFrictionParam] = {
  r"leg_(left|right)_1_joint": (0.883277, (0.383261, 0.943999)),
  r"leg_(left|right)_2_joint": (4.11424, (3.22315, 4.19275)),
  r"leg_(left|right)_3_joint": (1.27574, (0.475789, 1.44302)),
  r"leg_(left|right)_4_joint": (1.36257, (0.693442, 1.45564)),
  r"leg_(left|right)_5_joint": (0.402166, (0.221316, 0.463352)),
  LEG_LENGTH_JOINT_EXPR: (161.39, (140.891, 165.646)),
}
LEG_JOINT_FRICTIONLOSS: dict[str, LegFrictionParam] = {
  r"leg_(left|right)_1_joint": (0.464305, (0.305846, 0.48)),
  r"leg_(left|right)_2_joint": (0.69495, (0.615122, 0.739195)),
  r"leg_(left|right)_3_joint": (0.38778, (0.173468, 0.412296)),
  r"leg_(left|right)_4_joint": (0.402496, (0.287114, 0.425607)),
  r"leg_(left|right)_5_joint": (0.218706, (0.161079, 0.23386)),
  LEG_LENGTH_JOINT_EXPR: (3.2825, (2.86559, 3.36908)),
}
LEG_JOINT_ARMATURE: dict[str, LegFrictionParam] = {
  r"leg_(left|right)_1_joint": (0.0759319, (0.0329474, 0.0811519)),
  r"leg_(left|right)_2_joint": (0.502929, (0.394001, 0.512527)),
  r"leg_(left|right)_3_joint": (0.155948, (0.058161, 0.176397)),
  r"leg_(left|right)_4_joint": (0.166562, (0.0847671, 0.177939)),
  r"leg_(left|right)_5_joint": (0.0491612, (0.0270539, 0.0566406)),
  LEG_LENGTH_JOINT_EXPR: (3.79514, (2.89231, 3.99796)),
}

LEG_1_JOINT_PARAMS = dict(
  stiffness=100.0,
  effort=80.0,
  armature=0.0759319,
  frictionloss=0.464305,
  viscous_damping=0.883277,
)
LEG_2_JOINT_PARAMS = dict(
  stiffness=100.0,
  effort=230.0,
  armature=0.502929,
  frictionloss=0.69495,
  viscous_damping=4.11424,
)
LEG_3_JOINT_PARAMS = dict(
  stiffness=100.0,
  effort=139.0,
  armature=0.155948,
  frictionloss=0.38778,
  viscous_damping=1.27574,
)
LEG_4_JOINT_PARAMS = dict(
  stiffness=30.0,
  effort=140.0,
  armature=0.166562,
  frictionloss=0.402496,
  viscous_damping=1.36257,
)
LEG_5_JOINT_PARAMS = dict(
  stiffness=30.0,
  effort=82.0,
  armature=0.0491612,
  frictionloss=0.218706,
  viscous_damping=0.402166,
)
LEG_LENGTH_JOINT_PARAMS = dict(
  stiffness=1600.0,
  effort=1100.0,
  armature=3.79514,
  frictionloss=3.2825,
  viscous_damping=161.39,
)

LEG_1_SCREW_PARAMS = dict(
  stiffness=62500.0,
  effort=2000.0,
  armature=72.07,
  frictionloss=5.0,
  viscous_damping=200.0,
  saturation_effort=4334.0,
  velocity_limit=0.314,
)
LEG_2_SCREW_PARAMS = dict(
  stiffness=28000.0,
  effort=2000.0,
  armature=72.07,
  frictionloss=5.0,
  viscous_damping=200.0,
  saturation_effort=4334.0,
  velocity_limit=0.314,
)
LEG_3_SCREW_PARAMS = dict(
  stiffness=28000.0,
  effort=2000.0,
  armature=72.07,
  frictionloss=5.0,
  viscous_damping=200.0,
  saturation_effort=4334.0,
  velocity_limit=0.314,
)
LEG_4_SCREW_PARAMS = dict(
  stiffness=25000.0,
  effort=2000.0,
  armature=72.09,
  frictionloss=5.0,
  viscous_damping=200.0,
  saturation_effort=4334.0,
  velocity_limit=0.314,
)
LEG_5_SCREW_PARAMS = dict(
  stiffness=25000.0,
  effort=2000.0,
  armature=72.09,
  frictionloss=5.0,
  viscous_damping=200.0,
  saturation_effort=4334.0,
  velocity_limit=0.314,
)
LEG_LENGTH_SCREW_PARAMS = dict(
  stiffness=23000.0,
  effort=5000.0,
  armature=50.72,
  frictionloss=11.7333,
  viscous_damping=589.4603,
  saturation_effort=10443.0,
  velocity_limit=0.288,
)

LEG_SCREW_PARAMS: dict[str, dict] = {
  "leg_right_1_actuator": LEG_1_SCREW_PARAMS,
  "leg_right_2_actuator": LEG_2_SCREW_PARAMS,
  "leg_right_3_actuator": LEG_3_SCREW_PARAMS,
  "leg_right_4_actuator": LEG_4_SCREW_PARAMS,
  "leg_right_5_actuator": LEG_5_SCREW_PARAMS,
  "leg_right_length_actuator": LEG_LENGTH_SCREW_PARAMS,
}


def joint_pd_actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
  return (
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_1_joint",),
      **_calc_leg_params(**LEG_1_JOINT_PARAMS),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_2_joint",),
      **_calc_leg_params(**LEG_2_JOINT_PARAMS),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_3_joint",),
      **_calc_leg_params(**LEG_3_JOINT_PARAMS),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_4_joint",),
      **_calc_leg_params(**LEG_4_JOINT_PARAMS),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(r"leg_(left|right)_5_joint",),
      **_calc_leg_params(**LEG_5_JOINT_PARAMS),
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(LEG_LENGTH_JOINT_EXPR,),
      **_calc_leg_params(**LEG_LENGTH_JOINT_PARAMS),
    ),
  )


def actuator_pd_actuators(
  actuator_model: ActuatorModel,
) -> tuple[BuiltinPositionActuatorCfg | DcMotorActuatorCfg, ...]:
  actuator_cfg = (
    BuiltinPositionActuatorCfg if actuator_model == "builtin" else DcMotorActuatorCfg
  )
  calc_params = (
    _calc_builtin_leg_params if actuator_model == "builtin" else _calc_linear_leg_params
  )
  return (
    actuator_cfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"leg_(left|right)_1_actuator$",),
      **calc_params(**LEG_1_SCREW_PARAMS),
    ),
    actuator_cfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"leg_(left|right)_2_actuator$",),
      **calc_params(**LEG_2_SCREW_PARAMS),
    ),
    actuator_cfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"leg_(left|right)_3_actuator$",),
      **calc_params(**LEG_3_SCREW_PARAMS),
    ),
    actuator_cfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"leg_(left|right)_4_actuator$",),
      **calc_params(**LEG_4_SCREW_PARAMS),
    ),
    actuator_cfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r"leg_(left|right)_5_actuator$",),
      **calc_params(**LEG_5_SCREW_PARAMS),
    ),
    actuator_cfg(
      target_names_expr=(r"leg_(left|right)_length_actuator$",),
      **calc_params(**LEG_LENGTH_SCREW_PARAMS),
    ),
  )


def screw_actuator_cfg(
  target_names_expr: tuple[str, ...],
  map_actuator: str,
  actuator_model: ActuatorModel,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> ActuatorCfg:
  actuator_cfg = (
    BuiltinPositionActuatorCfg if actuator_model == "builtin" else DcMotorActuatorCfg
  )
  calc_params = (
    _calc_builtin_leg_params if actuator_model == "builtin" else _calc_linear_leg_params
  )
  return actuator_cfg(
    transmission_type=transmission_type,
    target_names_expr=target_names_expr,
    **calc_params(**LEG_SCREW_PARAMS[map_actuator]),
  )


def lut_actuator(
  actuator_model: ActuatorModel,
  screw_transmission_type: TransmissionType = TransmissionType.JOINT,
) -> LutTransmissionActuatorCfg:
  screw = partial(_calc_lut_screw_params, screw_transmission_type, actuator_model)
  knee_screw = partial(_calc_lut_screw_params, TransmissionType.JOINT, actuator_model)
  return LutTransmissionActuatorCfg(
    target_names_expr=(
      r"leg_(left|right)_1_joint",
      r"leg_(left|right)_2_joint",
      r"leg_(left|right)_3_joint",
      r"leg_(left|right)_4_joint",
      r"leg_(left|right)_5_joint",
      KNEE_JOINT_EXPR,
    ),
    transmission_type=TransmissionType.JOINT,
    maps=load_transmission_maps(),
    mechanisms={
      "hip_z": LutMechanismCfg(
        source_joint_names=("leg_right_1_joint",),
        target_actuator_names=("leg_right_1_actuator",),
      ),
      "hip_xy": LutMechanismCfg(
        source_joint_names=("leg_right_2_joint", "leg_right_3_joint"),
        target_actuator_names=("leg_right_2_actuator", "leg_right_3_actuator"),
      ),
      "ankle": LutMechanismCfg(
        source_joint_names=("leg_right_4_joint", "leg_right_5_joint"),
        target_actuator_names=("leg_right_4_actuator", "leg_right_5_actuator"),
        context_joint_names=("leg_right_knee_joint",),
      ),
      "leg_length": LutMechanismCfg(
        # The policy commands virtual leg_right_length_joint metres; this
        # knee source is mapped to that coordinate before applying the PD.
        source_joint_names=("leg_right_knee_joint",),
        target_actuator_names=("leg_right_length_actuator",),
      ),
    },
    sides=("left", "right"),
    reference_side="right",
    joint_sign={"leg_left_1_joint": -1.0},
    joint_params={
      r"leg_(left|right)_1_joint": _calc_lut_joint_params(**LEG_1_JOINT_PARAMS),
      r"leg_(left|right)_2_joint": _calc_lut_joint_params(**LEG_2_JOINT_PARAMS),
      r"leg_(left|right)_3_joint": _calc_lut_joint_params(**LEG_3_JOINT_PARAMS),
      r"leg_(left|right)_4_joint": _calc_lut_joint_params(**LEG_4_JOINT_PARAMS),
      r"leg_(left|right)_5_joint": _calc_lut_joint_params(**LEG_5_JOINT_PARAMS),
      KNEE_JOINT_EXPR: _calc_lut_joint_params(**LEG_LENGTH_JOINT_PARAMS),
    },
    screw_params={
      r"leg_(left|right)_1_actuator": screw(**LEG_1_SCREW_PARAMS),
      r"leg_(left|right)_2_actuator": screw(**LEG_2_SCREW_PARAMS),
      r"leg_(left|right)_3_actuator": screw(**LEG_3_SCREW_PARAMS),
      r"leg_(left|right)_4_actuator": screw(**LEG_4_SCREW_PARAMS),
      r"leg_(left|right)_5_actuator": screw(**LEG_5_SCREW_PARAMS),
      r"leg_(left|right)_length_actuator": knee_screw(**LEG_LENGTH_SCREW_PARAMS),
    },
  )


def _leg_actuators(
  transmission: Transmission,
  femur_closure: FemurClosure,
  actuator_model: ActuatorModel,
) -> tuple[ActuatorCfg, ...]:
  if transmission == "joint":
    if femur_closure == "linkage":
      raise ValueError(
        'transmission="joint" is not supported for femur_closure="linkage" '
        '(no leg_.*_length_joint to servo there) -- use "actuator" or "lut".'
      )
    return joint_pd_actuators()
  if transmission == "actuator":
    return actuator_pd_actuators(actuator_model)
  if transmission == "lut":
    return (lut_actuator(actuator_model, TransmissionType.TENDON),)
  raise ValueError(f"unknown transmission {transmission!r}")


_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
_LOWER_BODY_UPPER_BODY_ACTUATORS = (KANGAROO_PELVIS_ACTUATOR_CFG,)

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


def _build_action_scales(
  actuators: tuple[ActuatorCfg, ...],
  transmission_type: TransmissionType,
  exclude: frozenset[str] = frozenset(),
) -> tuple[dict[str, float], tuple[str, ...]]:
  scales: dict[str, float] = {}
  names: list[str] = []
  for actuator in actuators:
    if actuator.transmission_type != transmission_type:
      continue
    if isinstance(actuator, LutTransmissionActuatorCfg):
      for name in actuator.target_names_expr:
        if name in exclude:
          continue
        params = actuator.joint_params.get(name)
        if params is not None and params.stiffness:
          scales[name] = 0.25 * params.effort_limit / params.stiffness
          names.append(name)
      continue
    effort_limit, stiffness = actuator.effort_limit, actuator.stiffness
    for name in actuator.target_names_expr:
      if name in exclude:
        continue
      efforts = effort_limit if isinstance(effort_limit, dict) else {name: effort_limit}
      stiffnesses = stiffness if isinstance(stiffness, dict) else {name: stiffness}
      if name in efforts and stiffnesses.get(name):
        scales[name] = 0.25 * efforts[name] / stiffnesses[name]
        names.append(name)
  return scales, tuple(names)


def simple_model_action_names(
  joint_order: tuple[str, ...],
  joint_actuator_names: tuple[str, ...],
  leg_length_action: dict,
  leg_length_from_knee_joints: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
  knee_of_length = dict(leg_length_from_knee_joints)
  exprs = tuple(joint_actuator_names)
  knee_exprs = tuple(leg_length_action["actuator_names"])
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


def lut_leg_length_action_scale(
  leg_actuators: tuple[ActuatorCfg, ...],
) -> tuple[dict[str, float], tuple[str, ...]]:
  for actuator in leg_actuators:
    if isinstance(actuator, LutTransmissionActuatorCfg):
      params = _calc_leg_params(**LEG_LENGTH_JOINT_PARAMS)
      scale = 0.25 * params["effort_limit"] / params["stiffness"]
      expr = actuator.leg_length_servo_expr
      return {expr: scale}, (expr,)
  return {}, ()


@lru_cache(maxsize=None)
def get_kangaroo_full_model(
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
) -> dict:
  leg_actuators = _leg_actuators(transmission, femur_closure, actuator_model)
  upper_body_actuators = (
    _LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS
  )
  articulation = EntityArticulationInfoCfg(
    actuators=leg_actuators + upper_body_actuators,
    soft_joint_pos_limit_factor=0.99,
  )

  leg_length_scale, leg_length_names = lut_leg_length_action_scale(leg_actuators)
  joint_action_scale, joint_actuator_names = _build_action_scales(
    leg_actuators + upper_body_actuators,
    TransmissionType.JOINT,
    exclude=frozenset(leg_length_names),
  )
  # Only the leg mechanisms are ever tendon driven.
  tendon_scale, _ = _build_action_scales(leg_actuators, TransmissionType.TENDON)

  spec = get_kangaroo_full_spec(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
  )
  has_joint_equalities = any(
    eq.type == mujoco.mjtEq.mjEQ_JOINT for eq in spec.equalities
  )

  tendon_names = (
    HIP_Z_TENDON_NAMES + HIP_XY_TENDON_NAMES + ANKLE_TENDON_NAMES
    if transmission == "actuator"
    else ()
  )
  offsets = (
    _compute_tendon_lengths_at_init_state(spec, tendon_names, INIT_STATE.joint_pos)
    if tendon_names
    else {}
  )

  def _tendon_action(names: tuple[str, ...]) -> dict | None:
    if transmission != "actuator":
      return None
    return {
      "actuator_names": tuple(f"{name}$" for name in names),
      "scale": {
        expr: v
        for expr, v in tendon_scale.items()
        if any(re.fullmatch(expr, name) for name in names)
      },
      "offset": {name: offsets[name] for name in names},
    }

  return {
    "robot_cfg": EntityCfg(
      init_state=INIT_STATE,
      collisions=(FULL_COLLISION,),
      spec_fn=partial(
        get_kangaroo_full_spec,
        transmission=transmission,
        femur_closure=femur_closure,
        mjcf=mjcf,
        lower_body=lower_body,
      ),
      articulation=articulation,
    ),
    "lower_body": lower_body,
    "articulation": articulation,
    "joint_action_scale": joint_action_scale,
    "joint_actuator_names": joint_actuator_names,
    "hip_z_tendon_action": _tendon_action(HIP_Z_TENDON_NAMES),
    "hip_xy_tendon_action": _tendon_action(HIP_XY_TENDON_NAMES),
    "ankle_tendon_action": _tendon_action(ANKLE_TENDON_NAMES),
    "leg_length_action": (
      {"actuator_names": leg_length_names, "scale": leg_length_scale}
      if leg_length_names
      else None
    ),
    "has_knee_rod_tendons": transmission != "joint",
    "has_femur_linkage_tendons": femur_closure == "linkage",
    "has_butterfly_decoupler_coupling": transmission != "joint",
    "has_ankle_tibia_bar_tendons": transmission != "joint",
    "has_joint_equalities": has_joint_equalities,
    "has_leg_length_joint": femur_closure == "prismatic",
  }


def _pin_equality_tendon_lengths(model: mujoco.MjModel) -> None:
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
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
  launch_viewer: bool = True,
) -> None:
  model_cfg = get_kangaroo_full_model(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    actuator_model=actuator_model,
    lower_body=lower_body,
  )

  entity = Entity(model_cfg["robot_cfg"])
  spec = entity.spec
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  _pin_equality_tendon_lengths(model)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  mujoco.mj_forward(model, data)

  if launch_viewer:
    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
  import mjlab
  import tyro

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
