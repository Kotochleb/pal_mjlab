from __future__ import annotations

from functools import lru_cache, partial
from typing import Literal

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.actuator.actuator import TransmissionType
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
  ActuatorModel,
  LowerBody,
  _build_action_scales,
  lut_actuator,
  lut_leg_length_action_scale,
  screw_actuator_cfg,
)
from pal_mjlab.robots.pal_kangaroo_full.spec_edits import (
  add_collision_capsules,
  delete_subtrees,
)

Transmission = Literal["actuator", "lut"]
REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = (
  r"^(?!leg_.*_(?:[1-5]|length)_actuator$)(pelvis|arm|leg)_.*$"
)
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_(?:[1-5]|length)_actuator$)(pelvis|arm|leg)_.*$"

KANGAROO_FULL_FULL_PATH = (
  PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full_full" / "xmls"
)
KANGAROO_FULL_FULL_XML = KANGAROO_FULL_FULL_PATH / "kangaroo_full.xml"
assert KANGAROO_FULL_FULL_XML.exists(), f"Missing: {KANGAROO_FULL_FULL_XML}"


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
    "(leg_left_3_actuator|leg_right_2_actuator)": 0.0016356406,
    ".*_hip_xy_bracket_r": 0.0002925520,
    "(left_hip_xy_motor_r|right_hip_xy_motor_l)": 0.0074219151,
    "(leg_left_2_actuator|leg_right_3_actuator)": 0.0043910614,
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


def get_kangaroo_full_full_spec(lower_body: LowerBody = False) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(KANGAROO_FULL_FULL_XML))
  add_collision_capsules(spec)
  if lower_body:
    delete_subtrees(spec, _ARM_BASE_LINK_NAMES)
  return spec

_SLIDER_ACTUATOR_TARGETS: tuple[tuple[str, str], ...] = (
  (r"leg_(left|right)_1_actuator$", "leg_right_1_actuator"),
  (r"leg_(left|right)_[23]_actuator$", "leg_right_2_actuator"),
  (r"leg_(left|right)_[45]_actuator$", "leg_right_4_actuator"),
)

_SLIDER_ACTUATOR_KEYS = frozenset(expr for expr, _ in _SLIDER_ACTUATOR_TARGETS)


def _actuator_transmission_actuators(
  actuator_model: ActuatorModel,
) -> tuple[ActuatorCfg, ...]:
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
  if transmission == "actuator":
    return _actuator_transmission_actuators(actuator_model)
  if transmission == "lut":
    return (lut_actuator(actuator_model),)
  raise ValueError(f"unknown transmission {transmission!r}")


_UPPER_BODY_ACTUATORS = (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
_LOWER_BODY_UPPER_BODY_ACTUATORS = (KANGAROO_PELVIS_ACTUATOR_CFG,)



@lru_cache(maxsize=None)
def get_kangaroo_full_full_model(
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
) -> dict:
  leg_actuators = _leg_actuators(transmission, actuator_model)
  upper_body_actuators = (
    _LOWER_BODY_UPPER_BODY_ACTUATORS if lower_body else _UPPER_BODY_ACTUATORS
  )
  articulation = EntityArticulationInfoCfg(
    actuators=leg_actuators + upper_body_actuators,
    soft_joint_pos_limit_factor=0.99,
  )

  leg_length_scale, leg_length_names = lut_leg_length_action_scale(leg_actuators)
  scale, names = _build_action_scales(
    leg_actuators + upper_body_actuators,
    TransmissionType.JOINT,
    exclude=frozenset(leg_length_names),
  )
  slider_scale = {name: scale[name] for name in names if name in _SLIDER_ACTUATOR_KEYS}
  joint_action_scale = {
    name: scale[name] for name in names if name not in _SLIDER_ACTUATOR_KEYS
  }
  joint_actuator_names = tuple(name for name in names if name not in _SLIDER_ACTUATOR_KEYS)

  def _slider_action(names: tuple[str, ...], key: str) -> dict | None:
    if transmission != "actuator":
      return None
    return {
      "actuator_names": tuple(f"{name}$" for name in names),
      "scale": {k: v for k, v in slider_scale.items() if key in k},
    }

  return {
    "robot_cfg": EntityCfg(
      init_state=INIT_STATE,
      collisions=(FULL_COLLISION,),
      spec_fn=partial(get_kangaroo_full_full_spec, lower_body=lower_body),
      articulation=articulation,
    ),
    "lower_body": lower_body,
    "articulation": articulation,
    "joint_action_scale": joint_action_scale,
    "joint_actuator_names": joint_actuator_names,
    "hip_z_slider_action": _slider_action(HIP_Z_SLIDER_JOINT_NAMES, "_1_"),
    "hip_xy_slider_action": _slider_action(HIP_XY_SLIDER_JOINT_NAMES, "[23]"),
    "ankle_slider_action": _slider_action(ANKLE_SLIDER_JOINT_NAMES, "[45]"),
    "leg_length_action": (
      {"actuator_names": leg_length_names, "scale": leg_length_scale}
      if leg_length_names
      else None
    ),
  }


def main(
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  actuator_model: ActuatorModel = "builtin",
  launch_viewer: bool = True,
) -> None:
  model_cfg = get_kangaroo_full_full_model(
    transmission=transmission, lower_body=lower_body, actuator_model=actuator_model
  )

  entity = Entity(model_cfg["robot_cfg"])
  spec = entity.spec
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
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
