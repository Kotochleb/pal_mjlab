"""Pal Robotics KANGAROO FULL constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  FEET_ONLY_COLLISION,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  _build_action_scales,
  _calc_leg_params,
)

KANGAROO_FULL_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls"
KANGAROO_FULL_XML = KANGAROO_FULL_PATH / "kangaroo_full.xml"

for p in [
  KANGAROO_FULL_XML,
]:
  assert p.exists(), f"Missing: {p}"


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_kangaroo_full_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_FULL_XML)


##
# Actuator config.
##


KANGAROO_LEG_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_1_actuator$",),
    **_calc_leg_params(16000.0, 2000.0),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_2_actuator$",),
    **_calc_leg_params(16000.0, 2000.0),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_3_actuator$",),
    **_calc_leg_params(16000.0, 2000.0),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_4_actuator$",),
    **_calc_leg_params(16000.0, 2000.0),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_5_actuator$",),
    **_calc_leg_params(16000.0, 2000.0),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    **_calc_leg_params(200000.0, 5000.0),
  ),
)

COMMON_ACTUATORS = KANGAROO_LEG_ACTUATORS + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9053),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
    "left_hip_z_motor$": -0.000304605,
    "leg_left_1_actuator$": 0.000479008,
    "left_hip_z$": 0.0120757,
    "left_hip_xy_bracket_l$": -1.29227e-05,
    "left_hip_xy_motor_l$": -0.00517098,
    "leg_left_2_actuator$": 0.00314592,
    "left_hip_xy_bracket_r$": 1.27471e-05,
    "left_hip_xy_motor_r$": 0.00627606,
    "leg_left_3_actuator$": 0.00374401,
    "left_hip_xy_cross$": 0.0585147,
    "left_hip_xy$": -0.00839706,
    "left_hip_xy_cross_l$": 0.0150156,
    "left_hip_xy_cross_r$": -0.0150156,
    "left_ankle_motor_l$": 0.000800198,
    "leg_left_4_actuator$": 0.00255203,
    "left_ankle_motor_r$": 0.000741899,
    "leg_left_5_actuator$": 0.00255641,
    "left_ankle_crank_l$": -0.07332,
    "left_ankle_femur_bar_l$": 0.227459,
    "left_ankle_crank_r$": -0.0731852,
    "left_ankle_femur_bar_r$": -0.227596,
    "left_hip_xy_link$": 0.0835211,
    "left_femur$": -0.300511,
    "left_femur_triangle$": -0.608748,
    "left_femur_rod$": -0.608755,
    "left_butterfly_l$": 0.226768,
    "left_ankle_tibia_bar_l1$": 0.382068,
    "left_ankle_tibia_bar_l2$": 2.63367e-08,
    "left_butterfly_r$": 0.226902,
    "left_ankle_tibia_bar_r1$": -0.381934,
    "left_ankle_tibia_bar_r2$": -3.79657e-08,
    "leg_left_length_actuator$": 0.0283415,
    "left_knee_rods$": -0.211021,
    "left_tibia$": 0.608746,
    "leg_left_4_joint$": -0.381506,
    "leg_left_5_joint$": 0.000119179,
    "right_hip_z_motor$": 0.000304605,
    "leg_right_1_actuator$": 0.000479009,
    "right_hip_z$": -0.0120757,
    "right_hip_xy_bracket_r$": 1.32128e-05,
    "right_hip_xy_motor_r$": -0.00515681,
    "leg_right_3_actuator$": 0.00313853,
    "right_hip_xy_bracket_l$": -1.32281e-05,
    "right_hip_xy_motor_l$": 0.00627811,
    "leg_right_2_actuator$": 0.00374516,
    "right_hip_xy_cross$": 0.0584611,
    "right_hip_xy$": 0.00852229,
    "right_hip_xy_cross_r$": 9.45083e-08,
    "right_hip_xy_cross_l$": -0.0144655,
    "right_ankle_motor_l$": 0.000799322,
    "leg_right_4_actuator$": 0.00255109,
    "right_ankle_motor_r$": 0.000742914,
    "leg_right_5_actuator$": 0.00255753,
    "right_ankle_crank_l$": -0.0732876,
    "right_ankle_femur_bar_l$": 0.227492,
    "right_ankle_crank_r$": -0.0732228,
    "right_ankle_femur_bar_r$": -0.227558,
    "right_hip_xy_link$": 0.0835211,
    "right_femur$": -0.300511,
    "right_femur_triangle$": -0.608748,
    "right_femur_rod$": -0.608755,
    "right_butterfly_l$": 0.2268,
    "right_ankle_tibia_bar_l1$": 0.382036,
    "right_ankle_tibia_bar_l2$": 3.19145e-08,
    "right_butterfly_r$": 0.226864,
    "right_ankle_tibia_bar_r1$": -0.381972,
    "right_ankle_tibia_bar_r2$": -3.34347e-08,
    "leg_right_length_actuator$": 0.0283415,
    "right_knee_rods$": -0.211021,
    "right_tibia$": 0.608746,
    "leg_right_4_joint$": -0.381509,
    "leg_right_5_joint$": 5.71018e-05,
  },
  joint_vel={".*": 0.0},
)


KANGAROO_FULL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS),
  soft_joint_pos_limit_factor=0.99,
)


_ROBOT_CONFIGS = {
  "kangaroo_full": (
    get_kangaroo_full_spec,
    KANGAROO_FULL_ARTICULATION,
    FEET_ONLY_COLLISION,
  ),
}


def _make_robot_cfg(variant: str) -> EntityCfg:
  spec_fn, articulation, collision = _ROBOT_CONFIGS[variant]
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
  )


def get_kangaroo_full_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full")


##
# Final config.
##

KANGAROO_FULL_ACTION_SCALE, KANGAROO_FULL_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_FULL_ARTICULATION
)


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_full_robot_cfg())
  viewer.launch(robot.spec.compile())
