"""PAL Robotics kangaroo_full velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (  # noqa: F401
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANG_FULL_ACTION_SCALE,
  KANG_FULL_ACTUATOR_NAMES,
  get_kangaroo_full_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp as mdp_kgr_full


def pal_kangaroo_full_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  _LEG_ACTUATOR_RE = (
    r"leg_(left|right)_[1-5]_actuator$|leg_(left|right)_length_actuator$"
  )
  _ACTUATED_JOINT_RE = (
    _LEG_ACTUATOR_RE + r"|pelvis_1_joint$|pelvis_2_joint$|arm_.*_[1-4]_joint$"
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANG_FULL_ACTION_SCALE
  joint_pos_action.actuator_names = KANG_FULL_ACTUATOR_NAMES

  # -- Observations
  #
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=KANG_FULL_ACTUATOR_NAMES
  )
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=KANG_FULL_ACTUATOR_NAMES
  )

  # -- Events

  cfg.events["encoder_bias"].params["asset_cfg"].joint_names = [
    r"^(?!leg_(left|right)_length_actuator$).*"
  ]
  cfg.events["leg_length_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=[r"leg_(left|right)_length_actuator$"]
      ),
      "bias_range": (-0.005, 0.005),
    },
  )

  # -- Rewards

  cfg.rewards["pose"].params["asset_cfg"].joint_names = (_ACTUATED_JOINT_RE,)
  cfg.rewards["pose"].params["std_standing"] = {_ACTUATED_JOINT_RE: 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body. 1 = hip yaw, 2/3 = hip xy, 4/5 = ankle xy.
    r"leg_(left|right)_1_actuator$": 0.01,
    r"leg_(left|right)_2_actuator$": 0.01,
    r"leg_(left|right)_3_actuator$": 0.01,
    r"leg_(left|right)_length_actuator$": 0.05,
    r"leg_(left|right)_4_actuator$": 0.01,
    r"leg_(left|right)_5_actuator$": 0.01,
    # Waist.
    r"pelvis_1.*": 0.08,
    r"pelvis_2.*": 0.2,
    # Arms.
    r"arm_.*_1_.*": 0.2,  # pitch
    r"arm_.*_4_.*": 0.2,  # elbow
    r"arm_.*_(?![14]_joint)\d+_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body. 1 = hip yaw, 2/3 = hip xy, 4/5 = ankle xy.
    r"leg_(left|right)_1_actuator$": 0.015,
    r"leg_(left|right)_2_actuator$": 0.015,
    r"leg_(left|right)_3_actuator$": 0.015,
    r"leg_(left|right)_length_actuator$": 0.08,
    r"leg_(left|right)_4_actuator$": 0.015,
    r"leg_(left|right)_5_actuator$": 0.015,
    # Waist.
    r"pelvis_1.*": 0.08,
    r"pelvis_2.*": 0.3,
    # Arms.
    r"arm_.*_1_.*": 0.4,
    r"arm_.*_4_.*": 0.35,
    r"arm_.*_(?![14]_joint)\d+_joint": 0.15,
  }

  # pal_kangaroo penalises leg-length velocity with mjlab's linear joint_vel_limits
  # at -10.0 and a +/-1.6 limit. That limit is in its own leg-length coordinate,
  # which spans 0.582 m against 0.151 m of ball-screw travel here, so it does not
  # carry over. This term is the full model's own: quadratic past limits already
  # expressed in screw units, over all six leg actuators.
  cfg.rewards["joint_velocity_limit"] = RewardTermCfg(
    func=mdp_kgr_full.joint_vel_limit,
    weight=-0.02,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES)},
  )

  # The hull points should correspond to the respective joints defined in the joint_names_group order
  # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip roll
  cfg.rewards["convex_hull_joint_limits_hip"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "metrics_suffix": "hipXY",
      # Confirmed against pal_kangaroo by kinematics, not just joint axes: the
      # chain base -> femur is hip_z -> hip_xy_cross -> hip_xy here against
      # leg_*_1/2/3_joint there, and driving each of the last two moves the leg
      # the same way -- except roll, which is inverted. That flip is corrected in
      # HIP_XY_CONVEX_HULL_POINTS, not here.
      "joint_names_group": [
        [r"left_hip_xy_cross$", r"left_hip_xy$"],
        [r"right_hip_xy_cross$", r"right_hip_xy$"],
      ],
      "margin": 0.02,
      "hull_points": HIP_XY_CONVEX_HULL_POINTS,
    },
  )

  cfg.rewards["convex_hull_joint_limits_ankle"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "margin": 0.02,
      "metrics_suffix": "ankleXY",
      # Same joint names, axes and ranges as pal_kangaroo, and driving each moves
      # the foot the same way, so ANKLE_XY_CONVEX_HULL_POINTS transfers as-is.
      "joint_names_group": [
        [r"leg_left_4_joint$", r"leg_left_5_joint$"],
        [r"leg_right_4_joint$", r"leg_right_5_joint$"],
      ],
      "hull_points": ANKLE_XY_CONVEX_HULL_POINTS,
    },
  )

  cfg.rewards["electrical_power_cost"] = RewardTermCfg(
    func=mdp.electrical_power_cost,
    weight=0.0,  # To be defined
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES),
    },
  )

  return cfg


def pal_kangaroo_full_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
