"""PAL Robotics kangaroo_full velocity tracking environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots import (
  REGEX_SIMPLE_MODEL_JOINTS_ONLY,
  REGEX_ACTUATED_JOINTS_ONLY,
  REGEX_ALL_OBSERVABLE_JOINTS,
  REGEX_POSE_REVEOLUTE_JOINTS_ONLY,
  KANGAROO_FULL_JOINT_ACTION_SCALE,
  KANGAROO_FULL_ACTUATED_JOINTS_NAMES,
  KANGAROO_TENDON_LENGTHS,
  KANGAROO_INIT_STATE_SIMPLE_TO_FULL_JACOBIAN,
  KANGAROO_FULL_JOINT_ACTION_SCALE_LOW,
  get_kangaroo_full_robot_cfg,
  get_kangaroo_full_robot_low_pd_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths


def pal_kangaroo_full_rough_env_cfg(
  play: bool = False,
  joint_state_obs: str = "full_state",
  pose_in_actuator_space: bool = False,
  pd_mapping: str = "jacobian",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  if pd_mapping == "jacobian":
    cfg.scene.entities = {"robot": get_kangaroo_full_robot_cfg()}
    action_scale = KANGAROO_FULL_JOINT_ACTION_SCALE
  elif pd_mapping == "lowest":
    cfg.scene.entities = {"robot": get_kangaroo_full_robot_low_pd_cfg()}
    action_scale = KANGAROO_FULL_JOINT_ACTION_SCALE_LOW

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_ACTUATED_JOINTS_NAMES,
      scale=action_scale,
      use_default_offset=True,
    ),
    # "tendon_pos": TendonLengthActionCfg(
    #   entity_name="robot",
    #   actuator_names=KANGAROO_FULL_ACTUATED_TENDONS_NAMES,
    #   scale=KANGAROO_FULL_TENDON_ACTION_SCALE,
    #   offset=KANGAROO_TENDON_OFFSETS,
    # ),
  }

  # -- Observations

  observarion_space = {
    "simple_model": REGEX_SIMPLE_MODEL_JOINTS_ONLY,
    "actuator_space": REGEX_ACTUATED_JOINTS_ONLY,
    "full_state": REGEX_ALL_OBSERVABLE_JOINTS,
  }

  for obs in ["actor", "critic"]:
    for term in ["joint_pos", "joint_vel"]:
      cfg.observations[obs].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=observarion_space[joint_state_obs],
        # tendon_names=KANGAROO_FULL_ACTUATED_TENDONS_NAMES,
      )

  # -- Events

  cfg.events["tendon_lengths"] = EventTermCfg(
    mode="reset",
    func=enforce_tendon_lengths,
    params={"lengths": KANGAROO_TENDON_LENGTHS},
  )
  del cfg.events["encoder_bias"]
  del cfg.events["leg_length_encoder_bias"]

  # -- Rewards

  cfg.rewards["dof_pos_limits"] = RewardTermCfg(
    func=mdp.joint_pos_limits,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=REGEX_SIMPLE_MODEL_JOINTS_ONLY)
    },
  )

  if pose_in_actuator_space:
    simple_model_joints = (
      REGEX_SIMPLE_MODEL_JOINTS_ONLY  # Exclude femur and knee joints.
    )
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (simple_model_joints,)
    cfg.rewards["pose"].params["std_standing"] = {simple_model_joints: 0.05}
  else:
    leg_length_J = KANGAROO_INIT_STATE_SIMPLE_TO_FULL_JACOBIAN[
      r"leg_.*_length_actuator"
    ]
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (REGEX_ACTUATED_JOINTS_ONLY,)
    cfg.rewards["pose"].params["std_standing"] = {
      REGEX_POSE_REVEOLUTE_JOINTS_ONLY: 0.05,
      r"leg_.*_length_actuator": 0.05 * math.sqrt(leg_length_J),
    }
    # Remap simple model std to sull model std
    for param in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[param][r"leg_.*_length_actuator"] = cfg.rewards[
        "pose"
      ].params[param][r"leg_.*_length_.*"] * math.sqrt(leg_length_J)
      del cfg.rewards["pose"].params[param][r"leg_.*_length_.*"]

  # -- Metrics

  cfg.metrics["knee_rods_eq_mean_violation"] = MetricsTermCfg(
    func=mdp.tendon_equality_constraint_violation,
    params={
      "asset_cfg": SceneEntityCfg("robot", tendon_names=(r"(left|right)_knee_rods",)),
      "mode": "violation",
      "reduction": "mean",
    },
  )
  cfg.metrics["knee_rods_eq_max_violation"] = MetricsTermCfg(
    func=mdp.tendon_equality_constraint_violation,
    params={
      "asset_cfg": SceneEntityCfg("robot", tendon_names=(r"(left|right)_knee_rods",)),
      "mode": "violation",
      "reduction": "max",
    },
  )

  return cfg


def pal_kangaroo_full_flat_env_cfg(
  play: bool = False,
  joint_state_obs: str = "full_state",
  pose_in_actuator_space: bool = False,
  pd_mapping: str = "jacobian",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_rough_env_cfg(
    play=play,
    joint_state_obs=joint_state_obs,
    pose_in_actuator_space=pose_in_actuator_space,
  )

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
