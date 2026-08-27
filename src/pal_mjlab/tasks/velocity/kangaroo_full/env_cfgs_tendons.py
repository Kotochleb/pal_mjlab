"""PAL Robotics kangaroo_full velocity tracking environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots import (
  REGEX_ALL_OBSERVABLE_JOINTS,
  KANGAROO_TENDON_LENGTHS,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  KANGAROO_FULL_TENDON_HIPS_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES,
  KANGAROO_FULL_TENDON_HIPS_TENDON_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES,
  KANGAROO_TENDON_OFFSETS,
  get_kangaroo_full_robot_tendon_hips_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths


def pal_kangaroo_full_tendons_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  cfg.scene.entities = {"robot": get_kangaroo_full_robot_tendon_hips_cfg()}

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES,
      scale=KANGAROO_FULL_TENDON_HIPS_JOINT_ACTION_SCALE,
      use_default_offset=True,
    ),
    "tendon_pos": TendonLengthActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES,
      scale=KANGAROO_FULL_TENDON_HIPS_TENDON_ACTION_SCALE,
      offset=KANGAROO_TENDON_OFFSETS,
    ),
  }

  # -- Observations

  for obs in ["actor", "critic"]:
    for term in ["joint_pos", "joint_vel"]:
      cfg.observations[obs].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=REGEX_ALL_OBSERVABLE_JOINTS,
        tendon_names=KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES,
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
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
      )
    },
  )

  simple_model_joints = REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (simple_model_joints,)
  cfg.rewards["pose"].params["std_standing"] = {simple_model_joints: 0.05}

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


def pal_kangaroo_full_tendons_flat_env_cfg(
  play: bool = False,
  joint_state_obs: str = "full_state",
  pose_in_actuator_space: bool = False,
  pd_mapping: str = "jacobian",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_tendons_rough_env_cfg(play=play)

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
