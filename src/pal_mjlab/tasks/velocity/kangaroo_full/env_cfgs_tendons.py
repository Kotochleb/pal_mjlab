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
  KANGAROO_FULL_TENDON_HIPS_CL_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_ACTUATED_JOINTS_NAMES,
  KANGAROO_FULL_TENDON_HIPS_CL_MAIN_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_MAIN_ACTUATED_JOINTS_NAMES,
  KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_ACTUATED_JOINTS_NAMES,
  KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_ACTUATED_JOINTS_NAMES,
  KANGAROO_FULL_TENDON_HIPS_HIPZ_TENDON_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_HIPXY_TENDON_ACTION_SCALE,
  KANGAROO_INIT_STATE_HIPZ_TENDONS_OFFSETS,
  KANGAROO_INIT_STATE_HIPXY_TENDONS_OFFSETS,
  get_kangaroo_full_robot_tendon_hips_cfg,
  get_kangaroo_full_robot_tendon_hips_cl_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths


def pal_kangaroo_full_tendons_rough_env_cfg(
  play: bool = False,
  joint_state_obs: str = "full_state",
) -> ManagerBasedRlEnvCfg:
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
      # Explicit (left, right) order + preserve_order so this term's target
      # layout matches the CL variant's "hip_actuator_pos" term (whose JOINT
      # transmission always resolves in body-tree order, left leg first, and
      # has no preserve_order knob to override) -- keeps hip_z at the same
      # action index in both variants.
      actuator_names=(r"left_hip_z_slider$", r"right_hip_z_slider$"),
      preserve_order=True,
      scale=KANGAROO_FULL_TENDON_HIPS_HIPZ_TENDON_ACTION_SCALE,
      offset=KANGAROO_INIT_STATE_HIPZ_TENDONS_OFFSETS,
    ),
    "hip_xy_pos": TendonLengthActionCfg(
      entity_name="robot",
      # Order matches the CL variant's "hip_xy_actuator_pos" term (its JOINT
      # transmission's natural body-tree order): left_l, left_r, right_r,
      # right_l -- see the matching comment on "tendon_pos" above.
      actuator_names=(
        r"left_hip_xy_l_slider$",
        r"left_hip_xy_r_slider$",
        r"right_hip_xy_r_slider$",
        r"right_hip_xy_l_slider$",
      ),
      preserve_order=True,
      scale=KANGAROO_FULL_TENDON_HIPS_HIPXY_TENDON_ACTION_SCALE,
      offset=KANGAROO_INIT_STATE_HIPXY_TENDONS_OFFSETS,
    ),
  }

  # -- Observations
  #
  # "full_state" observes every joint plus the hip tendon lengths (the only
  # way this variant can expose hip position, since hip_z/hip_xy have no real
  # joint here). "simple_model" instead restricts to exactly the joint set
  # pal_kangaroo_flat_env_cfg observes (REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  # no tendon lengths), so a policy trained on the simple kangaroo model sees
  # an observation vector of the same shape here.
  observation_space = {
    "simple_model": SceneEntityCfg(
      "robot",
      joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
    ),
    "full_state": SceneEntityCfg(
      "robot",
      joint_names=REGEX_ALL_OBSERVABLE_JOINTS,
      tendon_names=KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES,
    ),
  }

  for obs in ["actor", "critic"]:
    for term in ["joint_pos", "joint_vel"]:
      cfg.observations[obs].terms[term].params["asset_cfg"] = observation_space[
        joint_state_obs
      ]

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


def pal_kangaroo_full_tendons_cl_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration.

  Uses the closed-loop (CL) hip mechanism: the hip_z rod is a real slide joint
  (``leg_.*_1_actuator``) kinematically closed onto the leg via a ``<connect>``
  equality constraint, instead of a spatial tendon. All actuated DOFs are JOINT
  transmission, so a single JointPositionActionCfg covers everything.
  """
  cfg = pal_kangaroo_baseline_env_cfg(play)

  cfg.scene.entities = {"robot": get_kangaroo_full_robot_tendon_hips_cl_cfg()}

  # Action layout matches the tendon-hips variant exactly: a 20-dim "joint_pos"
  # term (same joint set/order as KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES),
  # followed by a 2-dim hip_z term -- so both variants' policies share the same
  # action_dim (22) with hip_z always the last 2 entries, whether it routes
  # through a TENDON or a JOINT actuator internally.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_TENDON_HIPS_CL_MAIN_ACTUATED_JOINTS_NAMES,
      scale=KANGAROO_FULL_TENDON_HIPS_CL_MAIN_JOINT_ACTION_SCALE,
      use_default_offset=True,
    ),
    "hip_actuator_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_ACTUATED_JOINTS_NAMES,
      scale=KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_JOINT_ACTION_SCALE,
      use_default_offset=True,
    ),
    "hip_xy_actuator_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_ACTUATED_JOINTS_NAMES,
      scale=KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_JOINT_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # -- Observations
  #
  # Restrict to exactly the joint set the tendon-hips variant observes: real
  # output joints only (e.g. leg_.*_1_joint), not the CL mechanism's own input
  # DOFs (leg_.*_1_actuator, (left|right)_hip_xy_(l|r)_slider) or its phantom
  # passive hinges (left/right_hip_z_motor, the hip_xy bracket/motor hinges).
  # The tendon variant has no equivalent way to observe its own tendon length
  # either, so hiding these here keeps both variants' observation vectors
  # identical in shape *and* semantics, not just in dimension.
  _cl_hidden_from_obs = (
    r"leg_(left|right)_1_actuator"
    r"|(left|right)_hip_z_motor"
    r"|(left|right)_hip_xy_(l|r)_slider"
    r"|(left|right)_hip_xy_bracket_(l|r)"
    r"|(left|right)_hip_xy_motor_(l|r)"
  )
  cl_observable_joints = rf"^(?!({_cl_hidden_from_obs})$).*$"

  for obs in ["actor", "critic"]:
    for term in ["joint_pos", "joint_vel"]:
      cfg.observations[obs].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=cl_observable_joints,
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
  #
  # Unlike the tendon-hip variant, hip_z is a real joint here, so it is
  # automatically covered by the *_JOINTS_ONLY regexes below -- no separate
  # tendon-space limit/posture term is needed.
  #
  # The CL mechanism also introduces free, unactuated hinge joints that exist
  # only to let the slide-joint sub-bodies swing into place
  # (left/right_hip_z_motor, and the hip_xy bracket/motor hinges); they carry
  # no meaningful pose target. Excluded here so they don't leak into
  # dof_pos_limits/pose (std_walking/std_running, inherited from the base
  # kangaroo config, have no entry for these names and would otherwise
  # size-mismatch against std_standing). The actuator slide joints themselves
  # (leg_.*_1_actuator, hip_xy_(l|r)_slider) stay included -- unlike the
  # observation hiding above, the reward *should* regulate them.
  _cl_exclude_motor = (
    r"(left|right)_hip_z_motor$"
    r"|(left|right)_hip_xy_bracket_(l|r)$"
    r"|(left|right)_hip_xy_motor_(l|r)$"
  )
  cl_simple_model_observable = (
    rf"^(?!leg_.*_length_actuator$|{_cl_exclude_motor}).*$"
  )
  cl_simple_model_actuated = (
    rf"^(?!leg_.*_(femur|knee)_joint$|leg_.*_length_actuator$|{_cl_exclude_motor}).*$"
  )

  cfg.rewards["dof_pos_limits"] = RewardTermCfg(
    func=mdp.joint_pos_limits,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=cl_simple_model_observable,
      )
    },
  )

  cfg.rewards["pose"].params["asset_cfg"].joint_names = (cl_simple_model_actuated,)
  cfg.rewards["pose"].params["std_standing"] = {cl_simple_model_actuated: 0.05}
  # std_walking/std_running are inherited from the base kangaroo config and
  # have no pattern matching hip_xy_(l|r)_slider (they don't start with
  # "leg_"), which would otherwise leave those two targets uncovered and
  # size-mismatch against std_standing's catch-all at reward call time. Value
  # matches the sibling hip actuator pattern (leg_.*_1_.*: 0.15/0.2).
  _cl_hipxy_pattern = r"(left|right)_hip_xy_(l|r)_slider$"
  cfg.rewards["pose"].params["std_walking"][_cl_hipxy_pattern] = 0.15
  cfg.rewards["pose"].params["std_running"][_cl_hipxy_pattern] = 0.2

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


def pal_kangaroo_full_tendons_cl_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL (CL hip) flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_tendons_cl_rough_env_cfg(play=play)

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


def pal_kangaroo_full_tendons_flat_env_cfg(
  play: bool = False,
  joint_state_obs: str = "full_state",
  pose_in_actuator_space: bool = False,
  pd_mapping: str = "jacobian",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_tendons_rough_env_cfg(play=play, joint_state_obs=joint_state_obs)

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
