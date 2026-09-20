"""PAL Robotics KANGAROO FULL FULL velocity tracking environment configurations.

The baseline adapts the simple ``pal_kangaroo`` baseline to the connect-linkage
model's actuation, mapped joint observations and constraint metrics. Rough and
flat tasks extend that baseline independently. Rough-task overrides are shared
with the simple model, including its box terrain, sensors, rewards and command
setup. ``lower_body=True`` deletes both arms and removes their observation and
reward entries.

This mirrors ``pal_kangaroo_full``'s ``env_cfgs.py`` closely, but every
mechanism this MJCF has is a ``<connect>`` equality rather than a tendon, and
there is no ``leg_.*_length_joint`` to observe in any variant (this MJCF has
only the four-bar femur closure, never the straight-line stand-in) -- so the
branching that file does per-variant (tendon vs. joint action term, mapped vs.
direct leg-length observation, which metric groups exist) collapses to a
single unconditional path here.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  ANKLE_FEMUR_JOINT_PAIRS,
  ARM_ACTION_SCALE_FACTOR,
  KNEE_DISTANCE_MAP_CSV,
  KNEE_DISTANCE_MAP_LEGS,
  LEG_ACTION_SCALE_FACTOR,
  LEG_LENGTH_FROM_KNEE_JOINTS,
  LOWER_BODY_JOINT_ORDER,
  SIMPLE_MODEL_JOINT_ORDER,
  get_kangaroo_full_spec,
  simple_model_action_names,
)
from pal_mjlab.robots.pal_kangaroo_full_full.kangaroo_full_constants import (
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  ActuatorModel,
  LowerBody,
  Transmission,
  get_kangaroo_full_full_model,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import (
  configure_kangaroo_rough_env,
  pal_kangaroo_baseline_env_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.encoder_bias import (
  configure_simple_model_encoder_bias,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.leg_joint_friction import (
  configure_leg_joint_friction_dr,
)


def pal_kangaroo_full_full_baseline_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> ManagerBasedRlEnvCfg:
  """Create the shared PAL Robotics KANGAROO FULL FULL velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_full_model(
    transmission=transmission,
    lower_body=lower_body,
    actuator_model=actuator_model,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )
  cfg.scene.entities = {"robot": model.make_robot_cfg()}

  # -- Actions
  #
  # Same layout as pal_kangaroo_full: one JOINT term for everything commanded
  # on a joint, plus -- for the "actuator" transmission -- one term per hip
  # and ankle mechanism driven at its screws. Every actuator here is
  # JOINT-transmission (there is no tendon to target -- see the module
  # docstring on kangaroo_full_full_constants.py), but the screw terms are
  # still kept separate and order-preserved, with the slider names listed in
  # the order of the tendon they replace, so a mechanism occupies the same
  # action slots in both models and a checkpoint trained on one can be
  # played on the other. A plain JointPositionActionCfg would lay the sliders
  # out in MJCF tree order, which differs (the right ankle).
  #
  # The "lut" transmission is commanded on the simple-model joints, so it
  # sits in the catch-all term exactly like the "joint" one -- except the
  # leg length: its target joint is the knee but its targets are leg lengths
  # in metres. The knee is therefore commanded through the mapped term, which
  # offsets and de-biases in that coordinate -- and that term *is* the
  # catch-all term, with every target listed explicitly in the simple
  # model's joint order (the knee in leg_.*_length_joint's slot), so the
  # action vector is laid out exactly like the "joint" transmission's on
  # either MJCF. A separate knee term would trail the vector and permute
  # every slot after the left hip against a "joint"-trained checkpoint.

  joint_order = LOWER_BODY_JOINT_ORDER if model.lower_body else SIMPLE_MODEL_JOINT_ORDER
  if model.leg_length_action is None:
    cfg.actions = {
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=model.joint_actuator_names,
        scale=model.joint_action_scale,
        use_default_offset=True,
      )
    }
  else:
    cfg.actions = {
      "joint_pos": mdp.MappedLegLengthPositionActionCfg(
        entity_name="robot",
        actuator_names=simple_model_action_names(
          joint_order, model.joint_actuator_names, model.leg_length_action
        ),
        preserve_order=True,
        scale={**model.joint_action_scale, **model.leg_length_action.scale},
        use_default_offset=True,
        csv_path=KNEE_DISTANCE_MAP_CSV,
        mapped_joints=LEG_LENGTH_FROM_KNEE_JOINTS,
      )
    }
  for name, slider_action in (
    ("hip_z_pos", model.hip_z_slider_action),
    ("hip_xy_pos", model.hip_xy_slider_action),
    ("ankle_pos", model.ankle_slider_action),
  ):
    if slider_action is None:
      continue
    cfg.actions[name] = mdp.OrderedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=slider_action.actuator_names,
      preserve_order=True,
      scale=slider_action.scale,
      use_default_offset=True,
    )
  # -- Observations
  #
  # Exactly what the simple model's policy sees: the same 26 joints, in the
  # same order. This MJCF never has a leg_.*_length_joint to read (only the
  # four-bar femur closure exists -- see the module docstring), so unlike
  # pal_kangaroo_full this is unconditionally the knee-angle-mapped
  # reconstruction rather than a per-variant choice.

  configure_simple_model_encoder_bias(cfg, joint_order, has_leg_length_joint=False)
  configure_leg_joint_friction_dr(cfg, transmission)

  for group in ("actor", "critic"):
    for term, mode in (("joint_pos", "pos"), ("joint_vel", "vel")):
      term_cfg = cfg.observations[group].terms[term]
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_order": joint_order,
        "csv_path": KNEE_DISTANCE_MAP_CSV,
        "mapped_joints": LEG_LENGTH_FROM_KNEE_JOINTS,
        "mode": mode,
      }
      if "biased" in term_cfg.params:
        params["biased"] = term_cfg.params["biased"]
      term_cfg.func = mdp.joint_state_with_mapped_leg_length
      term_cfg.params = params

      # ankle_normalized re-expresses joint_pos's leg_.*_4_joint column
      # relative to the shank (leg_.*_4_joint - leg_.*_femur_joint) instead
      # of the femur link, matching the ankle hull reward's own correction --
      # see mdp.observations.ankle_femur_normalized. Velocity is untouched:
      # the hull reward this mirrors never normalizes a rate either. Unlike
      # pal_kangaroo_full, every variant here gets the correction when
      # requested -- there is no over-constrained MJCF that already pins
      # leg_.*_4_joint to the shank.
      if ankle_normalized and mode == "pos":
        inner_cfg = ObservationTermCfg(func=term_cfg.func, params=term_cfg.params)
        term_cfg.func = mdp.ankle_femur_normalized
        term_cfg.params = {
          "asset_cfg": SceneEntityCfg("robot"),
          "joint_order": joint_order,
          "ankle_femur_pairs": ANKLE_FEMUR_JOINT_PAIRS,
          "inner": inner_cfg,
        }

  if model.lower_body:
    # No arm_* joints in this variant: the arm-specific keys the baseline
    # config sets would otherwise match zero joints and
    # resolve_matching_names_values would raise -- same treatment as
    # pal_kangaroo_lower_body_flat_env_cfg gives the simple model's pose
    # reward.
    for pose_type in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_1_.*", None)
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_4_.*", None)
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_(?![14]_joint)\d+_joint", None)

  # -- Rewards
  #
  # Same terms and weights as the simple model, restricted to the same
  # joints: the six screws (leg_.*_length_actuator and the five
  # leg_.*_(1..5)_actuator sliders this MJCF adds) are this variant's own
  # input DOFs, not something the simple model has an opinion about, so they
  # stay out of both the joint limit penalty and the posture term -- see
  # kangaroo_full_full_constants.py's REGEX_SIMPLE_MODEL_*_JOINTS_ONLY.

  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY
  )
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  )
  cfg.rewards["pose"].params["std_standing"] = {
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY: 0.05
  }

  # The three baseline terms on leg_.*_length_joint, read through the knee map
  # instead (mdp.rewards): the posture term folds the mapped leg length into
  # its mean with the std_* keys the baseline already sets for it, and the
  # limit terms take the slider's range and rate limit from the variants that
  # have the slider, so all three score the same leg length the same way.
  mapped_leg_length = {
    "csv_path": KNEE_DISTANCE_MAP_CSV,
    "mapped_joints": LEG_LENGTH_FROM_KNEE_JOINTS,
  }
  cfg.rewards["pose"].func = mdp.variable_posture_with_mapped_leg_length
  cfg.rewards["pose"].params.update(mapped_leg_length)

  leg_length_joint = get_kangaroo_full_spec(lower_body=model.lower_body).joint(
    "leg_left_length_joint"
  )
  cfg.rewards["dof_pos_limits_leg_length"] = RewardTermCfg(
    func=mdp.mapped_leg_length_pos_limits,
    weight=cfg.rewards["dof_pos_limits"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_range": tuple(float(v) for v in leg_length_joint.range),
      "soft_limit_factor": model.articulation.soft_joint_pos_limit_factor,
      **mapped_leg_length,
    },
  )

  (velocity_limits,) = (
    cfg.rewards["joint_vel_limits"].params["velocity_limits"].values()
  )
  cfg.rewards["joint_vel_limits"].func = mdp.mapped_leg_length_vel_limits
  cfg.rewards["joint_vel_limits"].params = {
    "asset_cfg": SceneEntityCfg("robot"),
    "velocity_limits": velocity_limits,
    **mapped_leg_length,
  }

  # leg_.*_4_joint (ankle pitch) is measured off the shank, but here the
  # shank itself rotates on leg_.*_femur_joint through the four-bar femur
  # closure -- so the pitch the ankle hull was fit against is
  # leg_.*_4_joint - leg_.*_femur_joint, not leg_.*_4_joint alone. Unlike
  # pal_kangaroo_full, this MJCF has no over-constrained variant that already
  # normalizes it, so every full-full variant gets the normalized term.
  ankle_hull = cfg.rewards["convex_hull_joint_limits_ankle"]
  ankle_hull.func = mdp.joint_limits_convex_hull_ankle_femur_normalized
  ankle_hull.params["femur_joint_names"] = [
    "leg_left_femur_joint",
    "leg_right_femur_joint",
  ]

  # -- Metrics for the closed-loop constraints.
  #
  # One group of terms per mechanism, so a chain that pulls apart can be
  # traced to the joint that gave way instead of showing up as one lumped
  # number. Every mechanism in this MJCF is a <connect> (there is no tendon
  # left at all -- see the module docstring), so unlike pal_kangaroo_full
  # every group here uses the same connect-equality metric and every group
  # is unconditional: nothing is ever deleted from the spec (see
  # get_kangaroo_full_full_spec), so all 24 named equalities always exist.

  entity_cfg = SceneEntityCfg("robot")

  def _add_connect_eq_metrics(prefix: str, constraint_names: tuple[str, ...]) -> None:
    for axis in ("x", "y", "z"):
      cfg.metrics[f"{prefix}_eq_mean_violation_{axis}"] = MetricsTermCfg(
        func=mdp.connect_equality_constraint_violation,
        params={
          "asset_cfg": entity_cfg,
          "constraint_names": constraint_names,
          "axis": axis,
          "reduction": "mean",
        },
      )
    for reduction in ("mean", "max"):
      cfg.metrics[f"{prefix}_eq_{reduction}_violation"] = MetricsTermCfg(
        func=mdp.connect_equality_constraint_violation,
        params={
          "asset_cfg": entity_cfg,
          "constraint_names": constraint_names,
          "axis": None,
          "reduction": reduction,
        },
      )

  _add_connect_eq_metrics("hip_z_slider", (r"(left|right)_hip_z_slider",))
  _add_connect_eq_metrics("hip_xy_slider", (r"(left|right)_hip_xy_slider_(l|r)",))
  _add_connect_eq_metrics("hip_xy_link", (r"(left|right)_hip_xy_link",))
  _add_connect_eq_metrics("ankle_slider", (r"(left|right)_ankle_slider_(l|r)",))
  _add_connect_eq_metrics("ankle_tibia_bar", (r"(left|right)_ankle_tibia_bar_(l|r)",))
  _add_connect_eq_metrics("ankle_femur_bar", (r"(left|right)_ankle_femur_bar_(l|r)",))
  _add_connect_eq_metrics("femur_rod", (r"(left|right)_femur_rod",))
  _add_connect_eq_metrics("knee_rods", (r"(left|right)_knee_rods",))

  # -- Metrics for the knee displacement map.
  #
  # Independent of every variant axis: the two joints and the site it reads
  # are in the MJCF regardless of actuation choice, so this measures the
  # chain itself rather than any one constraint holding it together.

  for axis_name, axis in (("_x", "x"), ("_z", "z"), ("", None)):
    for reduction in ("mean", "max"):
      cfg.metrics[f"knee_distance_map_{reduction}_error{axis_name}"] = MetricsTermCfg(
        func=mdp.knee_distance_map_error,
        params={
          "asset_cfg": entity_cfg,
          "csv_path": KNEE_DISTANCE_MAP_CSV,
          "legs": KNEE_DISTANCE_MAP_LEGS,
          "axis": axis,
          "reduction": reduction,
        },
      )

  return cfg


def pal_kangaroo_full_full_rough_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
  top_speed: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL FULL rough terrain velocity configuration."""
  del top_speed  # configure_kangaroo_rough_env deletes the speed curriculum
  # outright -- rough terrain keeps its fixed, easier command range regardless.
  cfg = pal_kangaroo_full_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )
  cfg = configure_kangaroo_rough_env(cfg, play=play)
  return cfg


def pal_kangaroo_full_full_flat_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
  top_speed: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )

  cfg.sim.njmax = 550
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 64

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if top_speed:
    # TopSpeed: step lin_vel_x's upper bound up by +1 m/s every 2500 steps,
    # capped at 12 m/s. Registered after "command_vel" so it overrides that
    # term's own upper bound each step, while the lower bound and the y/yaw
    # ranges keep following command_vel's existing staged ramp unchanged.
    assert "command_vel" in cfg.curriculum
    cfg.curriculum["top_speed"] = CurriculumTermCfg(
      func=mdp.top_speed,
      params={
        "command_name": "twist",
        "step_interval": 2500 * 24,
        "increment": 1.0,
        "start_speed": 1.0,
        "max_speed": 12.0,
      },
    )

    # The joint velocity limits reward is tuned for the base command range --
    # TopSpeed pushes lin_vel_x well past it on purpose, so the penalty would
    # fight the curriculum instead of shaping gait.
    cfg.rewards.pop("joint_vel_limits", None)

  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]
    if top_speed:
      del cfg.curriculum["top_speed"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
