"""PAL Robotics KANGAROO FULL velocity tracking environment configurations.

The baseline adapts the simple ``pal_kangaroo`` baseline to the full model's
actuation, joint observations and constraint metrics. Rough and flat tasks
extend that baseline independently. Rough-task overrides are shared with the
simple model, including its box terrain, sensors, rewards and command setup.
``lower_body=True`` deletes both arms and removes their observation and reward
entries.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots import (
  ANKLE_FEMUR_JOINT_PAIRS,
  ARM_ACTION_SCALE_FACTOR,
  KANGAROO_TENDON_LENGTHS,
  KNEE_DISTANCE_MAP_CSV,
  KNEE_DISTANCE_MAP_LEGS,
  LEG_ACTION_SCALE_FACTOR,
  LEG_LENGTH_FROM_KNEE_JOINTS,
  LOWER_BODY_JOINT_ORDER,
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  SIMPLE_MODEL_JOINT_ORDER,
  FemurClosure,
  LowerBody,
  MjcfVariant,
  Transmission,
  get_kangaroo_full_model,
  simple_model_action_names,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import (
  configure_kangaroo_rough_env,
  pal_kangaroo_baseline_env_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.encoder_bias import (
  configure_simple_model_encoder_bias,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths


def pal_kangaroo_full_baseline_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> ManagerBasedRlEnvCfg:
  """Create the shared PAL Robotics KANGAROO FULL velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_model(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )
  cfg.scene.entities = {"robot": model.make_robot_cfg()}

  # -- Actions
  #
  # One JOINT term for everything commanded on a joint, plus -- for the
  # "actuator" transmission -- one TENDON term per mechanism. The tendon terms
  # use explicit, order-preserved names so a mechanism keeps the same action
  # indices in both full models. The "lut" transmission is commanded on the
  # simple model's joints, so it sits in the joint term like "joint" -- except
  # the knee servo of the "linkage" closure, whose targets are leg lengths in
  # metres. The knee is then commanded through the mapped term, which offsets
  # and de-biases in that coordinate -- and that term *is* the joint term,
  # with every target listed explicitly in the simple model's joint order
  # (the knee in leg_.*_length_joint's slot), so the action vector is laid
  # out exactly like the "joint" transmission's. A separate knee term would
  # trail the vector and permute every slot after the left hip against a
  # "joint"-trained checkpoint.

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
  for name, tendon_action in (
    ("hip_z_pos", model.hip_z_tendon_action),
    ("hip_xy_pos", model.hip_xy_tendon_action),
    ("ankle_pos", model.ankle_tendon_action),
  ):
    if tendon_action is None:
      continue
    cfg.actions[name] = TendonLengthActionCfg(
      entity_name="robot",
      actuator_names=tendon_action.actuator_names,
      preserve_order=True,
      scale=tendon_action.scale,
      offset=tendon_action.offset,
    )
  # -- Observations
  #
  # Exactly what the simple model's policy sees: the same 26 joints, in the
  # same order. Selecting by name with preserve_order matters as much as the
  # set does -- the full model both adds the mechanism DOFs and orders the
  # shared joints differently, so a regex would hand the policy the right
  # joints shuffled. Every other term is inherited from the baseline config
  # untouched.

  # lower_body=True has no arm_* joints at all (the whole arm subtree is
  # deleted), so the policy's joint vector drops those slots too rather than
  # reading zeros for a limb that doesn't exist.
  configure_simple_model_encoder_bias(cfg, joint_order, model.has_leg_length_joint)

  for group in ("actor", "critic"):
    for term, mode in (("joint_pos", "pos"), ("joint_vel", "vel")):
      term_cfg = cfg.observations[group].terms[term]
      if model.has_leg_length_joint:
        term_cfg.params["asset_cfg"] = SceneEntityCfg(
          "robot",
          joint_names=joint_order,
          preserve_order=True,
        )
      else:
        # No leg length joint to read in this variant: assemble the same
        # vector by hand, with those two slots reconstructed from their knee
        # angle through the displacement map.
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
      # the hull reward this mirrors never normalizes a rate either.
      # "tendons_over_constrained" keeps the raw term for the same reason the
      # reward does: its extra closed loops already pin leg_.*_4_joint to the
      # shank directly, so the correction would be wrong there.
      if ankle_normalized and mode == "pos" and mjcf != "tendons_over_constrained":
        inner_cfg = ObservationTermCfg(func=term_cfg.func, params=term_cfg.params)
        term_cfg.func = mdp.ankle_femur_normalized
        term_cfg.params = {
          "asset_cfg": SceneEntityCfg("robot"),
          "joint_order": joint_order,
          "ankle_femur_pairs": ANKLE_FEMUR_JOINT_PAIRS,
          "inner": inner_cfg,
        }

  if not model.has_leg_length_joint:
    # The map fills the observation slot, but the baseline terms that act on
    # the joint itself have nothing left to act on: there is no velocity to
    # limit and no posture to hold directly.
    cfg.rewards.pop("joint_vel_limits", None)
    for pose_type in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[pose_type].pop(r"leg_.*_length_.*", None)

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
  # Same terms and weights as the simple model, restricted to the same joints:
  # leg_.*_length_actuator is the variant's own input DOF, not something the
  # simple model has an opinion about, so it stays out of both the joint limit
  # penalty and the posture term.

  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY
  )
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  )
  # Keys may not overlap (resolve_matching_names_values raises on a joint two
  # keys match), so the joints with their own standing tolerance -- waist yaw
  # and hip pitch/roll -- are carved out of the catch-all regex.
  cfg.rewards["pose"].params["std_standing"] = {
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY: 0.05
  }

  # leg_.*_4_joint (ankle pitch) is measured off the shank, but on this model
  # the shank itself rotates on leg_.*_femur_joint through the four-bar (or
  # connect-loop) femur closure -- so the pitch the ankle hull was fit
  # against is leg_.*_4_joint - leg_.*_femur_joint, not leg_.*_4_joint alone.
  # "tendons_over_constrained" layers extra closed loops that already pin
  # leg_.*_4_joint to the shank directly, so that variant keeps the baseline
  # (un-normalized) term.
  if mjcf != "tendons_over_constrained":
    ankle_hull = cfg.rewards["convex_hull_joint_limits_ankle"]
    ankle_hull.func = mdp.joint_limits_convex_hull_ankle_femur_normalized
    ankle_hull.params["femur_joint_names"] = [
      "leg_left_femur_joint",
      "leg_right_femur_joint",
    ]

  # -- Metrics for the closed-loop constraints.
  #
  # One group of terms per mechanism, so a chain that pulls apart can be traced
  # to the constraint that gave way instead of showing up as one lumped number.
  # A tendon or joint residual is a scalar in its own units (metres, radians);
  # a <connect> residual is a displacement, so it also gets per-world-axis
  # terms saying which way the gap opens.

  entity_cfg = SceneEntityCfg("robot")

  def _add_tendon_eq_metrics(prefix: str, tendon_names: tuple[str, ...]) -> None:
    asset_cfg = SceneEntityCfg("robot", tendon_names=tendon_names)
    for reduction in ("mean", "max"):
      cfg.metrics[f"{prefix}_eq_{reduction}_violation"] = MetricsTermCfg(
        func=mdp.tendon_equality_constraint_violation,
        params={
          "asset_cfg": asset_cfg,
          "mode": "violation",
          "reduction": reduction,
        },
      )

  def _add_joint_eq_metrics(prefix: str, constraint_names: tuple[str, ...]) -> None:
    for reduction in ("mean", "max"):
      cfg.metrics[f"{prefix}_eq_{reduction}_violation"] = MetricsTermCfg(
        func=mdp.joint_equality_constraint_violation,
        params={
          "asset_cfg": entity_cfg,
          "constraint_names": constraint_names,
          "reduction": reduction,
        },
      )

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

  # The ankle bars and the decoupler's gearing to the knee are both deleted
  # when the ankle joints are servoed directly -- see
  # KangarooFullModel.has_ankle_tibia_bar_tendons / has_butterfly_decoupler_coupling.
  if model.has_ankle_tibia_bar_tendons:
    _add_tendon_eq_metrics("ankle_tibia_bars", (r"(left|right)_ankle_tibia_bar_(l|r)",))
  if model.has_butterfly_decoupler_coupling:
    _add_joint_eq_metrics(
      "butterfly_decoupler", (r"(left|right)_butterfly_decoupler_coupling",)
    )

  # The femur is closed one of two ways and never both -- see
  # KangarooFullModel.has_femur_linkage_tendons. hip_xy_link and femur_rod are
  # the two tendons of that same four-bar, so they are gated on the one
  # property together rather than two checks that could drift apart; each
  # still gets its own metric group so a chain that pulls apart can be traced
  # to which of the two gave way.
  if model.has_femur_linkage_tendons:
    _add_tendon_eq_metrics("hip_xy_link", (r"(left|right)_hip_xy_link",))
    _add_tendon_eq_metrics("femur_rod", (r"(left|right)_femur_rod",))
  else:
    _add_connect_eq_metrics("leg_length_connect", (r"leg_(left|right)_length_connect",))

  # -- Metrics for the knee displacement map.
  #
  # Independent of every variant axis: the two joints and the site it reads are
  # in the MJCF whichever way the femur is closed, so this measures the chain
  # itself rather than any one constraint holding it together.

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

  # -- Metrics for geared joint pairs.
  #
  # A catch-all over every <joint> equality, so a coupling that gets added to
  # the MJCF -- or the femur triangle locks the "prismatic" closure adds -- is
  # logged without a second edit here. The error is in the geared joint's own
  # units (radians for the revolutes), so it is not comparable with the
  # displacement terms above and gets its own terms.

  if model.has_joint_equalities:
    for metric_name, reduction in (
      ("joint_eq_mean_violation", "mean"),
      ("joint_eq_max_violation", "max"),
    ):
      cfg.metrics[metric_name] = MetricsTermCfg(
        func=mdp.joint_equality_constraint_violation,
        params={
          "asset_cfg": entity_cfg,
          "constraint_names": (r".*",),
          "reduction": reduction,
        },
      )

  # -- Events / metrics for the knee rod equality tendon.

  if model.has_knee_rod_tendons:
    cfg.events["tendon_lengths"] = EventTermCfg(
      mode="reset",
      func=enforce_tendon_lengths,
      params={"lengths": KANGAROO_TENDON_LENGTHS},
    )
    knee_rods_cfg = SceneEntityCfg("robot", tendon_names=(r"(left|right)_knee_rods",))
    for metric_name, reduction in (
      ("knee_rods_eq_mean_violation", "mean"),
      ("knee_rods_eq_max_violation", "max"),
    ):
      cfg.metrics[metric_name] = MetricsTermCfg(
        func=mdp.tendon_equality_constraint_violation,
        params={
          "asset_cfg": knee_rods_cfg,
          "mode": "violation",
          "reduction": reduction,
        },
      )

  return cfg


def pal_kangaroo_full_rough_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )
  return configure_kangaroo_rough_env(cfg, play=play)


def pal_kangaroo_full_flat_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  arm_action_scale_factor: float = ARM_ACTION_SCALE_FACTOR,
  leg_action_scale_factor: float = LEG_ACTION_SCALE_FACTOR,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    arm_action_scale_factor=arm_action_scale_factor,
    leg_action_scale_factor=leg_action_scale_factor,
  )

  cfg.sim.njmax = 300
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

  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
