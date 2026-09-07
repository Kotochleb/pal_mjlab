"""PAL Robotics KANGAROO FULL velocity tracking environment configurations.

Every variant is the simple ``pal_kangaroo`` velocity task with the full model
swapped in: identical rewards, identical observations, identical terrain and
command setup. The only thing a variant changes is *how the legs are actuated*
-- hip yaw through a tendon or a revolute motor, hip pitch/roll through tendons
or revolute motors, leg length through the prismatic screw or the leg length
joint directly. That is deliberate: it is what makes training results across
variants comparable.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots import (
  KANGAROO_TENDON_LENGTHS,
  KNEE_DISTANCE_MAP_CSV,
  KNEE_DISTANCE_MAP_LEGS,
  LEG_LENGTH_FROM_KNEE_JOINTS,
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  SIMPLE_MODEL_JOINT_ORDER,
  AnkleActuation,
  FemurClosure,
  HipXyActuation,
  HipZActuation,
  LegLengthActuation,
  MjcfVariant,
  get_kangaroo_full_model,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths


def pal_kangaroo_full_rough_env_cfg(
  play: bool = False,
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
  femur_closure: FemurClosure = "prismatic",
  ankle: AnkleActuation = "joint",
  mjcf: MjcfVariant = "tendons",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_model(
    hip_z=hip_z,
    hip_xy=hip_xy,
    leg_length=leg_length,
    femur_closure=femur_closure,
    ankle=ankle,
    mjcf=mjcf,
  )
  cfg.scene.entities = {"robot": model.make_robot_cfg()}

  # -- Actions
  #
  # One JOINT term for everything driven by a plain motor, plus one TENDON term
  # per hip mechanism that is tendon driven. The tendon terms use explicit,
  # order-preserved names so a mechanism keeps the same action indices whether
  # or not the other one is a tendon.

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=model.joint_actuator_names,
      scale=model.joint_action_scale,
      use_default_offset=True,
    )
  }
  for name, tendon_action in (
    ("hip_z_pos", model.hip_z_tendon_action),
    ("hip_xy_pos", model.hip_xy_tendon_action),
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

  for group in ("actor", "critic"):
    for term, mode in (("joint_pos", "pos"), ("joint_vel", "vel")):
      term_cfg = cfg.observations[group].terms[term]
      if model.has_leg_length_joint:
        term_cfg.params["asset_cfg"] = SceneEntityCfg(
          "robot",
          joint_names=SIMPLE_MODEL_JOINT_ORDER,
          preserve_order=True,
        )
        continue
      # No leg length joint to read in this variant: assemble the same vector
      # by hand, with those two slots reconstructed from their knee angle
      # through the displacement map.
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_order": SIMPLE_MODEL_JOINT_ORDER,
        "csv_path": KNEE_DISTANCE_MAP_CSV,
        "mapped_joints": LEG_LENGTH_FROM_KNEE_JOINTS,
        "mode": mode,
      }
      if "biased" in term_cfg.params:
        params["biased"] = term_cfg.params["biased"]
      term_cfg.func = mdp.joint_state_with_mapped_leg_length
      term_cfg.params = params

  if not model.has_leg_length_joint:
    # The map fills the observation slot, but the baseline terms that act on
    # the joint itself have nothing left to act on: there is no velocity to
    # limit, no encoder to bias, and no posture to hold.
    cfg.rewards.pop("joint_vel_limits", None)
    cfg.events.pop("leg_length_encoder_bias", None)
    for pose_type in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[pose_type].pop(r"leg_.*_length_.*", None)

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
  cfg.rewards["pose"].params["std_standing"] = {
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY: 0.05
  }

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

  # The ankle bars and the decoupler's gearing to the knee both only survive a
  # butterfly-actuated ankle -- see KangarooFullModel.has_ankle_tibia_bar_tendons
  # / has_butterfly_decoupler_coupling.
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


def pal_kangaroo_full_flat_env_cfg(
  play: bool = False,
  hip_z: HipZActuation = "tendon",
  hip_xy: HipXyActuation = "tendon",
  leg_length: LegLengthActuation = "actuator",
  femur_closure: FemurClosure = "prismatic",
  ankle: AnkleActuation = "joint",
  mjcf: MjcfVariant = "tendons",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_rough_env_cfg(
    play=play,
    hip_z=hip_z,
    hip_xy=hip_xy,
    leg_length=leg_length,
    femur_closure=femur_closure,
    ankle=ankle,
    mjcf=mjcf,
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
