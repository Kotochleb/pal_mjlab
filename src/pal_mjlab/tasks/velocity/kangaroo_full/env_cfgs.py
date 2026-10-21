from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import (
  TerrainGeneratorCfg,
  flat,
  pyramid_stairs,
  pyramid_stairs_inv,
)

from pal_mjlab.robots import (
  FEMUR_JOINT_REF,
  KANGAROO_TENDON_LENGTHS,
  LOWER_BODY_JOINT_ORDER,
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  SIMPLE_MODEL_JOINT_ORDER,
  ActuatorModel,
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
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.actuator_friction import (
  configure_actuator_friction_dr,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.encoder_bias import (
  configure_simple_model_encoder_bias,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.leg_joint_friction import (
  configure_leg_joint_friction_dr,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.dr.tendon import enforce_tendon_lengths
from pal_mjlab.tasks.velocity.mdp.lp_acrl import (
  Curriculum,
  configure_lp_acrl,
  configure_top_speed,
)

LEG_LENGTH_FROM_KNEE_JOINTS: tuple[tuple[str, str], ...] = (
  ("leg_left_length_joint", "leg_left_knee_joint"),
  ("leg_right_length_joint", "leg_right_knee_joint"),
)

ANKLE_FEMUR_JOINT_PAIRS: tuple[tuple[str, str], ...] = (
  ("leg_left_4_joint", "leg_left_femur_joint"),
  ("leg_right_4_joint", "leg_right_femur_joint"),
)


def pal_kangaroo_full_baseline_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_model(
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    actuator_model=actuator_model,
  )
  cfg.scene.entities = {"robot": model["robot_cfg"]}

  joint_order = (
    LOWER_BODY_JOINT_ORDER if model["lower_body"] else SIMPLE_MODEL_JOINT_ORDER
  )
  leg_length_action = model["leg_length_action"]
  if leg_length_action is None:
    cfg.actions = {
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=model["joint_actuator_names"],
        scale=model["joint_action_scale"],
        use_default_offset=True,
      )
    }
  else:
    cfg.actions = {
      "joint_pos": mdp.MappedLegLengthPositionActionCfg(
        entity_name="robot",
        actuator_names=simple_model_action_names(
          joint_order,
          model["joint_actuator_names"],
          leg_length_action,
          LEG_LENGTH_FROM_KNEE_JOINTS,
        ),
        preserve_order=True,
        scale={**model["joint_action_scale"], **leg_length_action["scale"]},
        use_default_offset=True,
        mapped_joints=LEG_LENGTH_FROM_KNEE_JOINTS,
      )
    }
  for name, tendon_action in (
    ("hip_z_pos", model["hip_z_tendon_action"]),
    ("hip_xy_pos", model["hip_xy_tendon_action"]),
    ("ankle_pos", model["ankle_tendon_action"]),
  ):
    if tendon_action is None:
      continue
    cfg.actions[name] = TendonLengthActionCfg(
      entity_name="robot",
      actuator_names=tendon_action["actuator_names"],
      preserve_order=True,
      scale=tendon_action["scale"],
      offset=tendon_action["offset"],
    )
  configure_simple_model_encoder_bias(cfg, joint_order, model["has_leg_length_joint"])
  configure_leg_joint_friction_dr(cfg, transmission)
  configure_actuator_friction_dr(cfg, transmission)

  for group in ("actor", "critic"):
    for term, mode in (("joint_pos", "pos"), ("joint_vel", "vel")):
      term_cfg = cfg.observations[group].terms[term]
      if model["has_leg_length_joint"]:
        term_cfg.params["asset_cfg"] = SceneEntityCfg(
          "robot",
          joint_names=joint_order,
          preserve_order=True,
        )
      else:
        params = {
          "asset_cfg": SceneEntityCfg("robot"),
          "joint_order": joint_order,
          "mapped_joints": LEG_LENGTH_FROM_KNEE_JOINTS,
          "mode": mode,
        }
        if "biased" in term_cfg.params:
          params["biased"] = term_cfg.params["biased"]
        term_cfg.func = mdp.joint_state_with_mapped_leg_length
        term_cfg.params = params

      if ankle_normalized and mode == "pos" and mjcf != "tendons_over_constrained":
        inner_cfg = ObservationTermCfg(func=term_cfg.func, params=term_cfg.params)
        term_cfg.func = mdp.ankle_femur_normalized
        term_cfg.params = {
          "asset_cfg": SceneEntityCfg("robot"),
          "joint_order": joint_order,
          "ankle_femur_pairs": ANKLE_FEMUR_JOINT_PAIRS,
          "inner": inner_cfg,
        }

  if not model["has_leg_length_joint"]:
    cfg.rewards.pop("joint_vel_limits", None)
    for pose_type in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[pose_type].pop(r"leg_.*_length_.*", None)

  if model["lower_body"]:
    for pose_type in ("std_walking", "std_running"):
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_1_.*", None)
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_4_.*", None)
      cfg.rewards["pose"].params[pose_type].pop(r"arm_.*_(?![14]_joint)\d+_joint", None)

  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY
  )
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  )
  cfg.rewards["pose"].params["std_standing"] = {
    REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY: 0.05
  }

  if mjcf != "tendons_over_constrained":
    ankle_hull = cfg.rewards["convex_hull_joint_limits_ankle"]
    ankle_hull.func = mdp.joint_limits_convex_hull_ankle_femur_normalized
    ankle_hull.params["femur_joint_names"] = [
      "leg_left_femur_joint",
      "leg_right_femur_joint",
    ]
    ankle_hull.params["femur_offset"] = FEMUR_JOINT_REF

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

  if model["has_ankle_tibia_bar_tendons"]:
    _add_tendon_eq_metrics("ankle_tibia_bars", (r"(left|right)_ankle_tibia_bar_(l|r)",))
  if model["has_butterfly_decoupler_coupling"]:
    _add_joint_eq_metrics(
      "butterfly_decoupler", (r"(left|right)_butterfly_decoupler_coupling",)
    )

  if model["has_femur_linkage_tendons"]:
    _add_tendon_eq_metrics("hip_xy_link", (r"(left|right)_hip_xy_link",))
    _add_tendon_eq_metrics("femur_rod", (r"(left|right)_femur_rod",))
  else:
    _add_connect_eq_metrics("leg_length_connect", (r"leg_(left|right)_length_connect",))

  if model["has_joint_equalities"]:
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

  if model["has_knee_rod_tendons"]:
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
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "default",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
  )
  cfg = configure_kangaroo_rough_env(cfg, play=play)
  return configure_lp_acrl(cfg, curriculum, play=play)


_STAIRS_TILE_SIZE = 8.0
_STAIRS_PLATFORM_WIDTH = 1.5
"""Side of the flat square on top of every peak and at the bottom of every pit [m]."""
_STAIRS_MIN_WIDTH_TO_HEIGHT = 1.5
"""Every tread is at least this many times as deep as the tallest step of its
staircase's height range."""


def _platform_pyramid_stairs(
  *,
  proportion: float,
  step_height_range: tuple[float, float],
  step_width: float,
  border_width: float = 0.5,
  inverted: bool = False,
):
  """Pyramid stairs whose summit is a ``_STAIRS_PLATFORM_WIDTH`` square.

  With ``inverted=True`` the pyramid goes down into the ground instead, ending in
  a pit bottom of the same size.

  mjlab fits ``int((inner - platform_width) / (2 * step_width))`` steps and turns
  the leftover into the top platform. The border is grown (from ``border_width``)
  so the inner width is the platform plus a whole number of tread pairs, leaving
  exactly ``_STAIRS_PLATFORM_WIDTH`` on top; passing ``platform_width`` half a
  tread smaller keeps the step count stable against float round-off.

  ``step_width`` is raised to ``_STAIRS_MIN_WIDTH_TO_HEIGHT`` times the top of
  ``step_height_range`` when it is shallower than that (one width serves every
  difficulty row).
  """
  step_width = max(step_width, _STAIRS_MIN_WIDTH_TO_HEIGHT * step_height_range[1])
  inner = _STAIRS_TILE_SIZE - 2 * border_width
  num_steps = int((inner - _STAIRS_PLATFORM_WIDTH) / (2 * step_width))
  make = pyramid_stairs_inv if inverted else pyramid_stairs
  return make(
    proportion=proportion,
    step_height_range=step_height_range,
    step_width=step_width,
    platform_width=_STAIRS_PLATFORM_WIDTH - 0.5 * step_width,
    border_width=0.5
    * (_STAIRS_TILE_SIZE - _STAIRS_PLATFORM_WIDTH - 2 * num_steps * step_width),
  )


def pal_kangaroo_full_stairs_rough_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "default",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
  )
  cfg = configure_kangaroo_rough_env(cfg, play=play)
  cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    func=mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / 2.0,
  )
  cfg.rewards.pop("foot_swing_height", None)
  # Stepping up and down stairs needs fast leg-length strokes; drop the
  # leg-length joint velocity penalty.
  cfg.rewards.pop("joint_vel_limits", None)
  cfg.rewards["pose"].weight *= 0.75
  # Robots spawn on the summit (or pit-bottom) platform, _STAIRS_PLATFORM_WIDTH wide.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (0.01, 0.05),
    "yaw": (-3.14, 3.14),
  }
  # (proportion, step_height_range, step_width, border_width) per staircase, with
  # step_width raised to 1.5x the tallest step where needed; each
  # one is generated both as a peak and, under "inv_<name>", as a pit.
  stairs = {
    "easy_stairs": (0.35, (0.02, 0.05), 0.40, 1.0),
    "moderate_stairs": (0.25, (0.05, 0.08), 0.35, 0.8),
    "challenging_stairs": (0.15, (0.08, 0.10), 0.30, 0.5),
    "very_challenging_stairs": (0.15, (0.10, 0.20), 0.30, 0.5),
    "more_very_challenging_stairs": (0.15, (0.20, 0.30), 0.30, 0.5),
    "more_more_very_challenging_stairs": (0.15, (0.30, 0.40), 0.30, 0.5),
    "more_more_more_very_challenging_stairs": (0.15, (0.40, 0.50), 0.30, 0.5),
  }
  sub_terrains = {"flat": flat(proportion=0.5)}
  for inverted in (False, True):
    for name, (proportion, heights, step_width, border_width) in stairs.items():
      sub_terrains[f"inv_{name}" if inverted else name] = _platform_pyramid_stairs(
        proportion=proportion,
        step_height_range=heights,
        step_width=step_width,
        border_width=border_width,
        inverted=inverted,
      )
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(_STAIRS_TILE_SIZE, _STAIRS_TILE_SIZE),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    curriculum=True,
    sub_terrains=sub_terrains,
    add_lights=True,
  )
  return configure_lp_acrl(cfg, curriculum, play=play)


def pal_kangaroo_full_flat_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "default",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 64

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return configure_lp_acrl(cfg, curriculum, play=play)


# Steps are env steps (common_step_counter); iterations * num_steps_per_env.
_STEPS_PER_ITERATION = 24
FLAT_FORWARD_LIN_VEL_X_MAX = 7.0
FLAT_FORWARD_LIN_VEL_X_STEP = 0.5
FLAT_FORWARD_STAGE_ITERATIONS = 500


def flat_forward_velocity_stages() -> list[dict]:
  """Forward vx grows 1.0 -> 6.0 m/s by 0.5 every 500 iterations.

  Backward vx, vy and yaw rate follow mjlab's default command_vel schedule.
  The vx range stays mostly positive, so uniform sampling is mostly forward.
  """

  def lower_x(iteration: int) -> float:
    return -2.0 if iteration >= 10000 else -1.5 if iteration >= 5000 else -1.0

  def upper_x(iteration: int) -> float:
    increments = iteration // FLAT_FORWARD_STAGE_ITERATIONS
    return min(
      1.0 + FLAT_FORWARD_LIN_VEL_X_STEP * increments, FLAT_FORWARD_LIN_VEL_X_MAX
    )

  num_increments = round(
    (FLAT_FORWARD_LIN_VEL_X_MAX - 1.0) / FLAT_FORWARD_LIN_VEL_X_STEP
  )
  iterations = sorted(
    {k * FLAT_FORWARD_STAGE_ITERATIONS for k in range(num_increments + 1)}
    | {5000, 10000}
  )
  stages: list[dict] = []
  for iteration in iterations:
    stage: dict = {
      "step": iteration * _STEPS_PER_ITERATION,
      "lin_vel_x": (lower_x(iteration), upper_x(iteration)),
    }
    if iteration == 0:
      stage["ang_vel_z"] = (-0.5, 0.5)
    elif iteration == 5000:
      stage["ang_vel_z"] = (-0.7, 0.7)
    stages.append(stage)
  return stages


def pal_kangaroo_full_flat_high_speed_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "default",
) -> ManagerBasedRlEnvCfg:
  """Flat task whose default curriculum ramps forward vx to 6 m/s."""
  if curriculum != "default":
    raise ValueError("FlatHighSpeed is only defined with the default curriculum")
  cfg = pal_kangaroo_full_flat_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
    curriculum=curriculum,
  )

  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  cfg.scene.entities["robot"] = replace(
    robot,
    articulation=replace(robot.articulation, soft_joint_pos_limit_factor=0.995),
  )

  cfg.rewards.pop("joint_vel_limits", None)
  cfg.rewards.pop("foot_swing_height", None)
  cfg.rewards["pose"].weight *= 0.5
  cfg.rewards["track_linear_velocity"].weight *= 1.25

  assert cfg.curriculum is not None
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-2.0, FLAT_FORWARD_LIN_VEL_X_MAX)
  else:
    cfg.curriculum["command_vel"].params["velocity_stages"] = (
      flat_forward_velocity_stages()
    )
  return cfg


def pal_kangaroo_full_top_speed_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "lp_acrl",
) -> ManagerBasedRlEnvCfg:
  """Flat LP-ACRL task with vx up to 10 m/s and doubled vy/yaw-rate ranges."""
  if curriculum != "lp_acrl":
    raise ValueError("TopSpeed is only defined with the LP-ACRL curriculum")
  cfg = pal_kangaroo_full_flat_env_cfg(
    play=play,
    transmission=transmission,
    femur_closure=femur_closure,
    mjcf=mjcf,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
    curriculum=curriculum,
  )
  return configure_top_speed(cfg)
