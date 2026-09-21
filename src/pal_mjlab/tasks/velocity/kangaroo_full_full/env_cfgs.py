from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
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
from pal_mjlab.tasks.velocity.kangaroo_full.env_cfgs import (
  ANKLE_FEMUR_JOINT_PAIRS,
  LEG_LENGTH_FROM_KNEE_JOINTS,
)
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
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_full_model(
    transmission=transmission,
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
  for name, slider_action in (
    ("hip_z_pos", model["hip_z_slider_action"]),
    ("hip_xy_pos", model["hip_xy_slider_action"]),
    ("ankle_pos", model["ankle_slider_action"]),
  ):
    if slider_action is None:
      continue
    cfg.actions[name] = mdp.OrderedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=slider_action["actuator_names"],
      preserve_order=True,
      scale=slider_action["scale"],
      use_default_offset=True,
    )

  configure_simple_model_encoder_bias(cfg, joint_order, has_leg_length_joint=False)
  configure_leg_joint_friction_dr(cfg, transmission)

  for group in ("actor", "critic"):
    for term, mode in (("joint_pos", "pos"), ("joint_vel", "vel")):
      term_cfg = cfg.observations[group].terms[term]
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

      if ankle_normalized and mode == "pos":
        inner_cfg = ObservationTermCfg(func=term_cfg.func, params=term_cfg.params)
        term_cfg.func = mdp.ankle_femur_normalized
        term_cfg.params = {
          "asset_cfg": SceneEntityCfg("robot"),
          "joint_order": joint_order,
          "ankle_femur_pairs": ANKLE_FEMUR_JOINT_PAIRS,
          "inner": inner_cfg,
        }

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

  mapped_leg_length = {
    "mapped_joints": LEG_LENGTH_FROM_KNEE_JOINTS,
  }
  cfg.rewards["pose"].func = mdp.variable_posture_with_mapped_leg_length
  cfg.rewards["pose"].params.update(mapped_leg_length)

  leg_length_joint = get_kangaroo_full_spec(lower_body=model["lower_body"]).joint(
    "leg_left_length_joint"
  )
  cfg.rewards["dof_pos_limits_leg_length"] = RewardTermCfg(
    func=mdp.mapped_leg_length_pos_limits,
    weight=cfg.rewards["dof_pos_limits"].weight,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_range": tuple(float(v) for v in leg_length_joint.range),
      "soft_limit_factor": model["articulation"].soft_joint_pos_limit_factor,
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

  ankle_hull = cfg.rewards["convex_hull_joint_limits_ankle"]
  ankle_hull.func = mdp.joint_limits_convex_hull_ankle_femur_normalized
  ankle_hull.params["femur_joint_names"] = [
    "leg_left_femur_joint",
    "leg_right_femur_joint",
  ]

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

  return cfg


def pal_kangaroo_full_full_rough_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_full_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
  )
  cfg = configure_kangaroo_rough_env(cfg, play=play)
  return cfg


def pal_kangaroo_full_full_flat_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_full_full_baseline_env_cfg(
    play=play,
    transmission=transmission,
    lower_body=lower_body,
    ankle_normalized=ankle_normalized,
    actuator_model=actuator_model,
  )

  cfg.sim.njmax = 550
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

  return cfg
