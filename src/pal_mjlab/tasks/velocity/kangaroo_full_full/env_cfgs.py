"""PAL Robotics KANGAROO FULL FULL velocity tracking environment configurations.

Every variant is the simple ``pal_kangaroo`` velocity task with the
connect-linkage full model (``pal_kangaroo_full_full``) swapped in: identical
rewards, identical observations, identical terrain and command setup. The
only thing a variant changes is *how the legs are actuated* -- hip yaw, hip
pitch/roll and ankle each through their linear actuator screw or the plain
revolute joint they drive (see
``pal_mjlab.robots.pal_kangaroo_full_full.kangaroo_full_constants``).
``lower_body=True`` is the one axis that isn't just an actuation choice: it
deletes both arms and makes every arm-related observation and reward term
drop out along with them.

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
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  KNEE_DISTANCE_MAP_CSV,
  KNEE_DISTANCE_MAP_LEGS,
  LEG_LENGTH_FROM_KNEE_JOINTS,
  LOWER_BODY_JOINT_ORDER,
  SIMPLE_MODEL_JOINT_ORDER,
)
from pal_mjlab.robots.pal_kangaroo_full_full.kangaroo_full_constants import (
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  AnkleActuation,
  HipXyActuation,
  HipZActuation,
  LowerBody,
  get_kangaroo_full_full_model,
)
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_baseline_env_cfg
from pal_mjlab.tasks.velocity.kangaroo_full import mdp


def pal_kangaroo_full_full_rough_env_cfg(
  play: bool = False,
  hip_z: HipZActuation = "slider",
  hip_xy: HipXyActuation = "slider",
  ankle: AnkleActuation = "joint",
  lower_body: LowerBody = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)
  cfg.sim.nconmax = 70

  model = get_kangaroo_full_full_model(
    hip_z=hip_z, hip_xy=hip_xy, ankle=ankle, lower_body=lower_body
  )
  cfg.scene.entities = {"robot": model.make_robot_cfg()}

  # -- Actions
  #
  # Every actuator in this model is JOINT-transmission -- there is no tendon
  # left to target (see the module docstring on
  # kangaroo_full_full_constants.py) -- so the whole action vector is one
  # JointPositionActionCfg, unlike pal_kangaroo_full's per-mechanism tendon
  # terms.

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=model.joint_actuator_names,
      scale=model.joint_action_scale,
      use_default_offset=True,
    )
  }

  # -- Observations
  #
  # Exactly what the simple model's policy sees: the same 26 joints, in the
  # same order. This MJCF never has a leg_.*_length_joint to read (only the
  # four-bar femur closure exists -- see the module docstring), so unlike
  # pal_kangaroo_full this is unconditionally the knee-angle-mapped
  # reconstruction rather than a per-variant choice.

  joint_order = LOWER_BODY_JOINT_ORDER if model.lower_body else SIMPLE_MODEL_JOINT_ORDER

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

  # The map fills the observation slot, but the baseline terms that act on
  # the joint itself have nothing left to act on: there is no velocity to
  # limit, no encoder to bias, and no posture to hold.
  cfg.rewards.pop("joint_vel_limits", None)
  cfg.events.pop("leg_length_encoder_bias", None)
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


def pal_kangaroo_full_full_flat_env_cfg(
  play: bool = False,
  hip_z: HipZActuation = "slider",
  hip_xy: HipXyActuation = "slider",
  ankle: AnkleActuation = "joint",
  lower_body: LowerBody = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_full_rough_env_cfg(
    play=play, hip_z=hip_z, hip_xy=hip_xy, ankle=ankle, lower_body=lower_body
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
