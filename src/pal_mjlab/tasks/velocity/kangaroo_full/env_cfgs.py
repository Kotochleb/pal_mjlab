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
  REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY,
  REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY,
  HipXyActuation,
  HipZActuation,
  LegLengthActuation,
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
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play)

  model = get_kangaroo_full_model(hip_z=hip_z, hip_xy=hip_xy, leg_length=leg_length)
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
  # Exactly the simple model's joint set: no tendon lengths, and no
  # leg_.*_length_actuator even when the variant has it. The policy sees the
  # same 26 joints it would on the simple model.

  for group in ("actor", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[group].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY
      )

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
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO FULL flat terrain velocity configuration."""
  cfg = pal_kangaroo_full_rough_env_cfg(
    play=play, hip_z=hip_z, hip_xy=hip_xy, leg_length=leg_length
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
