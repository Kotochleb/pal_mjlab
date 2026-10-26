"""PAL Robotics Kangaroo Full flat terrain tracking configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  LOWER_BODY_JOINT_ORDER,
  SIMPLE_MODEL_JOINT_ORDER,
  ActuatorModel,
  FemurClosure,
  LowerBody,
  MjcfVariant,
  Transmission,
)
from pal_mjlab.tasks.tracking import mdp as tracking_mdp
from pal_mjlab.tasks.tracking.kangaroo.env_cfgs import (
  pal_kangaroo_flat_tracking_env_cfg,
)
from pal_mjlab.tasks.tracking.kangaroo_lower_body.env_cfgs import (
  pal_kangaroo_lower_body_flat_tracking_env_cfg,
)
from pal_mjlab.tasks.velocity import mdp
from pal_mjlab.tasks.velocity.kangaroo_full import mdp as full_mdp
from pal_mjlab.tasks.velocity.kangaroo_full.env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
)

# The full model has no arm tip link; its tool link sits at the arm's end instead.
FULL_BODY_TRACKED_BODIES = (
  "base_link",
  "pelvis_2_link",
  "leg_left_3_link",
  "leg_left_4_link",
  "leg_left_5_link",
  "leg_right_3_link",
  "leg_right_4_link",
  "leg_right_5_link",
  "arm_left_2_link",
  "arm_left_3_link",
  "arm_left_tool_link",
  "arm_right_2_link",
  "arm_right_3_link",
  "arm_right_tool_link",
)
LOWER_BODY_TRACKED_BODIES = tuple(
  name for name in FULL_BODY_TRACKED_BODIES if not name.startswith("arm_")
)

# Source joints re-solved by loop closure rather than copied from the simple
# motion: whichever of femur/knee the leg-length DOF does not set.
_SOLVED_JOINTS: dict[FemurClosure, tuple[str, ...]] = {
  "prismatic": (r"leg_(left|right)_(femur|knee)_joint",),
  "linkage": (r"leg_(left|right)_femur_joint",),
}

# Proprioception and DR come from the velocity config, so the joint vectors
# match the full-model velocity policies (simple-model order, mapped leg length,
# femur-normalized ankle): its replacements for the simple joint observations.
_PROPRIOCEPTION_TERMS = ("joint_pos", "joint_vel", "actions")
# Actor terms that need a state estimator; the robot has none.
_STATE_ESTIMATION_TERMS = ("motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel")
# Reset is the motion command's; velocity-only events go.
_VELOCITY_EVENTS = ("reset_base", "reset_robot_joints", "randomize_terrain")


def configure_full_model_tracking(
  cfg: ManagerBasedRlEnvCfg,
  *,
  femur_closure: FemurClosure,
  lower_body: LowerBody,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Turns a fresh full-model flat velocity config into a motion tracking task.

  Mirrors ``pal_kangaroo_flat_tracking_env_cfg`` for the full body and
  ``pal_kangaroo_lower_body_flat_tracking_env_cfg`` for ``lower_body``. Motions
  are the simple-model ones (``kangaroo`` / ``kangaroo_lower_body`` npz), which
  ``FullModelMotionCommand`` converts to the full model at startup.
  """
  tracking = make_tracking_env_cfg()

  # The velocity task's replacement for the simple joint_pos observation; the
  # motion command observes the reference joints the same way.
  joint_pos_obs = cfg.observations["actor"].terms["joint_pos"]
  ankle_femur_pairs: tuple[tuple[str, str], ...] = ()
  if joint_pos_obs.func is full_mdp.ankle_femur_normalized:
    ankle_femur_pairs = tuple(joint_pos_obs.params["ankle_femur_pairs"])
    joint_pos_obs = joint_pos_obs.params["inner"]
  mapped_joints = tuple(joint_pos_obs.params.get("mapped_joints", ()))

  ## Commands
  motion_cmd = tracking.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd = tracking_mdp.FullModelMotionCommandCfg.from_motion_cfg(
    motion_cmd,
    source_joint_names=(
      LOWER_BODY_JOINT_ORDER if lower_body else SIMPLE_MODEL_JOINT_ORDER
    ),
    solved_joint_names=_SOLVED_JOINTS[femur_closure],
    mapped_joints=mapped_joints,
    ankle_femur_pairs=ankle_femur_pairs,
    tendon_lengths=(
      cfg.events["tendon_lengths"].params["lengths"]
      if "tendon_lengths" in cfg.events
      else None
    ),
  )
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    LOWER_BODY_TRACKED_BODIES if lower_body else FULL_BODY_TRACKED_BODIES
  )
  # Only reaches joints outside the closed loops (see FullModelMotionCommand).
  motion_cmd.joint_position_range = (-0.05, 0.05)
  cfg.commands = {"motion": motion_cmd}

  ## Observations
  for group_name, group in tracking.observations.items():
    velocity_terms = cfg.observations[group_name].terms
    for term in _PROPRIOCEPTION_TERMS:
      group.terms[term] = velocity_terms[term]
    cfg.observations[group_name] = group
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  actor_terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  )
  actor_terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  critic_terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
  )
  critic_terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
  )
  if lower_body:
    critic_terms["foot_air_time"] = ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
  # No state estimation on the robot: the actor sees neither the base pose
  # relative to the motion anchor nor the base linear velocity.
  for term in _STATE_ESTIMATION_TERMS:
    del actor_terms[term]

  ## Rewards
  # The simple model's tracking rewards, term for term, adapted only where the
  # full model's joints read differently.
  velocity_rewards = cfg.rewards
  simple_env_cfg = (
    pal_kangaroo_lower_body_flat_tracking_env_cfg
    if lower_body
    else pal_kangaroo_flat_tracking_env_cfg
  )
  cfg.rewards = simple_env_cfg(play=play).rewards
  # ".*" is the simple model's joints there; here it would add every passive
  # loop joint, so restrict it to those joints, as the velocity task does.
  cfg.rewards["joint_limit"].params["asset_cfg"] = deepcopy(
    velocity_rewards["dof_pos_limits"].params["asset_cfg"]
  )
  # leg_*_4_joint is measured off the shank here; the velocity task's ankle
  # hull carries the femur normalization that maps it back (if the MJCF needs it).
  ankle_hull = cfg.rewards["convex_hull_joint_limits_ankle"]
  velocity_ankle_hull = velocity_rewards["convex_hull_joint_limits_ankle"]
  ankle_hull.func = velocity_ankle_hull.func
  for param in ("femur_joint_names", "femur_offset"):
    if param in velocity_ankle_hull.params:
      ankle_hull.params[param] = velocity_ankle_hull.params[param]

  ## Terminations
  illegal_contacts = cfg.terminations["illegal_contacts"]
  cfg.terminations = tracking.terminations
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    ("leg_left_5_link", "leg_right_5_link")
    if lower_body
    else (
      "leg_left_5_link",
      "leg_right_5_link",
      "arm_left_tool_link",
      "arm_right_tool_link",
    )
  )
  if lower_body:
    cfg.terminations["illegal_contacts"] = illegal_contacts

  ## Events
  for name in _VELOCITY_EVENTS:
    cfg.events.pop(name, None)
  cfg.events["push_robot"] = tracking.events["push_robot"]

  cfg.curriculum = {}
  cfg.episode_length_s = tracking.episode_length_s
  cfg.viewer = tracking.viewer
  cfg.viewer.body_name = "base_link"

  ## Play mode overrides
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg


def pal_kangaroo_full_flat_tracking_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Kangaroo Full flat terrain tracking configuration."""
  return configure_full_model_tracking(
    pal_kangaroo_full_flat_env_cfg(
      play=play,
      transmission=transmission,
      femur_closure=femur_closure,
      mjcf=mjcf,
      lower_body=lower_body,
      ankle_normalized=ankle_normalized,
      actuator_model=actuator_model,
    ),
    femur_closure=femur_closure,
    lower_body=lower_body,
    play=play,
  )
