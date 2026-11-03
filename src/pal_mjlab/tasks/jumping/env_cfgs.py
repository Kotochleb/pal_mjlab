"""Vertical jumping on a plane, using Kangaroo Full's flat-task sensors/actions."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import builtin_sensor, time_out
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,  # noqa: F401  (the commented-out joint_symmetry term)
  TerminationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  ActuatorModel,
  FemurClosure,
  LowerBody,
  MjcfVariant,
  Transmission,
)
from pal_mjlab.tasks.jumping import mdp
from pal_mjlab.tasks.tracking.kangaroo_full.env_cfgs import (
  full_model_tracking_actor_observations,
)
from pal_mjlab.tasks.velocity.kangaroo_full.env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
)

# All jump heights are the world-frame z of JUMP_FRAME (ground plane at z = 0):
# a site between the front and rear upper torso cameras in x, at y = 0 and at
# their height, on pelvis_2_link.
# With both pelvis joints at zero it sits 0.4849 m above the base link
# (-0.0455 pelvis_base_link + 0.052 pelvis_1_link + 0.0591 pelvis_2_link, turned
# vertical by pelvis_1_link's +90 deg about y, + 0.4193 the site): every height
# below that was a base-link height is that 0.485 m higher, rounded to 5 mm.
JUMP_FRAME = "torso_up_cameras_frame"

# Environment steps per PPO iteration of the jump task (its rl_cfg's
# num_steps_per_env): the curricula below count PPO iterations in these.
ITERATION_STEPS = 64

# Sampled target apex range (minimum, maximum), world-frame metres of
# JUMP_FRAME, which stands at 1.415 m. Training starts at the apex of
# motions/jump_ocp_solution_mjlab.npz (base_link 1.278 m, frame 45, so about
# 1.763 m), rounded to 1.765 m, +-5 cm, held for the first 500 PPO iterations
# (HEIGHT_HOLD_STEPS). The whole 10 cm range then moves in stairs:
# one decision every HEIGHT_STAIR_STEPS, up by HEIGHT_STEP_UP (5 cm) while
# jumps succeed, never past HEIGHT_FINAL_RANGE's 3.0 m top (HEIGHT_STAIRS = 24
# stairs, the last one capped), and back by HEIGHT_STEP_DOWN (1 cm) while the
# jump success rate (episodes ending on jump_finished) is below
# HEIGHT_SUCCESS_THRESHOLD. The success rate is a moving average over about
# HEIGHT_SUCCESS_WINDOW episodes of the current stair. While jumps succeed but the smoothed |apex - target| (same
# window, jumps only) exceeds HEIGHT_MAX_APEX_ERROR, the range stays. Play
# samples HEIGHT_FINAL_RANGE.
HEIGHT_INITIAL_RANGE = (1.715, 1.815)
HEIGHT_FINAL_RANGE = (2.800, 3.000)
HEIGHT_HOLD_STEPS = 500 * ITERATION_STEPS
HEIGHT_STEP_UP = 0.05
HEIGHT_STEP_DOWN = 0.01
# With no setbacks the top stair is reached by PPO iteration 20,000: the first
# stair is decided when the hold ends, the other HEIGHT_STAIRS - 1 are spread
# evenly up to HEIGHT_FINAL_STEP (about 848 iterations apart).
HEIGHT_FINAL_STEP = 20_000 * ITERATION_STEPS
HEIGHT_STAIRS = math.ceil(
  (HEIGHT_FINAL_RANGE[1] - HEIGHT_INITIAL_RANGE[1]) / HEIGHT_STEP_UP - 1e-9
)
HEIGHT_STAIR_STEPS = (HEIGHT_FINAL_STEP - HEIGHT_HOLD_STEPS) // (HEIGHT_STAIRS - 1)
HEIGHT_SUCCESS_THRESHOLD = 0.5
HEIGHT_SUCCESS_WINDOW = 4096
HEIGHT_MAX_APEX_ERROR = 0.05

# JUMP_FRAME's world z with Kangaroo Full standing at its nominal 0.93 m base
# height is 1.415 m; the lowest apex that counts as a jump is 3 cm above it.
MIN_JUMP_HEIGHT = 1.445

# Seconds the landed robot must stay settled (both feet down, every joint
# slower than the settle speed; any pose) to end the episode at full settle
# tightness; landing_completed pays once, on that step, more the closer the
# pose is. Training (SETTLE_STAGES) only goes halfway to it.
LANDING_HOLD_TIME = 0.2
# landing_completed's pose kernel std, relative to the standing pose reward's.
SETTLE_POSE_STD_SCALE = 2.0
# (environment steps, settled joint speed limit in rad/s, settled hold time in
# s): control points in PPO iterations (ITERATION_STEPS), linearly interpolated in
# between (constant after the last). The loosest stage holds unchanged for the
# first 150 iterations; by iteration 800 the terminal state has tightened half
# of the way to full tightness (JumpHeightCommandCfg's default settle_joint_speed
# of 0.1 rad/s and a LANDING_HOLD_TIME hold): a 0.3 rad/s speed limit and a
# 0.1 s hold (5 control steps of 0.02 s). Play uses that last stage too.
SETTLE_STAGES = (
  (0, 0.5, 0.0),
  (150 * ITERATION_STEPS, 0.5, 0.0),
  (800 * ITERATION_STEPS, 0.3, LANDING_HOLD_TIME / 2),
)

# Frames of each observation term given to the actor and the critic.
OBSERVATION_HISTORY = 10

# Training only: the share of episodes restarted from a stored state of an
# earlier jump that finished (jump_state_replay), how many such states are kept,
# the best share of them (by episode return) that restarts sample from, the
# maximum random delay (s) after the drawn state transition at which an episode
# records its state, and the half-width of the uniform noise (m) on the target
# apex a restored state comes with.
REPLAY_PROBABILITY = 0.2
REPLAY_CAPACITY = 4096
REPLAY_TOP_FRACTION = 0.25
REPLAY_MAX_OFFSET = 0.3
REPLAY_TARGET_NOISE = 0.005
# Share of the replay buffer kept for states stored at earlier height
# curriculum stairs, while the curriculum is not failing.
REPLAY_KEEP_PREVIOUS_FRACTION = 0.5

# Base distance (m) from the environment origin that ends the episode: before
# landing (stance and flight), and once both feet are down.
MAX_JUMP_DRIFT = 1.0
MAX_LANDED_DRIFT = 3.0
# Base rotation (deg, about any axis) from the episode's start that ends it.
MAX_ROTATION_DEG = 30.0
# Standing too long before the jump ends the episode (idle_before_jump), only
# while still in stance (before takeoff): after MAX_STANCE_TIME s in stance, or
# MAX_IDLE_TIME s in total of standing still there (base slower than the
# command's idle_speed, 0.1 m/s).
MAX_STANCE_TIME = 2.5
MAX_IDLE_TIME = 0.5
# start_orientation reward: weight and kernel std (rad) of the base's rotation
# from the episode's start; exp(-1) at about 14 deg, ~0 near MAX_ROTATION_DEG.
START_ORIENTATION_WEIGHT = 1.0
START_ORIENTATION_STD = 0.25

# Leg joints the symmetry cost pairs; the pitch-like ones match left to right,
# the MIRRORED ones (hip yaw/roll, ankle roll) are equal and opposite.
LEG_SYMMETRY_JOINTS = r"leg_(left|right)_([1-5]|knee)_joint"
LEG_MIRRORED_JOINTS = r"leg_left_[135]_joint"
# The same for the arms, on full-body robots: shoulder 2 and elbow (4) match,
# shoulder 1 and 3 are equal and opposite (as their default positions are).
ARM_SYMMETRY_JOINTS = r"arm_(left|right)_[1-4]_joint"
ARM_MIRRORED_JOINTS = r"arm_left_[13]_joint"
# Free jumps with arms: the grounded posture reward's std (rad) for every arm
# joint, wide so the arms can swing through the countermovement and push-off
# (the other joints keep the standing 0.05 rad). Landing still settles towards
# the default arm pose: landing_completed's kernel is the shared settle one.
ARM_POSE_STD = 0.5
# Free jumps with arms: velocity-task rewards kept on top of VELOCITY_REWARDS,
# as swinging arms can hit the torso and legs.
ARM_VELOCITY_REWARDS = ("self_collisions",)

VELOCITY_REWARDS = (
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
  # Hip and ankle XY convex-hull joint limits (ankle femur-normalized per MJCF).
  "convex_hull_joint_limits_hip",
  "convex_hull_joint_limits_ankle",
)


# Half-width of the uniform error on the actor's base state: 3 cm in height,
# 3 cm/s per velocity axis.
BASE_STATE_NOISE = 0.03


def configure_jump_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Appends the base linear velocity and world-frame base height to the actor,
  each +-BASE_STATE_NOISE, and stacks OBSERVATION_HISTORY frames of every term
  in the actor and the critic (flattened).

  The velocity is built like mjlab's velocity-env ``base_lin_vel`` term (the
  IMU velocimeter, in the IMU frame), with this noise; the critics already observe
  it noise-free, so they only gain the clean height. Shared by the free and
  tracking jump tasks, so both actors keep one layout and the jump runner can
  still load the tracking-pretrained actor. Call it after the actor group is
  final: the terms go last, in this order.
  """
  actor = cfg.observations["actor"].terms
  actor["base_lin_vel"] = ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
    noise=Unoise(n_min=-BASE_STATE_NOISE, n_max=BASE_STATE_NOISE),
  )
  actor["base_height_world"] = ObservationTermCfg(
    func=mdp.base_height_world,
    noise=Unoise(n_min=-BASE_STATE_NOISE, n_max=BASE_STATE_NOISE),
  )
  cfg.observations["critic"].terms["base_height_world"] = ObservationTermCfg(
    func=mdp.base_height_world
  )
  for group in ("actor", "critic"):
    cfg.observations[group].history_length = OBSERVATION_HISTORY
    cfg.observations[group].flatten_history_dim = True


def jump_state_machine_params(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> dict:
  """JumpHeightCommandCfg fields shared by every task with this jump state
  machine, from a fresh full-model velocity config: settling on its joints'
  speed alone, for a hold time (both the first curriculum stage's, the last
  one's in play), the landing pose kernel (SETTLE_POSE_STD_SCALE x its standing std), the jump
  frame and the lowest apex that counts as a jump, in its world z."""
  pose = cfg.rewards["pose"]
  return {
    "settle_joint_names": tuple(pose.params["asset_cfg"].joint_names),
    "settle_pose_std": {
      pattern: SETTLE_POSE_STD_SCALE * std
      for pattern, std in pose.params["std_standing"].items()
    },
    "settle_joint_speed": SETTLE_STAGES[-1 if play else 0][1],
    "settle_hold_time": SETTLE_STAGES[-1 if play else 0][2],
    "settle_joint_speed_only": True,
    "min_jump_height": MIN_JUMP_HEIGHT,
    "height_site_name": JUMP_FRAME,
  }


def configure_vertical_jump(
  cfg: ManagerBasedRlEnvCfg, *, lower_body: LowerBody, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Adapt a fresh flat velocity config into a vertical jump task.

  The actor observes exactly what the full-model tracking actor does, with a
  scalar world-frame apex target (zero after landing) as its command, plus the
  noisy IMU base velocity and world-frame base height. The critic keeps the
  velocity task's observations, with that command and the clean base state.
  Both stack OBSERVATION_HISTORY frames.

  ``lower_body`` must match the config's robot. With arms, the policy may use
  them to jump: they are paired in the symmetry cost, loose (ARM_POSE_STD) in
  the grounded posture reward, and self-collisions are penalized.
  """
  pose = cfg.rewards["pose"]
  foot_slip = cfg.rewards["foot_slip"]
  cfg.commands = {
    "jump_height": mdp.JumpHeightCommandCfg(
      height_range=HEIGHT_FINAL_RANGE if play else HEIGHT_INITIAL_RANGE,
      **jump_state_machine_params(cfg, play=play),
    )
  }
  actor = full_model_tracking_actor_observations(cfg)
  actor.enable_corruption = cfg.observations["actor"].enable_corruption
  cfg.observations["actor"] = actor
  for group in cfg.observations.values():
    group.terms["command"].params = {"command_name": "jump_height"}
  # Privileged: the jump frame's world-frame height, which targets and apex use.
  cfg.observations["critic"].terms["jump_frame_height"] = ObservationTermCfg(
    func=mdp.jump_frame_height, params={"command_name": "jump_height"}
  )
  configure_jump_observations(cfg)

  kept_rewards = (
    VELOCITY_REWARDS if lower_body else VELOCITY_REWARDS + ARM_VELOCITY_REWARDS
  )
  cfg.rewards = {name: cfg.rewards[name] for name in kept_rewards}
  posture_std = deepcopy(pose.params["std_standing"])
  # Used by the (commented-out) joint_symmetry term below.
  symmetry_joints, mirrored_joints = (LEG_SYMMETRY_JOINTS,), (LEG_MIRRORED_JOINTS,)
  if not lower_body:
    # The standing std's regex also matches the arms; take them out of it.
    posture_std = {rf"(?!arm_){pattern}": std for pattern, std in posture_std.items()}
    posture_std[r"arm_.*"] = ARM_POSE_STD
    symmetry_joints += (ARM_SYMMETRY_JOINTS,)
    mirrored_joints += (ARM_MIRRORED_JOINTS,)
  cfg.rewards.update(
    {
      "jump_height": RewardTermCfg(
        func=mdp.jump_height,
        weight=45.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "estimated_jump_height": RewardTermCfg(
        func=mdp.estimated_jump_height,
        weight=30.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "pose": RewardTermCfg(
        # Only while a foot touches the ground; nothing in the air.
        func=mdp.grounded_posture,
        weight=0.1,
        params={
          "command_name": "jump_height",
          "asset_cfg": deepcopy(pose.params["asset_cfg"]),
          "std": posture_std,
        },
      ),
      # Keep the base's orientation from the episode start, in every phase.
      "start_orientation": RewardTermCfg(
        func=mdp.start_orientation,
        weight=START_ORIENTATION_WEIGHT,
        params={"command_name": "jump_height", "std": START_ORIENTATION_STD},
      ),
      "horizontal_velocity": RewardTermCfg(
        func=mdp.horizontal_velocity,
        weight=2.0,
        params={"command_name": "jump_height", "std": 0.2},
      ),
      # The velocity task's foot-slip cost on its foot sites, in every phase.
      "foot_slip": RewardTermCfg(
        func=mdp.feet_slip,
        weight=foot_slip.weight,
        params={
          "sensor_name": foot_slip.params["sensor_name"],
          "asset_cfg": deepcopy(foot_slip.params["asset_cfg"]),
        },
      ),
      # Disabled for now (user, 2026-11-03); uncomment to restore.
      # Left/right axes are identical: pitch-like joints match, the others
      # are equal and opposite (LEG_/ARM_MIRRORED_JOINTS).
      # "joint_symmetry": RewardTermCfg(
      #   func=mdp.joint_symmetry_l2,
      #   weight=-7.0,
      #   params={
      #     "asset_cfg": SceneEntityCfg("robot", joint_names=symmetry_joints),
      #     "mirrored_joints": mirrored_joints,
      #   },
      # ),
      "landing_completed": RewardTermCfg(
        func=mdp.landing_completed,
        weight=120.0,
        params={"command_name": "jump_height"},
      ),
      # Keeping the touchdown's downward base speed in the first 0.3 s after it.
      "catch_landing": RewardTermCfg(
        func=mdp.catch_landing,
        weight=3.0,
        params={"command_name": "jump_height", "window": 0.3},
      ),
    }
  )
  cfg.curriculum = (
    {}
    if play
    else {
      "jump_height": CurriculumTermCfg(
        func=mdp.jump_height_curriculum,
        params={
          "command_name": "jump_height",
          "termination_name": "jump_finished",
          "initial_range": HEIGHT_INITIAL_RANGE,
          "max_height": HEIGHT_FINAL_RANGE[1],
          "hold_steps": HEIGHT_HOLD_STEPS,
          "stair_steps": HEIGHT_STAIR_STEPS,
          "step_up": HEIGHT_STEP_UP,
          "step_down": HEIGHT_STEP_DOWN,
          "success_threshold": HEIGHT_SUCCESS_THRESHOLD,
          "success_window": HEIGHT_SUCCESS_WINDOW,
          "max_apex_error": HEIGHT_MAX_APEX_ERROR,
        },
      ),
      "settle": CurriculumTermCfg(
        func=mdp.settle_curriculum,
        params={"command_name": "jump_height", "stages": SETTLE_STAGES},
      ),
    }
  )
  cfg.episode_length_s = 25.0
  cfg.terminations.pop("fell_over", None)  # Replaced by per-axis attitude limits.
  # A vertical jump on a plane: horizontal_drift already bounds the travel.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.update(
    {
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      # The task's successful end: a true terminal state (time_out=False), so
      # PPO does not bootstrap a value after it.
      "jump_finished": TerminationTermCfg(
        func=mdp.jump_finished,
        time_out=False,
        params={"command_name": "jump_height"},
      ),
      # "bad_touchdown": TerminationTermCfg(
      #   func=mdp.bad_touchdown,
      #   params={
      #     "command_name": "jump_height",
      #     "max_height_error": 0.3,
      #     "min_base_height": 0.2,
      #   },
      # ),
      "sideways_takeoff": TerminationTermCfg(
        func=mdp.sideways_takeoff,
        params={"command_name": "jump_height", "max_speed": 0.75},
      ),
      # "walking_before_jump": TerminationTermCfg(
      #   func=mdp.walking_before_jump,
      #   params={"command_name": "jump_height", "max_distance": 0.2},
      # ),
      # Distance from the env origin: MAX_JUMP_DRIFT until both feet land,
      # MAX_LANDED_DRIFT after.
      "horizontal_drift": TerminationTermCfg(
        func=mdp.horizontal_drift,
        params={
          "command_name": "jump_height",
          "max_distance": MAX_JUMP_DRIFT,
          "max_landed_distance": MAX_LANDED_DRIFT,
        },
      ),
      "excessive_rotation": TerminationTermCfg(
        func=mdp.excessive_rotation,
        params={"command_name": "jump_height", "max_angle_deg": MAX_ROTATION_DEG},
      ),
      # One jump per episode: a second takeoff after landing is a failure.
      "jumped_again": TerminationTermCfg(
        func=mdp.jumped_again,
        params={"command_name": "jump_height"},
      ),
      "idle_before_jump": TerminationTermCfg(
        func=mdp.idle_before_jump,
        params={
          "command_name": "jump_height",
          "max_stance_time": MAX_STANCE_TIME,
          "max_idle_time": MAX_IDLE_TIME,
        },
      ),
    }
  )
  # External impulses would contaminate ballistic height tracking.
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  # Every episode starts standing still at the nominal pose and heading.
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_base"].params["velocity_range"] = {}
  if not play:
    cfg.events["jump_state_replay"] = EventTermCfg(
      func=mdp.jump_state_replay,
      mode="step",
      params={
        "command_name": "jump_height",
        "termination_name": "jump_finished",
        "probability": REPLAY_PROBABILITY,
        "capacity": REPLAY_CAPACITY,
        "top_fraction": REPLAY_TOP_FRACTION,
        "max_offset": REPLAY_MAX_OFFSET,
        "target_noise": REPLAY_TARGET_NOISE,
        "curriculum_name": "jump_height",
        "keep_previous_fraction": REPLAY_KEEP_PREVIOUS_FRACTION,
      },
    )
  return cfg


def pal_kangaroo_full_vertical_jump_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
) -> ManagerBasedRlEnvCfg:
  return configure_vertical_jump(
    pal_kangaroo_full_flat_env_cfg(
      play=play,
      transmission=transmission,
      femur_closure=femur_closure,
      mjcf=mjcf,
      lower_body=lower_body,
      ankle_normalized=ankle_normalized,
      actuator_model=actuator_model,
    ),
    lower_body=lower_body,
    play=play,
  )
