"""Vertical jumping on a plane, using Kangaroo Full's flat-task sensors/actions."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import builtin_sensor, time_out
from mjlab.managers import (
  CurriculumTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
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

# (environment steps, minimum apex, maximum apex), world-frame metres of
# JUMP_FRAME, which stands at 1.415 m. Training starts at the apex of
# motions/jump_ocp_solution_mjlab.npz (base_link 1.278 m, frame 45, so about
# 1.763 m), rounded to 1.765 m, +-5 cm, held constant for the first 800
# iterations. The lower bound never drops below that start (1.715 m) while the
# upper bound rises 10 cm by iteration 1,800, then 5 cm per 1,000 iterations.
# From iteration 2,800 the range narrows 2.5 cm per 1,000 iterations, so the
# lower bound rises 7.5 cm, until the range is 7.5 cm wide (iteration 10,800).
# The 7.5 cm range then keeps rising 5 cm per 1,000 iterations, capped at a
# 2.985 m upper bound (2.5 m at the base link). Bounds end in 0 or 5 mm. After
# the hold, control points every 1,000 PPO iterations of 24 steps, up to 23,800,
# linearly interpolated in between.
HEIGHT_STAGES = (
  (0, 1.715, 1.815),
  (800 * 24, 1.715, 1.815),
  (1_800 * 24, 1.715, 1.915),
  (2_800 * 24, 1.715, 1.965),
  (3_800 * 24, 1.765, 2.015),
  (4_800 * 24, 1.840, 2.065),
  (5_800 * 24, 1.915, 2.115),
  (6_800 * 24, 1.990, 2.165),
  (7_800 * 24, 2.065, 2.215),
  (8_800 * 24, 2.140, 2.265),
  (9_800 * 24, 2.215, 2.315),
  (10_800 * 24, 2.290, 2.365),
  (11_800 * 24, 2.340, 2.415),
  (12_800 * 24, 2.390, 2.465),
  (13_800 * 24, 2.440, 2.515),
  (14_800 * 24, 2.490, 2.565),
  (15_800 * 24, 2.540, 2.615),
  (16_800 * 24, 2.590, 2.665),
  (17_800 * 24, 2.640, 2.715),
  (18_800 * 24, 2.690, 2.765),
  (19_800 * 24, 2.740, 2.815),
  (20_800 * 24, 2.790, 2.865),
  (21_800 * 24, 2.840, 2.915),
  (22_800 * 24, 2.890, 2.965),
  (23_800 * 24, 2.910, 2.985),
)

# JUMP_FRAME's world z with Kangaroo Full standing at its nominal 0.93 m base
# height is 1.415 m. The posture reward blends from ground to airborne stds over
# its first 5 cm above that, and the lowest apex that counts as a jump is 3 cm
# above it.
POSTURE_BLEND_HEIGHTS = (1.415, 1.465)
MIN_JUMP_HEIGHT = 1.445

# Final seconds the landed robot must stay settled (both feet down, every
# joint slower than the settle speed; any pose) to end the episode;
# landing_completed pays once, on that step, more the closer the pose is.
LANDING_HOLD_TIME = 0.2
# landing_completed's pose kernel std, relative to the standing pose reward's.
SETTLE_POSE_STD_SCALE = 2.0
# Every joint's airborne std in the pose reward, in rad (m for prismatic
# joints): large enough that the posture is effectively free in flight.
AIRBORNE_JOINT_STD = 1e4
# (environment steps, settled joint speed limit in rad/s, settled hold time in
# s): control points in PPO iterations of 24 steps, linearly interpolated in
# between (constant after the last). The loosest stage holds unchanged for the
# first 150 iterations; by iteration 1,100 the speed limit has tightened down to
# JumpHeightCommandCfg's default settle_joint_speed (0.1 rad/s) while the hold
# has grown from 0 to LANDING_HOLD_TIME (10 control steps of 0.02 s).
SETTLE_STAGES = (
  (0, 0.5, 0.0),
  (150 * 24, 0.5, 0.0),
  (800 * 24, 0.1, LANDING_HOLD_TIME),
)

# Frames of each observation term given to the actor and the critic.
OBSERVATION_HISTORY = 10

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
  cfg: ManagerBasedRlEnvCfg, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Adapt a fresh flat velocity config into a vertical jump task.

  The actor observes exactly what the full-model tracking actor does, with a
  scalar world-frame apex target (zero after landing) as its command, plus the
  noisy IMU base velocity and world-frame base height. The critic keeps the
  velocity task's observations, with that command and the clean base state.
  Both stack OBSERVATION_HISTORY frames.
  """
  pose = cfg.rewards["pose"]
  foot_slip = cfg.rewards["foot_slip"]
  cfg.commands = {
    "jump_height": mdp.JumpHeightCommandCfg(
      height_range=HEIGHT_STAGES[-1][1:] if play else HEIGHT_STAGES[0][1:],
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

  cfg.rewards = {name: cfg.rewards[name] for name in VELOCITY_REWARDS}
  cfg.rewards.update(
    {
      "jump_height": RewardTermCfg(
        func=mdp.jump_height,
        weight=40.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "estimated_jump_height": RewardTermCfg(
        func=mdp.estimated_jump_height,
        weight=30.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "pose": RewardTermCfg(
        func=mdp.height_variable_posture,
        weight=0.1,
        params={
          "command_name": "jump_height",
          "asset_cfg": deepcopy(pose.params["asset_cfg"]),
          "std_ground": deepcopy(pose.params["std_standing"]),
          "std_airborne": dict.fromkeys(pose.params["std_running"], AIRBORNE_JOINT_STD),
          "height_range": POSTURE_BLEND_HEIGHTS,
        },
      ),
      "horizontal_velocity": RewardTermCfg(
        func=mdp.horizontal_velocity,
        weight=1.0,
        params={"command_name": "jump_height", "std": 0.2},
      ),
      "ground_force_l2": RewardTermCfg(
        func=mdp.ground_force_l2,
        weight=-0.05,
        params={"sensor_name": "feet_ground_contact", "force_scale": 500.0},
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
      # Left/right axes are identical: pitch-like joints match, the others
      # (hip yaw/roll, ankle roll) are equal and opposite.
      "joint_symmetry": RewardTermCfg(
        func=mdp.joint_symmetry_l2,
        weight=-6.0,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=(r"leg_(left|right)_([1-5]|knee)_joint",)
          ),
          "mirrored_joints": (r"leg_left_[135]_joint",),
        },
      ),
      "landing_completed": RewardTermCfg(
        func=mdp.landing_completed,
        weight=200.0,
        params={"command_name": "jump_height"},
      ),
    }
  )
  cfg.curriculum = (
    {}
    if play
    else {
      "jump_height": CurriculumTermCfg(
        func=mdp.jump_height_curriculum,
        params={"command_name": "jump_height", "stages": HEIGHT_STAGES},
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
      "jump_finished": TerminationTermCfg(
        func=mdp.jump_finished,
        time_out=True,
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
      "horizontal_drift": TerminationTermCfg(
        func=mdp.horizontal_drift,
        params={"command_name": "jump_height", "max_distance": 0.6},
      ),
    }
  )
  # External impulses would contaminate ballistic height tracking.
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  # Every episode starts standing still at the nominal pose and heading.
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_base"].params["velocity_range"] = {}
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
    play=play,
  )
