"""Vertical jumping on a plane, using Kangaroo Full's flat-task sensors/actions."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers import (
  CurriculumTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)

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

# All jump heights are the base link's world-frame z (ground plane at z = 0).

# (environment steps, minimum apex, maximum apex), world-frame metres; Kangaroo
# Full stands with its base at 0.93 m. Training starts at the world-frame apex of
# motions/jump_ocp_solution_mjlab.npz (base_link 1.278 m, frame 45), rounded to
# 1.280 m, +-5 cm, then widens to +-15 cm. Each later stage raises the upper
# bound 5 cm and narrows the range 2.5 cm, so the lower bound rises 7.5 cm,
# until the range is 7.5 cm wide (iteration 10,000). The 7.5 cm range then keeps
# rising 5 cm per stage, capped at a 2 m upper bound. Bounds end in 0 or 5 mm.
# A new stage every 1,000 PPO iterations of 24 steps, up to 13,000.
HEIGHT_STAGES = (
  (0, 1.230, 1.330),
  (1_000 * 24, 1.130, 1.430),
  (2_000 * 24, 1.205, 1.480),
  (3_000 * 24, 1.280, 1.530),
  (4_000 * 24, 1.355, 1.580),
  (5_000 * 24, 1.430, 1.630),
  (6_000 * 24, 1.505, 1.680),
  (7_000 * 24, 1.580, 1.730),
  (8_000 * 24, 1.655, 1.780),
  (9_000 * 24, 1.730, 1.830),
  (10_000 * 24, 1.805, 1.880),
  (11_000 * 24, 1.855, 1.930),
  (12_000 * 24, 1.905, 1.980),
  (13_000 * 24, 1.925, 2.000),
)

# Former rise-based thresholds, now offsets above the robot's nominal standing
# base height, converted to world z per robot: the posture blend band and the
# lowest apex that counts as a jump.
POSTURE_BLEND_RISE = (0.0, 0.05)
MIN_JUMP_RISE = 0.03

# Final seconds the landed robot must stay settled (both feet down, every
# joint slower than the settle speed; any pose) to end the episode;
# landing_completed pays once, on that step, more the closer the pose is.
LANDING_HOLD_TIME = 0.5
# landing_completed's pose kernel std, relative to the standing pose reward's.
SETTLE_POSE_STD_SCALE = 2.0
# Every joint's airborne std in the pose reward, in rad (m for prismatic
# joints): large enough that the posture is effectively free in flight.
AIRBORNE_JOINT_STD = 1e4
# (environment steps, settled joint speed limit in rad/s, settled hold time in
# s): control points every 500 PPO iterations of 24 steps, linearly
# interpolated in between (constant after the last). The speed limit tightens
# down to JumpHeightCommandCfg's default settle_joint_speed (0.1 rad/s) while
# the hold grows from 0 to LANDING_HOLD_TIME; control-point holds are rounded to
# the 0.02 s control step (0, 8, 17, 25 steps).
SETTLE_STAGES = (
  (0, 0.5, 0.0),
  (500 * 24, 0.25, 0.16),
  (1_000 * 24, 0.15, 0.34),
  (1_500 * 24, 0.1, LANDING_HOLD_TIME),
)

# Frames of each observation term given to the critic.
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


def jump_state_machine_params(cfg: ManagerBasedRlEnvCfg, play: bool = False) -> dict:
  """JumpHeightCommandCfg fields shared by every task with this jump state
  machine, from a fresh full-model velocity config: settling on its joints'
  speed alone, for a hold time (both the first curriculum stage's, the last
  one's in play), the landing pose kernel (SETTLE_POSE_STD_SCALE x its standing std) and the lowest
  apex that counts as a jump, in world z."""
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
    "min_jump_height": cfg.scene.entities["robot"].init_state.pos[2] + MIN_JUMP_RISE,
  }


def configure_vertical_jump(
  cfg: ManagerBasedRlEnvCfg, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Adapt a fresh flat velocity config into a vertical jump task.

  The actor observes exactly what the full-model tracking actor does, with a
  scalar world-frame apex target (zero after landing) as its command. The critic keeps the
  velocity task's observations, with that command and stacked history.
  """
  pose = cfg.rewards["pose"]
  foot_slip = cfg.rewards["foot_slip"]
  standing_height = cfg.scene.entities["robot"].init_state.pos[2]
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
  # Privileged: the world-frame base height the targets and apex are measured in.
  cfg.observations["critic"].terms["base_height"] = ObservationTermCfg(
    func=mdp.base_height, params={"command_name": "jump_height"}
  )
  cfg.observations["critic"].history_length = OBSERVATION_HISTORY

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
          "height_range": tuple(standing_height + h for h in POSTURE_BLEND_RISE),
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
        weight=120.0,
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
