"""Vertical jumping on a plane, using Kangaroo Full's flat-task sensors/actions."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers import (
  CurriculumTermCfg,
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
from pal_mjlab.tasks.velocity.kangaroo_full.env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
)
from pal_mjlab.tasks.velocity.mdp.lp_acrl import (
  Curriculum,
  LpAcrlCfg,
  lp_acrl_curriculum,
)

# (environment steps, minimum rise, maximum rise), metres above standing.
# Keep easier commands in the distribution as the upper bound increases.
HEIGHT_STAGES = (
  (0, 0.30, 0.50),
  (2_000 * 24, 0.20, 0.55),
  (5_000 * 24, 0.10, 0.70),
  (10_000 * 24, 0.05, 0.90),
)

# (environment steps, seconds after touchdown that end the episode and pay the
# landing reward). Training lengthens the landing stage; play uses the last.
# 10 stages, linear from 0.1 s to 5.0 s, every 250 PPO iterations of 24 steps.
LANDING_STAGES = tuple(
  (i * 250 * 24, round(0.1 + i * (5.0 - 0.1) / 9, 3)) for i in range(10)
)

# Frames of each observation term given to actor and critic.
OBSERVATION_HISTORY = 10

# LP-ACRL splits the final stage's range into equal-width tasks instead.
HEIGHT_BINS = 15

VELOCITY_REWARDS = (
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
)


def configure_vertical_jump(
  cfg: ManagerBasedRlEnvCfg, play: bool = False, curriculum: Curriculum = "default"
) -> ManagerBasedRlEnvCfg:
  """Adapt a fresh flat velocity config without changing its proprioception.

  Only the existing command observation changes: twist is replaced by a scalar
  target rise (zero after landing). No height, contact, phase, or terrain-scan
  observations are added to the actor or critic.
  """
  pose = cfg.rewards["pose"]
  landing_time = LANDING_STAGES[-1][1] if play else LANDING_STAGES[0][1]
  cfg.commands = {
    "jump_height": mdp.JumpHeightCommandCfg(
      height_range=HEIGHT_STAGES[-1][1:] if play else HEIGHT_STAGES[0][1:],
    )
  }
  for group in cfg.observations.values():
    group.terms["command"].params = {"command_name": "jump_height"}
  # Per-foot clearance above the ground, as the critic already observes.
  cfg.observations["actor"].terms["foot_height"] = deepcopy(
    cfg.observations["critic"].terms["foot_height"]
  )
  # Stack the last frames of every term, flattened, for both groups.
  for group in cfg.observations.values():
    group.history_length = OBSERVATION_HISTORY

  cfg.rewards = {name: cfg.rewards[name] for name in VELOCITY_REWARDS}
  cfg.rewards.update(
    {
      "jump_height": RewardTermCfg(
        func=mdp.jump_height,
        weight=30.0,
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
          "std_ground": deepcopy(pose.params["std_running"]),
          "std_airborne": deepcopy(pose.params["std_walking"]),
          "height_range": (0.0, 0.05),
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
      # Left/right axes are identical: pitch-like joints match, the others
      # (hip yaw/roll, ankle roll) are equal and opposite.
      "joint_symmetry": RewardTermCfg(
        func=mdp.joint_symmetry_l2,
        weight=-3.0,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=(r"leg_(left|right)_([1-5]|knee)_joint",)
          ),
          "mirrored_joints": (r"leg_left_[135]_joint",),
        },
      ),
      "soft_impact": RewardTermCfg(
        func=mdp.soft_impact,
        weight=1.0,
        params={"command_name": "jump_height", "max_acceleration": 100.0},
      ),
      "landing_completed": RewardTermCfg(
        func=mdp.landing_completed,
        weight=50.0,
        params={"command_name": "jump_height", "landing_time": landing_time},
      ),
      "catch_landing": RewardTermCfg(
        func=mdp.catch_landing,
        weight=3.0,
        params={"command_name": "jump_height", "window": 0.3},
      ),
      "damp_landing": RewardTermCfg(
        func=mdp.damp_landing,
        weight=10.0,
        params={
          "command_name": "jump_height",
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=(r"leg_(left|right)_knee_joint",)
          ),
          "window": 0.3,
        },
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
      )
    }
  )
  if curriculum == "lp_acrl":
    _configure_lp_acrl(cfg, play=play)
  elif curriculum != "default":
    raise ValueError(f"Unknown curriculum: {curriculum!r}")
  if not play:
    cfg.curriculum["landing_time"] = CurriculumTermCfg(
      func=mdp.landing_time_curriculum,
      params={
        "reward_name": "landing_completed",
        "termination_name": "jump_finished",
        "stages": LANDING_STAGES,
      },
    )
  cfg.episode_length_s = 20.0
  cfg.terminations.pop("fell_over", None)  # Replaced by per-axis attitude limits.
  cfg.terminations.update(
    {
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "jump_finished": TerminationTermCfg(
        func=mdp.jump_finished,
        time_out=True,
        params={"command_name": "jump_height", "landing_time": landing_time},
      ),
      "bad_touchdown": TerminationTermCfg(
        func=mdp.bad_touchdown,
        params={
          "command_name": "jump_height",
          "max_height_error": 0.3,
          "min_base_height": 0.2,
        },
      ),
      "jumped_again": TerminationTermCfg(
        func=mdp.jumped_again,
        params={"command_name": "jump_height"},
      ),
      "sideways_takeoff": TerminationTermCfg(
        func=mdp.sideways_takeoff,
        params={"command_name": "jump_height", "max_speed": 0.75},
      ),
      # "walking_before_jump": TerminationTermCfg(
      #   func=mdp.walking_before_jump,
      #   params={"command_name": "jump_height", "max_distance": 0.2},
      # ),
      "idle_before_jump": TerminationTermCfg(
        func=mdp.idle_before_jump,
        params={
          "command_name": "jump_height",
          "max_stance_time": 5.0,
          "max_idle_time": 1.0,
        },
      ),
      "excessive_attitude": TerminationTermCfg(
        func=mdp.excessive_attitude,
        params={"command_name": "jump_height", "max_angle_deg": 25.0},
      ),
      "horizontal_drift": TerminationTermCfg(
        func=mdp.horizontal_drift,
        params={"command_name": "jump_height", "max_distance": 0.2},
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


def _configure_lp_acrl(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Replace the time-based height schedule with LP-ACRL over height bins.

  Training and play share the full final-stage range; play samples bins
  uniformly. The reset term is needed in play too, to assign each episode a bin.
  """
  cfg.commands["jump_height"] = mdp.LpAcrlJumpHeightCommandCfg(
    height_range=HEIGHT_STAGES[-1][1:],
    height_bins=HEIGHT_BINS,
    lp_acrl=LpAcrlCfg(adaptive=not play),
  )
  cfg.curriculum = {
    "lp_acrl": CurriculumTermCfg(
      func=lp_acrl_curriculum, params={"command_name": "jump_height"}
    )
  }


def pal_kangaroo_full_vertical_jump_env_cfg(
  play: bool = False,
  transmission: Transmission = "actuator",
  femur_closure: FemurClosure = "prismatic",
  mjcf: MjcfVariant = "tendons",
  lower_body: LowerBody = False,
  ankle_normalized: bool = False,
  actuator_model: ActuatorModel = "builtin",
  curriculum: Curriculum = "default",
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
    curriculum=curriculum,
  )
