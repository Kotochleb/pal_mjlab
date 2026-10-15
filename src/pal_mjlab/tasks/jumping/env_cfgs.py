"""Vertical jumping on a plane, using Kangaroo Full's flat-task sensors/actions."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg, TerminationTermCfg

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
  (0, 0.05, 0.15),
  (2_000 * 24, 0.05, 0.25),
  (5_000 * 24, 0.05, 0.40),
  (10_000 * 24, 0.05, 0.60),
  (15_000 * 24, 0.05, 0.80),
)

# LP-ACRL splits the final stage's range into equal-width tasks instead.
HEIGHT_BINS = 15

VELOCITY_REWARDS = (
  "upright",
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
  "foot_slip",
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
  cfg.commands = {
    "jump_height": mdp.JumpHeightCommandCfg(
      height_range=HEIGHT_STAGES[-1][1:] if play else HEIGHT_STAGES[0][1:],
      flight_reset_probability=0.0 if play else 0.2,
    )
  }
  for group in cfg.observations.values():
    group.terms["command"].params = {"command_name": "jump_height"}

  cfg.rewards = {name: cfg.rewards[name] for name in VELOCITY_REWARDS}
  # These costs must remain active after the height input becomes zero, and
  # must not interpret a one-dimensional height as a three-dimensional twist.
  slip = cfg.rewards["foot_slip"]
  slip.func = mdp.feet_slip
  slip.params.pop("command_name", None)
  slip.params.pop("command_threshold", None)
  cfg.rewards.update(
    {
      "jump_height": RewardTermCfg(
        func=mdp.jump_height,
        weight=20.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "estimated_jump_height": RewardTermCfg(
        func=mdp.estimated_jump_height,
        weight=20.0,
        params={"command_name": "jump_height", "std": 0.1, "laplace_scale": 0.05},
      ),
      "pose": RewardTermCfg(
        func=mdp.height_variable_posture,
        weight=1.0,
        params={
          "command_name": "jump_height",
          "asset_cfg": deepcopy(pose.params["asset_cfg"]),
          "std_ground": deepcopy(pose.params["std_running"]),
          "std_airborne": deepcopy(pose.params["std_walking"]),
          "height_range": (0.0, 0.25),
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
      "soft_impact": RewardTermCfg(
        func=mdp.soft_impact,
        weight=1.0,
        params={"command_name": "jump_height", "max_acceleration": 100.0},
      ),
      "landing_drift": RewardTermCfg(
        func=mdp.landing_drift_l2,
        weight=-0.01,
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
      )
    }
  )
  if curriculum == "lp_acrl":
    _configure_lp_acrl(cfg, play=play)
  elif curriculum != "default":
    raise ValueError(f"Unknown curriculum: {curriculum!r}")
  cfg.episode_length_s = 15.0
  cfg.terminations["jump_finished"] = TerminationTermCfg(
    func=mdp.jump_finished,
    time_out=True,
    params={"command_name": "jump_height", "settle_time": 2.0},
  )
  # External impulses would contaminate ballistic height tracking.
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (0.0, 0.5),
    "yaw": (-3.14, 3.14),
  }
  cfg.events["reset_base"].params["velocity_range"] = {}
  return cfg


def _configure_lp_acrl(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Replace the time-based height schedule with LP-ACRL over height bins.

  Training and play share the full final-stage range; play samples bins
  uniformly. The reset term is needed in play too, to assign each episode a bin.
  """
  old = cfg.commands["jump_height"]
  cfg.commands["jump_height"] = mdp.LpAcrlJumpHeightCommandCfg(
    height_range=HEIGHT_STAGES[-1][1:],
    flight_reset_probability=old.flight_reset_probability,
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
