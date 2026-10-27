"""Kangaroo Full tracking of a jump motion, observing a target jump height."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg

from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots import FemurClosure
from pal_mjlab.tasks.jumping.env_cfgs import jump_state_machine_params
from pal_mjlab.tasks.tracking.kangaroo_full.env_cfgs import (
  configure_full_model_tracking,
)

from . import mdp

# Simple-model lower-body jump (LOWER_BODY_JOINT_ORDER); its base_link peaks at
# 1.278 m in the world frame. Override with --env.commands.motion.motion-file.
JUMP_MOTION_FILE = (
  PAL_MJLAB_SRC_PATH.parents[1] / "motions" / "jump_ocp_solution_mjlab.npz"
)

# Half-width (m) of the uniform noise on each episode's target apex.
APEX_NOISE = 0.1


def configure_full_model_jump_tracking(
  cfg: ManagerBasedRlEnvCfg,
  *,
  femur_closure: FemurClosure,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Turns a fresh lower-body full-model flat velocity config into jump tracking.

  This is ``configure_full_model_tracking`` (rewards, critic, terminations,
  events) on the jump motion, except for what the policy observes: the actor's
  command is the vertical-jump task's target apex (``jump_height``: the motion's
  apex + U(-APEX_NOISE, APEX_NOISE), zero once both feet land) instead of the
  motion's reference joints. The actor and actions are thus exactly the
  ``Mjlab-Jump-Vertical-*`` task's, whose runner loads this policy's actor.
  Every episode is one pass through the motion, from its first frame.
  """
  jump_params = jump_state_machine_params(cfg, play=play)
  cfg = configure_full_model_tracking(
    cfg, femur_closure=femur_closure, lower_body=True, play=play
  )

  motion = cfg.commands["motion"]
  motion.motion_file = str(JUMP_MOTION_FILE)
  # The jump state machine needs every episode to start standing, before takeoff.
  motion.sampling_mode = "start"
  # After "motion": its reset reads the reset pose the motion command writes.
  cfg.commands["jump_height"] = mdp.MotionApexJumpHeightCommandCfg(
    apex_noise=APEX_NOISE, **jump_params
  )
  cfg.observations["actor"].terms["command"].params = {"command_name": "jump_height"}

  cfg.terminations["motion_finished"] = TerminationTermCfg(
    func=mdp.motion_finished, time_out=True, params={"command_name": "motion"}
  )
  return cfg
