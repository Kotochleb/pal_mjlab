from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.jumping.env_cfgs import ITERATION_STEPS
from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import (
  pal_kangaroo_full_ppo_runner_cfg,
)

# Frames of motions/jump_ocp_solution_mjlab.npz (50 fps, 4.3 s: the jump and
# its 2 s of held final pose). The policy also steps at 0.02 s, so this is the
# whole reference jump in policy steps.
JUMP_MOTION_FRAMES = 215

# Discount whose effective horizon 1 / (1 - gamma) is the reference jump's
# length, so the landing bonus at its end still weighs about e^-1 at the
# crouch that starts it (0.99 gave a 100-step, 2 s horizon).
JUMP_GAMMA = 1.0 - 1.0 / JUMP_MOTION_FRAMES
# GAE lambda: spreads the sparse apex and landing payouts further back.
JUMP_LAM = 0.98


def pal_kangaroo_full_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The full-model velocity PPO with a jump-length credit horizon: gamma from
  JUMP_MOTION_FRAMES, lambda JUMP_LAM, and ITERATION_STEPS-long rollouts
  (1.28 s instead of 0.48 s), so the push-off and its apex payout usually fall
  in one rollout."""
  cfg = pal_kangaroo_full_ppo_runner_cfg()
  cfg.experiment_name = "kangaroo_full_vertical_jump"
  cfg.algorithm.gamma = JUMP_GAMMA
  cfg.algorithm.lam = JUMP_LAM
  cfg.num_steps_per_env = ITERATION_STEPS
  return cfg
