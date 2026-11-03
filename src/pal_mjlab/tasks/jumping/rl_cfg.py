from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.jumping.env_cfgs import ITERATION_STEPS
from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import (
  pal_kangaroo_full_ppo_runner_cfg,
)

# Frames of motions/jump_ocp_solution_mjlab.npz (50 fps, 4.3 s: the jump and
# its 2 s of held final pose). The policy also steps at 0.02 s, so this is the
# whole reference jump in policy steps.
JUMP_MOTION_FRAMES = 215

# Effective discount horizon 1 / (1 - gamma), relative to the reference jump's
# length: 1.5 x 215 = 322.5 policy steps (6.45 s), so the landing bonus at the
# end of a jump as long as the reference still weighs (1 - 1/322.5)^215 ~ 0.51
# at the crouch that starts it (0.99 gave a 100-step, 2 s horizon).
JUMP_HORIZON_SCALE = 1.5
JUMP_GAMMA = 1.0 - 1.0 / (JUMP_HORIZON_SCALE * JUMP_MOTION_FRAMES)
# GAE lambda: spreads the sparse apex and landing payouts further back.
JUMP_LAM = 0.98


def pal_kangaroo_full_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The full-model velocity PPO with a jump-length credit horizon: gamma for
  JUMP_HORIZON_SCALE x JUMP_MOTION_FRAMES steps, lambda JUMP_LAM, and
  ITERATION_STEPS-long rollouts (1.28 s instead of 0.48 s), so the push-off and
  its apex payout usually fall in one rollout."""
  cfg = pal_kangaroo_full_ppo_runner_cfg()
  cfg.experiment_name = "kangaroo_full_vertical_jump"
  cfg.algorithm.gamma = JUMP_GAMMA
  cfg.algorithm.lam = JUMP_LAM
  cfg.num_steps_per_env = ITERATION_STEPS
  return cfg
