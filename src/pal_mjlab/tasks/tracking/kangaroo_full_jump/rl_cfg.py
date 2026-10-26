"""RL configuration for Kangaroo Full jump-motion tracking."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.tracking.kangaroo.rl_cfg import (
  pal_kangaroo_tracking_ppo_runner_cfg,
)


def pal_kangaroo_full_jump_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Tracking PPO; its actor network matches the vertical-jump task's, so the
  jump runner can load the trained actor."""
  cfg = pal_kangaroo_tracking_ppo_runner_cfg()
  cfg.experiment_name = "kangaroo_full_jump_tracking"
  return cfg
