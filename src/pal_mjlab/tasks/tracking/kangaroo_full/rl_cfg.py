"""RL configuration for PAL Robotics' Kangaroo Full motion imitation task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.tracking.kangaroo.rl_cfg import (
  pal_kangaroo_tracking_ppo_runner_cfg,
)


def pal_kangaroo_full_tracking_ppo_runner_cfg(
  lower_body: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL Kangaroo Full tracking task."""
  cfg = pal_kangaroo_tracking_ppo_runner_cfg()
  cfg.experiment_name = (
    "kangaroo_full_lower_body_tracking" if lower_body else "kangaroo_full_tracking"
  )
  return cfg
