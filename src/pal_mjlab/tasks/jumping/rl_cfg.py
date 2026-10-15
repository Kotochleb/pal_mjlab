from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import (
  pal_kangaroo_full_ppo_runner_cfg,
)


def pal_kangaroo_full_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pal_kangaroo_full_ppo_runner_cfg()
  cfg.experiment_name = "kangaroo_full_vertical_jump"
  return cfg
