from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .env_cfgs_tendons import pal_kangaroo_full_tendons_flat_env_cfg
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Full",
  env_cfg=pal_kangaroo_full_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_full_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Tendons",
  env_cfg=pal_kangaroo_full_tendons_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_full_tendons_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

for state in ("full_state", "actuator_space", "simple_model"):
  for pose_space in (False, True):
    for pd_mapping in ("jacobian", "lowest", "semi_serial"):
      task_id_base = "Mjlab-Velocity-Flat-Pal-Kangaroo-Full"
      state_name = state.replace("_", " ").title().replace(" ", "-")
      pose_space_name = "-Pose-In-Actuator" if pose_space else ""
      pd_map_name = pd_mapping.replace("_", " ").title().replace(" ", "-")
      register_mjlab_task(
        task_id=f"{task_id_base}-{state_name}-PD-{pd_map_name}{pose_space_name}",
        env_cfg=pal_kangaroo_full_flat_env_cfg(
          joint_state_obs=state,
          pose_in_actuator_space=pose_space,
          pd_mapping=pd_mapping,
        ),
        play_env_cfg=pal_kangaroo_full_flat_env_cfg(
          joint_state_obs=state,
          pose_in_actuator_space=pose_space,
          pd_mapping=pd_mapping,
          play=True,
        ),
        rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
      )
