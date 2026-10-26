"""Separate vertical-jump tasks for the existing Kangaroo Full model variants."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task

from .env_cfgs import configure_vertical_jump, pal_kangaroo_full_vertical_jump_env_cfg
from .rl_cfg import pal_kangaroo_full_jump_ppo_runner_cfg
from .runner import JumpOnPolicyRunner

# Importing env_cfgs registers the Full velocity family first. Reuse its valid
# model combinations so jump tasks cannot drift from the supported robot axes.
for _task in list_tasks():
  if not _task.startswith("Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Transmission-"):
    continue
  if "-Curriculum-" in _task:
    continue
  register_mjlab_task(
    task_id=_task.replace("Mjlab-Velocity-Flat-", "Mjlab-Jump-Vertical-"),
    env_cfg=configure_vertical_jump(load_env_cfg(_task)),
    play_env_cfg=configure_vertical_jump(load_env_cfg(_task, play=True), play=True),
    rl_cfg=pal_kangaroo_full_jump_ppo_runner_cfg(),
    runner_cls=JumpOnPolicyRunner,
  )

register_mjlab_task(
  task_id="Mjlab-Jump-Vertical-Pal-Kangaroo-Full",
  env_cfg=pal_kangaroo_full_vertical_jump_env_cfg(),
  play_env_cfg=pal_kangaroo_full_vertical_jump_env_cfg(play=True),
  rl_cfg=pal_kangaroo_full_jump_ppo_runner_cfg(),
  runner_cls=JumpOnPolicyRunner,
)
