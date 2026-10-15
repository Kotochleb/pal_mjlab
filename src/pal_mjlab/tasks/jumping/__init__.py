"""Separate vertical-jump tasks for the existing Kangaroo Full model variants."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import configure_vertical_jump, pal_kangaroo_full_vertical_jump_env_cfg
from .lp_acrl_runner import LpAcrlJumpOnPolicyRunner
from .rl_cfg import pal_kangaroo_full_jump_ppo_runner_cfg

# Keep existing ids for the time-based height schedule; add an LP-ACRL suffix.
_CURRICULUM_VARIANTS = {"": "default", "LpAcrl": "lp_acrl"}

# Importing env_cfgs registers the Full velocity family first. Reuse its valid
# model combinations so jump tasks cannot drift from the supported robot axes.
for _task in list_tasks():
  if not _task.startswith("Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Transmission-"):
    continue
  if "-Curriculum-" in _task:
    continue
  for _curriculum_name, _curriculum in _CURRICULUM_VARIANTS.items():
    _curriculum_suffix = f"-Curriculum-{_curriculum_name}" if _curriculum_name else ""
    register_mjlab_task(
      task_id=_task.replace("Mjlab-Velocity-Flat-", "Mjlab-Jump-Vertical-")
      + _curriculum_suffix,
      env_cfg=configure_vertical_jump(load_env_cfg(_task), curriculum=_curriculum),
      play_env_cfg=configure_vertical_jump(
        load_env_cfg(_task, play=True), play=True, curriculum=_curriculum
      ),
      rl_cfg=pal_kangaroo_full_jump_ppo_runner_cfg(),
      runner_cls=(
        LpAcrlJumpOnPolicyRunner if _curriculum == "lp_acrl" else VelocityOnPolicyRunner
      ),
    )

for _curriculum_name, _curriculum in _CURRICULUM_VARIANTS.items():
  register_mjlab_task(
    task_id="Mjlab-Jump-Vertical-Pal-Kangaroo-Full"
    + (f"-Curriculum-{_curriculum_name}" if _curriculum_name else ""),
    env_cfg=pal_kangaroo_full_vertical_jump_env_cfg(curriculum=_curriculum),
    play_env_cfg=pal_kangaroo_full_vertical_jump_env_cfg(
      play=True, curriculum=_curriculum
    ),
    rl_cfg=pal_kangaroo_full_jump_ppo_runner_cfg(),
    runner_cls=(
      LpAcrlJumpOnPolicyRunner if _curriculum == "lp_acrl" else VelocityOnPolicyRunner
    ),
  )
