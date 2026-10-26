"""Motion tracking tasks for the existing Kangaroo Full model variants."""

from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import configure_full_model_tracking
from .rl_cfg import pal_kangaroo_full_tracking_ppo_runner_cfg

# Importing env_cfgs registers the Full velocity family first. Reuse its valid
# model combinations so tracking tasks cannot drift from the supported robot
# axes; the curriculum axis has no meaning here.
for _task in list_tasks():
  if not _task.startswith("Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Transmission-"):
    continue
  if "-Curriculum-" in _task:
    continue
  _variant = {
    "femur_closure": "linkage" if "-Femur-Linkage-" in _task else "prismatic",
    "lower_body": "-Body-LowerBody-" in _task,
  }
  register_mjlab_task(
    task_id=_task.replace("Mjlab-Velocity-Flat-", "Mjlab-Tracking-Flat-"),
    env_cfg=configure_full_model_tracking(load_env_cfg(_task), **_variant),
    play_env_cfg=configure_full_model_tracking(
      load_env_cfg(_task, play=True), play=True, **_variant
    ),
    rl_cfg=pal_kangaroo_full_tracking_ppo_runner_cfg(_variant["lower_body"]),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
