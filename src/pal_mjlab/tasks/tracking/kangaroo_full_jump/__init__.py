"""Jump-motion tracking tasks that pretrain policies for the vertical-jump task.

``Mjlab-Tracking-Jump-<variant>`` pairs with ``Mjlab-Jump-Vertical-<variant>``:
same robot, actions and actor observations. The jump motion is lower body, so
only the lower-body Kangaroo Full variants are registered.
"""

from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import configure_full_model_jump_tracking
from .rl_cfg import pal_kangaroo_full_jump_tracking_ppo_runner_cfg

for _task in list_tasks():
  if not _task.startswith("Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Transmission-"):
    continue
  if "-Curriculum-" in _task or "-Body-LowerBody-" not in _task:
    continue
  _femur_closure = "linkage" if "-Femur-Linkage-" in _task else "prismatic"
  register_mjlab_task(
    task_id=_task.replace("Mjlab-Velocity-Flat-", "Mjlab-Tracking-Jump-"),
    env_cfg=configure_full_model_jump_tracking(
      load_env_cfg(_task), femur_closure=_femur_closure
    ),
    play_env_cfg=configure_full_model_jump_tracking(
      load_env_cfg(_task, play=True), femur_closure=_femur_closure, play=True
    ),
    rl_cfg=pal_kangaroo_full_jump_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
