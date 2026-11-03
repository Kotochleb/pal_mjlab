"""Jump-motion tracking tasks that pretrain policies for the vertical-jump task.

``Mjlab-Tracking-Jump-<variant>`` pairs with ``Mjlab-Jump-Vertical-<variant>``:
same robot, actions and actor observations. Lower-body variants track the
lower-body jump motion; full-body ones its copy with the arms held at their
default pose (JUMP_ARMS_MOTION_FILE).
"""

from mjlab.tasks.registry import list_tasks, load_env_cfg, register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import configure_full_model_jump_tracking
from .rl_cfg import pal_kangaroo_full_jump_tracking_ppo_runner_cfg

for _task in list_tasks():
  if not _task.startswith("Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Transmission-"):
    continue
  if "-Curriculum-" in _task:
    continue
  _femur_closure = "linkage" if "-Femur-Linkage-" in _task else "prismatic"
  _lower_body = "-Body-LowerBody-" in _task
  register_mjlab_task(
    task_id=_task.replace("Mjlab-Velocity-Flat-", "Mjlab-Tracking-Jump-"),
    env_cfg=configure_full_model_jump_tracking(
      load_env_cfg(_task), femur_closure=_femur_closure, lower_body=_lower_body
    ),
    play_env_cfg=configure_full_model_jump_tracking(
      load_env_cfg(_task, play=True),
      femur_closure=_femur_closure,
      lower_body=_lower_body,
      play=True,
    ),
    rl_cfg=pal_kangaroo_full_jump_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
