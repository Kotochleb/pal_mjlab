from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import pal_kangaroo_full_ppo_runner_cfg

from .env_cfgs import (
  pal_kangaroo_full_full_flat_env_cfg,
  pal_kangaroo_full_full_rough_env_cfg,
)

# One task per actuation variant, on flat and on rough terrain. Unlike
# pal_kangaroo_full this has no femur_closure or mjcf axis (this MJCF only
# has the one four-bar femur closure and the one XML -- see the module
# docstring on kangaroo_full_full_constants.py) and no leg_length axis (that
# axis's non-"actuator" values all target a leg_.*_length_joint this MJCF
# doesn't have), so four axes -- hip_z, hip_xy, ankle, lower_body -- spell out
# a task's name instead of pal_kangaroo_full's six.
_HIP_Z_VARIANTS = {"Slider": "slider", "Joint": "joint"}
_HIP_XY_VARIANTS = {"Slider": "slider", "Joint": "joint"}
_ANKLE_VARIANTS = {"Butterfly": "butterfly", "Joint": "joint", "Slider": "slider"}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_full_rough_env_cfg),
):
  for _hip_z_name, _hip_z in _HIP_Z_VARIANTS.items():
    for _hip_xy_name, _hip_xy in _HIP_XY_VARIANTS.items():
      for _ankle_name, _ankle in _ANKLE_VARIANTS.items():
        for _body_name, _lower_body in _BODY_VARIANTS.items():
          _variant = {
            "hip_z": _hip_z,
            "hip_xy": _hip_xy,
            "ankle": _ankle,
            "lower_body": _lower_body,
          }
          register_mjlab_task(
            task_id=(
              f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full-Full"
              f"-HipZ-{_hip_z_name}"
              f"-HipXY-{_hip_xy_name}"
              f"-Ankle-{_ankle_name}"
              f"-Body-{_body_name}"
            ),
            env_cfg=_env_cfg_fn(**_variant),
            play_env_cfg=_env_cfg_fn(play=True, **_variant),
            rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
            runner_cls=VelocityOnPolicyRunner,
          )
