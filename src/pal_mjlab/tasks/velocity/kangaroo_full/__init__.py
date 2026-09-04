from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

# One task per actuation variant, on flat and on rough terrain. The name spells
# out all three axes so a run is identifiable from its task id alone.
_HIP_Z_VARIANTS = {"Tendon": "tendon", "Joint": "joint"}
_HIP_XY_VARIANTS = {"Tendon": "tendon", "Joint": "joint"}
_LEG_LENGTH_VARIANTS = {"Actuator": "actuator", "Joint": "joint"}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_rough_env_cfg),
):
  for _hip_z_name, _hip_z in _HIP_Z_VARIANTS.items():
    for _hip_xy_name, _hip_xy in _HIP_XY_VARIANTS.items():
      for _leg_length_name, _leg_length in _LEG_LENGTH_VARIANTS.items():
        _variant = {
          "hip_z": _hip_z,
          "hip_xy": _hip_xy,
          "leg_length": _leg_length,
        }
        register_mjlab_task(
          task_id=(
            f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full"
            f"-HipZ-{_hip_z_name}"
            f"-HipXY-{_hip_xy_name}"
            f"-LegLength-{_leg_length_name}"
          ),
          env_cfg=_env_cfg_fn(**_variant),
          play_env_cfg=_env_cfg_fn(play=True, **_variant),
          rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
          runner_cls=VelocityOnPolicyRunner,
        )
