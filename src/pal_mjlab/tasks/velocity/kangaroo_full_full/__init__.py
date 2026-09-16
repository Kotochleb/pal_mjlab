import warnings

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.robots.pal_kangaroo_full_full.kangaroo_full_constants import (
  LUT_TRANSMISSION_DIR,
  transmission_maps_available,
)
from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import (
  pal_kangaroo_full_ppo_runner_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo_full.runner import KangarooFullOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_full_flat_env_cfg,
  pal_kangaroo_full_full_rough_env_cfg,
)

# One task per actuation variant, on flat and on rough terrain. Unlike
# pal_kangaroo_full this has no femur_closure or mjcf axis (this MJCF only
# has the one four-bar femur closure and the one XML -- see the module
# docstring on kangaroo_full_full_constants.py), and its leg_length axis is
# the two-valued "actuator" / "transmission" one rather than
# pal_kangaroo_full's four (the others all target a leg_.*_length_joint this
# MJCF doesn't have), so five axes -- hip_z, hip_xy, ankle, leg_length,
# lower_body -- spell out a task's name instead of pal_kangaroo_full's six.
# Every "Transmission" value needs the maps in
# robots/pal_kangaroo_full/lut_transmission/; without them those tasks are
# left unregistered rather than taking every other task down with them.
_TRANSMISSION_AVAILABLE = transmission_maps_available()
if not _TRANSMISSION_AVAILABLE:
  warnings.warn(
    f"No transmission maps in {LUT_TRANSMISSION_DIR}: the Kangaroo-Full-Full "
    "*-Transmission-* tasks are not registered.",
    stacklevel=1,
  )
_HIP_Z_VARIANTS = {
  "Slider": "slider",
  "Joint": "joint",
  "Transmission": "transmission",
}
_HIP_XY_VARIANTS = {
  "Slider": "slider",
  "Joint": "joint",
  "Transmission": "transmission",
}
_ANKLE_VARIANTS = {
  "Butterfly": "butterfly",
  "Joint": "joint",
  "Slider": "slider",
  "Transmission": "transmission",
}
_LEG_LENGTH_VARIANTS = {
  "Actuator": "actuator",
  "Transmission": "transmission",
}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_full_rough_env_cfg),
):
  for _hip_z_name, _hip_z in _HIP_Z_VARIANTS.items():
    for _hip_xy_name, _hip_xy in _HIP_XY_VARIANTS.items():
      for _ankle_name, _ankle in _ANKLE_VARIANTS.items():
        for _leg_length_name, _leg_length in _LEG_LENGTH_VARIANTS.items():
          for _body_name, _lower_body in _BODY_VARIANTS.items():
            _variant = {
              "hip_z": _hip_z,
              "hip_xy": _hip_xy,
              "ankle": _ankle,
              "leg_length": _leg_length,
              "lower_body": _lower_body,
            }
            if "transmission" in _variant.values() and not _TRANSMISSION_AVAILABLE:
              continue
            register_mjlab_task(
              task_id=(
                f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full-Full"
                f"-HipZ-{_hip_z_name}"
                f"-HipXY-{_hip_xy_name}"
                f"-Ankle-{_ankle_name}"
                f"-LegLength-{_leg_length_name}"
                f"-Body-{_body_name}"
              ),
              env_cfg=_env_cfg_fn(**_variant),
              play_env_cfg=_env_cfg_fn(play=True, **_variant),
              rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
              runner_cls=KangarooFullOnPolicyRunner,
            )
