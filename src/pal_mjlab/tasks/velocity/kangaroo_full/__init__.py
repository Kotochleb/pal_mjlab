from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

# One task per actuation variant, on flat and on rough terrain. The name spells
# out all five axes so a run is identifiable from its task id alone.
_TRANSMISSION_VARIANTS = {"Joint": "joint", "Actuator": "actuator", "Lut": "lut"}
_FEMUR_CLOSURE_VARIANTS = {"Prismatic": "prismatic", "Linkage": "linkage"}
_MJCF_VARIANTS = {
  "Tendons": "tendons",
  "TendonsOverConstrained": "tendons_over_constrained",
}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}
_ANKLE_OBS_VARIANTS = {"AnkleRaw": False, "AnkleNormalized": True}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_rough_env_cfg),
):
  for _transmission_name, _transmission in _TRANSMISSION_VARIANTS.items():
    for _femur_closure_name, _femur_closure in _FEMUR_CLOSURE_VARIANTS.items():
      for _mjcf_name, _mjcf in _MJCF_VARIANTS.items():
        for _body_name, _lower_body in _BODY_VARIANTS.items():
          for _ankle_obs_name, _ankle_normalized in _ANKLE_OBS_VARIANTS.items():
            _variant = {
              "transmission": _transmission,
              "femur_closure": _femur_closure,
              "mjcf": _mjcf,
              "lower_body": _lower_body,
              "ankle_normalized": _ankle_normalized,
            }
            register_mjlab_task(
              task_id=(
                f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full"
                f"-Transmission-{_transmission_name}"
                f"-Femur-{_femur_closure_name}"
                f"-Mjcf-{_mjcf_name}"
                f"-Body-{_body_name}"
                f"-AnkleObs-{_ankle_obs_name}"
              ),
              env_cfg=_env_cfg_fn(**_variant),
              play_env_cfg=_env_cfg_fn(play=True, **_variant),
              rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
              runner_cls=VelocityOnPolicyRunner,
            )
