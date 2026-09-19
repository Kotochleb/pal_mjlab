from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

# One task per actuation variant, on flat and on rough terrain. The name spells
# out all seven axes so a run is identifiable from its task id alone.
_TRANSMISSION_VARIANTS = {"Joint": "joint", "Actuator": "actuator", "Lut": "lut"}
_FEMUR_CLOSURE_VARIANTS = {"Prismatic": "prismatic", "Linkage": "linkage"}
_MJCF_VARIANTS = {
  "Tendons": "tendons",
  "TendonsOverConstrained": "tendons_over_constrained",
}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}
_ANKLE_OBS_VARIANTS = {"AnkleRaw": False, "AnkleNormalized": True}
# Only the "Actuator" transmission's screws read this: the native <position>
# element ("Builtin") or DcMotorActuatorCfg's velocity-saturated torque-speed
# curve ("DcMotor"). "Joint" and "Lut" have no screw actuator in the model, so
# the two values are identical for them -- kept as a full axis anyway so a
# task id always spells out every variant the same way.
_ACTUATOR_MODEL_VARIANTS = {"Builtin": "builtin", "DcMotor": "dc_motor"}
# Only the "Flat" terrain function reads this: rough terrain already deletes
# the speed curriculum outright and runs a fixed, easier command range, so
# "On" is a no-op there -- kept as a full axis anyway, same as actuator_model
# above.
_TOP_SPEED_VARIANTS = {"Off": False, "On": True}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_rough_env_cfg),
):
  for _transmission_name, _transmission in _TRANSMISSION_VARIANTS.items():
    for _femur_closure_name, _femur_closure in _FEMUR_CLOSURE_VARIANTS.items():
      for _mjcf_name, _mjcf in _MJCF_VARIANTS.items():
        for _body_name, _lower_body in _BODY_VARIANTS.items():
          for _ankle_obs_name, _ankle_normalized in _ANKLE_OBS_VARIANTS.items():
            for (
              _actuator_model_name,
              _actuator_model,
            ) in _ACTUATOR_MODEL_VARIANTS.items():
              for _top_speed_name, _top_speed in _TOP_SPEED_VARIANTS.items():
                _variant = {
                  "transmission": _transmission,
                  "femur_closure": _femur_closure,
                  "mjcf": _mjcf,
                  "lower_body": _lower_body,
                  "ankle_normalized": _ankle_normalized,
                  "actuator_model": _actuator_model,
                  "top_speed": _top_speed,
                }
                register_mjlab_task(
                  task_id=(
                    f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full"
                    f"-Transmission-{_transmission_name}"
                    f"-Femur-{_femur_closure_name}"
                    f"-Mjcf-{_mjcf_name}"
                    f"-Body-{_body_name}"
                    f"-AnkleObs-{_ankle_obs_name}"
                    f"-ActuatorModel-{_actuator_model_name}"
                    f"-TopSpeed-{_top_speed_name}"
                  ),
                  env_cfg=_env_cfg_fn(**_variant),
                  play_env_cfg=_env_cfg_fn(play=True, **_variant),
                  rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
                  runner_cls=VelocityOnPolicyRunner,
                )
