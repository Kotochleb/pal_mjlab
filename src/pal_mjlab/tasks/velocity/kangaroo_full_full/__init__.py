from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from pal_mjlab.tasks.velocity.kangaroo_full.rl_cfg import (
  pal_kangaroo_full_ppo_runner_cfg,
)
from pal_mjlab.tasks.velocity.lp_acrl_runner import LpAcrlVelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_full_flat_env_cfg,
  pal_kangaroo_full_full_rough_env_cfg,
)

# One task per actuation variant, on flat and on rough terrain. Unlike
# pal_kangaroo_full this has no femur_closure or mjcf axis (this MJCF only
# has the one four-bar femur closure and the one XML -- see the module
# docstring on kangaroo_full_full_constants.py), so four model axes --
# transmission, lower_body, ankle_normalized and actuator_model -- plus the
# curriculum identify each task (the legacy default has no curriculum suffix).
# No "Joint" here: this MJCF has no leg_.*_length_joint for it to servo, so
# kangaroo_full_full_constants.py's Transmission only allows "actuator"/"lut"
# (see its module docstring).
_TRANSMISSION_VARIANTS = {"Actuator": "actuator", "Lut": "lut"}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}
_ANKLE_OBS_VARIANTS = {"AnkleRaw": False, "AnkleNormalized": True}
# Both "Actuator" and "Lut" read this -- see pal_kangaroo_full's __init__.py.
_ACTUATOR_MODEL_VARIANTS = {"Builtin": "builtin", "DcMotor": "dc_motor"}

# Keep existing ids for the default curriculum; add an independent LP-ACRL axis.
_CURRICULUM_VARIANTS = {"": "default", "LpAcrl": "lp_acrl"}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_full_rough_env_cfg),
):
  for _transmission_name, _transmission in _TRANSMISSION_VARIANTS.items():
    for _body_name, _lower_body in _BODY_VARIANTS.items():
      for _ankle_obs_name, _ankle_normalized in _ANKLE_OBS_VARIANTS.items():
        for _actuator_model_name, _actuator_model in _ACTUATOR_MODEL_VARIANTS.items():
          _variant = {
            "transmission": _transmission,
            "lower_body": _lower_body,
            "ankle_normalized": _ankle_normalized,
            "actuator_model": _actuator_model,
          }
          for _curriculum_name, _curriculum in _CURRICULUM_VARIANTS.items():
            _curriculum_suffix = (
              f"-Curriculum-{_curriculum_name}" if _curriculum_name else ""
            )
            register_mjlab_task(
              task_id=(
                f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full-Full"
                f"-Transmission-{_transmission_name}"
                f"-Body-{_body_name}"
                f"-AnkleObs-{_ankle_obs_name}"
                f"-ActuatorModel-{_actuator_model_name}"
                f"{_curriculum_suffix}"
              ),
              env_cfg=_env_cfg_fn(curriculum=_curriculum, **_variant),
              play_env_cfg=_env_cfg_fn(play=True, curriculum=_curriculum, **_variant),
              rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
              runner_cls=(
                LpAcrlVelocityOnPolicyRunner
                if _curriculum == "lp_acrl"
                else VelocityOnPolicyRunner
              ),
            )
