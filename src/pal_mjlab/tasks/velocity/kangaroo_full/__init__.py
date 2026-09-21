from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

# One task per actuation variant, on flat and on rough terrain. The name spells
# out all six axes so a run is identifiable from its task id alone.
_TRANSMISSION_VARIANTS = {"Joint": "joint", "Actuator": "actuator", "Lut": "lut"}
_FEMUR_CLOSURE_VARIANTS = {"Prismatic": "prismatic", "Linkage": "linkage"}
_MJCF_VARIANTS = {
  "Tendons": "tendons",
  "TendonsOverConstrained": "tendons_over_constrained",
}
_BODY_VARIANTS = {"FullBody": False, "LowerBody": True}
_ANKLE_OBS_VARIANTS = {"AnkleRaw": False, "AnkleNormalized": True}
# The native <position> element / an ideal clamped PD ("Builtin") or
# DcMotorActuatorCfg's velocity-saturated torque-speed curve ("DcMotor"), on
# the "Actuator" transmission's screws directly and, the same way, on the
# "Lut" transmission's own screw <motor>s. "Joint" has no screw actuator in
# the model -- it's always the builtin native actuator -- so it doesn't get
# this axis at all; see the _transmission == "joint" special case below.
_ACTUATOR_MODEL_VARIANTS = {"Builtin": "builtin", "DcMotor": "dc_motor"}

for _terrain, _env_cfg_fn in (
  ("Flat", pal_kangaroo_full_flat_env_cfg),
  ("Rough", pal_kangaroo_full_rough_env_cfg),
):
  for _transmission_name, _transmission in _TRANSMISSION_VARIANTS.items():
    for _femur_closure_name, _femur_closure in _FEMUR_CLOSURE_VARIANTS.items():
      if _transmission == "joint" and _femur_closure == "linkage":
        # No leg_.*_length_joint under "linkage" for "joint" to servo --
        # only "actuator" and "lut" are valid there (kangaroo_full_
        # constants.py's _leg_actuators raises for this combo).
        continue
      for _mjcf_name, _mjcf in _MJCF_VARIANTS.items():
        if _femur_closure == "linkage" and _mjcf == "tendons_over_constrained":
          # kangaroo_full_tendons_over_constarined.xml never defines the
          # hip_xy_link/femur_rod tendons that close the leg-length loop
          # under "linkage" (only kangaroo_full_tendons.xml does) -- with
          # them absent, get_kangaroo_full_spec still deletes the
          # leg_.*_length_connect equality and leg_.*_length_joint for
          # "linkage", leaving that loop with nothing to close it at all.
          # Not a valid combination until the XML gets those tendons.
          continue
        # ankle_normalized is only a real choice for "tendons" -- env_cfgs.py
        # ignores it for "tendons_over_constrained" (its extra closed loops
        # already pin leg_.*_4_joint to the shank directly, see the
        # ankle_normalized comment there), so registering both would just
        # produce two task ids for the same env. Collapse to the one raw
        # variant there instead.
        _ankle_obs_items = (
          _ANKLE_OBS_VARIANTS.items()
          if _mjcf == "tendons"
          else (("AnkleRaw", False),)
        )
        for _body_name, _lower_body in _BODY_VARIANTS.items():
          for _ankle_obs_name, _ankle_normalized in _ankle_obs_items:
            # "joint" transmission is always the builtin native actuator --
            # it has no screw for ActuatorModel to vary -- so don't spell
            # out that axis in its task id at all.
            _actuator_model_items = (
              (("", "builtin"),)
              if _transmission == "joint"
              else _ACTUATOR_MODEL_VARIANTS.items()
            )
            for (
              _actuator_model_name,
              _actuator_model,
            ) in _actuator_model_items:
              _variant = {
                "transmission": _transmission,
                "femur_closure": _femur_closure,
                "mjcf": _mjcf,
                "lower_body": _lower_body,
                "ankle_normalized": _ankle_normalized,
                "actuator_model": _actuator_model,
              }
              _actuator_model_suffix = (
                f"-ActuatorModel-{_actuator_model_name}"
                if _actuator_model_name
                else ""
              )
              register_mjlab_task(
                task_id=(
                  f"Mjlab-Velocity-{_terrain}-Pal-Kangaroo-Full"
                  f"-Transmission-{_transmission_name}"
                  f"-Femur-{_femur_closure_name}"
                  f"-Mjcf-{_mjcf_name}"
                  f"-Body-{_body_name}"
                  f"-AnkleObs-{_ankle_obs_name}"
                  f"{_actuator_model_suffix}"
                ),
                env_cfg=_env_cfg_fn(**_variant),
                play_env_cfg=_env_cfg_fn(play=True, **_variant),
                rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
                runner_cls=VelocityOnPolicyRunner,
              )
