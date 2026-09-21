from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  LEG_JOINT_ARMATURE,
  LEG_JOINT_FRICTIONLOSS,
  LEG_JOINT_VISCOUS_DAMPING,
  Transmission,
)


def configure_leg_joint_friction_dr(
  cfg: ManagerBasedRlEnvCfg, transmission: Transmission
) -> None:
  """Randomize each leg joint's viscous damping, frictionloss and armature
  within the measured min/max range, once per environment at startup (like
  manufacturing variance, not something that should shift mid-episode).

  Only meaningful for the "joint" transmission: elsewhere these DOFs are
  unactuated and their physical friction lives on the screw actuator instead
  (LEG_SCREW_PD / screw_pd_params), so this is a no-op for "actuator"/"lut".
  """
  if transmission != "joint":
    return

  asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  def _ranges(per_joint: dict) -> dict[str, tuple[float, float]]:
    return {expr: rng for expr, (_, rng) in per_joint.items()}

  cfg.events["leg_joint_viscous_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": asset_cfg,
      "operation": "abs",
      "ranges": _ranges(LEG_JOINT_VISCOUS_DAMPING),
    },
  )
  cfg.events["leg_joint_frictionloss"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": asset_cfg,
      "operation": "abs",
      "ranges": _ranges(LEG_JOINT_FRICTIONLOSS),
    },
  )
  cfg.events["leg_joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": asset_cfg,
      "operation": "abs",
      "ranges": _ranges(LEG_JOINT_ARMATURE),
    },
  )
