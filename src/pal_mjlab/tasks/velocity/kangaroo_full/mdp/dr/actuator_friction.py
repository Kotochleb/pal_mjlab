from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import Transmission

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
ACTUATOR_FRICTION_SCALE_RANGE: tuple[float, float] = (0.95, 1.05)


def _actuator_friction_targets(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[list[int], list[int]]:
  """DOF addresses and tendon ids that the entity's actuators drive, i.e. where
  create_*_actuator put each actuator's frictionloss."""
  mj_model = env.sim.mj_model
  dof_adrs: set[int] = set()
  tendon_ids: set[int] = set()
  for actuator_id in env.scene[asset_cfg.name].indexing.ctrl_ids.tolist():
    trn_type = int(mj_model.actuator_trntype[actuator_id])
    trn_id = int(mj_model.actuator_trnid[actuator_id, 0])
    if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
      dof_adrs.add(int(mj_model.jnt_dofadr[trn_id]))
    elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
      tendon_ids.add(trn_id)
  return sorted(dof_adrs), sorted(tendon_ids)


@requires_model_fields("dof_frictionloss", "tendon_frictionloss")
def actuator_frictionloss(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float] = ACTUATOR_FRICTION_SCALE_RANGE,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale every actuator's frictionloss (on its target joint DOF or tendon) by
  an independent U(scale_range) factor of the compiled nominal value."""
  env_ids = resolve_env_ids(env, env_ids).to(torch.long)
  dof_adrs, tendon_ids = _actuator_friction_targets(env, asset_cfg)
  model = env.sim.model
  for field, indices in (
    ("dof_frictionloss", dof_adrs),
    ("tendon_frictionloss", tendon_ids),
  ):
    if not indices:
      continue
    idx = torch.tensor(indices, dtype=torch.long, device=env.device)
    nominal = env.sim.get_default_field(field)[idx]
    scale = sample_uniform(
      scale_range[0], scale_range[1], (len(env_ids), len(indices)), device=env.device
    )
    env_grid, idx_grid = torch.meshgrid(env_ids, idx, indexing="ij")
    getattr(model, field)[env_grid, idx_grid] = nominal.unsqueeze(0) * scale


def configure_actuator_friction_dr(
  cfg: ManagerBasedRlEnvCfg, transmission: Transmission
) -> None:
  """Re-sample every actuator's frictionloss within ±5% of nominal at the start
  of each episode, for the screw-driven "actuator"/"lut" transmissions. The
  "joint" transmission keeps the measured per-joint table instead (see
  configure_leg_joint_friction_dr)."""
  if transmission == "joint":
    return
  cfg.events["actuator_frictionloss"] = EventTermCfg(
    mode="reset",
    func=actuator_frictionloss,
    params={"scale_range": ACTUATOR_FRICTION_SCALE_RANGE},
  )
