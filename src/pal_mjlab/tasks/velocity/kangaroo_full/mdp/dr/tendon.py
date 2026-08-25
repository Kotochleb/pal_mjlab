"""Reset events for tendon equality constraints."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def enforce_tendon_lengths(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lengths: dict[str, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  env_ids = resolve_env_ids(env, env_ids)

  asset = env.scene[asset_cfg.name]
  model = env.sim.model

  eq_type = model.eq_type
  eq_obj1id = model.eq_obj1id

  for local_id, tendon_name in enumerate(asset.tendon_names):
    matches = [
      length
      for pattern, length in lengths.items()
      if re.fullmatch(pattern, tendon_name)
    ]
    if not matches:
      continue
    if len(matches) > 1:
      raise ValueError(
        f"Tendon '{tendon_name}' matches multiple length patterns: {matches}"
      )
    target_length = matches[0]

    tendon_id = int(asset.indexing.tendon_ids[local_id])
    eq_rows = [
      i
      for i in range(model.neq)
      if int(eq_type[i]) == int(mujoco.mjtEq.mjEQ_TENDON)
      and int(eq_obj1id[i]) == tendon_id
    ]
    if not eq_rows:
      continue

    length_at_qpos0 = model.tendon_length0[env_ids, tendon_id]
    model.eq_data[env_ids, eq_rows[0], 0] = target_length - length_at_qpos0
