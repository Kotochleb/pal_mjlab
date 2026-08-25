"""Metrics for kangaroo_full closed-loop tendon equality constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import mujoco
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class tendon_equality_constraint_violation:
  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    model = env.sim.model

    eq_type = model.eq_type
    eq_obj1id = model.eq_obj1id

    tendon_ids: list[int] = []
    eq_ids: list[int] = []
    for local_id, tendon_name in enumerate(asset.tendon_names):
      if tendon_name not in asset_cfg.tendon_names:
        continue

      tendon_id = int(asset.indexing.tendon_ids[local_id])
      eq_rows = [
        i
        for i in range(model.neq)
        if int(eq_type[i]) == int(mujoco.mjtEq.mjEQ_TENDON)
        and int(eq_obj1id[i]) == tendon_id
      ]
      if not eq_rows:
        continue

      tendon_ids.append(local_id)
      eq_ids.append(eq_rows[0])

    self.tendon_ids = torch.as_tensor(tendon_ids, device=env.device, dtype=torch.long)
    self.global_tendon_ids = asset.indexing.tendon_ids[self.tendon_ids]
    self.eq_ids = torch.as_tensor(eq_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mode: Literal["violation", "length"] = "violation",
    reduction: Literal["sum", "mean", "max"] = "sum",
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    tendon_len = asset.data.tendon_len[:, self.tendon_ids]

    if mode == "length":
      return self._reduce(tendon_len, reduction)  # (num_envs,)

    model = env.sim.model
    tendon_len0 = model.tendon_length0[:, self.global_tendon_ids]
    target_offset = model.eq_data[:, self.eq_ids, 0]

    violation = (tendon_len - tendon_len0) - target_offset
    return self._reduce(violation.abs(), reduction)  # (num_envs,)

  @staticmethod
  def _reduce(values: torch.Tensor, reduction: Literal["sum", "mean", "max"]) -> torch.Tensor:
    if reduction == "sum":
      return values.sum(dim=-1)
    if reduction == "mean":
      return values.mean(dim=-1)
    if reduction == "max":
      return values.max(dim=-1).values
    raise ValueError(f"Unknown reduction '{reduction}'.")
