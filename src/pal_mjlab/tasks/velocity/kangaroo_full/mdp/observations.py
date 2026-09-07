"""Observations that keep the full model's policy input identical to the
simple ``pal_kangaroo`` model's.

The full model does not always carry a ``leg_.*_length_joint``: the "linkage"
femur closure describes the femur with the real four-bar instead, and deletes
the straight-line slider. The policy still has to see a leg length in that slot
-- same 26 joints, same order, same layout -- so it is reconstructed from the
knee angle through the swept ``knee_distance_map.csv``, which is exactly the
relation the deleted slider encodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def load_knee_leg_length_map(csv_path: str | Path) -> torch.Tensor:
  """Load ``knee_distance_map.csv`` as a ``(N, 2)`` ``(knee_rad, length_m)``.

  ``distance_m`` -- the femur joint to connect site distance -- *is* the leg
  length the slider would have measured, and it runs the same way as the
  simple model's ``leg_.*_length_joint``: longest with the knee straight.
  """
  data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
  table = torch.from_numpy(data[:, [0, 4]])
  return table[torch.argsort(table[:, 0])]


def _interpolate(
  x: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Linear interpolation of ``table``'s value column at ``x``.

  Returns:
    ``(value, slope)`` -- the interpolated value and the slope of the segment
    it landed in, the latter being what turns a knee velocity into a leg
    length rate. Queries outside the swept range clamp to the end rows, whose
    slope is that of the last segment.
  """
  keys = table[:, 0].contiguous()
  values = table[:, 1]

  x_clamped = torch.clamp(x, keys[0], keys[-1])
  idx = torch.clamp(
    torch.searchsorted(keys, x_clamped.contiguous()), 1, keys.numel() - 1
  )

  lo, hi = keys[idx - 1], keys[idx]
  slope = (values[idx] - values[idx - 1]) / (hi - lo)
  return values[idx - 1] + (x_clamped - lo) * slope, slope


class joint_state_with_mapped_leg_length:
  """``joint_pos_rel`` / ``joint_vel_rel`` over an explicit joint vector, with
  any joint the model lacks filled in from the knee displacement map.

  ``joint_order`` names the vector the policy expects, in order. Every name
  the entity has is read straight off it, exactly as the stock terms do --
  relative to the default pose, biased for the actor's positions. The rest
  must be leg length joints, and are reconstructed from their leg's knee:

  * positions as ``map(knee) - map(knee_default)``, matching the stock term's
    relative-to-default convention, which also cancels the constant offset
    between the map's absolute distance and the slider's own zero;
  * velocities as ``d map/d knee * knee_vel``, the map's local slope carrying
    the knee rate into the slider's units.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    joint_order: Sequence[str] = cfg.params["joint_order"]
    self.num_joints = len(joint_order)
    self.table = load_knee_leg_length_map(cfg.params["csv_path"]).to(env.device)

    present_cols: list[int] = []
    present_ids: list[int] = []
    mapped_cols: list[int] = []
    knee_ids: list[int] = []
    knee_joints = dict(cfg.params["mapped_joints"])
    for column, name in enumerate(joint_order):
      if name in asset.joint_names:
        present_cols.append(column)
        present_ids.append(asset.joint_names.index(name))
        continue
      if name not in knee_joints:
        raise ValueError(
          f"Entity '{asset_cfg.name}' has no joint '{name}' and no knee joint "
          "is mapped to it; add it to 'mapped_joints' or drop it from "
          "'joint_order'."
        )
      mapped_cols.append(column)
      knee_ids.append(asset.joint_names.index(knee_joints[name]))

    def _ids(values: list[int]) -> torch.Tensor:
      return torch.as_tensor(values, device=env.device, dtype=torch.long)

    self.present_cols = _ids(present_cols)
    self.present_ids = _ids(present_ids)
    self.mapped_cols = _ids(mapped_cols)
    self.knee_ids = _ids(knee_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_order: Sequence[str],
    csv_path: str | Path,
    mapped_joints: Sequence[tuple[str, str]],
    mode: Literal["pos", "vel"] = "pos",
    biased: bool = False,
  ) -> torch.Tensor:
    del joint_order, csv_path, mapped_joints  # Resolved once, in __init__.
    asset = env.scene[asset_cfg.name]
    data = asset.data

    if mode == "pos":
      measured = data.joint_pos_biased if biased else data.joint_pos
      default = data.default_joint_pos
    else:
      measured = data.joint_vel
      default = data.default_joint_vel
    assert default is not None

    out = torch.zeros(
      measured.shape[0], self.num_joints, device=measured.device, dtype=measured.dtype
    )
    out[:, self.present_cols] = (
      measured[:, self.present_ids] - default[:, self.present_ids]
    )

    if self.mapped_cols.numel():
      knee = data.joint_pos[:, self.knee_ids]
      length, slope = _interpolate(knee, self.table)
      if mode == "pos":
        default_knee = data.default_joint_pos
        assert default_knee is not None
        length_default, _ = _interpolate(default_knee[:, self.knee_ids], self.table)
        out[:, self.mapped_cols] = length - length_default
      else:
        out[:, self.mapped_cols] = slope * data.joint_vel[:, self.knee_ids]

    return out
