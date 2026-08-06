"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def imu_projected_gravity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Get projected gravity from IMU sensor orientation (accounts for IMU mounting)."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)

  # Get IMU orientation (already includes mounting offset)
  imu_quat = sensor.data  # or however you access orientation

  # Gravity in world frame
  gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=imu_quat.device).expand(
    imu_quat.shape[0], -1
  )
  # print(f"imu proj{quat_apply_inverse(imu_quat, gravity_w)}")
  # asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  # print(f"proj{asset.data.projected_gravity_b}")
  # Project to IMU frame (same as your C++ code)
  return quat_apply_inverse(imu_quat, gravity_w)


##
# Exteroception.
##


def clipped_depth(
  env: ManagerBasedRlEnv, sensor_name: str, max_depth: float
) -> torch.Tensor:
  """Depth image from a `CameraSensor`, flattened row-major with row 0 at the top.

  mujoco_warp already renders planar depth in metres, so no conversion is needed. The
  two guards are the renderer's conventions rather than ours:

  - A ray that hits nothing writes ``0.0``, which raw would tell the policy there is a
    surface against the lens. Filling with ``max_depth`` makes "nothing there" and "far
    away" the same number, which is what they mean for locomotion.
  - A hit is unbounded -- there is no far clip -- so over a gap or a downslope the value
    grows without limit and the caller's ``1 / max_depth`` scaling stops normalising.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, CameraSensor)
  depth = sensor.data.depth
  assert depth is not None, f"sensor '{sensor_name}' has no depth data type enabled"
  depth = depth.squeeze(-1).flatten(1)  # [B, height * width]
  return torch.where((depth > 0.0) & (depth < max_depth), depth, max_depth)
