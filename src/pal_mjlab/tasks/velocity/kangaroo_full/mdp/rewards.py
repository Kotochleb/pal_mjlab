from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture
from mjlab.utils.lab_api.string import resolve_matching_names_values

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  load_transmission_maps,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.observations import (
  _leg_length_and_slope,
)
from pal_mjlab.tasks.velocity.mdp.rewards import joint_limits_convex_hull

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class _MappedLegLength:
  """The knee-to-leg-length map, resolved once against an entity.

  ``mapped_joints`` pairs each absent leg length joint with the knee that
  stands in for it, exactly as the observation term takes it.
  """

  def __init__(
    self,
    asset: Entity,
    mapped_joints: Sequence[tuple[str, str]],
    device: str,
  ) -> None:
    self.names = [length for length, _ in mapped_joints]
    self.leg = load_transmission_maps().leg_length.to(device)
    self.knee_ids = torch.as_tensor(
      [asset.joint_names.index(knee) for _, knee in mapped_joints],
      device=device,
      dtype=torch.long,
    )
    default = asset.data.default_joint_pos
    assert default is not None
    self.length_default, _ = _leg_length_and_slope(self.leg, default[:, self.knee_ids])
    # The slider's zero is the fully extended leg, which is the knee at zero:
    # its range tops out at 0.0 exactly where the knee's bottoms out.
    self.length_at_zero, _ = _leg_length_and_slope(
      self.leg, torch.zeros(1, device=device)
    )

  def length(self, asset: Entity) -> torch.Tensor:
    """Leg length in the slider's own coordinate: zero fully extended."""
    length, _ = _leg_length_and_slope(self.leg, asset.data.joint_pos[:, self.knee_ids])
    return length - self.length_at_zero

  def length_rel(self, asset: Entity) -> torch.Tensor:
    """Leg length relative to its default, as the observation reports it."""
    length, _ = _leg_length_and_slope(self.leg, asset.data.joint_pos[:, self.knee_ids])
    return length - self.length_default

  def velocity(self, asset: Entity) -> torch.Tensor:
    """Leg length rate: the map's local slope carrying the knee rate."""
    _, slope = _leg_length_and_slope(self.leg, asset.data.joint_pos[:, self.knee_ids])
    return slope * asset.data.joint_vel[:, self.knee_ids]


class variable_posture_with_mapped_leg_length(variable_posture):
  """``variable_posture`` with the leg length joints read through the knee map.

  Same reward -- ``exp(-mean(error² / std²))`` over every joint in
  ``asset_cfg`` plus one entry per mapped leg length -- so the ``std_*``
  dicts take the same ``leg_.*_length_.*`` keys the baseline sets, and a
  variant without the joint holds its leg length to the same tolerance as
  one with it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.mapped = _MappedLegLength(asset, cfg.params["mapped_joints"], env.device)
    # The base resolves each std dict against the entity's joint names; do
    # the same here against those names plus the mapped ones, in the order
    # the error vector is assembled below (present joints, then mapped).
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    names = list(joint_names) + self.mapped.names
    for regime in ("std_standing", "std_walking", "std_running"):
      _, _, std = resolve_matching_names_values(
        data=cfg.params[regime], list_of_strings=names
      )
      setattr(self, regime, torch.tensor(std, device=env.device, dtype=torch.float32))
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    mapped_joints: Sequence[tuple[str, str]],
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running, mapped_joints

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()
    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    error = torch.cat(
      [
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - self.default_joint_pos[:, asset_cfg.joint_ids],
        self.mapped.length_rel(asset),
      ],
      dim=1,
    )
    return torch.exp(-torch.mean(torch.square(error) / (std**2), dim=1))


class mapped_leg_length_pos_limits:
  """``joint_pos_limits`` for a leg length read through the knee map.

  ``joint_range`` is the slider's range on the variants that have it, and
  ``soft_limit_factor`` the articulation's ``soft_joint_pos_limit_factor``,
  so the soft limits come out as the entity would have computed them.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.mapped = _MappedLegLength(asset, cfg.params["mapped_joints"], env.device)
    lo, hi = cfg.params["joint_range"]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * cfg.params["soft_limit_factor"]
    self.soft_limits = (mid - half, mid + half)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mapped_joints: Sequence[tuple[str, str]],
    joint_range: tuple[float, float],
    soft_limit_factor: float,
  ) -> torch.Tensor:
    del mapped_joints, joint_range, soft_limit_factor
    length = self.mapped.length(env.scene[asset_cfg.name])
    out_of_limits = -(length - self.soft_limits[0]).clip(max=0.0)
    out_of_limits += (length - self.soft_limits[1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


class mapped_leg_length_vel_limits:
  """``joint_vel_limits`` for a leg length read through the knee map.

  Same 0.9 soft factor and the same ``Metrics/joint_vel_*`` log keys as the
  baseline term, so the two variants' curves line up.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.mapped = _MappedLegLength(asset, cfg.params["mapped_joints"], env.device)
    lo, hi = cfg.params["velocity_limits"]
    self.soft_vel_limits = (0.9 * lo, 0.9 * hi)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mapped_joints: Sequence[tuple[str, str]],
    velocity_limits: tuple[float, float],
  ) -> torch.Tensor:
    del mapped_joints, velocity_limits
    vel = self.mapped.velocity(env.scene[asset_cfg.name])
    out_of_limits = -(vel - self.soft_vel_limits[0]).clip(max=0.0)
    out_of_limits += (vel - self.soft_vel_limits[1]).clip(min=0.0)
    penalty = torch.sum(out_of_limits, dim=1)

    env.extras["log"]["Metrics/joint_vel_max"] = torch.max(torch.abs(vel)).item()
    env.extras["log"]["Metrics/joint_vel_limit_violation"] = torch.mean(penalty).item()
    return penalty


class joint_limits_convex_hull_ankle_femur_normalized(joint_limits_convex_hull):
  """``joint_limits_convex_hull``, for the ankle, normalized by the femur joint.

  The simple model's ``leg_.*_4_joint`` (ankle pitch) is measured off the
  shank, but on the full and full-full models the shank itself is free to
  rotate on ``leg_.*_femur_joint`` -- the four-bar (or connect-loop) femur
  closure -- so the ankle bar mechanism's pitch relative to the *shank*, the
  quantity the hull was fit against, is ``leg_.*_4_joint - leg_.*_femur_joint``,
  not ``leg_.*_4_joint`` alone. ``femur_joint_names`` pairs one femur joint
  with each entry of ``joint_names_group``, in the same order; only the first
  joint of each group (joint 4) is shifted, the second (joint 5, ankle roll)
  is untouched.
  """

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    metrics_suffix: str,
    margin: float,
    joint_names_group: list[list[str]],
    hull_points: torch.Tensor,
    femur_joint_names: list[str],
  ) -> torch.Tensor:
    del margin, hull_points
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    metrics_violation_dist = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    for joint_group, femur_joint_name in zip(joint_names_group, femur_joint_names):
      asset: Entity = env.scene[asset_cfg.name]
      target_ids, _ = asset.find_joints(joint_group)
      femur_id, _ = asset.find_joints([femur_joint_name])

      joint_pos = asset.data.joint_pos[:, target_ids].clone()
      joint_pos[:, 0] = joint_pos[:, 0] - asset.data.joint_pos[:, femur_id[0]]

      dot_product_res = (
        torch.matmul(joint_pos, self.equation_coeff_A.T) + self.equation_coeff_b
      )
      violation_dist = torch.clamp(dot_product_res, min=0.0).max(dim=1)[0]
      penalty += torch.square(violation_dist)
      metrics_violation_dist += violation_dist

    env.extras["log"][f"Metrics/joint_limits_hull_{metrics_suffix}"] = torch.mean(
      metrics_violation_dist
    )
    return penalty
