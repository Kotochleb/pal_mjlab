from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .metrics import resolve_tendon_eq_targets, tendon_length_violation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_ACTUATED_LEG_JOINT_RE_1 = r"leg_(left|right)_length_actuator$"

_ACTUATED_LEG_JOINT_RE_2 = r"leg_(left|right)_[1-5]_actuator$"


def joint_vel_limit(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  limit_leg_length = 0.26
  limit_leg_sliders = 0.25  # Experiments showed 30 is probable limit

  # Only work with the actuated joints
  leg_ids_1, _ = asset.find_joints_by_actuator_names(_ACTUATED_LEG_JOINT_RE_1)
  leg_ids_2, _ = asset.find_joints_by_actuator_names(_ACTUATED_LEG_JOINT_RE_2)

  joint_velocity_1 = asset.data.joint_vel[:, leg_ids_1]
  joint_velocity_2 = asset.data.joint_vel[:, leg_ids_2]

  # Flat penalty in between the limits, quadratic outside
  over_limit_1 = torch.where(
    joint_velocity_1 > limit_leg_length,
    joint_velocity_1 - limit_leg_length,
    torch.zeros_like(joint_velocity_1),
  ) + torch.where(
    joint_velocity_1 < -limit_leg_length,
    -limit_leg_length - joint_velocity_1,
    torch.zeros_like(joint_velocity_1),
  )
  over_limit_squared_1 = torch.square(over_limit_1)

  over_limit_2 = torch.where(
    joint_velocity_2 > limit_leg_sliders,
    joint_velocity_2 - limit_leg_sliders,
    torch.zeros_like(joint_velocity_2),
  ) + torch.where(
    joint_velocity_2 < -limit_leg_sliders,
    -limit_leg_sliders - joint_velocity_2,
    torch.zeros_like(joint_velocity_2),
  )
  over_limit_squared_2 = torch.square(over_limit_2)

  penalty = torch.sum(over_limit_squared_1, dim=1) + torch.sum(
    over_limit_squared_2, dim=1
  )
  return penalty  # Positive, scaled to negative with weight


class tendon_equality_violation_exp:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.tendon_ids, self.global_tendon_ids, self.eq_ids = resolve_tendon_eq_targets(
      env, asset_cfg
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    deadzone: float,
    scale: float,
  ) -> torch.Tensor:
    violation = tendon_length_violation(
      env, asset_cfg, self.tendon_ids, self.global_tendon_ids, self.eq_ids
    ).abs()  # (num_envs, num_tendons)

    max_violation = violation.max(dim=-1).values  # (num_envs,)
    excess = torch.relu(max_violation - deadzone)
    penalty = torch.expm1(excess / scale)  # exp(excess / scale) - 1

    env.extras["log"]["Metrics/tendon_eq_violation_max"] = torch.max(violation)
    env.extras["log"]["Metrics/tendon_eq_violation_median"] = torch.median(violation)

    return penalty  # Positive, scaled to negative with weight
