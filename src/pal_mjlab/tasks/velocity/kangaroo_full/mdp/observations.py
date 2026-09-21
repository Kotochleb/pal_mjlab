from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  load_transmission_maps,
)
from pal_mjlab.robots.pal_kangaroo_full.lut_maps import LegLengthMap

from .dr.encoder_bias import mapped_leg_length_encoder_bias

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _leg_length_and_slope(
  leg: LegLengthMap, knee: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Leg length (in ``leg_.*_length_joint``'s own metres) and its slope with
  respect to the knee angle, from ``leg_length_map.npz``.

  Returns:
    ``(value, slope)`` -- the leg length at ``knee`` and ``d(length)/d(knee)``,
    the latter being what turns a knee velocity into a leg length rate.
  """
  s = leg.slider_of_knee(knee)
  slope = 1.0 / (leg.jacobian(s) * leg.dknee_dslider(s))
  return leg.distance(s), slope


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

  The actor adds an independent, persistent leg-length encoder bias after
  reconstruction, in metres, just as for a real leg-length joint.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    joint_order: Sequence[str] = cfg.params["joint_order"]
    self.num_joints = len(joint_order)
    self.leg = load_transmission_maps().leg_length.to(env.device)

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
    self.length_encoder_bias: torch.Tensor | None = None
    if mapped_cols and cfg.params.get("biased", False):
      bias_event = env.event_manager.get_term_cfg("leg_length_encoder_bias").func
      if not isinstance(bias_event, mapped_leg_length_encoder_bias):
        raise TypeError("Mapped actor positions require mapped leg-length encoder bias")
      self.length_encoder_bias = bias_event.bias
      self.length_bias_ids = _ids(
        [bias_event.joint_names.index(joint_order[col]) for col in mapped_cols]
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_order: Sequence[str],
    mapped_joints: Sequence[tuple[str, str]],
    mode: Literal["pos", "vel"] = "pos",
    biased: bool = False,
  ) -> torch.Tensor:
    del joint_order, mapped_joints  # Resolved once, in __init__.
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
      length, slope = _leg_length_and_slope(self.leg, knee)
      if mode == "pos":
        default_knee = data.default_joint_pos
        assert default_knee is not None
        length_default, _ = _leg_length_and_slope(
          self.leg, default_knee[:, self.knee_ids]
        )
        out[:, self.mapped_cols] = length - length_default
        if biased and self.length_encoder_bias is not None:
          out[:, self.mapped_cols] += self.length_encoder_bias[:, self.length_bias_ids]
      else:
        out[:, self.mapped_cols] = slope * data.joint_vel[:, self.knee_ids]

    return out


class ankle_femur_normalized:
  """Another joint-position observation term, with each ankle joint 4 column
  shifted by its leg's femur joint -- the observation-side counterpart to
  ``mdp.rewards.joint_limits_convex_hull_ankle_femur_normalized``.

  ``leg_.*_4_joint`` (ankle pitch) is measured off the shank, but on the full
  and full-full models the shank itself rotates on ``leg_.*_femur_joint``
  through the four-bar (or connect-loop) femur closure, so the ankle bar
  mechanism's pitch relative to the shank -- the quantity a policy trained
  against the simple model expects -- is ``leg_.*_4_joint - leg_.*_femur_joint``,
  not ``leg_.*_4_joint`` alone.

  ``inner`` is the ``ObservationTermCfg`` this variant would otherwise use for
  the term -- stock ``joint_pos_rel`` when the model still has every
  simple-model joint, ``joint_state_with_mapped_leg_length`` when the leg
  length slot is reconstructed from the knee -- called unchanged first, so
  the vector's order and every value but the ones ``ankle_femur_pairs`` names
  are exactly what that term would already produce. The femur term is taken
  relative to its own default, the same convention every other slot in the
  vector already follows, so this column is still zero at the default pose
  rather than picking up a constant offset equal to the default femur angle.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    inner: ObservationTermCfg = cfg.params["inner"]
    for value in inner.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    self._inner_func = (
      inner.func(cfg=inner, env=env) if inspect.isclass(inner.func) else inner.func
    )
    self._inner_params = inner.params

    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    joint_order: Sequence[str] = cfg.params["joint_order"]
    ankle_femur_pairs: Sequence[tuple[str, str]] = cfg.params["ankle_femur_pairs"]

    self.ankle_cols = torch.as_tensor(
      [joint_order.index(ankle) for ankle, _ in ankle_femur_pairs],
      device=env.device,
      dtype=torch.long,
    )
    self.femur_ids = torch.as_tensor(
      [asset.joint_names.index(femur) for _, femur in ankle_femur_pairs],
      device=env.device,
      dtype=torch.long,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_order: Sequence[str],
    ankle_femur_pairs: Sequence[tuple[str, str]],
    inner: ObservationTermCfg,
  ) -> torch.Tensor:
    del joint_order, ankle_femur_pairs, inner  # Resolved once, in __init__.
    out = self._inner_func(env, **self._inner_params).clone()

    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    femur_rel = (
      asset.data.joint_pos[:, self.femur_ids] - default_joint_pos[:, self.femur_ids]
    )
    out[:, self.ankle_cols] -= femur_rel
    return out
