"""Encoder calibration errors in the simple Kangaroo's joint coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.managers.event_manager import EventTermCfg


class mapped_leg_length_encoder_bias:
  """Persistent startup bias for leg-length encoders absent from the MJCF."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self.joint_names = tuple(cfg.params["joint_names"])
    self.bias = torch.zeros(env.num_envs, len(self.joint_names), device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    joint_names: tuple[str, ...],
  ) -> None:
    del joint_names  # Resolved in __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    self.bias[env_ids] = sample_uniform(
      bias_range[0], bias_range[1], (len(env_ids), len(self.joint_names)), env.device
    )


def configure_simple_model_encoder_bias(
  cfg: ManagerBasedRlEnvCfg,
  joint_order: tuple[str, ...],
  has_leg_length_joint: bool,
) -> None:
  """Keep baseline bias ranges, selecting only simple-model coordinates."""
  length_names = tuple(n for n in joint_order if n.endswith("_length_joint"))
  cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=tuple(n for n in joint_order if n not in length_names),
    preserve_order=True,
  )
  length_event = cfg.events["leg_length_encoder_bias"]
  if has_leg_length_joint:
    length_event.params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=length_names, preserve_order=True
    )
  else:
    length_event.func = mapped_leg_length_encoder_bias
    length_event.params = {
      "bias_range": length_event.params["bias_range"],
      "joint_names": length_names,
    }
