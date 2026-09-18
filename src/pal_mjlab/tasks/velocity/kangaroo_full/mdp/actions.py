"""Joint position action term that keeps the configured target order.

mjlab's ``JointPositionAction`` resolves its ``actuator_names`` through
``Entity.find_joints_by_actuator_names``, which always returns the matched
joints in the MJCF's natural order and ignores ``preserve_order`` (only the
TENDON and SITE transmissions honour it). That is fine for one catch-all
term, but not when a slider-driven mechanism has to occupy the same action
slots as the tendon-driven version of the same mechanism in
``pal_kangaroo_full`` -- there the tendon terms are explicit, order-preserved
name lists, and the sliders' natural order in ``kangaroo_full.xml`` does not
always agree with them (the right ankle's ``leg_right_4_actuator`` precedes
``leg_right_5_actuator`` in the tree, the tendon order is r then l).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class OrderedJointPositionAction(JointPositionAction):
  """``JointPositionAction`` whose action slots follow ``cfg.actuator_names``.

  With ``cfg.preserve_order`` set, each entry of ``actuator_names`` is resolved
  in turn and its matches keep that position in the action vector (the same
  contract ``TendonLengthActionCfg(preserve_order=True)`` gives). Every match
  still has to be an actuated joint. With ``preserve_order`` unset this is
  exactly the parent term.
  """

  def _find_targets(self, cfg: JointPositionActionCfg) -> tuple[list[int], list[str]]:
    if cfg.transmission_type != TransmissionType.JOINT or not cfg.preserve_order:
      return super()._find_targets(cfg)
    entity = self._entity
    actuated: set[str] = set()
    for act in entity._actuators:
      actuated.update(act.target_names)
    actuated_in_natural_order = [n for n in entity.joint_names if n in actuated]
    _, names = entity.find_joints(
      cfg.actuator_names, joint_subset=actuated_in_natural_order, preserve_order=True
    )
    name_to_entity_idx = {name: i for i, name in enumerate(entity.joint_names)}
    return [name_to_entity_idx[name] for name in names], names


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
  """``JointPositionActionCfg`` that honours ``preserve_order`` (default on)."""

  preserve_order: bool = True

  def build(self, env: ManagerBasedRlEnv) -> OrderedJointPositionAction:
    return OrderedJointPositionAction(self, env)


class MappedLegLengthPositionAction(OrderedJointPositionAction):
  """Order-preserved joint position targets, with the knee joints commanded as
  leg lengths in metres.

  For the "lut" transmission's leg length on the MJCFs whose only leg-length
  DOF is the knee: the actuator's target joint is ``leg_.*_knee_joint``, but
  it servos the knee-to-length map of that joint, so its position target is
  a leg length. This term commands it in the coordinate the simple model's
  ``leg_.*_length_joint`` term would -- the same map the observation term
  reconstructs the missing joint through -- while every other target it
  carries is an ordinary joint position target, so the whole leg can sit in
  **one** term laid out like the simple model's action vector (the knee in
  the leg-length joint's slot) rather than the knees trailing in a term of
  their own. For the knee columns:

  * the offset is the map evaluated at the knee's default position, so a
    zero action holds the default leg length (``use_default_offset``, in the
    mapped coordinate);
  * the encoder bias subtracted before the target is written is the mapped
    leg-length bias the actor's observation carries, not the knee's own.

  ``actuator_names`` is the ordered target list; ``mapped_joints`` pairs each
  leg-length joint name with the knee that takes its slot and bias. Targets
  not named as a knee there are left to the parent term.
  """

  def __init__(self, cfg: MappedLegLengthPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    from .observations import _interpolate, load_knee_leg_length_map

    length_of_knee = {knee: length for length, knee in cfg.mapped_joints}
    knee_cols = [i for i, n in enumerate(self._target_names) if n in length_of_knee]
    if not knee_cols:
      raise ValueError(
        f"{type(self).__name__}: none of {self._target_names} is a knee of "
        f"mapped_joints {cfg.mapped_joints}"
      )
    self._knee_cols = torch.as_tensor(knee_cols, dtype=torch.long, device=self.device)
    self._knee_ids = self._target_ids[self._knee_cols]

    self._table = load_knee_leg_length_map(cfg.csv_path).to(self.device)
    self._interpolate = _interpolate
    if cfg.use_default_offset:
      # The parent set every column to the joint's default position; the
      # knee columns are re-expressed through the map.
      assert isinstance(self._offset, torch.Tensor)
      offset = self._offset.clone()
      default_knee = self._entity.data.default_joint_pos[:, self._knee_ids]
      offset[:, self._knee_cols], _ = self._interpolate(default_knee, self._table)
      self._offset = offset

    from .dr.encoder_bias import mapped_leg_length_encoder_bias

    self._length_bias: torch.Tensor | None = None
    try:
      bias_event = env.event_manager.get_term_cfg("leg_length_encoder_bias").func
    except ValueError:
      bias_event = None  # No such event: nothing to subtract.
    if isinstance(bias_event, mapped_leg_length_encoder_bias):
      self._length_bias = bias_event.bias
      self._length_bias_ids = torch.as_tensor(
        [
          bias_event.joint_names.index(length_of_knee[self._target_names[c]])
          for c in knee_cols
        ],
        dtype=torch.long,
        device=self.device,
      )

  def apply_actions(self) -> None:
    target = self._processed_actions
    if self._length_bias is not None:
      target = target.clone()
      target[:, self._knee_cols] -= self._length_bias[:, self._length_bias_ids]
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class MappedLegLengthPositionActionCfg(OrderedJointPositionActionCfg):
  """``OrderedJointPositionActionCfg`` whose knee targets are leg lengths
  applied through ``knee_distance_map.csv``."""

  csv_path: str | Path
  """The knee-to-leg-length map (see ``observations.load_knee_leg_length_map``)."""

  mapped_joints: tuple[tuple[str, str], ...]
  """``(leg length joint, knee joint)`` pairs, as for the observation term;
  the knees among ``actuator_names`` are the mapped targets."""

  def build(self, env: ManagerBasedRlEnv) -> MappedLegLengthPositionAction:
    return MappedLegLengthPositionAction(self, env)
