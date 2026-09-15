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
from typing import TYPE_CHECKING

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
