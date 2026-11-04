"""The actuators a policy observes, on either full model.

Both full models expose the same twelve leg screws, as tendons on
``pal_kangaroo_full`` and slide joints on ``pal_kangaroo_full_full``. Pelvis and
arm joints are driven directly, so they are their own actuators.
"""

from __future__ import annotations

from typing import Sequence

from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  SIMPLE_MODEL_JOINT_ORDER,
)


def actuator_order(joint_order: Sequence[str]) -> tuple[str, ...]:
  """``joint_order`` with each leg joint replaced by the screw in its slot.

  The passive femur and knee joints have no actuator and are dropped.
  """
  names: list[str] = []
  for name in joint_order:
    if name.endswith(("_femur_joint", "_knee_joint")):
      continue
    if name.startswith("leg_"):
      name = name.removesuffix("_joint") + "_actuator"
    names.append(name)
  return tuple(names)


ACTUATOR_ORDER = actuator_order(SIMPLE_MODEL_JOINT_ORDER)
LOWER_BODY_ACTUATOR_ORDER = tuple(
  name for name in ACTUATOR_ORDER if not name.startswith("arm_")
)
