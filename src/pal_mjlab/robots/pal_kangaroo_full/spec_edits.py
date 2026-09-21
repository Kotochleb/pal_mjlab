from __future__ import annotations

import mujoco


def delete_tendons(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete spatial tendons and any equality constraint that references them."""
  targets = set(names)
  for eq in list(spec.equalities):
    if eq.type == mujoco.mjtEq.mjEQ_TENDON and eq.name1 in targets:
      spec.delete(eq)
  for tendon in list(spec.tendons):
    if tendon.name in targets:
      spec.delete(tendon)


def delete_equalities(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  targets = set(names)
  for eq in list(spec.equalities):
    if eq.name in targets:
      spec.delete(eq)


def delete_joints(spec: mujoco.MjSpec, names: tuple[str, ...]) -> None:
  """Delete joints by name, welding their body to its parent."""
  targets = set(names)
  for joint in list(spec.joints):
    if joint.name in targets:
      spec.delete(joint)


def delete_subtrees(spec: mujoco.MjSpec, body_names: tuple[str, ...]) -> None:
  for name in body_names:
    body = spec.body(name)
    if body is None:
      raise ValueError(f"MJCF has no body '{name}' to delete")
    spec.delete(body)


def _add_capsule(spec: mujoco.MjSpec, body_name: str, geom_name: str, **shape) -> None:
  body = spec.body(body_name)
  if body is None:
    raise ValueError(f"MJCF has no body '{body_name}' to hang '{geom_name}' off")
  # density=0: every body these hang off declares an explicit <inertial>, so a
  # collision proxy should not add mass.
  body.add_geom(
    name=geom_name,
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    group=3,
    contype=1,
    conaffinity=1,
    density=0.0,
    material="bright_orange",
    **shape,
  )


def add_collision_capsules(spec: mujoco.MjSpec) -> None:
  """Add the pelvis/forearm/femur/tibia/hip-xy-motor capsules the hand-written
  pal_kangaroo model has that these MJCFs don't -- lifted verbatim from
  kangaroo.xml so both robots present the same contact geometry to a policy."""
  _add_capsule(
    spec,
    "pelvis_2_link",
    "pelvis_2_collision",
    pos=(4.77798e-07, -1.67441e-06, 0.234507),
    quat=(1.0, -3.97075e-06, -1.19191e-05, 0.0),
    size=[0.163724, 0.162985, 0.0],
  )
  _add_capsule(
    spec,
    "arm_left_4_link",
    "arm_left_4_collision",
    pos=(0.210246, -0.0167497, -0.0171982),
    quat=(0.733112, 0.0532001, 0.678025, -2.2842e-07),
    size=[0.054849, 0.219217, 0.0],
  )
  _add_capsule(
    spec,
    "arm_right_4_link",
    "arm_right_4_collision",
    pos=(0.210649, -0.0172373, -0.0265895),
    quat=(0.717524, 0.0513886, 0.694636, -1.38234e-06),
    size=[0.0559132, 0.220143, 0.0],
  )
  for side in ("left", "right"):
    motor = f"{side}_hip_xy_motor_{'r' if side == 'left' else 'l'}"
    _add_capsule(
      spec,
      motor,
      f"{motor}_collision",
      fromto=(-0.01, 0.0, -0.015, -0.01, 0.0, 0.0),
      size=[0.04, 0.0, 0.0],
    )
    _add_capsule(
      spec,
      f"leg_{side}_femur_link",
      f"leg_{side}_femur_collision",
      pos=(0.03, 0.2, 0.0),
      quat=(0.0308436, 0.0308436, 0.7064338, 0.7064338),
      size=[0.08, 0.2, 0.0],
    )
    _add_capsule(
      spec,
      f"leg_{side}_knee_link",
      f"leg_{side}_knee_collision",
      fromto=(0.03896, 0.02628, 0.0, -0.16904, 0.24934, 0.0),
      size=[0.04, 0.0, 0.0],
    )
    _add_capsule(
      spec,
      f"leg_{side}_knee_link",
      f"leg_{side}_knee_bar_collision",
      fromto=(-0.063, 0.0, 0.0, -0.184, 0.23889, 0.0),
      size=[0.05, 0.0, 0.0],
    )
