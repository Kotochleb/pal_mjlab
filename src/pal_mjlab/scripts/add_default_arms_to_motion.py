"""Turn a lower-body Kangaroo motion npz into a full-body one, arms held still.

The input is a ``csv_to_npz.py`` motion recorded on the simple lower-body
Kangaroo (``LOWER_BODY_JOINT_ORDER``). The output has the simple full-body
model's layout (``SIMPLE_MODEL_JOINT_ORDER``, its bodies in tree order), with
every arm joint fixed at its Kangaroo Full default position (``INIT_STATE``)
and zero velocity in every frame. Body poses and velocities are recomputed by
forward kinematics of the simple full-body model from the root pose and
velocity and the joints; the bodies both simple MJCFs share are checked
against the input. Other entries (fps, dt, torques, actuated_names,
contact_wrenches) are copied unchanged, so torques still cover only the
lower-body actuators.

  uv run python -m pal_mjlab.scripts.add_default_arms_to_motion \\
    --input-file motions/jump_ocp_solution_mjlab.npz \\
    --output-file motions/jump_ocp_solution_mjlab_arms.npz
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
import tyro
from pal_mjlab.robots import LOWER_BODY_JOINT_ORDER, SIMPLE_MODEL_JOINT_ORDER
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import get_kangaroo_spec
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import INIT_STATE

ARM_JOINTS = tuple(name for name in SIMPLE_MODEL_JOINT_ORDER if name.startswith("arm_"))

# Largest tolerated difference between the input's lower-body body poses and
# the recomputed ones (m, and quaternion components).
_FK_TOLERANCE = 1e-5
# The same for their velocities (m/s, rad/s), a looser bound: the input's are
# simulated, these come from the joint velocities.
_FK_VELOCITY_TOLERANCE = 1e-3
# Input bodies checked: those kangaroo.xml and kangaroo_lower_body.xml both
# place alike. Below leg_*_5_link the two MJCFs model the foot differently
# (other bodies, the sole ~6 cm apart); leg_*_4_link sits at the same place
# but its frame is turned 20 deg about y, so only its position is checked.
_CHECKED_BODIES = (
  r"base_link|pelvis_[12]_link|leg_(left|right)_([1-5]|length|femur|knee)_link"
)
_ROTATED_BODIES = r"leg_(left|right)_4_link"


def default_arm_positions(
  joint_pos: Mapping[str, float] = INIT_STATE.joint_pos,
) -> np.ndarray:
  """Each ARM_JOINTS entry's default position, from INIT_STATE's regex keys."""
  values = []
  for name in ARM_JOINTS:
    matches = [v for expr, v in joint_pos.items() if re.fullmatch(expr, name)]
    if len(matches) != 1:
      raise ValueError(f"{name}: {len(matches)} INIT_STATE entries match")
    values.append(float(matches[0]))
  return np.asarray(values)


def add_default_arms(motion: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
  """The full-body motion, as described in the module docstring."""
  joint_names = tuple(str(n) for n in motion["joint_names"])
  if joint_names != LOWER_BODY_JOINT_ORDER:
    raise ValueError(f"expected a lower-body motion, got joints {joint_names}")
  num_frames = motion["joint_pos"].shape[0]

  src = {name: i for i, name in enumerate(joint_names)}
  arms = default_arm_positions()
  joint_pos = np.zeros((num_frames, len(SIMPLE_MODEL_JOINT_ORDER)))
  joint_vel = np.zeros_like(joint_pos)
  for column, name in enumerate(SIMPLE_MODEL_JOINT_ORDER):
    if name in src:
      joint_pos[:, column] = motion["joint_pos"][:, src[name]]
      joint_vel[:, column] = motion["joint_vel"][:, src[name]]
    else:
      joint_pos[:, column] = arms[ARM_JOINTS.index(name)]

  spec = get_kangaroo_spec()
  # Kinematics only: drop the contact excludes, some name the scene's terrain.
  for exclude in list(spec.excludes):
    spec.delete(exclude)
  model = spec.compile()
  data = mujoco.MjData(model)
  free = [
    j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free) != 1:
    raise ValueError("expected one free joint in the simple Kangaroo model")
  free_q = model.jnt_qposadr[free[0]]
  free_v = model.jnt_dofadr[free[0]]
  q_adr = [model.joint(n).qposadr[0] for n in SIMPLE_MODEL_JOINT_ORDER]
  v_adr = [model.joint(n).dofadr[0] for n in SIMPLE_MODEL_JOINT_ORDER]
  # The entity's bodies: every body but the world, in tree (id) order.
  body_ids = np.arange(1, model.nbody)
  body_names = [model.body(i).name for i in body_ids]

  root_pos = motion["body_pos_w"][:, 0]
  root_quat = motion["body_quat_w"][:, 0]
  root_lin_vel = motion["body_lin_vel_w"][:, 0]
  root_ang_vel = motion["body_ang_vel_w"][:, 0]
  shape = (num_frames, len(body_ids))
  body_pos = np.zeros((*shape, 3))
  body_quat = np.zeros((*shape, 4))
  body_lin_vel = np.zeros((*shape, 3))
  body_ang_vel = np.zeros((*shape, 3))
  quat_inv = np.zeros(4)
  velocity = np.zeros(6)
  for t in range(num_frames):
    data.qpos[free_q : free_q + 3] = root_pos[t]
    data.qpos[free_q + 3 : free_q + 7] = root_quat[t]
    data.qpos[q_adr] = joint_pos[t]
    data.qvel[:] = 0.0
    # Free joint: world-frame linear, body-frame angular velocity.
    data.qvel[free_v : free_v + 3] = root_lin_vel[t]
    mujoco.mju_negQuat(quat_inv, root_quat[t])
    mujoco.mju_rotVecQuat(data.qvel[free_v + 3 : free_v + 6], root_ang_vel[t], quat_inv)
    data.qvel[v_adr] = joint_vel[t]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    body_pos[t] = data.xpos[body_ids]
    body_quat[t] = data.xquat[body_ids]
    # mjOBJ_XBODY: at the link frame origin, like body_link_lin_vel_w.
    for i, body_id in enumerate(body_ids):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, int(body_id), velocity, 0
      )
      body_ang_vel[t, i] = velocity[:3]
      body_lin_vel[t, i] = velocity[3:]

  # The legs and torso do not depend on the arms: they must reproduce the input
  # wherever the two simple MJCFs place the same frame.
  checked = 0
  for k, name in enumerate(str(n) for n in motion["body_names"]):
    if not re.fullmatch(_CHECKED_BODIES, name):
      continue
    i = body_names.index(name)
    checked += 1
    for key, ours, tol in (
      ("body_pos_w", body_pos, _FK_TOLERANCE),
      ("body_quat_w", body_quat, _FK_TOLERANCE),
      ("body_lin_vel_w", body_lin_vel, _FK_VELOCITY_TOLERANCE),
      ("body_ang_vel_w", body_ang_vel, _FK_VELOCITY_TOLERANCE),
    ):
      if key == "body_quat_w" and re.fullmatch(_ROTATED_BODIES, name):
        continue
      theirs = motion[key][:, k]
      if key == "body_quat_w":  # q and -q are the same rotation.
        theirs = theirs * np.sign(np.sum(theirs * ours[:, i], axis=-1, keepdims=True))
      error = float(np.abs(ours[:, i] - theirs).max())
      if error > tol:
        raise ValueError(f"{name} {key} differs from the input by {error:.2e}")
  if checked != 2 * 8 + 3:
    raise ValueError(f"checked {checked} input bodies, expected the 19 shared ones")

  out = {key: np.asarray(motion[key]) for key in motion.keys()}
  out.update(
    joint_names=np.asarray(SIMPLE_MODEL_JOINT_ORDER),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_names=np.asarray(body_names),
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
  )
  return out


def main(input_file: Path, output_file: Path) -> None:
  """Write the full-body copy of input_file to output_file."""
  if input_file.resolve() == output_file.resolve():
    raise ValueError("output_file must differ from input_file (it is a copy)")
  with np.load(input_file) as motion:
    out = add_default_arms(motion)
  np.savez(output_file, **out)
  print(
    f"Wrote {output_file}: {out['joint_pos'].shape[0]} frames,"
    f" {len(out['joint_names'])} joints, {len(out['body_names'])} bodies"
  )


if __name__ == "__main__":
  tyro.cli(main)
