"""Motion tracking command for the closed-loop Kangaroo Full model."""

from __future__ import annotations

import copy
import io
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

from pal_mjlab.robots import SIMPLE_ANKLE_PITCH_OFFSET
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  load_transmission_maps,
)
from pal_mjlab.tasks.velocity.kangaroo_full.mdp.observations import (
  _leg_length_and_slope,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

ANKLE_PITCH_JOINTS: tuple[tuple[str, str, str], ...] = tuple(
  (f"leg_{side}_4_joint", f"leg_{side}_femur_joint", f"leg_{side}_knee_joint")
  for side in ("left", "right")
)
"""(ankle pitch, femur, knee) per leg, for the simple -> full leg_*_4_joint shift."""

_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1e-9
_RESIDUAL_WARN = 1e-4


def _equality_rows(data: mujoco.MjData) -> np.ndarray:
  return np.flatnonzero(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)


def _pin_tendon_lengths(model: mujoco.MjModel, lengths: Mapping[str, float]) -> None:
  """Same eq_data as ``enforce_tendon_lengths`` writes into the batched model."""
  for i in range(model.neq):
    if model.eq_type[i] != mujoco.mjtEq.mjEQ_TENDON:
      continue
    tendon_id = int(model.eq_obj1id[i])
    name = model.tendon(tendon_id).name.split("/")[-1]
    for pattern, length in lengths.items():
      if re.fullmatch(pattern, name):
        model.eq_data[i, 0] = length - model.tendon_length0[tendon_id]


def simple_motion_to_full_model(
  motion: Mapping[str, np.ndarray],
  model: mujoco.MjModel,
  *,
  body_ids: np.ndarray,
  joint_names: Sequence[str],
  joint_q_adr: np.ndarray,
  joint_v_adr: np.ndarray,
  free_joint_q_adr: np.ndarray,
  free_joint_v_adr: np.ndarray,
  default_joint_pos: np.ndarray,
  source_joint_names: Sequence[str],
  solved_joint_names: Sequence[str] = (),
  tendon_lengths: Mapping[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  """Replays a simple-model motion kinematically on the full model.

  ``motion`` is a ``csv_to_npz.py`` motion recorded on the simple Kangaroo
  model; only its root (body 0) and its joints, laid out as
  ``source_joint_names``, are read. Every source joint the full model also has
  is copied over -- except those matching ``solved_joint_names`` -- with
  leg_*_4_joint shifted to the full model's convention. The remaining joints
  (closed-loop passive joints, plus the solved ones) are Newton-projected onto
  the equality constraints, starting from ``default_joint_pos`` and then warm
  started frame to frame; their velocities are the ones that keep the
  constraints satisfied. Body poses and velocities come from the full model's
  forward kinematics.

  The joint and body arguments are the entity's ``EntityIndexing`` addresses
  in ``model``, in ``joint_names`` / body order.

  Returns:
    ``(arrays, loop_joints)``: the motion as ``MotionLoader`` reads it, with
    joints and bodies laid out like the entity, and a mask of the entity joints
    that any equality constraint touches.
  """
  model = copy.deepcopy(model)
  model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
  if tendon_lengths:
    _pin_tendon_lengths(model, tendon_lengths)
  data = mujoco.MjData(model)

  src_q = np.asarray(motion["joint_pos"], dtype=np.float64)
  src_qd = np.asarray(motion["joint_vel"], dtype=np.float64)
  if src_q.shape[1] != len(source_joint_names):
    raise ValueError(
      f"motion has {src_q.shape[1]} joints, expected {len(source_joint_names)}"
      f" ({source_joint_names[0]} ... {source_joint_names[-1]}); was it recorded"
      " on the matching simple Kangaroo model (full body vs lower body)?"
    )
  src = {name: i for i, name in enumerate(source_joint_names)}
  joint_index = {name: i for i, name in enumerate(joint_names)}

  # Source joint -> entity joint copies, with the ankle pitch shifted.
  copied = [
    name
    for name in source_joint_names
    if name in joint_index
    and not any(re.fullmatch(expr, name) for expr in solved_joint_names)
  ]
  q = src_q[:, [src[n] for n in copied]].copy()
  qd = src_qd[:, [src[n] for n in copied]].copy()
  for ankle, femur, knee in ANKLE_PITCH_JOINTS:
    if ankle in copied:
      k = copied.index(ankle)
      q[:, k] += -src_q[:, src[femur]] + src_q[:, src[knee]] - SIMPLE_ANKLE_PITCH_OFFSET
      qd[:, k] += -src_qd[:, src[femur]] + src_qd[:, src[knee]]
  copied_q_adr = joint_q_adr[[joint_index[n] for n in copied]]
  copied_v_adr = joint_v_adr[[joint_index[n] for n in copied]]
  free_v_adr = np.setdiff1d(joint_v_adr, copied_v_adr)

  root_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)[:, 0]
  root_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)[:, 0]
  root_lin_vel = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)[:, 0]
  root_ang_vel = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)[:, 0]

  num_frames = src_q.shape[0]
  out_q = np.zeros((num_frames, len(joint_names)))
  out_qd = np.zeros_like(out_q)
  body_pos = np.zeros((num_frames, len(body_ids), 3))
  body_quat = np.zeros((num_frames, len(body_ids), 4))
  body_lin_vel = np.zeros_like(body_pos)
  body_ang_vel = np.zeros_like(body_pos)
  loop_dofs = np.zeros(model.nv, dtype=bool)
  worst = (0.0, 0)

  data.qpos[joint_q_adr] = default_joint_pos
  ang_local = np.zeros(3)
  quat_inv = np.zeros(4)
  velocity = np.zeros(6)
  for t in range(num_frames):
    data.qpos[free_joint_q_adr[:3]] = root_pos[t]
    data.qpos[free_joint_q_adr[3:7]] = root_quat[t]
    data.qpos[copied_q_adr] = q[t]

    # Newton projection of the free joints onto the equality constraints.
    for _ in range(_NEWTON_ITERATIONS):
      mujoco.mj_forward(model, data)
      rows = _equality_rows(data)
      residual = data.efc_pos[rows]
      if len(rows) == 0 or np.abs(residual).max() < _NEWTON_TOLERANCE:
        break
      jac = data.efc_J.reshape(data.nefc, model.nv)[np.ix_(rows, free_v_adr)]
      step = np.zeros(model.nv)
      step[free_v_adr] = -np.linalg.lstsq(jac, residual, rcond=None)[0]
      mujoco.mj_integratePos(model, data.qpos, step, 1.0)
    mujoco.mj_forward(model, data)
    rows = _equality_rows(data)
    if len(rows):
      err = float(np.abs(data.efc_pos[rows]).max())
      worst = max(worst, (err, t))

    # Free joint velocities that keep the constraints satisfied.
    data.qvel[:] = 0.0
    data.qvel[free_joint_v_adr[:3]] = root_lin_vel[t]
    mujoco.mju_negQuat(quat_inv, root_quat[t])
    mujoco.mju_rotVecQuat(ang_local, root_ang_vel[t], quat_inv)
    data.qvel[free_joint_v_adr[3:6]] = ang_local
    data.qvel[copied_v_adr] = qd[t]
    if len(rows):
      jac = data.efc_J.reshape(data.nefc, model.nv)[rows]
      loop_dofs |= np.any(np.abs(jac) > 1e-9, axis=0)
      data.qvel[free_v_adr] = -np.linalg.lstsq(
        jac[:, free_v_adr], jac @ data.qvel, rcond=None
      )[0]
    mujoco.mj_forward(model, data)

    out_q[t] = data.qpos[joint_q_adr]
    out_qd[t] = data.qvel[joint_v_adr]
    body_pos[t] = data.xpos[body_ids]
    body_quat[t] = data.xquat[body_ids]
    # mjOBJ_XBODY: at the link frame origin, like body_link_lin_vel_w (mjOBJ_BODY
    # would give the COM's).
    for i, body_id in enumerate(body_ids):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_XBODY, int(body_id), velocity, 0
      )
      body_ang_vel[t, i] = velocity[:3]
      body_lin_vel[t, i] = velocity[3:]

  if worst[0] > _RESIDUAL_WARN:
    warnings.warn(
      f"full-model loop closure residual {worst[0]:.2e} at motion frame {worst[1]}",
      stacklevel=2,
    )

  arrays = {
    "fps": np.asarray(motion["fps"]),
    "joint_pos": out_q,
    "joint_vel": out_qd,
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_lin_vel,
    "body_ang_vel_w": body_ang_vel,
  }
  return arrays, loop_dofs[joint_v_adr]


class FullModelMotionCommand(MotionCommand):
  """``MotionCommand`` that tracks a simple-model motion on the full model.

  ``motion_file`` is a ``csv_to_npz.py`` motion recorded on the simple Kangaroo
  (``kangaroo`` or ``kangaroo_lower_body``), so the full-model tasks share the
  simple tasks' motion registry. It is converted once, at construction, by
  ``simple_motion_to_full_model``; everything else is ``MotionCommand`` on the
  converted motion, apart from:

  - ``command`` is the motion's joint state in the full-model velocity task's
    joint observation space instead of every passive loop joint: the source
    joints in source order, relative to the default pose like
    ``joint_state_with_mapped_leg_length`` (``mapped_joints`` rebuilt from
    their knee through the leg-length map) and, for ``ankle_femur_pairs``,
    with the ankle pitch shifted by the femur like ``ankle_femur_normalized``.
    It so has the simple tracking command's layout.
  - Reset clipping and ``joint_position_range`` noise apply only to joints
    outside the closed loops; loop joints are written as converted, so a reset
    never starts with a violated constraint.
  """

  cfg: FullModelMotionCommandCfg

  def __init__(self, cfg: FullModelMotionCommandCfg, env: ManagerBasedRlEnv):
    robot: Entity = env.scene[cfg.entity_name]
    indexing = robot.indexing
    arrays, loop_joints = simple_motion_to_full_model(
      np.load(cfg.motion_file),
      env.sim.mj_model,
      body_ids=indexing.body_ids.cpu().numpy(),
      joint_names=robot.joint_names,
      joint_q_adr=indexing.joint_q_adr.cpu().numpy(),
      joint_v_adr=indexing.joint_v_adr.cpu().numpy(),
      free_joint_q_adr=indexing.free_joint_q_adr.cpu().numpy(),
      free_joint_v_adr=indexing.free_joint_v_adr.cpu().numpy(),
      default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy(),
      source_joint_names=cfg.source_joint_names,
      solved_joint_names=cfg.solved_joint_names,
      tendon_lengths=cfg.tendon_lengths,
    )
    # MotionLoader np.load()s motion_file, which also takes a file object.
    converted = io.BytesIO()
    np.savez(converted, **arrays)
    converted.seek(0)
    super().__init__(replace(cfg, motion_file=converted), env)  # type: ignore[arg-type]

    self._loop_joints = torch.tensor(loop_joints, device=self.device)
    self._command_joint_pos, self._command_joint_vel = self._observed_joint_state()

  def _observed_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole motion's joint positions and velocities, per frame, as the
    full-model velocity task observes the robot's (without bias or noise)."""
    joint_names = self.robot.joint_names
    knee_of = dict(self.cfg.mapped_joints)
    q = self.motion.joint_pos
    qd = self.motion.joint_vel
    default = self.robot.data.default_joint_pos[0]
    pos = torch.zeros(q.shape[0], len(self.cfg.source_joint_names), device=self.device)
    vel = torch.zeros_like(pos)
    leg = None
    for column, name in enumerate(self.cfg.source_joint_names):
      if name in joint_names:
        i = joint_names.index(name)
        pos[:, column] = q[:, i] - default[i]
        vel[:, column] = qd[:, i]
        continue
      if name not in knee_of:
        raise ValueError(f"no joint '{name}' and no knee mapped to it")
      if leg is None:
        leg = load_transmission_maps().leg_length.to(self.device)
      k = joint_names.index(knee_of[name])
      length, slope = _leg_length_and_slope(leg, q[:, k])
      length_default, _ = _leg_length_and_slope(leg, default[k : k + 1])
      pos[:, column] = length - length_default
      vel[:, column] = slope * qd[:, k]
    for ankle, femur in self.cfg.ankle_femur_pairs:
      f = joint_names.index(femur)
      pos[:, self.cfg.source_joint_names.index(ankle)] -= q[:, f] - default[f]
    return pos, vel

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self._command_joint_pos[self.time_steps],
        self._command_joint_vel[self.time_steps],
      ],
      dim=1,
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.where(
      self._loop_joints,
      self.joint_pos[env_ids],
      torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1]),
    )
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)


@dataclass(kw_only=True)
class FullModelMotionCommandCfg(MotionCommandCfg):
  source_joint_names: tuple[str, ...]
  """Joint layout of ``motion_file``: the simple model's joints, in tree order."""
  solved_joint_names: tuple[str, ...] = ()
  """Source joints (regex) solved by loop closure instead of copied."""
  tendon_lengths: dict[str, float] | None = None
  """Tendon equality lengths the env pins at reset (``enforce_tendon_lengths``)."""
  mapped_joints: tuple[tuple[str, str], ...] = ()
  """(missing joint, knee) pairs, as ``joint_state_with_mapped_leg_length``."""
  ankle_femur_pairs: tuple[tuple[str, str], ...] = ()
  """(ankle, femur) pairs, as ``ankle_femur_normalized``; empty for raw ankles."""

  @classmethod
  def from_motion_cfg(
    cls, cfg: MotionCommandCfg, **kwargs
  ) -> FullModelMotionCommandCfg:
    return cls(
      **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}, **kwargs
    )

  def build(self, env: ManagerBasedRlEnv) -> FullModelMotionCommand:
    return FullModelMotionCommand(self, env)
