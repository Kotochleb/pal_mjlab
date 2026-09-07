"""Metrics for kangaroo_full closed-loop equality constraints.

Three kinds of equality can hold this model together, and their violations are
not the same shape, hence one metric term each:

* ``<tendon>`` -- a scalar length, how far the tendon is from its rest length.
* ``<connect>`` -- a 3-D displacement between the two anchor points it pins
  together, reported per world axis or as a total.
* ``<joint>`` -- a scalar in the coupled joint's own units (radians for a
  hinge), how far it has drifted from the angle its partner prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

Axis = Literal["x", "y", "z"]
Reduction = Literal["sum", "mean", "max"]

_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
  """Reduce per-constraint values (num_envs, num_constraints) to (num_envs,)."""
  if reduction == "sum":
    return values.sum(dim=-1)
  if reduction == "mean":
    return values.mean(dim=-1)
  if reduction == "max":
    return values.max(dim=-1).values
  raise ValueError(f"Unknown reduction '{reduction}'.")


def resolve_tendon_eq_targets(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  asset = env.scene[asset_cfg.name]
  model = env.sim.model

  eq_type = model.eq_type
  eq_obj1id = model.eq_obj1id

  tendon_ids: list[int] = []
  eq_ids: list[int] = []
  for local_id, tendon_name in enumerate(asset.tendon_names):
    if tendon_name not in asset_cfg.tendon_names:
      continue

    tendon_id = int(asset.indexing.tendon_ids[local_id])
    eq_rows = [
      i
      for i in range(model.neq)
      if int(eq_type[i]) == int(mujoco.mjtEq.mjEQ_TENDON)
      and int(eq_obj1id[i]) == tendon_id
    ]
    if not eq_rows:
      continue

    tendon_ids.append(local_id)
    eq_ids.append(eq_rows[0])

  local_tendon_ids = torch.as_tensor(tendon_ids, device=env.device, dtype=torch.long)
  global_tendon_ids = asset.indexing.tendon_ids[local_tendon_ids]
  eq_row_ids = torch.as_tensor(eq_ids, device=env.device, dtype=torch.long)
  return local_tendon_ids, global_tendon_ids, eq_row_ids


def tendon_length_violation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  tendon_ids: torch.Tensor,
  global_tendon_ids: torch.Tensor,
  eq_ids: torch.Tensor,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  tendon_len = asset.data.tendon_len[:, tendon_ids]

  model = env.sim.model
  tendon_len0 = model.tendon_length0[:, global_tendon_ids]
  target_offset = model.eq_data[:, eq_ids, 0]

  return (tendon_len - tendon_len0) - target_offset


class tendon_equality_constraint_violation:
  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.tendon_ids, self.global_tendon_ids, self.eq_ids = resolve_tendon_eq_targets(
      env, asset_cfg
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mode: Literal["violation", "length"] = "violation",
    reduction: Literal["sum", "mean", "max"] = "sum",
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    if mode == "length":
      tendon_len = asset.data.tendon_len[:, self.tendon_ids]
      return _reduce(tendon_len, reduction)  # (num_envs,)

    violation = tendon_length_violation(
      env, asset_cfg, self.tendon_ids, self.global_tendon_ids, self.eq_ids
    )
    return _reduce(violation.abs(), reduction)  # (num_envs,)


def _index_tensor(env: ManagerBasedRlEnv, values: Sequence[int]) -> torch.Tensor:
  return torch.as_tensor(list(values), device=env.device, dtype=torch.long)


def _resolve_eq_rows(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  constraint_names: Sequence[str],
  eq_type: mujoco.mjtEq,
  kind: str,
) -> list[int]:
  """Global rows of ``asset_cfg``'s ``eq_type`` equalities matching the regexes.

  Read off the host-side ``mj_model`` because the mjwarp model carries no
  names. Only indices are resolved here; the values themselves are read per
  step from the batched model and data, so per-env randomization still shows
  through.

  ``constraint_names`` are entity-local: scene assembly prefixes every element
  with its entity's name (``robot/leg_left_closed_kin_joint``), and that prefix
  is stripped before matching, so the same patterns work here and against a
  spec compiled standalone.
  """
  mj_model = env.sim.mj_model
  prefix = f"{asset_cfg.name}/"
  rows = [
    i
    for i in range(mj_model.neq)
    if int(mj_model.eq_type[i]) == int(eq_type)
    and (mj_model.eq(i).name.startswith(prefix) or "/" not in mj_model.eq(i).name)
  ]
  if not rows:
    raise ValueError(f"Entity '{asset_cfg.name}' has no <{kind}> equality constraints.")
  matched_ids, _ = resolve_matching_names(
    list(constraint_names),
    [mj_model.eq(i).name.removeprefix(prefix) for i in rows],
  )
  return [rows[i] for i in matched_ids]


@dataclass(frozen=True)
class ConnectEqTargets:
  """Index tensors for the two ways a ``<connect>`` can name its anchors.

  MuJoCo writes body-anchored and site-anchored connects into different
  fields -- the first pair of offsets in ``eq_data`` for bodies, a pair of
  site ids for sites -- so they are resolved into separate groups and their
  displacements computed separately. Every reduction over the result is
  order-independent, so the two groups are simply concatenated.
  """

  body_eq_ids: torch.Tensor
  body1_ids: torch.Tensor
  body2_ids: torch.Tensor
  site1_ids: torch.Tensor
  site2_ids: torch.Tensor


def resolve_connect_eq_targets(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  constraint_names: Sequence[str],
) -> ConnectEqTargets:
  """Rows and anchor pairs of the ``<connect>`` equalities named by regex."""
  mj_model = env.sim.mj_model
  eq_ids = _resolve_eq_rows(
    env, asset_cfg, constraint_names, mujoco.mjtEq.mjEQ_CONNECT, "connect"
  )

  body_rows: list[int] = []
  site_rows: list[int] = []
  for i in eq_ids:
    objtype = int(mj_model.eq_objtype[i])
    if objtype == int(mujoco.mjtObj.mjOBJ_BODY):
      body_rows.append(i)
    elif objtype == int(mujoco.mjtObj.mjOBJ_SITE):
      site_rows.append(i)
    else:
      raise ValueError(
        f"Connect equality '{mj_model.eq(i).name}' anchors on "
        f"{mujoco.mjtObj(objtype).name}; only bodies and sites are supported."
      )

  return ConnectEqTargets(
    body_eq_ids=_index_tensor(env, body_rows),
    body1_ids=_index_tensor(env, [int(mj_model.eq_obj1id[i]) for i in body_rows]),
    body2_ids=_index_tensor(env, [int(mj_model.eq_obj2id[i]) for i in body_rows]),
    site1_ids=_index_tensor(env, [int(mj_model.eq_obj1id[i]) for i in site_rows]),
    site2_ids=_index_tensor(env, [int(mj_model.eq_obj2id[i]) for i in site_rows]),
  )


def connect_equality_displacement(
  env: ManagerBasedRlEnv,
  targets: ConnectEqTargets,
) -> torch.Tensor:
  """World-frame displacement between each connect equality's two anchors.

  A ``<connect>`` pins an anchor point given in body1's frame to one given in
  body2's frame -- or, when it names sites instead, pins two site origins
  together. Whatever gap is left once the solver has run is the constraint
  residual, the same vector MuJoCo reports in the constraint's ``efc_pos``
  rows.

  Returns:
    Signed displacement, shape (num_envs, num_constraints, 3), body-anchored
    constraints first.
  """
  data = env.sim.data
  parts: list[torch.Tensor] = []

  if targets.body_eq_ids.numel():
    eq_data = env.sim.model.eq_data
    anchor1 = eq_data[:, targets.body_eq_ids, 0:3].unsqueeze(-1)
    anchor2 = eq_data[:, targets.body_eq_ids, 3:6].unsqueeze(-1)
    body1, body2 = targets.body1_ids, targets.body2_ids
    pos1 = data.xpos[:, body1] + (data.xmat[:, body1] @ anchor1).squeeze(-1)
    pos2 = data.xpos[:, body2] + (data.xmat[:, body2] @ anchor2).squeeze(-1)
    parts.append(pos1 - pos2)

  if targets.site1_ids.numel():
    site_xpos = data.site_xpos
    parts.append(site_xpos[:, targets.site1_ids] - site_xpos[:, targets.site2_ids])

  return torch.cat(parts, dim=1)


class connect_equality_constraint_violation:
  """Violation of ``<connect>`` equality constraints, per axis or as a total.

  A closed kinematic chain the solver cannot hold exactly pulls apart, and the
  gap it leaves has a direction: ``axis`` reports one world-frame component of
  it (x forward, y left, z up), ``axis=None`` its Euclidean norm -- the total
  displacement, whichever way it points. Per-axis values are magnitudes unless
  ``signed`` is set, so that two legs pulling apart in opposite directions
  cannot cancel each other out under ``reduction``.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    if cfg.params.get("signed") and cfg.params.get("axis") is None:
      raise ValueError(
        "connect_equality_constraint_violation: 'signed' needs an axis -- the "
        "total displacement is a norm, and has no sign to report."
      )
    self.targets = resolve_connect_eq_targets(
      env, cfg.params["asset_cfg"], cfg.params["constraint_names"]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    constraint_names: Sequence[str],
    axis: Axis | None = None,
    reduction: Reduction = "mean",
    signed: bool = False,
  ) -> torch.Tensor:
    del asset_cfg, constraint_names  # Resolved once, in __init__.
    displacement = connect_equality_displacement(env, self.targets)
    if axis is None:
      values = torch.linalg.vector_norm(displacement, dim=-1)
    else:
      values = displacement[..., _AXIS_INDEX[axis]]
      if not signed:
        values = values.abs()
    return _reduce(values, reduction)  # (num_envs,)


def resolve_joint_eq_targets(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  constraint_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Rows and qpos addresses of the ``<joint>`` equalities named by regex.

  Returns:
    ``(eq_ids, qpos_adr1, qpos_adr2, has_second)``. ``has_second`` is 0.0 for a
    one-joint equality, whose ``qpos_adr2`` is a placeholder that the zeroed
    partner displacement never actually reads.
  """
  mj_model = env.sim.mj_model
  eq_ids = _resolve_eq_rows(
    env, asset_cfg, constraint_names, mujoco.mjtEq.mjEQ_JOINT, "joint"
  )

  qpos_adr1: list[int] = []
  qpos_adr2: list[int] = []
  has_second: list[float] = []
  for i in eq_ids:
    joint_ids = (int(mj_model.eq_obj1id[i]), int(mj_model.eq_obj2id[i]))
    for joint_id in joint_ids:
      # A joint equality reads one qpos per joint, so a ball or free joint --
      # whose qpos address opens a quaternion -- has no scalar to couple.
      if joint_id >= 0 and int(mj_model.jnt_type[joint_id]) not in (
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
      ):
        raise ValueError(
          f"Joint equality '{mj_model.eq(i).name}' couples "
          f"'{mj_model.joint(joint_id).name}', which is not a hinge or slide."
        )
    qpos_adr1.append(int(mj_model.jnt_qposadr[joint_ids[0]]))
    qpos_adr2.append(
      int(mj_model.jnt_qposadr[joint_ids[1]]) if joint_ids[1] >= 0 else 0
    )
    has_second.append(float(joint_ids[1] >= 0))

  return (
    _index_tensor(env, eq_ids),
    _index_tensor(env, qpos_adr1),
    _index_tensor(env, qpos_adr2),
    torch.as_tensor(has_second, device=env.device, dtype=torch.float),
  )


def joint_equality_error(
  env: ManagerBasedRlEnv,
  eq_ids: torch.Tensor,
  qpos_adr1: torch.Tensor,
  qpos_adr2: torch.Tensor,
  has_second: torch.Tensor,
) -> torch.Tensor:
  """Residual of each ``<joint>`` equality, in the first joint's own units.

  MuJoCo couples the pair by a quartic in the second joint's displacement from
  its reference pose::

    err = (q1 - q1_0) - sum_k polycoef[k] * (q2 - q2_0)**k

  With no second joint the polynomial degenerates to its constant term, pinning
  the first joint to a fixed offset from its reference; that falls out of the
  same expression once the partner displacement is zeroed.

  Returns:
    Signed error, shape (num_envs, num_constraints).
  """
  qpos = env.sim.data.qpos
  qpos0 = env.sim.model.qpos0
  pos1 = qpos[:, qpos_adr1] - qpos0[:, qpos_adr1]
  pos2 = (qpos[:, qpos_adr2] - qpos0[:, qpos_adr2]) * has_second
  coefficients = env.sim.model.eq_data[:, eq_ids, 0:5]
  powers = torch.stack([pos2**k for k in range(5)], dim=-1)
  return pos1 - (coefficients * powers).sum(dim=-1)


class joint_equality_constraint_violation:
  """Violation of ``<joint>`` equality constraints.

  A joint equality gears one joint to another: how far the geared joint has
  drifted from the angle (or displacement, for a slide) its partner prescribes
  is a scalar in that joint's own units, so unlike a connect there is no axis
  to pick -- only whether to keep the sign. Values are magnitudes unless
  ``signed`` is set, so that two couplings drifting opposite ways cannot cancel
  each other out under ``reduction``.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    (
      self.eq_ids,
      self.qpos_adr1,
      self.qpos_adr2,
      self.has_second,
    ) = resolve_joint_eq_targets(
      env, cfg.params["asset_cfg"], cfg.params["constraint_names"]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    constraint_names: Sequence[str],
    reduction: Reduction = "mean",
    signed: bool = False,
  ) -> torch.Tensor:
    del asset_cfg, constraint_names  # Resolved once, in __init__.
    error = joint_equality_error(
      env, self.eq_ids, self.qpos_adr1, self.qpos_adr2, self.has_second
    )
    return _reduce(error if signed else error.abs(), reduction)  # (num_envs,)


def load_knee_distance_map(csv_path: str | Path) -> torch.Tensor:
  """Load ``knee_distance_map.csv`` as a ``(N, 3)`` table, sorted by knee angle.

  Columns are ``(knee_rad, displacement_x_m, displacement_z_m)``; the CSV's
  ``knee_deg`` and ``distance_m`` columns are redundant and dropped.
  """
  data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
  table = torch.from_numpy(data[:, [0, 2, 3]])
  return table[torch.argsort(table[:, 0])]


def _interpolate_rows(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
  """Linearly interpolate every value column of ``table`` at ``x``.

  ``table`` is ``(N, 1 + C)``: an ascending key column followed by C value
  columns. Queries outside the key range are clamped to the end rows rather
  than extrapolated, so a knee angle past the mapped sweep reports the
  deviation from the nearest mapped point instead of a fabricated one.

  Returns:
    Interpolated values, shape ``x.shape + (C,)``.
  """
  keys = table[:, 0].contiguous()
  values = table[:, 1:]

  x_clamped = torch.clamp(x, keys[0], keys[-1])
  idx = torch.clamp(
    torch.searchsorted(keys, x_clamped.contiguous()), 1, keys.numel() - 1
  )

  lo, hi = keys[idx - 1], keys[idx]
  t = ((x_clamped - lo) / (hi - lo)).unsqueeze(-1)
  return values[idx - 1] + t * (values[idx] - values[idx - 1])


class knee_distance_map_error:
  """Deviation of the knee mechanism from its measured displacement map.

  ``knee_distance_map.csv`` records where ``leg_.*_length_connect_b`` sits
  relative to the ``leg_.*_femur_joint`` anchor as the knee folds, swept over
  the model itself: one row per knee angle, holding the displacement's x and z
  components. This metric measures the same displacement live and reports how
  far it has drifted from the row the current knee angle interpolates to --
  i.e. how far the simulated mechanism has departed from its own kinematics,
  whether because a constraint is being violated or because a variant's spec
  edits changed the chain.

  Both components are world-frame, as the map is: the displacement rotates
  with the base, so the numbers are only comparable to the map while the base
  holds the pose the sweep was taken in.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    if cfg.params.get("signed") and cfg.params.get("axis") is None:
      raise ValueError(
        "knee_distance_map_error: 'signed' needs an axis -- the total "
        "deviation is a norm, and has no sign to report."
      )
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    self.table = load_knee_distance_map(cfg.params["csv_path"]).to(env.device)

    femur_ids: list[int] = []
    knee_q_adr: list[int] = []
    site_ids: list[int] = []
    for femur_joint, knee_joint, site in cfg.params["legs"]:
      femur_ids.append(
        int(asset.indexing.joint_ids[asset.joint_names.index(femur_joint)])
      )
      knee_q_adr.append(
        int(asset.indexing.joint_q_adr[asset.joint_names.index(knee_joint)])
      )
      site_ids.append(int(asset.indexing.site_ids[asset.site_names.index(site)]))

    self.femur_ids = _index_tensor(env, femur_ids)
    self.knee_q_adr = _index_tensor(env, knee_q_adr)
    self.site_ids = _index_tensor(env, site_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    csv_path: str | Path,
    legs: Sequence[tuple[str, str, str]],
    axis: Literal["x", "z"] | None = None,
    reduction: Reduction = "mean",
    signed: bool = False,
  ) -> torch.Tensor:
    del asset_cfg, csv_path, legs  # Resolved once, in __init__.
    data = env.sim.data

    # (num_envs, num_legs, 3), femur joint anchor to connect site.
    displacement = data.site_xpos[:, self.site_ids] - data.xanchor[:, self.femur_ids]
    measured = displacement[..., [0, 2]]
    expected = _interpolate_rows(data.qpos[:, self.knee_q_adr], self.table)
    error = measured - expected  # (num_envs, num_legs, 2)

    if axis is None:
      values = torch.linalg.vector_norm(error, dim=-1)
    else:
      values = error[..., 0 if axis == "x" else 1]
      if not signed:
        values = values.abs()
    return _reduce(values, reduction)  # (num_envs,)
