"""Torch port of ``lut_transmission/transmission_maps.py``: the leg
transmissions of the KANGAROO screw mechanisms as lookup tables, evaluated
batched on the GPU. The maps ship in ``pal_kangaroo_full/lut_transmission/``
(:class:`TransmissionMaps` loads the set) and are driven by the
``"lut"`` transmission of ``pal_kangaroo_full.lut_actuator`` on both MJCFs of
the full robot.

The maps store the actuation Jacobian ``J = d(actuators)/d(joints)`` (and,
for the 1-D mechanisms, the actuator position and a few companion tables)
on a non-uniform tensor grid, and the reference reads them with
tensor-product natural cubic splines (``method="cubic"``) or multilinear
interpolation (``method="linear"``), then solves ``tau = J^T F`` for the
actuator forces. Everything here is that computation, item for item, with
one difference in *how* the spline is evaluated: the reference recomputes
the natural-spline second derivatives of the reduced array along each axis
per query, while :class:`SplineTensorMap` precomputes the ``2^K`` mixed
second-derivative tables once and evaluates a local ``4^K``-term stencil --
the two are the same linear map (see the class docstring), so they agree to
rounding. The map classes keep the reference's names and conventions:

    hip = HipXyMap.load(path)                 # hip_xy_jacobian_map.npz
    F = hip.forces(q, tau)                    # q, tau (..., 2) -> F (..., 2)
    ankle = AnkleMap.load(path)               # ankle_xy_jacobian_map.npz
    F = ankle.forces(q, s, tau)               # s: leg_*_length_actuator position (m)
    leg = LegLengthMap.load(path)             # leg_length_map.npz
    s = leg.slider_of_knee(knee); d = leg.distance(s); F_s = leg.actuator_force(s, F_d)
    yaw = HipZMap.load(path)                  # hip_z_map.npz
    F = yaw.force(q, tau)                     # tau / J(q)

Queries outside a map are clamped to its edge, as in the reference. Every
map lives on one device (``to``) and is shared by both legs -- the files
are built from the right leg, and the callers handle the mirroring.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

InterpolationMethod = Literal["cubic", "linear"]


def spline_second_derivatives(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
  """Second derivatives of the natural cubic spline through ``y`` along
  ``axis`` (knots ``x``), for every line of ``y`` at once (Thomas algorithm).

  Verbatim from the reference, so the spline is the same one.
  """
  y = np.moveaxis(y, axis, 0)
  n = len(x)
  M = np.zeros_like(y)
  if n < 3:
    return np.moveaxis(M, 0, axis)
  h = np.diff(x)
  shape = (n - 2,) + (1,) * (y.ndim - 1)
  a = h[:-1].reshape(shape)  # sub-diagonal
  b = (2 * (h[:-1] + h[1:])).reshape(shape)  # diagonal
  c = h[1:].reshape(shape)  # super-diagonal
  d = 6 * (
    (y[2:] - y[1:-1]) / h[1:].reshape(shape)
    - (y[1:-1] - y[:-2]) / h[:-1].reshape(shape)
  )
  # forward sweep
  cp = np.zeros_like(d)
  dp = np.zeros_like(d)
  cp[0] = c[0] / b[0]
  dp[0] = d[0] / b[0]
  for i in range(1, n - 2):
    denom = b[i] - a[i] * cp[i - 1]
    cp[i] = c[i] / denom
    dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
  # back substitution
  M[n - 2] = dp[-1]
  for i in range(n - 4, -1, -1):
    M[i + 1] = dp[i] - cp[i] * M[i + 2]
  return np.moveaxis(M, 0, axis)


class SplineTensorMap:
  """One array on a non-uniform tensor grid, interpolated batched in torch.

  ``values`` has shape ``(len(axes[0]), ..., len(axes[K-1]), *item)``.
  Evaluating the reference's tensor-product natural cubic spline at
  ``(t_0, ..., t_{K-1})`` applies, axis by axis, the linear operator
  ``A_k(t_k) = P_k(t_k) + Q_k(t_k) D_k``: ``P_k`` weights the two bracketing
  knots by ``(p, q)``, ``Q_k`` weights them by ``((p^3 - p), (q^3 - q)) h^2/6``
  and ``D_k`` is the natural-spline second-derivative operator along axis
  ``k`` -- linear, and acting on a different axis than every other ``D_j``,
  so they all commute. Expanding the product over the axes,

      value = sum over subsets S of the axes of
              [prod_{k in S} Q_k][prod_{k not in S} P_k] (D_S values),  D_S = prod_{k in S} D_k,

  so the ``2^K`` tables ``D_S values`` are computed once at construction (in
  float64, with the reference's own :func:`spline_second_derivatives`) and a
  query only touches the ``2^K`` grid corners around it: ``4^K`` gathers
  with products of per-axis weights, no per-query loop and no host sync.
  ``method="linear"`` keeps the ``S = {}`` term with the ``P`` weights only,
  which is the reference's multilinear branch. Queries are clamped to the
  grid.
  """

  def __init__(
    self,
    axes: Sequence[np.ndarray],
    values: np.ndarray,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    if method not in ("cubic", "linear"):
      raise ValueError(f"method must be 'cubic' or 'linear', got {method!r}")
    axes = [np.asarray(a, dtype=np.float64) for a in axes]
    for a in axes:
      if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0):
        raise ValueError("every axis must be 1-D, strictly increasing, >= 2 nodes")
    values = np.asarray(values, dtype=np.float64)
    n_axes = len(axes)
    grid_shape = tuple(len(a) for a in axes)
    if values.shape[:n_axes] != grid_shape:
      raise ValueError(f"values {values.shape} do not sit on the grid {grid_shape}")
    if not np.all(np.isfinite(values)):
      raise ValueError("values have non-finite entries")
    self.method = method
    self.n_axes = n_axes
    self.grid_shape = grid_shape
    self.item_shape = tuple(values.shape[n_axes:])
    n_nodes = int(np.prod(grid_shape))
    item_size = int(np.prod(self.item_shape)) if self.item_shape else 1

    tables = []
    n_subsets = 2**n_axes if method == "cubic" else 1
    for subset in range(n_subsets):
      table = values
      for k in range(n_axes):
        if subset >> k & 1:
          table = spline_second_derivatives(axes[k], table, k)
      tables.append(table.reshape(n_nodes, item_size))
    # (n_subsets, n_nodes, item_size); subset bit k set <=> D_k applied.
    self.tables = torch.as_tensor(np.stack(tables), dtype=dtype)
    self.axes = [torch.as_tensor(a, dtype=dtype) for a in axes]
    self.strides = [int(np.prod(grid_shape[k + 1 :])) for k in range(n_axes)]
    # The stencil: every combination, per axis, of (corner bit, derivative
    # bit) -- 4 choices per axis for cubic, the 2 corners for linear.
    n_choices = 4 if method == "cubic" else 2
    self._stencil = list(itertools.product(range(n_choices), repeat=n_axes))

  @property
  def bounds(self) -> list[tuple[float, float]]:
    return [(float(a[0]), float(a[-1])) for a in self.axes]

  @property
  def device(self) -> torch.device:
    return self.tables.device

  @property
  def dtype(self) -> torch.dtype:
    return self.tables.dtype

  def to(self, device: str | torch.device) -> SplineTensorMap:
    clone = object.__new__(SplineTensorMap)
    clone.__dict__.update(self.__dict__)
    clone.tables = self.tables.to(device)
    clone.axes = [a.to(device) for a in self.axes]
    return clone

  def inside(self, *coords: torch.Tensor) -> torch.Tensor:
    """True where every coordinate lies within the grid."""
    ok = torch.ones(
      torch.broadcast_shapes(*[c.shape for c in coords]),
      dtype=torch.bool,
      device=self.device,
    )
    for a, c in zip(self.axes, coords, strict=True):
      ok &= (c >= a[0]) & (c <= a[-1])
    return ok

  def __call__(self, *coords: torch.Tensor) -> torch.Tensor:
    """The array interpolated at the coordinates (one tensor per axis,
    broadcastable); the result has the coordinates' shape + item shape."""
    if len(coords) != self.n_axes:
      raise ValueError(f"{self.n_axes} coordinates expected, got {len(coords)}")
    coords = torch.broadcast_tensors(*[c.to(self.dtype) for c in coords])
    shape = coords[0].shape
    lower: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for a, c in zip(self.axes, coords, strict=True):
      t = torch.clamp(c, a[0], a[-1])
      i = torch.clamp(
        torch.searchsorted(a, t.contiguous(), right=True) - 1, 0, a.numel() - 2
      )
      x0, x1 = a[i], a[i + 1]
      h = x1 - x0
      p, q = (x1 - t) / h, (t - x0) / h
      if self.method == "cubic":
        h2 = h * h / 6.0
        w = torch.stack((p, q, (p**3 - p) * h2, (q**3 - q) * h2), dim=-1)
      else:
        w = torch.stack((p, q), dim=-1)
      lower.append(i)
      weights.append(w)

    out = torch.zeros(
      shape + (self.tables.shape[-1],), dtype=self.dtype, device=self.device
    )
    for choice in self._stencil:
      subset = 0
      flat = torch.zeros(shape, dtype=torch.long, device=self.device)
      weight = torch.ones(shape, dtype=self.dtype, device=self.device)
      for k, c in enumerate(choice):
        subset |= (c >> 1) << k
        flat = flat + (lower[k] + (c & 1)) * self.strides[k]
        weight = weight * weights[k][..., c]
      out = out + weight.unsqueeze(-1) * self.tables[subset][flat]
    return out.reshape(shape + self.item_shape)


def solve_forces(J: torch.Tensor, tau: torch.Tensor, rcond: float) -> torch.Tensor:
  """``F = J^-T tau`` for 2x2 Jacobians, batched over leading dimensions,
  with a pseudo-inverse where ``J`` is (near) singular.

  The reference inverts ``J^T`` exactly where ``|det| > rcond * max|J|^2``
  and falls back to ``numpy.linalg.pinv(J^T, rcond)`` elsewhere. The
  fallback here is the closed-form rank-1 pseudo-inverse ``J / ||J||_F^2``:
  in that branch the small singular value is at most ``~rcond`` times the
  large one, so it equals the truncated-SVD pseudo-inverse to relative
  ``O(rcond)`` -- and it keeps the whole solve free of host syncs.
  """
  tau = tau.to(J.dtype)
  JT = J.transpose(-1, -2)
  det = JT[..., 0, 0] * JT[..., 1, 1] - JT[..., 0, 1] * JT[..., 1, 0]
  scale = JT.abs().amax(dim=(-1, -2))
  regular = det.abs() > rcond * scale * scale
  # exact 2x2 inverse where regular: adj(J^T) tau / det
  adj = torch.stack(
    (
      torch.stack((JT[..., 1, 1], -JT[..., 0, 1]), dim=-1),
      torch.stack((-JT[..., 1, 0], JT[..., 0, 0]), dim=-1),
    ),
    dim=-2,
  )
  safe_det = torch.where(regular, det, torch.ones_like(det))
  F_exact = (adj @ tau.unsqueeze(-1)).squeeze(-1) / safe_det.unsqueeze(-1)
  # rank-1 pseudo-inverse of J^T elsewhere: J tau / ||J||_F^2 (0 when J = 0)
  norm2 = (JT * JT).sum(dim=(-1, -2))
  safe_norm2 = torch.where(norm2 > 0, norm2, torch.ones_like(norm2))
  F_pinv = (J @ tau.unsqueeze(-1)).squeeze(-1) / safe_norm2.unsqueeze(-1)
  F_pinv = torch.where((norm2 > 0).unsqueeze(-1), F_pinv, torch.zeros_like(F_pinv))
  return torch.where(regular.unsqueeze(-1), F_exact, F_pinv)


def _scalar(d, key: str, default=None):
  if key not in d.files:
    return default
  return d[key].item() if d[key].ndim == 0 else d[key]


class _LutMap:
  """Shared: the file, the interpolation method, ``to`` over every table."""

  _table_attrs: tuple[str, ...] = ()

  def __init__(self, path: str | Path | None, method: InterpolationMethod) -> None:
    self.path = Path(path) if path is not None else None
    self.method = method

  def to(self, device: str | torch.device):
    clone = object.__new__(type(self))
    clone.__dict__.update(self.__dict__)
    for name in self._table_attrs:
      table = getattr(self, name)
      setattr(clone, name, table.to(device) if table is not None else None)
    return clone

  @property
  def device(self) -> torch.device:
    return getattr(self, self._table_attrs[0]).device

  @property
  def dtype(self) -> torch.dtype:
    return getattr(self, self._table_attrs[0]).dtype


class HipZMap(_LutMap):
  """Hip yaw: ``leg_right_1_joint`` <-> ``leg_right_1_actuator``, from
  ``hip_z_map.npz`` (spline in the joint angle q, rad):

      jacobian(q)            J = d(actuator)/d(joint), m/rad
      actuator_position(q)   m
      force(q, tau)          actuator force for a joint torque: F = tau / J
      torque(q, F)           joint torque from an actuator force: tau = J F
  """

  _table_attrs = ("_J", "_actuator")

  def __init__(
    self,
    joint: np.ndarray,
    J: np.ndarray,
    actuator: np.ndarray,
    *,
    joint_name: str = "leg_right_1_joint",
    actuator_name: str = "leg_right_1_actuator",
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
    path: str | Path | None = None,
  ) -> None:
    super().__init__(path, method)
    self.joint_name, self.actuator_name = joint_name, actuator_name
    self._J = SplineTensorMap([joint], J, method, dtype)
    self._actuator = SplineTensorMap([joint], actuator, method, dtype)

  @classmethod
  def load(
    cls,
    path: str | Path,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
  ) -> HipZMap:
    with np.load(path) as d:
      return cls(
        d["joint"],
        d["J"],
        d["actuator"],
        joint_name=str(d["joint_name"]),
        actuator_name=str(d["actuator_name"]),
        method=method,
        dtype=dtype,
        path=path,
      )

  @property
  def joint_names(self) -> tuple[str, ...]:
    return (self.joint_name,)

  @property
  def actuator_names(self) -> tuple[str, ...]:
    return (self.actuator_name,)

  @property
  def bounds(self) -> list[tuple[float, float]]:
    return self._J.bounds

  def jacobian(self, q: torch.Tensor) -> torch.Tensor:
    return self._J(q)

  def actuator_position(self, q: torch.Tensor) -> torch.Tensor:
    return self._actuator(q)

  def force(self, q: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return tau.to(self.dtype) / self.jacobian(q)

  def torque(self, q: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return self.jacobian(q) * F.to(self.dtype)

  def inside(self, q: torch.Tensor) -> torch.Tensor:
    return self._J.inside(q)


class LegLengthMap(_LutMap):
  """Leg-length slider <-> knee angle and femur-ankle distance, from
  ``leg_length_map.npz``. Everything is a spline in the slider position
  ``s`` (``leg_right_length_actuator``, m):

      knee(s)                knee joint angle, rad
      distance(s)            femur joint -> ankle (joint 4) distance, m -- the
                             simple model's leg length
      jacobian(s)            J = d(slider)/d(distance), m/m (negative)
      dknee_dslider(s)       rad/m
      distance_rate(s, knee_rate)   d(distance)/dt from the knee rate
      force_on_distance(s, F_s)     F_d = J F_s
      actuator_force(s, F_d)        F_s = F_d / J
      knee_torque(s, F_s)           tau = F_s / (dknee/dslider)
      slider_of_knee(k), slider_of_distance(d)   inverse lookups

  The reference inverts ``knee(s)`` and ``distance(s)`` by bisection on the
  spline; here each spline is sampled once on a fine uniform ``s`` grid
  (``inverse_samples`` points, checked monotone) and inverted by linear
  interpolation of that table, then the splines are evaluated at the
  recovered ``s``. The inversion error is ``O(ds^2 f'')`` -- below float32
  at the default sampling.
  """

  _table_attrs = (
    "_knee",
    "_distance",
    "_J",
    "_dknee_dslider",
    "_inverse_knee",
    "_inverse_distance",
  )

  def __init__(
    self,
    slider: np.ndarray,
    knee: np.ndarray,
    distance: np.ndarray,
    J: np.ndarray,
    dknee_dslider: np.ndarray,
    *,
    slider_name: str = "leg_right_length_actuator",
    knee_name: str = "leg_right_knee_joint",
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
    inverse_samples: int = 4096,
    path: str | Path | None = None,
  ) -> None:
    super().__init__(path, method)
    self.slider_name, self.knee_name = slider_name, knee_name
    self._knee = SplineTensorMap([slider], knee, method, dtype)
    self._distance = SplineTensorMap([slider], distance, method, dtype)
    self._J = SplineTensorMap([slider], J, method, dtype)
    self._dknee_dslider = SplineTensorMap([slider], dknee_dslider, method, dtype)
    # Inverse tables (value -> slider), sampled from the same splines.
    self._inverse_knee = self._inverse_table(
      slider, knee, "knee", method, dtype, inverse_samples
    )
    self._inverse_distance = self._inverse_table(
      slider, distance, "distance", method, dtype, inverse_samples
    )

  @staticmethod
  def _inverse_table(slider, values, what, method, dtype, n) -> torch.Tensor:
    """``(2, n)`` rows ``(value, slider)`` with increasing value, from the
    spline of ``values`` sampled on a uniform slider grid in float64."""
    fine = SplineTensorMap([slider], values, method, torch.float64)
    s_fine = torch.linspace(float(slider[0]), float(slider[-1]), n, dtype=torch.float64)
    v_fine = fine(s_fine)
    steps = torch.diff(v_fine)
    if not (torch.all(steps > 0) or torch.all(steps < 0)):
      raise ValueError(f"{what}(slider) is not monotone; cannot invert it")
    if steps[0] < 0:  # searchsorted needs an increasing key
      s_fine, v_fine = s_fine.flip(0), v_fine.flip(0)
    return torch.stack((v_fine, s_fine)).to(dtype).contiguous()

  @classmethod
  def load(
    cls,
    path: str | Path,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
    inverse_samples: int = 4096,
  ) -> LegLengthMap:
    with np.load(path) as d:
      return cls(
        d["slider"],
        d["knee"],
        d["distance"],
        d["J"],
        d["dknee_dslider"],
        slider_name=str(_scalar(d, "slider_name", "leg_right_length_actuator")),
        knee_name=str(_scalar(d, "knee_name", "leg_right_knee_joint")),
        method=method,
        dtype=dtype,
        inverse_samples=inverse_samples,
        path=path,
      )

  @property
  def bounds(self) -> list[tuple[float, float]]:
    return self._J.bounds

  @property
  def knee_bounds(self) -> tuple[float, float]:
    return float(self._inverse_knee[0, 0]), float(self._inverse_knee[0, -1])

  @property
  def distance_bounds(self) -> tuple[float, float]:
    return float(self._inverse_distance[0, 0]), float(self._inverse_distance[0, -1])

  def knee(self, s: torch.Tensor) -> torch.Tensor:
    return self._knee(s)

  def distance(self, s: torch.Tensor) -> torch.Tensor:
    return self._distance(s)

  def jacobian(self, s: torch.Tensor) -> torch.Tensor:
    return self._J(s)

  def dknee_dslider(self, s: torch.Tensor) -> torch.Tensor:
    return self._dknee_dslider(s)

  def distance_rate(self, s: torch.Tensor, knee_rate: torch.Tensor) -> torch.Tensor:
    """d(distance)/dt = knee_rate * (dd/ds) / (dk/ds) = knee_rate / (J dk/ds)."""
    return knee_rate.to(self.dtype) / (self.jacobian(s) * self.dknee_dslider(s))

  def force_on_distance(self, s: torch.Tensor, F_s: torch.Tensor) -> torch.Tensor:
    return self.jacobian(s) * F_s.to(self.dtype)

  def actuator_force(self, s: torch.Tensor, F_d: torch.Tensor) -> torch.Tensor:
    return F_d.to(self.dtype) / self.jacobian(s)

  def knee_torque(self, s: torch.Tensor, F_s: torch.Tensor) -> torch.Tensor:
    return F_s.to(self.dtype) / self.dknee_dslider(s)

  def inside(self, s: torch.Tensor) -> torch.Tensor:
    return self._J.inside(s)

  @staticmethod
  def _invert(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    keys, values = table[0], table[1]
    x = torch.clamp(x.to(keys.dtype), keys[0], keys[-1])
    i = torch.clamp(
      torch.searchsorted(keys, x.contiguous(), right=True) - 1, 0, keys.numel() - 2
    )
    t = (x - keys[i]) / (keys[i + 1] - keys[i])
    return values[i] + t * (values[i + 1] - values[i])

  def slider_of_knee(self, knee: torch.Tensor) -> torch.Tensor:
    """Slider position at a knee angle (clamped to the map's knee range)."""
    return self._invert(self._inverse_knee, knee)

  def slider_of_distance(self, distance: torch.Tensor) -> torch.Tensor:
    """Slider position at a femur-ankle distance (clamped to the map's range)."""
    return self._invert(self._inverse_distance, distance)


class _JacobianMap(_LutMap):
  """Shared by the 2x2 mechanisms: the J grid and the force solve."""

  _table_attrs = ("_J", "_actuators")

  def __init__(
    self,
    axes: Sequence[np.ndarray],
    J: np.ndarray,
    actuators: np.ndarray | None,
    *,
    joint_names: Sequence[str],
    actuator_names: Sequence[str],
    rcond: float,
    method: InterpolationMethod,
    dtype: torch.dtype,
    path: str | Path | None,
  ) -> None:
    super().__init__(path, method)
    self.joint_names = tuple(str(n) for n in joint_names)
    self.actuator_names = tuple(str(n) for n in actuator_names)
    self.rcond = rcond
    if J.shape[-2:] != (2, 2):
      raise ValueError(f"J must be a grid of 2x2 matrices, got {J.shape}")
    self._J = SplineTensorMap(axes, J, method, dtype)
    # The actuator-position table is diagnostic; the ankle file has holes
    # (unsolved nodes), and the spline cannot carry NaN, so it is optional.
    self._actuators = (
      SplineTensorMap(axes, actuators, method, dtype)
      if actuators is not None and np.all(np.isfinite(actuators))
      else None
    )

  @property
  def bounds(self) -> list[tuple[float, float]]:
    return self._J.bounds


class HipXyMap(_JacobianMap):
  """Hip pitch/roll (``leg_right_2_joint``, ``leg_right_3_joint``) <->
  ``leg_right_2_actuator``, ``leg_right_3_actuator``, from
  ``hip_xy_jacobian_map.npz`` (axes: joint 2, joint 3)."""

  @classmethod
  def load(
    cls,
    path: str | Path,
    rcond: float = 1e-6,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
  ) -> HipXyMap:
    with np.load(path) as d:
      return cls(
        [d["axis0"], d["axis1"]],
        d["J"],
        d["actuators"] if "actuators" in d.files else None,
        joint_names=d["joint_names" if "joint_names" in d.files else "joints"],
        actuator_names=d["actuator_names"],
        rcond=rcond,
        method=method,
        dtype=dtype,
        path=path,
      )

  def jacobian(self, q: torch.Tensor) -> torch.Tensor:
    """``J = d(act 2, act 3)/d(joint 2, joint 3)`` at ``q`` (..., 2), m/rad."""
    return self._J(q[..., 0], q[..., 1])

  def forces(self, q: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Actuator forces (N) producing the joint torques ``tau`` (N m) at ``q``."""
    return solve_forces(self.jacobian(q), tau, self.rcond)

  def torques(self, q: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Joint torques from actuator forces: ``tau = J^T F``."""
    return (self.jacobian(q).transpose(-1, -2) @ F.unsqueeze(-1)).squeeze(-1)

  def actuator_positions(self, q: torch.Tensor) -> torch.Tensor:
    assert self._actuators is not None, "this map has no actuator-position table"
    return self._actuators(q[..., 0], q[..., 1])

  def inside(self, q: torch.Tensor) -> torch.Tensor:
    return self._J.inside(q[..., 0], q[..., 1])


class AnkleMap(_JacobianMap):
  """Ankle pitch/roll (``leg_right_4_joint``, ``leg_right_5_joint``) <->
  ``leg_right_4_actuator``, ``leg_right_5_actuator`` at a leg length, from
  ``ankle_xy_jacobian_map.npz`` (axes: joint 4, joint 5, length). The length
  axis is the slider position ``leg_right_length_actuator`` (m); the file
  says so in ``length_coordinate`` and anything else is refused, since the
  callers derive it from the knee through :class:`LegLengthMap`."""

  @classmethod
  def load(
    cls,
    path: str | Path,
    rcond: float = 1e-6,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
  ) -> AnkleMap:
    with np.load(path) as d:
      length_coordinate = str(_scalar(d, "length_coordinate", "slider"))
      if length_coordinate != "slider":
        raise ValueError(
          f"{path}: length axis is {length_coordinate!r}; only 'slider' "
          "(leg_*_length_actuator position) is supported"
        )
      self = cls(
        [d["joint4"], d["joint5"], d["length"]],
        d["J"],
        d["actuators"] if "actuators" in d.files else None,
        joint_names=d["joint_names" if "joint_names" in d.files else "joints"],
        actuator_names=d["actuator_names"],
        rcond=rcond,
        method=method,
        dtype=dtype,
        path=path,
      )
    self.length_coordinate = length_coordinate
    return self

  def jacobian(self, q: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """``J = d(act 4, act 5)/d(joint 4, joint 5)`` at ankle joints ``q``
    (..., 2) and slider position ``length`` (...), m/rad."""
    return self._J(q[..., 0], q[..., 1], length)

  def forces(
    self, q: torch.Tensor, length: torch.Tensor, tau: torch.Tensor
  ) -> torch.Tensor:
    return solve_forces(self.jacobian(q, length), tau, self.rcond)

  def torques(
    self, q: torch.Tensor, length: torch.Tensor, F: torch.Tensor
  ) -> torch.Tensor:
    return (self.jacobian(q, length).transpose(-1, -2) @ F.unsqueeze(-1)).squeeze(-1)

  def actuator_positions(self, q: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    assert self._actuators is not None, "this map has no actuator-position table"
    return self._actuators(q[..., 0], q[..., 1], length)

  def inside(self, q: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    return self._J.inside(q[..., 0], q[..., 1], length)


@dataclass(frozen=True)
class TransmissionMaps:
  """The four maps of one leg's screw mechanisms, loaded from one directory:
  ``hip_z_map.npz``, ``hip_xy_jacobian_map.npz``, ``ankle_xy_jacobian_map.npz``
  and ``leg_length_map.npz`` (see :data:`TRANSMISSION_MAP_NAMES`)."""

  hip_z: HipZMap
  hip_xy: HipXyMap
  ankle: AnkleMap
  leg_length: LegLengthMap

  @classmethod
  def load(
    cls,
    directory: str | Path,
    method: InterpolationMethod = "cubic",
    dtype: torch.dtype = torch.float32,
  ) -> TransmissionMaps:
    directory = Path(directory)
    missing = [
      name for name in TRANSMISSION_MAP_NAMES if not (directory / name).exists()
    ]
    if missing:
      raise FileNotFoundError(f"{directory} lacks the transmission maps {missing}")
    return cls(
      hip_z=HipZMap.load(directory / "hip_z_map.npz", method=method, dtype=dtype),
      hip_xy=HipXyMap.load(
        directory / "hip_xy_jacobian_map.npz", method=method, dtype=dtype
      ),
      ankle=AnkleMap.load(
        directory / "ankle_xy_jacobian_map.npz", method=method, dtype=dtype
      ),
      leg_length=LegLengthMap.load(
        directory / "leg_length_map.npz", method=method, dtype=dtype
      ),
    )

  @staticmethod
  def available(directory: str | Path) -> bool:
    """Whether every map is in ``directory``."""
    return all((Path(directory) / name).exists() for name in TRANSMISSION_MAP_NAMES)

  def to(self, device: str | torch.device) -> TransmissionMaps:
    return TransmissionMaps(
      hip_z=self.hip_z.to(device),
      hip_xy=self.hip_xy.to(device),
      ankle=self.ankle.to(device),
      leg_length=self.leg_length.to(device),
    )

  @property
  def joint_names(self) -> tuple[str, ...]:
    """The reference side's joints the maps are keyed by, in the order the
    actuator lays them out: hip yaw, hip pitch, hip roll, ankle pitch, ankle
    roll -- then the leg length, whose servo joint the actuator picks."""
    return self.hip_z.joint_names + self.hip_xy.joint_names + self.ankle.joint_names

  @property
  def actuator_names(self) -> tuple[str, ...]:
    """The reference side's actuators (screws), in the same order, with the
    leg-length screw last."""
    return (
      self.hip_z.actuator_names
      + self.hip_xy.actuator_names
      + self.ankle.actuator_names
      + (self.leg_length.slider_name,)
    )


TRANSMISSION_MAP_NAMES = (
  "hip_z_map.npz",
  "hip_xy_jacobian_map.npz",
  "ankle_xy_jacobian_map.npz",
  "leg_length_map.npz",
)
"""The files :meth:`TransmissionMaps.load` reads, relative to its directory."""


def reside(name: str, side: str, reference_side: str = "right") -> str:
  """``leg_right_2_joint`` -> ``leg_left_2_joint``, ``right_hip_z_slider`` ->
  ``left_hip_z_slider`` for ``side="left"``: the first ``_``-separated
  component equal to ``reference_side`` is replaced."""
  parts = name.split("_")
  if reference_side not in parts:
    raise ValueError(f"{name!r} does not name the {reference_side} side")
  parts[parts.index(reference_side)] = side
  return "_".join(parts)


__all__ = [
  "TRANSMISSION_MAP_NAMES",
  "AnkleMap",
  "HipXyMap",
  "HipZMap",
  "InterpolationMethod",
  "LegLengthMap",
  "SplineTensorMap",
  "TransmissionMaps",
  "reside",
  "solve_forces",
  "spline_second_derivatives",
]
