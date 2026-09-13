"""Ranges of the hip_xy ground-truth tables ``joint_to_actuator_1/2.npy``.

The tables map (joint 3, joint 2) -> (l, r) hip_xy actuator position on a
201 x 321 grid; array 1 is the l slider, array 2 the r slider, in metres of
slider travel (tendon length = slider + HIP_XY_TENDON_OFFSET in the tendon
model). The files carry no axes; the grid convention used here is rows =
joint 3 from -0.471239 to +0.471239, columns = joint 2 from +0.663225 to
-0.741765 (see :mod:`pal_mjlab.scripts.plot_hip_xy_rom`).

Printed, per table and for the pair:

1. actuator (tendon extension) min / max -- raw, and excluding the entries
   saturated at +-0.04 m, with the joint angles where they occur;
2. joint 2 / joint 3 min / max of the table domain, and of the region where
   neither actuator is saturated (the part of the domain the tables actually
   resolve);
3. the actuator positions at the joint limits: origin, the four axis
   extremes and the four corners.

Run::

    python -m pal_mjlab.scripts.hip_xy_lut_ranges            # tables in cwd
    python -m pal_mjlab.scripts.hip_xy_lut_ranges --lut a.npy b.npy --offset 0.09344327156
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pal_mjlab.scripts.hip_xy_rom_sweep import HIP_XY_TENDON_OFFSET
from pal_mjlab.scripts.plot_hip_xy_rom import LUT_FILES, LUT_J2, LUT_J3, LUT_SATURATION


def load(paths: tuple[Path, Path]) -> dict:
  gl, gr = (np.load(p).astype(float) for p in paths)
  if gl.shape != gr.shape or gl.ndim != 2:
    raise ValueError(f"unexpected shapes {gl.shape} / {gr.shape}")
  j3 = np.linspace(*LUT_J3, gl.shape[0])
  j2 = np.linspace(*LUT_J2, gl.shape[1])
  J3, J2 = np.meshgrid(j3, j2, indexing="ij")
  sat = (np.abs(gl) >= LUT_SATURATION - 1e-4) | (np.abs(gr) >= LUT_SATURATION - 1e-4)
  return {"l": gl, "r": gr, "J2": J2, "J3": J3, "j2": j2, "j3": j3, "sat": sat}


def fmt(v: float, offset: float) -> str:
  return f"{v * 1e3:+8.2f} mm (tendon {v + offset:.5f} m)"


def print_actuator_ranges(t: dict, offset: float) -> None:
  print("\n-- 1. actuator (tendon extension) min / max")
  print(
    f"saturated entries (|value| >= {LUT_SATURATION} m): {t['sat'].sum()} of {t['sat'].size}"
  )
  for name in ("l", "r"):
    a = t[name]
    print(
      f"{name} slider, all entries:      min {fmt(a.min(), offset)}   max {fmt(a.max(), offset)}"
    )
    m = ~t["sat"]
    imin = np.unravel_index(np.argmin(np.where(m, a, np.inf)), a.shape)
    imax = np.unravel_index(np.argmax(np.where(m, a, -np.inf)), a.shape)
    print(
      f"{name} slider, unsaturated only: min {fmt(a[imin], offset)} at j2={t['J2'][imin]:+.4f}"
      f" j3={t['J3'][imin]:+.4f}   max {fmt(a[imax], offset)} at j2={t['J2'][imax]:+.4f}"
      f" j3={t['J3'][imax]:+.4f}"
    )
  i0 = np.unravel_index(np.argmin(np.abs(t["J2"]) + np.abs(t["J3"])), t["l"].shape)
  print(
    f"at the origin (j2={t['J2'][i0]:+.4f}, j3={t['J3'][i0]:+.4f}): l {fmt(t['l'][i0], offset)}"
    f"   r {fmt(t['r'][i0], offset)}"
  )


def print_joint_ranges(t: dict) -> None:
  print("\n-- 2. joint min / max")
  print(
    f"table domain (grid convention): joint 2 [{t['j2'].min():+.6f}, {t['j2'].max():+.6f}]"
    f" ({t['j2'].size} columns, step {abs(t['j2'][1] - t['j2'][0]):.5f} rad),"
    f" joint 3 [{t['j3'].min():+.6f}, {t['j3'].max():+.6f}]"
    f" ({t['j3'].size} rows, step {abs(t['j3'][1] - t['j3'][0]):.5f} rad)"
  )
  m = ~t["sat"]
  print(
    f"unsaturated region:             joint 2 [{t['J2'][m].min():+.4f}, {t['J2'][m].max():+.4f}],"
    f" joint 3 [{t['J3'][m].min():+.4f}, {t['J3'][m].max():+.4f}]"
  )
  i3_0 = int(np.argmin(np.abs(t["j3"])))
  i2_0 = int(np.argmin(np.abs(t["j2"])))
  row, col = m[i3_0, :], m[:, i2_0]
  print(
    f"unsaturated at joint 3 = {t['j3'][i3_0]:+.4f}: joint 2 [{t['j2'][row].min():+.4f}, {t['j2'][row].max():+.4f}];"
    f"  at joint 2 = {t['j2'][i2_0]:+.4f}: joint 3 [{t['j3'][col].min():+.4f}, {t['j3'][col].max():+.4f}]"
  )


def print_limit_points(t: dict, offset: float) -> None:
  print("\n-- 3. actuator positions at the joint limits of the table")
  j2min, j2max = t["j2"].min(), t["j2"].max()
  j3min, j3max = t["j3"].min(), t["j3"].max()
  points = [
    ("origin", 0.0, 0.0),
    ("j2 min", j2min, 0.0),
    ("j2 max", j2max, 0.0),
    ("j3 min", 0.0, j3min),
    ("j3 max", 0.0, j3max),
    ("j2 min, j3 min", j2min, j3min),
    ("j2 min, j3 max", j2min, j3max),
    ("j2 max, j3 min", j2max, j3min),
    ("j2 max, j3 max", j2max, j3max),
  ]
  print(
    f"{'point':16s} {'joint2':>8s} {'joint3':>8s} | {'l [mm]':>8s} {'r [mm]':>8s} | {'tendon l [m]':>12s} {'tendon r [m]':>12s}"
  )
  for name, j2, j3 in points:
    i = int(np.argmin(np.abs(t["j3"] - j3)))
    k = int(np.argmin(np.abs(t["j2"] - j2)))
    vl, vr = t["l"][i, k], t["r"][i, k]
    line = (
      f"{name:16s} {t['j2'][k]:+8.4f} {t['j3'][i]:+8.4f} | {vl * 1e3:+8.2f} {vr * 1e3:+8.2f}"
      f" | {vl + offset:12.5f} {vr + offset:12.5f}"
    )
    if t["sat"][i, k]:
      line += "   (saturated)"
    print(line)


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "--lut",
    type=Path,
    nargs=2,
    metavar=("L_NPY", "R_NPY"),
    default=[Path.cwd() / n for n in LUT_FILES],
    help="the two tables (default: joint_to_actuator_1/2.npy in the cwd)",
  )
  ap.add_argument(
    "--offset",
    type=float,
    default=HIP_XY_TENDON_OFFSET,
    help="tendon length at slider = 0, used to also print tendon lengths",
  )
  args = ap.parse_args()

  t = load(tuple(args.lut))
  print(f"tables: {args.lut[0]} (l), {args.lut[1]} (r); shape {t['l'].shape}")
  print_actuator_ranges(t, args.offset)
  print_joint_ranges(t)
  print_limit_points(t, args.offset)


if __name__ == "__main__":
  main()
