"""Overlay the hip_xy ROM surfaces of the tendon and slider models.

Reads the ``.npy`` written by :mod:`pal_mjlab.scripts.hip_xy_rom_sweep` and
draws, for every leg present, joint 2 and joint 3 as 3D surfaces over the
common slider-space (l, r) grid -- the tendon lengths are already converted
with ``HIP_XY_TENDON_OFFSET`` in the file, so both models share the axes.
Both models are drawn in the same axes: a translucent surface per model plus a
sparse wireframe of the same colour, so the two sheets stay readable where
they coincide (they do over most of the range).

The ground-truth lookup tables ``joint_to_actuator_1.npy`` / ``_2.npy`` (repo
root) are overlaid as a gray wireframe when found. They are the *inverse* map,
(joint 3, joint 2) -> (l, r) slider position on a 201 x 321 grid with no
metadata; the file carries no axes, so the convention below was fixed by
matching against the sweep (joint 3 agrees to ~0.01 rad only with it):
array 1 = l slider, array 2 = r slider, rows run joint 3 from -0.471239 to
+0.471239, columns run joint 2 from +0.663225 to -0.741765. The table is
mirror-symmetric (``a1[i] == a2[-1 - i]``), so swapping the arrays is the same
as reversing the rows. Entries saturated at +-0.04 m are masked. The sweep's
joint values are interpolated at every table point and the mean error per
model is printed.

Run::

    python -m pal_mjlab.scripts.plot_hip_xy_rom hip_xy_rom.npy --out hip_xy_rom.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Fixed hue per model, never cycled (validated CVD-safe pair).
COLORS = {"tendons": "#1d4ed8", "sliders": "#ea580c"}
LUT_COLOR = "#4b5563"
LUT_FILES = ("joint_to_actuator_1.npy", "joint_to_actuator_2.npy")
LUT_J3 = (-0.471239, 0.471239)  # rows
LUT_J2 = (0.663225, -0.741765)  # columns
LUT_SATURATION = 0.04
LABELS = {
  "tendons": "tendons (kangaroo_full_tendons.xml)",
  "sliders": "sliders (kangaroo_full_full)",
}


def _draw_surface(ax, r: dict, key: str, color: str, stride: int) -> None:
  x = r["l_cmd_slider"] * 1e3
  y = r["r_cmd_slider"] * 1e3
  z = r[key]
  ax.plot_surface(
    x, y, z, color=color, alpha=0.45, linewidth=0, antialiased=True, shade=True
  )
  ax.plot_wireframe(x, y, z, color=color, linewidth=0.6, rstride=stride, cstride=stride)


def load_lut(paths: tuple[Path, Path]) -> dict:
  """Return the table as slider-space (l, r) grids plus the joint grids."""
  lg, rg = (np.load(p).astype(float) for p in paths)
  if lg.shape != rg.shape or lg.ndim != 2:
    raise ValueError(f"unexpected LUT shapes {lg.shape} / {rg.shape}")
  j3, j2 = np.meshgrid(
    np.linspace(*LUT_J3, lg.shape[0]), np.linspace(*LUT_J2, lg.shape[1]), indexing="ij"
  )
  sat = (np.abs(lg) >= LUT_SATURATION - 1e-4) | (np.abs(rg) >= LUT_SATURATION - 1e-4)
  lg, rg = lg.copy(), rg.copy()
  lg[sat] = rg[sat] = np.nan
  return {"l": lg, "r": rg, "joint2": j2, "joint3": j3}


def lut_error(r: dict, lut: dict, key: str) -> tuple[float, float, int]:
  """Mean |error| and signed mean of the sweep vs the table, at table points."""
  from scipy.interpolate import RegularGridInterpolator

  f = RegularGridInterpolator(
    (r["l_cmd_slider"][:, 0], r["r_cmd_slider"][0, :]), r[key], bounds_error=False
  )
  pred = f(np.stack([lut["l"].ravel(), lut["r"].ravel()], 1))
  # Only where the sweep is not clipped by its own joint limits.
  jr = r["joint_range"][0 if key == "joint2" else 1]
  ok = (
    ~np.isnan(pred)
    & (lut[key].ravel() > jr[0] + 0.03)
    & (lut[key].ravel() < jr[1] - 0.03)
  )
  err = pred[ok] - lut[key].ravel()[ok]
  return float(np.abs(err).mean()), float(err.mean()), int(ok.sum())


def _draw_lut(ax, lut: dict, key: str) -> None:
  n0, n1 = lut["l"].shape
  ax.plot_wireframe(
    lut["l"] * 1e3,
    lut["r"] * 1e3,
    lut[key],
    color=LUT_COLOR,
    linewidth=0.7,
    rstride=max(1, n0 // 14),
    cstride=max(1, n1 // 14),
  )


def plot(
  res: dict,
  out: Path,
  models: list[str],
  elev: float,
  azims: tuple[float, float],
  lut: dict | None = None,
) -> None:
  models = [m for m in models if m in res]
  sides = [s for s in ("left", "right") if any(s in res[m] for m in models)]
  joints = ("joint2", "joint3")

  fig = plt.figure(figsize=(6.2 * len(joints), 5.4 * len(sides)))
  for i, side in enumerate(sides):
    for j, key in enumerate(joints):
      ax = fig.add_subplot(
        len(sides), len(joints), i * len(joints) + j + 1, projection="3d"
      )
      for m in models:
        r = res[m].get(side)
        if r is None:
          continue
        stride = max(1, r[key].shape[0] // 10)
        _draw_surface(ax, r, key, COLORS[m], stride)
        if lut is not None:
          mean_abs, mean_signed, n = lut_error(r, lut, key)
          print(
            f"{m:8s} {side:5s} {key}: vs LUT mean|err| {mean_abs:.4f} rad "
            f"(signed {mean_signed:+.4f}, n={n})"
          )
      if lut is not None:
        _draw_lut(ax, lut, key)
      jname = next(res[m][side]["joint_names"][j] for m in models if side in res[m])
      ax.set_title(f"{side} leg — {jname}", fontsize=11)
      ax.set_xlabel("l slider [mm]", labelpad=8)
      ax.set_ylabel("r slider [mm]", labelpad=8)
      ax.set_zlabel("angle [rad]", labelpad=6)
      ax.view_init(elev=elev, azim=azims[j])
      ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
      ax.grid(True, alpha=0.3)

  handles = [Line2D([], [], color=COLORS[m], lw=3, label=LABELS[m]) for m in models]
  if lut is not None:
    handles.append(
      Line2D(
        [],
        [],
        color=LUT_COLOR,
        lw=1.2,
        label="ground truth (joint_to_actuator_1/2.npy)",
      )
    )
  fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=len(handles),
    frameon=False,
    fontsize=10,
  )
  n = next(iter(res[models[0]].values()))["joint2"].shape[0]
  fig.suptitle(
    f"hip_xy range of motion: (l, r) slider position → joint angle ({n}×{n} grid)",
    y=0.995,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.5)
  fig.savefig(out, dpi=150)
  print(f"saved -> {out.resolve()}")


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("npy", type=Path, nargs="?", default=Path("hip_xy_rom.npy"))
  ap.add_argument("--out", type=Path, default=None, help="default: <npy stem>.png")
  ap.add_argument("--models", nargs="+", default=list(COLORS), choices=list(COLORS))
  ap.add_argument(
    "--lut",
    type=Path,
    nargs=2,
    metavar=("L_NPY", "R_NPY"),
    help="ground-truth tables (default: joint_to_actuator_1/2.npy next to the "
    "input .npy or in the cwd, if present); pass --no-lut to skip",
  )
  ap.add_argument("--no-lut", action="store_true")
  ap.add_argument("--elev", type=float, default=28.0)
  # joint 2 varies along l + r, joint 3 along l - r: each wants its own view so
  # the surface is not seen edge-on along its ridge.
  ap.add_argument(
    "--azim2", type=float, default=-60.0, help="view azimuth, joint 2 panels"
  )
  ap.add_argument(
    "--azim3", type=float, default=30.0, help="view azimuth, joint 3 panels"
  )
  args = ap.parse_args()

  res = np.load(args.npy, allow_pickle=True).item()
  lut = None
  if not args.no_lut:
    candidates = (
      [tuple(args.lut)]
      if args.lut
      else [
        tuple(d / n for n in LUT_FILES) for d in (args.npy.resolve().parent, Path.cwd())
      ]
    )
    paths = next((c for c in candidates if all(p.exists() for p in c)), None)
    if paths is not None:
      lut = load_lut(paths)
      print(f"ground truth: {paths[0]} / {paths[1]}")
    elif args.lut:
      raise FileNotFoundError(args.lut)
  plot(
    res,
    args.out or args.npy.with_suffix(".png"),
    args.models,
    args.elev,
    (args.azim2, args.azim3),
    lut,
  )


if __name__ == "__main__":
  main()
