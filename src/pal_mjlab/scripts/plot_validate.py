"""Find the latest ``validate`` result and plot its velocity tracking and
success rate over time.

Searches ``--root`` (``logs`` by default) for the ``.npz`` files ``validate``
writes (``<checkpoint dir>/validate/*.npz`` and ``logs/validate/*.npz``), opens
the most recently written one and shows a matplotlib window with, against
simulated time: the fastest forward command over the alive envs with the
measured forward velocity of the envs still alive at each step (min-max band
and mean, narrowing as envs fail), the lateral error, and at the bottom the
fraction of envs that have not failed yet; a black line marks the first
failure. An env that only hit the episode time limit counts as a success,
matching ``failed`` in the ``.npz``. Stairs results run with a finish distance
also show the finish rate: the fraction of envs that have walked past the
finish line by then.

Results with actuation traces (``validate --trace-envs``, on by default) also
get a window per leg (left, then right) for each space plotted, then one for
the pelvis (in joint space when both are plotted): a row per signal with its
position, velocity and force over time for up to three of the traced envs:
the one reaching the highest forward velocity, the longest surviving one (the
fastest of them if several reached the end) and the first one to fail (``--envs
first`` draws the first ``--num-envs`` instead). Also on each row: the
signal's static bounds as dashed lines and, filled, the range of the position
over the envs alive at each step and ± the peak |velocity| and |force| over
them (over the alive traced envs for results from before validate recorded
these). Saturation (the force at the limit in effect) is not drawn, only
its share given in the force title. DC motor signals also show the
velocity-dependent force limit per env (thin, the env's colour) and, on the velocity axis, the
velocity up to which the full effort limit is available and the no-load
velocity where the available force reaches zero. Signals simulated without a
torque-speed curve (builtin actuators, the LUT's joint-space side) get the
same two velocities of the Kangaroo motor behind them, fainter and marked
"hardware, not simulated", so a velocity the real motor can't reach shows.

The simple model and the "joint" transmission plot joint positions,
velocities and torques; the "actuator" transmission plots actuator lengths,
velocities and forces; the "lut" transmission plots both the screw actuators
(with the force the LUT requested before the motor clip, dashed) and the
joint-space PD torques the LUT maps onto them, in map coordinates (the leg
length is the actuator distance). ``--space`` overrides; its joint pages also
show the leg joints an actuator-space model drives through its linkage. Run::

    uv run plot-validate
    uv run plot-validate --envs first --num-envs 8 --signals "leg_left_.*"
    uv run plot-validate --scenario stairs --task Stairs
    uv run plot-validate --path logs/validate/flat_ramp_2026-09-15_04-20-49.npz
    uv run plot-validate --list 10
    uv run plot-validate --batch "results/<sweep>/*/*.npz" --workers 16
    uv run plot-validate --aggregate "results/<sweep>/*_e2e_*/*.npz" --save agg.png
"""

from __future__ import annotations

import dataclasses
import glob
import json
import multiprocessing
import re
import textwrap
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mjlab
import numpy as np
import tyro
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  LEG_JOINT_DC_MOTOR_CURVES,
  LEG_LENGTH_JOINT_EXPR,
  LEG_SCREW_PARAMS,
)

BLUE, GRAY, ORANGE = "#2a78d6", "#6b6b66", "#e06c00"
DC_LIMIT_LABEL = (
  "DC force limit at the env's velocity\n(thin, env colour; dips toward 0 at speed)"
)


def hardware_dc_curve(name: str) -> tuple[float, float, float]:
  """(effort_limit, saturation_effort, velocity_limit) of the Kangaroo DC
  motor behind a signal, whatever actuator model the run simulated; NaN for
  signals without one (the pelvis joints).

  Screws (``leg_<side>_<n>_actuator``, both models) use the ballscrew limits,
  joints (``leg_<side>_<n>_joint``, ``leg_<side>_length_joint``) the screw
  curves reflected to them; a LUT ``... (distance)`` signal is the leg length.
  The left leg mirrors the right one."""
  name = str(name)
  screw = re.fullmatch(r"leg_(?:left|right)_(.+)_actuator", name)
  if screw:
    params = LEG_SCREW_PARAMS.get(f"leg_right_{screw.group(1)}_actuator")
    if params is None:
      return (np.nan, np.nan, np.nan)
    return (params["effort"], params["saturation_effort"], params["velocity_limit"])
  joint = name.removesuffix(" (distance)")
  if joint != name:
    curve = LEG_JOINT_DC_MOTOR_CURVES[LEG_LENGTH_JOINT_EXPR]
  else:
    curve = next(
      (c for expr, c in LEG_JOINT_DC_MOTOR_CURVES.items() if re.fullmatch(expr, joint)),
      None,
    )
    if curve is None:
      return (np.nan, np.nan, np.nan)
  effort, knee_velocity, velocity_limit = curve
  return (effort, effort / (1.0 - knee_velocity / velocity_limit), velocity_limit)


@dataclass(frozen=True)
class PlotValidateConfig:
  path: str | None = None
  """A result .npz to plot, or a directory to search instead of --root."""
  root: str = "logs"
  """Directory searched recursively for validate results."""
  scenario: str | None = None
  """Only consider results of this scenario (file-name prefix, e.g. "stairs",
  "stairs_peak", "flat_ramp")."""
  task: str | None = None
  """Only consider results whose task id contains this substring."""
  list: int = 0
  """Print the N most recent matching results before plotting."""
  save: str | None = None
  """Also write the figure to this path (actuation pages go next to it as
  <stem>_<actuator|joint>_<page><suffix>)."""
  actuation: bool = True
  """Plot the actuation traces when the result has them."""
  envs: Literal["extremes", "first"] = "extremes"
  """Envs whose actuation traces are drawn: "extremes" (of the traced envs,
  the one with the highest peak forward velocity, the longest surviving one
  and the first one to fail)
  or "first" (the first --num-envs traced envs)."""
  num_envs: int = 4
  """Envs drawn with --envs first."""
  space: Literal["auto", "joint", "actuator", "both"] = "auto"
  """Signals to plot: "joint" (joint position, velocity, torque of every
  actuated joint and every leg joint behind a Kangaroo motor, or the LUT's
  joint-space torques), "actuator" (actuator
  length, velocity, force), "both", or "auto" (the space the task's actuators
  act in; both for the LUT transmission)."""
  signals: str | None = None
  """Regex: only plot the signals whose name matches it (re.search)."""
  show: bool = True
  """Open the matplotlib window."""
  workers: int = 8
  """Processes rendering figures (each result's summary and actuation pages)
  in parallel when saving without --show; 1 renders them one by one."""
  batch: str | None = None
  """Glob of result .npz files (quoted) to plot in one go, each saved next to
  it as <stem>.png plus its pages, all figures sharing the --workers pool
  (one process start-up instead of one per result)."""
  skip_plotted: bool = False
  """With --batch, skip results whose .png is newer than both the .npz and
  this script."""
  aggregate: str | None = None
  """Glob of result .npz files (quoted, matches sorted) to overlay in one
  figure instead of plotting a single result: per result, the mean forward
  velocity of the alive envs with their min-max band (each bound smoothed
  separately, see --smooth), the lateral error mean ± std and the success
  rate. Saved to --save."""
  title: str | None = None
  """Title of the --aggregate figure."""
  smooth: int = 50
  """Moving-average window [steps] of the min and max v_x bounds in the
  --aggregate figure."""
  lateral: bool = True
  """Include the lateral-error panel in the --aggregate figure."""
  cmd_of: str | None = None
  """Draw only the velocity command of the --aggregate results whose file
  name contains this text, as a black dashed line (default: every result's
  command in its own colour)."""
  crop_to_shortest: bool = False
  """End the --aggregate figure's time axis where the shortest result ends
  (e.g. when all of its envs have failed) instead of the longest."""
  alpha_by_alive: bool = False
  """Fade the speed and lateral-error mean and band over the alive envs with
  the share of all envs still alive at each step (opaque while all are,
  invisible once none is), in the summary and --aggregate figures."""


def _summary(npz_path: Path) -> dict:
  json_path = npz_path.with_suffix(".json")
  if not json_path.exists():
    return {}
  return json.loads(json_path.read_text())


def find_results(cfg: PlotValidateConfig) -> list[Path]:
  """Matching validate results, newest first."""
  search = Path(cfg.path) if cfg.path is not None else Path(cfg.root)
  if search.is_file():
    return [search]
  if not search.is_dir():
    raise FileNotFoundError(f"No such file or directory: {search}")
  found = [p for p in search.rglob("*.npz") if p.parent.name == "validate"]
  if cfg.scenario is not None:
    found = [p for p in found if p.stem.startswith(cfg.scenario + "_")]
  if cfg.task is not None:
    found = [p for p in found if cfg.task in _summary(p).get("task_id", "")]
  return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _fraction_by(z, event_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(time, fraction of all envs whose event_time has passed), from t=0."""
  time = z["time"]
  step_dt = float(time[1] - time[0]) if len(time) > 1 else 0.02
  t_end = time + step_dt
  n = np.searchsorted(np.sort(event_time), t_end + 1e-6, side="right")
  frac = n / len(z["failed"])
  return np.concatenate([[0.0], t_end]), np.concatenate([[0.0], frac])


def success_rate(z) -> tuple[np.ndarray, np.ndarray]:
  """(time, fraction of envs not failed by then), starting from 100% at t=0."""
  t, failed = _fraction_by(z, z["survival_time"][z["failed"]])
  return t, 1.0 - failed


def finish_rate(z) -> tuple[np.ndarray, np.ndarray]:
  """(time, fraction of envs past the stairs finish line by then)."""
  return _fraction_by(z, z["finish_time"][z["finished"]])


def _has_finish(z, summary: dict) -> bool:
  """Whether the result is a stairs run with a finish distance (older results
  and the other scenarios have no finish line)."""
  if "finished" not in z:
    return False
  config = summary.get("config")
  if config is None:
    return bool(z["finished"].any())
  return (
    config.get("scenario") == "stairs"
    and config.get("stairs", {}).get("finish_distance") is not None
  )


def alive_fraction(z) -> np.ndarray:
  """[steps] share of all envs still alive after each step (the envs the
  post-step state statistics cover)."""
  if "vel_x_per_env" in z:
    return (~np.isnan(z["vel_x_per_env"])).mean(0)
  # Envs alive after step k are the ones entering step k + 1.
  count = z["alive_count"]
  return np.concatenate([count[1:], count[-1:]]) / len(z["failed"])


def _line(ax, t, y, fade: np.ndarray | None, alpha: float = 1.0, **kw) -> None:
  """ax.plot(t, y), or with ``fade`` ([steps] in [0, 1]) a line whose segment
  into step k has alpha * fade[k] (plus an empty line as its legend key)."""
  if fade is None:
    ax.plot(t, y, alpha=alpha, **kw)
    return
  pts = np.stack([t, y], -1)
  seg = np.stack([pts[:-1], pts[1:]], 1)
  ok = np.isfinite(seg).all((1, 2))
  rgba = np.tile(to_rgba(kw["color"]), (len(seg), 1))
  rgba[:, 3] = alpha * fade[1:]
  ax.add_collection(
    LineCollection(seg[ok], colors=rgba[ok], linewidths=kw.get("linewidth", 1.5))
  )
  ax.autoscale_view()
  ax.plot([], [], alpha=alpha, **kw)


def _band(ax, t, lo, hi, fade: np.ndarray | None, alpha: float, **kw) -> None:
  """ax.fill_between(t, lo, hi), or with ``fade`` one quad per step with alpha
  * fade[k] (plus an empty band as its legend key)."""
  if fade is None:
    ax.fill_between(t, lo, hi, alpha=alpha, linewidth=0, **kw)
    return
  quads = np.stack(
    [
      np.stack([t[:-1], lo[:-1]], -1),
      np.stack([t[1:], lo[1:]], -1),
      np.stack([t[1:], hi[1:]], -1),
      np.stack([t[:-1], hi[:-1]], -1),
    ],
    1,
  )
  ok = np.isfinite(quads).all((1, 2))
  rgba = np.tile(to_rgba(kw["color"]), (len(quads), 1))
  rgba[:, 3] = alpha * fade[1:]
  # Not antialiased: neighbouring quads would otherwise leave seams.
  ax.add_collection(
    PolyCollection(quads[ok], facecolors=rgba[ok], linewidths=0, antialiaseds=False)
  )
  ax.autoscale_view()
  ax.fill_between([], [], [], alpha=alpha, linewidth=0, **kw)


def speed_per_env(z) -> tuple[np.ndarray, str]:
  """([envs, steps] measured planar base speed |v_xy| = hypot(v_x, v_y), NaN
  once an env has ended; its name for labels). Older results without per-env
  v_y fall back to the signed forward v_x."""
  if "vel_y_per_env" in z:
    return np.hypot(z["vel_x_per_env"], z["vel_y_per_env"]), "|v_xy|"
  return z["vel_x_per_env"], "v_x"


def _has_speed(z) -> bool:
  """Whether the result recorded per-env v_y, i.e. its planar speed."""
  return "vel_x_per_env" in z and "vel_y_per_env" in z


def plot_velocity(ax, z, fade: np.ndarray | None = None) -> None:
  """Speed over time: the fastest command over the alive envs (forward only,
  so its v_x is its speed) and the measured planar base speed |v_xy| of the
  envs still alive at each step (all of them at first, fewer as envs fail),
  as their min-max band and mean. Results without per-env v_y show v_x
  instead; older ones without per-env velocities fall back to the mean
  command and the v_x mean ± std over the alive envs. With ``fade`` ([steps]
  in [0, 1]) the band and mean fade with it. State is recorded after each
  step, hence the one-step shift."""
  time = z["time"]
  step_dt = float(time[1] - time[0]) if len(time) > 1 else 0.02
  t_end = time + step_dt
  if "cmd_vel_x_max" in z:
    ax.plot(
      time,
      z["cmd_vel_x_max"],
      color=GRAY,
      linewidth=1.5,
      linestyle="--",
      label="cmd v_x = cmd speed (max over alive envs)",
    )
  else:
    ax.plot(
      time,
      z["cmd_vel_x"],
      color=GRAY,
      linewidth=1.5,
      linestyle="--",
      label="cmd v_x (mean over alive envs; max not recorded)",
    )

  ylabel = "forward velocity v_x [m/s]"
  if "vel_x_per_env" in z:
    # NaN once an env has ended: each step covers the envs still alive.
    vel, what = speed_per_env(z)
    if what != "v_x":
      ylabel = f"speed {what} [m/s]"
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)  # steps no env reached
      lo, hi, mean = np.nanmin(vel, 0), np.nanmax(vel, 0), np.nanmean(vel, 0)
    band_label = f"{what} min-max, alive envs"
    mean_label = f"mean {what}, alive envs"
  else:
    count = z["alive_count"]
    alive = np.concatenate([count[1:], count[-1:]]) > 0
    col = list(z["state_names"]).index("base_lin_vel_x_b")
    mean = np.where(alive, z["state_mean"][:, col], np.nan)
    std = np.where(alive, z["state_std"][:, col], np.nan)
    lo, hi = mean - std, mean + std
    band_label = "v_x mean ± std, alive envs (per-env v_x not recorded)"
    mean_label = "mean v_x, alive envs"
  if fade is not None:
    band_label += ", faded by share alive"
  _band(ax, t_end, lo, hi, fade, 0.2, color=BLUE, label=band_label)
  _line(ax, t_end, mean, fade, color=BLUE, linewidth=2, label=mean_label)
  ax.grid(True, color="#e2e2dd", linewidth=0.8)
  ax.set_axisbelow(True)
  ax.set_ylim(bottom=0.0)
  ax.set_ylabel(ylabel)
  ax.legend(loc="upper left", frameon=False)


def plot_success_rate(ax, z, has_finish: bool) -> None:
  """Fraction of envs not failed by each time and, for stairs runs with a
  finish line, the fraction past it."""
  n_env = len(z["failed"])
  t, rate = success_rate(z)
  ax.step(t, 100 * rate, where="post", color=BLUE, linewidth=2, label="success rate")
  if has_finish:
    t_fin, fin = finish_rate(z)
    ax.step(
      t_fin, 100 * fin, where="post", color=ORANGE, linewidth=2, label="finish rate"
    )
  ax.set_ylim(-2, 102)
  ax.grid(True, color="#e2e2dd", linewidth=0.8)
  ax.set_axisbelow(True)
  ax.set_ylabel(
    f"envs [%] (of {n_env})" if has_finish else f"envs not failed [%] (of {n_env})"
  )
  ax.legend(loc="lower left", frameon=False)


def plot_lateral_error(ax, z, log_every: int, fade: np.ndarray | None = None) -> None:
  """Offset from the steering line over time: signed mean ± std over the envs
  still alive (faded with ``fade`` if given), plus every env's trace when
  per-env steps were saved. State is recorded after each step, hence the
  one-step shift."""
  time = z["time"]
  step_dt = float(time[1] - time[0]) if len(time) > 1 else 0.02
  t_end = time + step_dt
  col = list(z["state_names"]).index("lateral_error_y")
  # Envs alive after step k are the ones entering step k + 1; a step that left
  # none alive averages nothing (validate stores 0 there), so hide it.
  count = z["alive_count"]
  alive = np.concatenate([count[1:], count[-1:]]) > 0

  if "per_env_steps" in z:
    lat_env = z["per_env_steps"][:, :, col]
    t_env = t_end[::log_every][: lat_env.shape[1]]
    # On the step an env ends, its row already holds the post-reset state.
    done = z["failed"] | z["truncated"]
    ended = done[:, None] & (t_env[None, :] >= z["survival_time"][:, None] - 1e-6)
    lat_env = np.where(ended, np.nan, lat_env)
    for trace in lat_env:
      ax.plot(t_env, trace, color=GRAY, linewidth=0.5, alpha=0.25)

  mean = np.where(alive, z["state_mean"][:, col], np.nan)
  std = np.where(alive, z["state_std"][:, col], np.nan)
  _band(ax, t_end, mean - std, mean + std, fade, 0.15, color=BLUE)
  _line(ax, t_end, mean, fade, color=BLUE, linewidth=1.5, label="signed mean ± std")
  ax.axhline(0.0, color="#9a9a94", linewidth=0.8)
  ax.grid(True, color="#e2e2dd", linewidth=0.8)
  ax.set_axisbelow(True)
  ax.set_xlabel("time [s]")
  ax.set_ylabel("lateral error y [m]")
  ax.legend(loc="upper left", frameon=False)


@dataclass
class _Signals:
  """One set of actuation signals: [envs, steps, 3 (position, velocity,
  force), n] traces and what to draw around them."""

  space: str
  quantities: tuple[str, str, str]
  names: np.ndarray
  trace: np.ndarray
  bounds: np.ndarray
  """[3, n, 2] static bounds."""
  limit: np.ndarray
  """[envs, steps, 2, n] force limits in effect."""
  dc: np.ndarray
  """[n, 3] effort_limit, saturation_effort, velocity_limit (NaN: no curve)."""
  request: np.ndarray | None = None
  """[envs, steps, n] force requested before the motor clip."""
  envelope: np.ndarray | None = None
  """[steps, 2 (min, max), 3 (quantity), n] min and max over the alive envs."""
  envelope_envs: str = ""
  """The envs the envelope covers, for the legend."""

  def select(self, keep: np.ndarray, e: np.ndarray) -> _Signals:
    """The signals in ``keep`` of the traced envs at positions ``e``."""
    return _Signals(
      space=self.space,
      quantities=self.quantities,
      names=self.names[keep],
      trace=self.trace[e][..., keep],
      bounds=self.bounds[:, keep],
      limit=self.limit[e][..., keep],
      dc=self.dc[keep],
      request=None if self.request is None else self.request[e][..., keep],
      envelope=None if self.envelope is None else self.envelope[..., keep],
      envelope_envs=self.envelope_envs,
    )


def _envelope(z, key: str | None, trace: np.ndarray) -> tuple[np.ndarray, str]:
  """[steps, 2 (min, max), 3, n] min and max of each quantity over the envs
  alive at each step,
  as validate recorded it under ``key``, and the envs it covers. Results from
  before validate recorded it fall back to the traced envs."""
  if key is not None and key in z:
    return z[key], "alive envs"
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # steps no env reached
    envelope = np.stack([np.nanmin(trace, axis=0), np.nanmax(trace, axis=0)], 1)
  return envelope, f"alive of {trace.shape[0]} traced envs"


def trace_extremes(z) -> list[tuple[int, str]]:
  """(position in the traces, legend label) of the traced env with the
  highest peak forward velocity, of the longest surviving one (the fastest of
  them when several lasted as long, e.g. to the end) and of the first one to
  fail (if any); one entry per env, with every role it plays in its label.
  Matches validate's ``--trace-keep extremes``. Results without per-env
  velocities rank the envs by their mean speed (distance over survival time)
  instead."""
  n = z["trace_actuator"].shape[0]
  ids = z["trace_env_ids"] if "trace_env_ids" in z else np.arange(n)
  if "vel_x_per_env" in z:
    vel = z["vel_x_per_env"][ids]
    speed = np.where(np.isnan(vel), -np.inf, vel).max(1)
    what = "peak v_x"
  else:
    speed = z["distance_x"][ids] / np.maximum(z["survival_time"][ids], 1e-6)
    what = "mean v_x"
  fastest = int(np.argmax(speed))
  labels = {fastest: [f"fastest, {what} {speed[fastest]:.2f} m/s"]}
  failed = z["failed"][ids]
  survival = z["survival_time"][ids]
  longest = survival >= survival.max() - 1e-6
  best = int(np.argmax(np.where(longest, speed, -np.inf)))
  lasted = f"longest survival {survival[best]:.2f} s"
  if best == fastest:
    labels[best].append(lasted)
  else:
    labels[best] = [f"{lasted}, {what} {speed[best]:.2f} m/s"]
  if failed.any():
    first = int(np.argmin(np.where(failed, z["survival_time"][ids], np.inf)))
    survived = z["survival_time"][ids][first]
    labels.setdefault(first, []).append(f"first to fail at {survived:.2f} s")
  return [(e, f"env {ids[e]} ({', '.join(lbl)})") for e, lbl in labels.items()]


def _plot_envs(z, cfg: PlotValidateConfig) -> list[tuple[int, str]]:
  """(position in the traces, legend label) of the envs to draw."""
  if cfg.envs == "extremes":
    return trace_extremes(z)
  n = min(max(cfg.num_envs, 1), z["trace_actuator"].shape[0])
  ids = z["trace_env_ids"] if "trace_env_ids" in z else np.arange(n)
  return [(e, f"env {ids[e]}") for e in range(n)]


def _signal_sets(z, cfg: PlotValidateConfig, envs: np.ndarray) -> list[_Signals]:
  stored = str(z["actuation_space"]) if "actuation_space" in z else "actuator"
  lut = "trace_lut_joint" in z
  space = cfg.space
  if space == "auto":
    space = "both" if stored == "lut" else stored
  sets = []
  if space in ("actuator", "both"):
    sets.append(
      _Signals(
        space="actuator",
        quantities=("length", "velocity", "force"),
        names=z["actuator_names"],
        trace=z["trace_actuator"],
        bounds=z["actuator_bounds"],
        limit=z["trace_actuator_limit"],
        dc=z["actuator_dc"],
        request=z["trace_actuator_request"] if lut else None,
      )
    )
    sets[-1].envelope, sets[-1].envelope_envs = _envelope(
      z, "actuator_envelope", z["trace_actuator"]
    )
  if space in ("joint", "both") and lut:
    bounds = z["lut_joint_bounds"]
    trace = z["trace_lut_joint"]
    sets.append(
      _Signals(
        space="LUT joint (map coordinates)",
        quantities=("position", "velocity", "torque"),
        names=z["lut_joint_names"],
        trace=trace,
        bounds=bounds,
        limit=np.broadcast_to(bounds[2].T, (*trace.shape[:2], *bounds[2].T.shape)),
        dc=np.full((len(z["lut_joint_names"]), 3), np.nan),
      )
    )
    sets[-1].envelope, sets[-1].envelope_envs = _envelope(z, None, trace)
  elif space in ("joint", "both"):
    # The actuated joints plus the ones a Kangaroo motor stands behind, which
    # actuator-space models (tendons, screws) drive only through the linkage.
    behind_motor = [np.isfinite(hardware_dc_curve(n)).all() for n in z["joint_names"]]
    cols = np.flatnonzero(z["joint_actuated"] | np.array(behind_motor, dtype=bool))
    sets.append(
      _Signals(
        space="joint",
        quantities=("position", "velocity", "torque"),
        names=z["joint_names"][cols],
        trace=z["trace_joint"][..., cols],
        bounds=z["joint_bounds"][:, cols],
        limit=z["trace_joint_limit"][..., cols],
        dc=z["joint_dc"][cols],
      )
    )
    envelope, label = _envelope(z, "joint_envelope", z["trace_joint"])
    sets[-1].envelope, sets[-1].envelope_envs = envelope[..., cols], label
  out = []
  for sig in sets:
    keep = np.array(
      [cfg.signals is None or bool(re.search(cfg.signals, str(n))) for n in sig.names],
      dtype=bool,
    )
    if keep.any():
      out.append(sig.select(keep, envs))
    else:
      print(f"[WARN] no {sig.space} signals match --signals {cfg.signals!r}")
  return out


def saturated(force: np.ndarray, limit: np.ndarray) -> np.ndarray:
  """Where ``force`` [...] sits at the limit in effect ([..., 2 (lo, hi)]),
  within 0.5% of the limit (the DC curve of a non-LUT DcMotor is evaluated at
  the post-step velocity, not the one its last substep used)."""
  lo, hi = limit[..., 0], limit[..., 1]
  with np.errstate(invalid="ignore"):
    tol_hi = 5e-3 * np.abs(hi) + 1e-6
    tol_lo = 5e-3 * np.abs(lo) + 1e-6
    return (force >= hi - tol_hi) | (force <= lo + tol_lo)


def _plot_signal_row(
  axes, sig: _Signals, i: int, t_end, colors, env_labels: list[str]
) -> None:
  force = sig.trace[:, :, 2, i]
  sat = saturated(force, sig.limit[..., i]) & np.isfinite(force)
  dc = sig.dc[i]
  has_dc = bool(np.isfinite(dc).all())
  # Without a simulated curve, the real motor's velocities for reference.
  hw_dc = np.asarray(hardware_dc_curve(sig.names[i]))
  has_hw_dc = not has_dc and bool(np.isfinite(hw_dc).all())
  for q, ax in enumerate(axes):
    values = sig.trace[:, :, q, i]
    for e, v in enumerate(values):
      ax.plot(
        t_end, v, color=colors[e % len(colors)], linewidth=0.9, label=env_labels[e]
      )
    lo, hi = sig.bounds[q, i]
    if not (q == 1 and has_dc):
      for b in (lo, hi):
        if np.isfinite(b):
          ax.axhline(b, color="black", linewidth=1.0, linestyle="--", label="bounds")
    label = f"{sig.names[i]} {sig.quantities[q]}"
    if np.isfinite(values).any():
      label += f"  [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]"
    if q == 1 and has_dc:
      effort, saturation, v_max = dc
      v_full = v_max * max(1.0 - effort / saturation, 0.0)
      for sign in (-1, 1):
        ax.axhline(
          sign * v_max,
          color="#d62728",
          linewidth=1.0,
          linestyle="-.",
          label="max velocity (available force -> 0)",
        )
        ax.axhline(
          sign * v_full,
          color="#2ca02c",
          linewidth=1.0,
          linestyle=":",
          label="unsaturated velocity (full effort limit)",
        )
      label += f"  full ±{v_full:.3g}, max ±{v_max:.3g}"
    elif q == 1 and has_hw_dc:
      effort, saturation, v_max = hw_dc
      v_full = v_max * max(1.0 - effort / saturation, 0.0)
      for sign in (-1, 1):
        ax.axhline(
          sign * v_max,
          color="#d62728",
          linewidth=1.0,
          linestyle="-.",
          alpha=0.5,
          label="hardware max velocity (not simulated)",
        )
        ax.axhline(
          sign * v_full,
          color="#2ca02c",
          linewidth=1.0,
          linestyle=":",
          alpha=0.5,
          label="hardware unsaturated velocity (not simulated)",
        )
      label += f"  hw full ±{v_full:.3g}, max ±{v_max:.3g}"
    elif np.isfinite(lo) or np.isfinite(hi):
      label += f"  lim [{lo:.3g}, {hi:.3g}]"
    if sig.envelope is not None:
      # Position: its range over the envs; velocity and force: ± their peak
      # magnitude.
      lo_env, hi_env = sig.envelope[:, 0, q, i], sig.envelope[:, 1, q, i]
      band_label = f"min-max {sig.quantities[q]}, {sig.envelope_envs}"
      if q > 0:
        hi_env = np.maximum(np.abs(lo_env), np.abs(hi_env))
        lo_env = -hi_env
        band_label = f"peak |{sig.quantities[q]}|, {sig.envelope_envs}"
      ax.fill_between(
        t_end,
        lo_env,
        hi_env,
        color=GRAY,
        alpha=0.1,
        linewidth=0,
        label=band_label,
      )
    if q == 2:
      for e in range(len(values)):
        color = colors[e % len(colors)]
        if sig.request is not None:
          ax.plot(
            t_end,
            sig.request[e, :, i],
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
            label="requested (before motor clip)",
          )
        if has_dc:
          for b in range(2):
            ax.plot(
              t_end,
              sig.limit[e, :, b, i],
              color=color,
              linewidth=0.6,
              alpha=0.6,
              label="_nolegend_",
            )
      if has_dc:
        # One grey key for the per-env limit lines, which would otherwise
        # take the first env's colour and read as another trace.
        ax.plot([], [], color=GRAY, linewidth=0.8, label=DC_LIMIT_LABEL)
      alive = np.isfinite(force)
      if alive.any():
        label += f"  sat {100 * sat.sum() / alive.sum():.1f}%"
    ax.set_title(label, fontsize=8, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e2e2dd", linewidth=0.8)
    ax.set_axisbelow(True)


def _leg_side(name: str) -> str:
  """ "left" or "right" for a leg signal, "" for anything else (the pelvis)."""
  m = re.search(r"leg_(left|right)", str(name))
  return m.group(1) if m else ""


def _pages(sets: list[_Signals]) -> list[tuple[str, str, _Signals]]:
  """(file tag, title, signals) of each page: every set's left-leg signals,
  then its right-leg ones, then one page for the rest (the pelvis), taken
  from the last set that has any (joint space when both are plotted, so the
  pelvis isn't drawn twice)."""
  pages, rest = [], None
  for sig in sets:
    tag = "joint" if sig.space != "actuator" else "actuator"
    side = np.array([_leg_side(n) for n in sig.names])
    envs = np.arange(sig.trace.shape[0])
    for leg in ("left", "right"):
      if (side == leg).any():
        title = f"{sig.space} space, {leg} leg"
        pages.append((f"{tag}_{leg}", title, sig.select(side == leg, envs)))
    if (side == "").any():
      rest = (f"{sig.space} space, pelvis", sig.select(side == "", envs))
  if rest is not None:
    pages.append(("pelvis", *rest))
  return pages


def _actuation_pages(
  z, cfg: PlotValidateConfig
) -> tuple[list[str], list[tuple[str, str, _Signals]]]:
  """(legend labels of the drawn envs, the actuation pages) of a result; no
  pages when it has no traces or cfg.actuation is off."""
  if "trace_actuator" not in z or not cfg.actuation:
    return [], []
  # Envs alive after step k are the ones entering step k + 1.
  count = z["alive_count"]
  alive = np.concatenate([count[1:], count[-1:]]) > 0
  envs = _plot_envs(z, cfg)
  env_pos = np.array([e for e, _ in envs])
  sets = _signal_sets(z, cfg, env_pos)
  for sig in sets:
    if sig.envelope is not None:
      sig.envelope = np.where(alive[:, None, None, None], sig.envelope, np.nan)
  return [label for _, label in envs], _pages(sets)


def plot_actuation(
  npz_path: Path, cfg: PlotValidateConfig, page: int | None = None
) -> list:
  """Position, velocity and force over time of each signal for the traced
  envs cfg.envs picks, with the bounds and the share saturated: a figure
  per leg in each space, then one for the pelvis (only the ``page``-th of
  them when given)."""
  z = np.load(npz_path)
  env_labels, pages = _actuation_pages(z, cfg)
  if page is not None:
    pages = pages[page : page + 1]
  time = z["time"]
  step_dt = float(time[1] - time[0]) if len(time) > 1 else 0.02
  t_end = time + step_dt
  # Red is kept for the velocity limits.
  colors = [
    c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"] if c != "#d62728"
  ]
  figs = []
  for tag, title, sig in pages:
    rows = range(len(sig.names))
    fig, axes = plt.subplots(
      len(rows), 3, figsize=(16, 1.9 * len(rows) + 1.0), sharex=True, squeeze=False
    )
    for r, i in enumerate(rows):
      _plot_signal_row(axes[r], sig, i, t_end, colors, env_labels)
    for q in range(3):
      axes[-1, q].set_xlabel("time [s]")
    legend = {}
    for ax in axes.flat:
      for h, lbl in zip(*ax.get_legend_handles_labels(), strict=True):
        legend.setdefault(lbl, h)
    fig.legend(
      legend.values(), legend.keys(), loc="upper right", frameon=False, fontsize=7
    )
    fig.suptitle(
      f"{npz_path.stem}: {title} (traces of {sig.trace.shape[0]} env(s))",
      fontsize=10,
      x=0.02,
      ha="left",
    )
    # Room on the right for the figure legend.
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    if cfg.save is not None:
      save = Path(cfg.save)
      path = save.with_name(f"{save.stem}_{tag}{save.suffix}")
      fig.savefig(path, dpi=150)
      print(f"[INFO] saved {path}")
    figs.append(fig)
  return figs


def plot_summary(npz_path: Path, cfg: PlotValidateConfig):
  """The result's first figure: velocity tracking, lateral error and success
  rate over time."""
  save = cfg.save
  z = np.load(npz_path)
  summary = _summary(npz_path)
  t, rate = success_rate(z)
  n_env = len(z["failed"])
  has_lateral = "lateral_error_abs_mean" in z
  has_contacts = "illegal_contact_logged" in z and bool(z["illegal_contact_logged"])
  has_finish = _has_finish(z, summary)

  # Velocity tracking on top, lateral error, success rate at the bottom.
  ratios = [3] + [2] * has_lateral + [2]
  fig, axes = plt.subplots(
    len(ratios),
    1,
    figsize=(10, 5 + 3 * (len(ratios) - 1)),
    sharex=True,
    height_ratios=ratios,
    squeeze=False,
  )
  ax = axes[0, 0]
  fade = alive_fraction(z) if cfg.alpha_by_alive else None
  plot_velocity(ax, z, fade)
  if has_lateral:
    log_every = summary.get("config", {}).get("log_every", 1)
    plot_lateral_error(axes[1, 0], z, log_every, fade)
  plot_success_rate(axes[-1, 0], z, has_finish)
  if z["failed"].any():
    first_failure = float(z["survival_time"][z["failed"]].min())
    for a in axes[:, 0]:
      a.axvline(
        first_failure,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="first failure",
      )
    ax.legend(loc="upper left", frameon=False)
  ax.set_xlim(0, t[-1])
  axes[-1, 0].set_xlabel("time [s]")

  title = npz_path.stem
  if "task_id" in summary:
    title += "\n" + "\n".join(textwrap.wrap(summary["task_id"], 100))
  if summary.get("end_to_end"):
    title += "\nend to end, trained on:\n" + "\n".join(
      textwrap.wrap(summary["train_task_id"], 100)
    )
  title += f"\nfinal success rate {100 * rate[-1]:.1f}%"
  if z["failed"].any():
    title += f", first failure at {z['survival_time'][z['failed']].min():.2f} s"
  if has_finish:
    finished = z["finished"]
    title += f"\nfinish rate {100 * finished.mean():.1f}%"
    if finished.any():
      title += f", mean finish time {np.nanmean(z['finish_time']):.2f} s"
  if has_lateral:
    title += (
      f"\nlateral error mean |y| {summary.get('lateral_error_abs_mean', np.nan):.3f} m"
      f", max {summary.get('lateral_error_abs_max', np.nan):.3f} m"
    )
  if has_contacts:
    events = z["illegal_contact_events_per_env"]
    title += (
      f"\nillegal contacts (not terminating): {int((events > 0).sum())}/{n_env} envs"
      f", {int(events.sum())} events"
    )
  for a in axes[:-1, 0]:
    a.set_xlabel("")
  ax.set_title(title, fontsize=10)
  fig.tight_layout()

  if save is not None:
    fig.savefig(save, dpi=150)
    print(f"[INFO] saved {save}")
  return fig


def _save_figure(npz_path: Path, cfg: PlotValidateConfig, page: int | None) -> None:
  """Worker: render and save the summary figure (``page`` None) or one
  actuation page."""
  if page is None:
    figs = [plot_summary(npz_path, cfg)]
  else:
    figs = plot_actuation(npz_path, cfg, page)
  for f in figs:
    plt.close(f)


def save_parallel(results: list[tuple[Path, PlotValidateConfig]], workers: int) -> bool:
  """Render and save every figure of each (result, its config) in a pool of
  ``workers`` forked processes (no re-import per worker), one figure per task.
  A figure that fails is reported and skipped; False if any did."""
  jobs = []
  for npz_path, cfg in results:
    _, pages = _actuation_pages(np.load(npz_path), cfg)
    jobs += [(npz_path, cfg, page) for page in [None, *range(len(pages))]]
  if not jobs:
    return True
  ok = True
  with ProcessPoolExecutor(
    max(1, min(workers, len(jobs))), mp_context=multiprocessing.get_context("fork")
  ) as pool:
    futures = [(job, pool.submit(_save_figure, *job)) for job in jobs]
    for (npz_path, _, page), future in futures:
      try:
        future.result()
      except Exception as e:  # noqa: BLE001 -- keep plotting the rest
        what = "summary" if page is None else f"actuation page {page}"
        print(f"[WARN] plotting failed for {npz_path} ({what}): {e!r}")
        ok = False
  return ok


def plot_batch(cfg: PlotValidateConfig) -> bool:
  """cfg.batch: every matching result saved next to it, in parallel."""
  todo = []
  for npz_path in sorted(Path(p) for p in glob.glob(cfg.batch)):
    png = npz_path.with_suffix(".png")
    newest_input = max(npz_path.stat().st_mtime, Path(__file__).stat().st_mtime)
    if cfg.skip_plotted and png.exists() and png.stat().st_mtime > newest_input:
      print(f"=== {npz_path.name}: already plotted")
      continue
    print(f"=== Plotting {npz_path.name}")
    todo.append((npz_path, dataclasses.replace(cfg, save=str(png), show=False)))
  return save_parallel(todo, cfg.workers)


def plot(npz_path: Path, cfg: PlotValidateConfig) -> None:
  if cfg.save is not None and not cfg.show and cfg.workers > 1:
    if not save_parallel([(npz_path, cfg)], cfg.workers):
      raise SystemExit(1)
    return
  figs = [plot_summary(npz_path, cfg), *plot_actuation(npz_path, cfg)]
  if cfg.show:
    plt.show()
  for f in figs:
    plt.close(f)


def moving_average(x: np.ndarray, size: int) -> np.ndarray:
  """Centred moving average over ``size`` samples that skips NaNs (a window
  averages the finite samples in it); NaN where ``x`` is NaN."""
  finite = np.isfinite(x)
  if size <= 1 or not finite.any():
    return x.astype(float)
  kernel = np.ones(size)
  start = (size - 1) // 2
  total = np.convolve(np.where(finite, x, 0.0), kernel, "full")[start : start + len(x)]
  count = np.convolve(finite.astype(float), kernel, "full")[start : start + len(x)]
  with np.errstate(invalid="ignore", divide="ignore"):
    return np.where(finite, total / count, np.nan)


def _model_label(task_id: str) -> str:
  """The model and transmission a task runs ("full_full LUT", "full
  actuator", "simple") from its id."""
  if "Kangaroo-Full-Full" in task_id:
    model = "full_full"
  elif "Kangaroo-Full" in task_id:
    model = "full"
  else:
    return "simple"
  m = re.search(r"-Transmission-([A-Za-z]+)", task_id)
  if m is None:
    return model
  transmission = m.group(1).lower()
  return f"{model} {'LUT' if transmission == 'lut' else transmission}"


def _result_label(npz_path: Path) -> str:
  """A sweep test's run and checkpoint ("Tendons model_14750") from its
  NN_<stage>_<run>_<model> name (other results keep their file stem), with the
  model and transmission it ran on when its summary names the task
  ("Joints model_15500 on full_full LUT")."""
  label = re.sub(
    r"_(model_\d+)$", r" \1", re.sub(r"^\d+_(?:trained|e2e)_", "", npz_path.stem)
  )
  task_id = _summary(npz_path).get("task_id")
  return label if task_id is None else f"{label} on {_model_label(task_id)}"


def plot_aggregate(npz_paths: list[Path], cfg: PlotValidateConfig):
  """One figure overlaying several results, a colour each: the speed
  (command dashed, mean |v_xy| of the alive envs, or v_x for results without
  per-env v_y, and the min-max band
  over them with min and max each smoothed by a cfg.smooth-step moving
  average), the lateral error (signed mean ± std over the alive envs) and the
  success rate over time."""
  if cfg.lateral:
    fig, (ax_v, ax_y, ax_s) = plt.subplots(
      3, 1, figsize=(11, 12), sharex=True, height_ratios=[3, 2, 2]
    )
  else:
    fig, (ax_v, ax_s) = plt.subplots(
      2, 1, figsize=(11, 9), sharex=True, height_ratios=[3, 2]
    )
    ax_y = None
  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  # Speed |v_xy| where per-env v_y was recorded, v_x otherwise; with a mix,
  # the v_x results say so in the speed legend.
  has_vy = [_has_speed(np.load(p)) for p in npz_paths]
  what = "|v_xy|" if any(has_vy) else "v_x"
  t_ends = []
  for k, npz_path in enumerate(npz_paths):
    z = np.load(npz_path)
    color = colors[k % len(colors)]
    time = z["time"]
    step_dt = float(time[1] - time[0]) if len(time) > 1 else 0.02
    t_end = time + step_dt
    t_ends.append(float(t_end[-1]))
    count = z["alive_count"]
    alive = np.concatenate([count[1:], count[-1:]]) > 0
    t, rate = success_rate(z)
    label = f"{_result_label(npz_path)} ({100 * rate[-1]:.0f}% success)"
    fade = alive_fraction(z) if cfg.alpha_by_alive else None

    cmd = z["cmd_vel_x_max"] if "cmd_vel_x_max" in z else z["cmd_vel_x"]
    if cfg.cmd_of is None:
      ax_v.plot(time, cmd, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
    elif cfg.cmd_of in npz_path.name:
      ax_v.plot(time, cmd, color="black", linewidth=1.2, linestyle="--")
    speed_label = label
    if any(has_vy) and not has_vy[k]:
      speed_label += " (v_x; v_y not recorded)"
    if "vel_x_per_env" in z:
      vel, _ = speed_per_env(z)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # steps no env reached
        lo, hi, mean = np.nanmin(vel, 0), np.nanmax(vel, 0), np.nanmean(vel, 0)
      lo, hi = moving_average(lo, cfg.smooth), moving_average(hi, cfg.smooth)
      _band(ax_v, t_end, lo, hi, fade, 0.15, color=color)
    else:
      col = list(z["state_names"]).index("base_lin_vel_x_b")
      mean = np.where(alive, z["state_mean"][:, col], np.nan)
    _line(ax_v, t_end, mean, fade, color=color, linewidth=2, label=speed_label)

    if ax_y is not None and "state_names" in z and "lateral_error_y" in list(z["state_names"]):
      col = list(z["state_names"]).index("lateral_error_y")
      lat = np.where(alive, z["state_mean"][:, col], np.nan)
      std = np.where(alive, z["state_std"][:, col], np.nan)
      _band(ax_y, t_end, lat - std, lat + std, fade, 0.15, color=color)
      _line(ax_y, t_end, lat, fade, color=color, linewidth=1.5, label=label)

    ax_s.step(t, 100 * rate, where="post", color=color, linewidth=2, label=label)

  # Style keys for what the colours don't say.
  ax_v.plot(
    [],
    [],
    color=GRAY if cfg.cmd_of is None else "black",
    linestyle="--",
    label="cmd v_x = cmd speed (max over alive envs)"
    + ("" if cfg.cmd_of is None else f", {cfg.cmd_of}"),
  )
  ax_v.fill_between(
    [],
    [],
    [],
    color=GRAY,
    alpha=0.2,
    linewidth=0,
    label=f"{what} min-max, alive envs (min, max each {cfg.smooth}-step moving avg.)",
  )
  if cfg.alpha_by_alive:
    ax_v.plot([], [], color=GRAY, linewidth=2, label="faded by share of envs alive")
  ax_v.set_ylim(bottom=0.0)
  quantity = "speed |v_xy|" if what == "|v_xy|" else "forward velocity v_x"
  ax_v.set_ylabel(f"{quantity} [m/s]\n(solid: mean over alive envs)")
  if ax_y is not None:
    ax_y.axhline(0.0, color="#9a9a94", linewidth=0.8)
    ax_y.set_ylabel("lateral error y [m]\n(signed mean ± std, alive envs)")
    ax_y.legend(loc="upper left", frameon=False, fontsize=8)
  ax_s.set_ylim(-2, 102)
  ax_s.set_ylabel("envs not failed [%]")
  ax_s.set_xlabel("time [s]")
  ax_s.set_xlim(0, min(t_ends) if cfg.crop_to_shortest else max(t_ends))
  for ax in (ax_v, ax_y, ax_s):
    if ax is None:
      continue
    ax.grid(True, color="#e2e2dd", linewidth=0.8)
    ax.set_axisbelow(True)
  ax_v.legend(loc="upper left", frameon=False, fontsize=8)
  ax_s.legend(loc="lower left", frameon=False, fontsize=8)
  ax_v.set_title(cfg.title or f"{len(npz_paths)} validate results", fontsize=11)
  fig.tight_layout()
  if cfg.save is not None:
    fig.savefig(cfg.save, dpi=150)
    print(f"[INFO] saved {cfg.save}")
  return fig


def main():
  cfg = tyro.cli(PlotValidateConfig, config=mjlab.TYRO_FLAGS)
  if cfg.batch is not None:
    if not glob.glob(cfg.batch):
      raise SystemExit(f"No results match {cfg.batch!r}.")
    if not plot_batch(cfg):
      raise SystemExit(1)
    return
  if cfg.aggregate is not None:
    paths = sorted(Path(p) for p in glob.glob(cfg.aggregate))
    if not paths:
      raise SystemExit(f"No results match {cfg.aggregate!r}.")
    fig = plot_aggregate(paths, cfg)
    if cfg.show:
      plt.show()
    plt.close(fig)
    return
  results = find_results(cfg)
  if not results:
    raise SystemExit("No matching validate results found.")
  for p in results[: cfg.list]:
    print(f"  {p}  ({_summary(p).get('task_id', '?')})")
  print(f"[INFO] plotting {results[0]}")
  plot(results[0], cfg)


if __name__ == "__main__":
  main()
