"""Silence selected MuJoCo Warp overflow prints for every mjlab Simulation.

Since mujoco-warp 3.16, every overflow type prints from inside the GPU kernels
by default (``Option.warn_overflow = OverflowType.ALL``), including the two
solver *iteration caps*::

    linesearch iterations limit reached - please increase ls_iterations beyond 20
    solver iterations limit reached - please increase iterations beyond 10

Those are not buffer overflows: the affected world just converged less
tightly, exactly as it did silently on mujoco-warp 3.11. mujoco_warp's own G1
benchmarks hide them the same way (mujoco_warp PR #1626). The per-world flag is
still recorded in ``data.overflow`` whatever the mask, so it can be logged.

mjlab's ``SimulationCfg`` does not expose ``warn_overflow``, and mujoco_warp
bakes the mask into the solver kernels when they are built, so it must be set
after ``put_model`` and before the CUDA graph is captured. ``install()`` wraps
``SimulationCfg.apply_wp_opt``, which mjlab calls exactly there for both the
plain and the per-world-variant model paths.

Which prints are silenced is controlled by the ``PAL_MJLAB_SUPPRESS_OVERFLOW``
environment variable, a comma-separated list of ``mujoco_warp.OverflowType``
names (case-insensitive):

* unset                       -> ``ITERATIONS,LS_ITERATIONS`` (the default)
* ``LS_ITERATIONS``           -> only the linesearch print
* ``ALL``                     -> no overflow prints at all
* ``NONE`` or empty string    -> mujoco_warp default, every print enabled
"""

from __future__ import annotations

import functools
import os

import mujoco_warp as mjwarp
from mjlab.sim.sim import SimulationCfg

ENV_VAR = "PAL_MJLAB_SUPPRESS_OVERFLOW"
DEFAULT_SUPPRESSED = "ITERATIONS,LS_ITERATIONS"

_PATCH_MARKER = "_pal_mjlab_overflow_patch"


def parse_suppressed_overflow(spec: str) -> mjwarp.OverflowType:
  """Parse a comma-separated list of ``OverflowType`` names into one mask."""
  mask = mjwarp.OverflowType.NONE
  for raw in spec.split(","):
    name = raw.strip().upper()
    if not name:
      continue
    try:
      mask |= mjwarp.OverflowType[name]
    except KeyError:
      valid = ", ".join(f.name for f in mjwarp.OverflowType)
      raise ValueError(
        f"{ENV_VAR}: unknown OverflowType {raw.strip()!r}; valid names: {valid}"
      ) from None
  return mask


def suppressed_overflow_mask() -> mjwarp.OverflowType:
  """The mask of overflow prints to silence, read from the environment."""
  return parse_suppressed_overflow(os.environ.get(ENV_VAR, DEFAULT_SUPPRESSED))


def install() -> None:
  """Wrap ``SimulationCfg.apply_wp_opt`` to clear the suppressed print bits.

  Idempotent. The environment variable is read on every Simulation build, so
  it can be changed between envs in one process.
  """
  original = SimulationCfg.apply_wp_opt
  if getattr(original, _PATCH_MARKER, False):
    return

  @functools.wraps(original)
  def apply_wp_opt(self: SimulationCfg, wp_opt: mjwarp.Option) -> None:
    original(self, wp_opt)
    wp_opt.warn_overflow = int(wp_opt.warn_overflow) & ~int(
      suppressed_overflow_mask()
    )

  setattr(apply_wp_opt, _PATCH_MARKER, True)
  SimulationCfg.apply_wp_opt = apply_wp_opt
