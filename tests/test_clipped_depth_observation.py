"""Tests for the `clipped_depth` observation term.

The sensor is stock mjlab, so the only thing here that is ours is the miss-fill and the
far clamp. Both are silent when wrong: a raw 0.0 miss reads as a surface against the
lens, and an unclamped hit breaks the caller's 1 / max_depth scaling.
"""

import math
import types

import torch
from conftest import get_test_device, make_scene_and_sim
from mjlab.sensor import CameraSensorCfg
from pal_mjlab.tasks.velocity.mdp import clipped_depth

ROWS, COLS = 12, 16
CAM_HEIGHT = 1.5
FOVY_DEG = 60.0
# Pitched enough that the top rows clear the horizon (misses) while the bottom rows hit
# the ground inside MAX_DEPTH, so one scene exercises both branches.
PITCH_DEG = 30.0
MAX_DEPTH = 2.0

SCENE_XML = f"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" pos="0 0 0"/>
    <body name="base" pos="0 0 {CAM_HEIGHT}">
      <geom name="base_geom" type="box" size="0.05 0.05 0.05" mass="1.0"/>
      <camera name="cam" mode="fixed" resolution="{COLS} {ROWS}" fovy="{FOVY_DEG}"
              xyaxes="0 -1 0  {math.sin(math.radians(PITCH_DEG))} 0
                      {math.cos(math.radians(PITCH_DEG))}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _observe(device: str) -> torch.Tensor:
  cfg = CameraSensorCfg(
    name="low_res_depth_cam",
    camera_name="robot/cam",
    width=COLS,
    height=ROWS,
    data_types=("depth",),
  )
  scene, sim = make_scene_and_sim(device, SCENE_XML, sensors=(cfg,))
  sim.forward()
  sim.sense()
  env = types.SimpleNamespace(scene=scene)
  return clipped_depth(env, "low_res_depth_cam", max_depth=MAX_DEPTH)  # type: ignore[arg-type]


def test_clipped_depth_is_bounded_and_fills_misses() -> None:
  device = get_test_device()
  obs = _observe(device)

  assert obs.shape == (1, ROWS * COLS)
  assert torch.isfinite(obs).all()
  assert (obs > 0.0).all(), "a raw 0.0 miss reached the policy"
  assert (obs <= MAX_DEPTH).all()

  grid = obs.view(ROWS, COLS)
  # Top rows clear the horizon, so they must read exactly max_depth; bottom rows see
  # ground well inside it. Both branches present means the fill is doing something.
  assert torch.allclose(grid[0], torch.full_like(grid[0], MAX_DEPTH))
  assert (grid[-1] < MAX_DEPTH).all()
