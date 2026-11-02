"""Replay the traced envs of a ``validate`` run in viser.

``validate`` stores, for the first ``--trace-envs`` envs, every joint's
position at every step (``trace_joint``) and the base pose (``trace_root``).
This script rebuilds the very scenario the run used (task id and validate
config come from the ``.json`` summary next to the ``.npz``; one env, on the
CPU) so the robot and the terrain match, then writes the recorded qpos frame
by frame and serves it in viser: pick a traced env, scrub or play back.
``--all-envs True`` (or unticking Scene > Environment > Hide others) shows every
traced env at once, each where it ran (``--spread`` spaces them sideways);
the ones that ended earlier stay frozen on their last recorded step.

``--trace-keep extremes`` (the high-speed sweep's default) keeps only up to
four envs per file; for a crowd, validate with ``--trace-keep all`` and a
small ``--trace-envs`` (``TRACE_KEEP=all TRACE_ENVS=64`` for the sweep).

Nothing is simulated: the replay shows the recorded states only, ending at
the env's last recorded step (its termination, finish or the end of the run).

Files written before ``trace_root`` existed hold no base pose; their envs are
replayed with the base pinned at the spawn pose, so only the joint motion is
shown.

    uv run replay-validate logs/rsl_rl/.../validate/flat_ramp_<stamp>.npz
    uv run replay-validate results/.../08_e2e_Joints-DcMotor_model_11750.npz \\
        --env 3 --speed 0.25
    uv run replay-validate <file>.npz --all-envs True --spread 0.6
"""

from __future__ import annotations

import json
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import mjlab
import mujoco
import numpy as np
import tyro
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.viser import MjlabViserScene
from pal_mjlab.scripts.validate import ValidateConfig, build_scenario_env_cfg


def _as_tuples(value):
  """JSON turns tuples into lists; the frozen configs want them back."""
  if isinstance(value, list):
    return tuple(_as_tuples(v) for v in value)
  return value


def _config_from_dict(default, values: dict):
  """Rebuild a (nested, frozen) config dataclass from its ``asdict`` form.
  Keys the current config no longer has are skipped; fields the file
  predates keep their defaults."""
  names = {f.name for f in fields(default)}
  kwargs = {}
  for key, value in values.items():
    if key not in names:
      print(f"[WARN] config field {key!r} no longer exists; ignored")
      continue
    current = getattr(default, key)
    if is_dataclass(current) and isinstance(value, dict):
      kwargs[key] = _config_from_dict(current, value)
    else:
      kwargs[key] = _as_tuples(value)
  return replace(default, **kwargs)


class ValidateReplay:
  def __init__(self, npz_path: Path, port: int):
    self.npz_path = npz_path
    summary_path = npz_path.with_suffix(".json")
    if not summary_path.exists():
      raise FileNotFoundError(
        f"{summary_path} not found: the task id and scenario come from the "
        "validate summary next to the .npz"
      )
    summary = json.loads(summary_path.read_text())
    self.task_id: str = summary["task_id"]

    npz = np.load(npz_path)
    if "trace_joint" not in npz:
      raise ValueError(f"{npz_path} holds no traces (validate ran --trace-envs 0)")
    self.joint_pos = npz["trace_joint"][:, :, 0, :]  # [K, T, J]
    self.root = npz["trace_root"] if "trace_root" in npz else None  # [K, T, 7]
    if self.root is None:
      print(
        "[WARN] no trace_root in this file (written before validate traced the "
        "base pose): the base stays pinned at the spawn pose"
      )
    num_traced = self.joint_pos.shape[0]
    self.env_ids = (
      npz["trace_env_ids"] if "trace_env_ids" in npz else np.arange(num_traced)
    )
    self.time = npz["time"]
    self.step_dt = float(summary["step_dt"])
    self.cmd_vel_x = npz["cmd_vel_x"]
    self.vel_x = npz["vel_x_per_env"][self.env_ids]
    self.termination_names = [str(n) for n in npz["termination_names"]]
    self.termination_idx = npz["termination_idx"][self.env_ids]
    self.failed = npz["failed"][self.env_ids]
    self.finished = npz["finished"][self.env_ids]
    self.survival_time = npz["survival_time"][self.env_ids]
    self.joint_names = [str(n) for n in npz["joint_names"]]
    # Last recorded step of each traced env (the traces are NaN after it).
    valid = ~np.isnan(self.joint_pos).any(-1)
    self.last_frame = np.array(
      [np.flatnonzero(v)[-1] if v.any() else -1 for v in valid]
    )

    self._build_model(summary)
    self.server = viser.ViserServer(port=port, label="Validate Replay")
    # One viser world per traced env; the scene's env_idx is the selected one
    # and its show_only_selected hides the others.
    K, nbody = num_traced, self.mj_model.nbody
    self.scene = MjlabViserScene(self.server, self.mj_model, num_envs=K)
    self.qpos = np.tile(self.spawn_qpos, (K, 1))
    self.xpos = np.zeros((K, nbody, 3))
    self.xmat = np.tile(np.eye(3), (K, nbody, 1, 1))
    self.frame = 0
    self.shown: tuple | None = None
    self.playing = False

  def _build_model(self, summary: dict) -> None:
    """The scenario's env with one env on the CPU: its compiled model has the
    robot and the scenario terrain; its qpos after reset is the spawn pose."""
    cfg = _config_from_dict(ValidateConfig(), summary["config"])
    cfg = replace(cfg, num_envs=1, viewer="none", video=False, trace_envs=0)
    env_cfg, _, _ = build_scenario_env_cfg(self.task_id, cfg)
    print(f"[INFO] building {self.task_id} ({cfg.scenario}) on the CPU")
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    robot = env.scene["robot"]
    if list(robot.joint_names) != self.joint_names:
      raise ValueError(
        "the task's joints no longer match the recorded ones "
        f"({len(robot.joint_names)} vs {len(self.joint_names)} joints)"
      )
    self.mj_model = env.sim.mj_model
    self.mj_data = mujoco.MjData(self.mj_model)
    self.spawn_qpos = env.sim.data.qpos[0].cpu().numpy().copy()
    self.env_origin = env.scene.env_origins[0].cpu().numpy().copy()
    self.free_q_adr = robot.indexing.free_joint_q_adr.cpu().numpy()
    self.joint_q_adr = robot.indexing.joint_q_adr.cpu().numpy()
    env.close()

  # GUI.

  def _env_label(self, i: int) -> str:
    env_id = int(self.env_ids[i])
    if self.finished[i]:
      outcome = "finished"
    elif self.failed[i]:
      outcome = f"failed: {self.termination_names[self.termination_idx[i]]}"
    elif self.termination_idx[i] >= 0:
      outcome = f"truncated: {self.termination_names[self.termination_idx[i]]}"
    else:
      outcome = "survived"
    return f"env {env_id} ({outcome} @ {self.survival_time[i]:.2f} s)"

  @property
  def trace(self) -> int:
    return int(self.scene.env_idx)

  @property
  def show_all(self) -> bool:
    return len(self.env_ids) > 1 and not self.scene.show_only_selected

  def _end_frame(self) -> int:
    """Last step to play: the selected env's last one, or the last of any
    env when they are all shown (the ones that ended earlier stay put)."""
    last = self.last_frame.max() if self.show_all else self.last_frame[self.trace]
    return max(int(last), 0)

  def setup(self, trace: int, speed: float, show_all: bool, spread: float) -> None:
    labels = [self._env_label(i) for i in range(len(self.env_ids))]
    self.scene.env_idx = trace
    self.scene.show_only_selected = not show_all
    self.info = self.server.gui.add_html("")
    with self.server.gui.add_folder("Playback"):
      self.env_dropdown = self.server.gui.add_dropdown(
        "Env", options=labels, initial_value=labels[trace]
      )
      self.slider = self.server.gui.add_slider(
        "Step", min=0, max=self._end_frame(), step=1, initial_value=0
      )
      play = self.server.gui.add_button("Play / pause")
      self.speed = self.server.gui.add_slider(
        "Speed", min=0.05, max=2.0, step=0.05, initial_value=speed
      )
      self.loop = self.server.gui.add_checkbox("Loop", initial_value=True)
      self.spread = self.server.gui.add_slider(
        "Spread (m)",
        min=0.0,
        max=2.0,
        step=0.05,
        initial_value=spread,
        hint="Lateral spacing between the envs when all are shown.",
      )
    self.labels = labels

    @self.env_dropdown.on_update
    def _(_) -> None:
      i = labels.index(self.env_dropdown.value)
      if i != self.trace:
        self.scene._apply_visualization_change(
          lambda: setattr(self.scene, "env_idx", i)
        )

    @self.slider.on_update
    def _(_) -> None:
      self.frame = int(self.slider.value)

    @play.on_click
    def _(_) -> None:
      self.playing = not self.playing

    # With more than one traced env this adds the Environment folder: its
    # "Select" slider and "Hide others" checkbox (off = all envs shown) drive
    # the scene directly, the dropdown above follows them.
    with self.server.gui.add_folder("Scene"):
      self.scene.create_scene_gui(show_debug_viz_control=False)

  def _info_html(self) -> str:
    i, k = self.trace, self.frame
    lines = [
      f"<strong>Task:</strong> {self.task_id}",
      f"<strong>Env:</strong> {self._env_label(i)}",
      f"<strong>t:</strong> {self.time[k] + self.step_dt:.2f} s (step {k})",
      f"<strong>cmd vx:</strong> {self.cmd_vel_x[k]:.2f} m/s",
      f"<strong>vx:</strong> {self.vel_x[i, k]:.2f} m/s",
    ]
    if self.show_all:
      alive = int((self.last_frame >= k).sum())
      lines.append(f"<strong>Recording:</strong> {alive}/{len(self.env_ids)} envs")
    if self.root is None:
      lines.append("<em>no base pose in this file: base pinned at spawn</em>")
    return (
      '<div style="font-size: 0.85em; line-height: 1.3; padding: 0 1em 0.5em 1em;">'
      + "<br/>".join(lines)
      + "</div>"
    )

  def _pose(self, i: int, k: int, offset_y: float) -> None:
    """Env i at step k (its last recorded one once it has ended) into
    self.qpos[i], self.xpos[i] and self.xmat[i]."""
    qpos = self.spawn_qpos.copy()
    k = min(k, int(self.last_frame[i]))
    if k >= 0:
      if self.root is not None:
        root = self.root[i, k].copy()
        root[:3] += self.env_origin
        qpos[self.free_q_adr] = root
      qpos[self.joint_q_adr] = self.joint_pos[i, k]
    qpos[self.free_q_adr[1]] += offset_y
    self.mj_data.qpos[:] = qpos
    mujoco.mj_kinematics(self.mj_model, self.mj_data)
    self.qpos[i] = qpos
    self.xpos[i] = self.mj_data.xpos
    self.xmat[i] = self.mj_data.xmat.reshape(-1, 3, 3)

  def _show(self) -> None:
    sel, k, show_all = self.trace, self.frame, self.show_all
    spread = float(self.spread.value) if show_all else 0.0
    # Spread around the selected env, so it stays where it ran.
    for i in range(len(self.env_ids)) if show_all else (sel,):
      self._pose(i, k, (i - sel) * spread)
    self.scene.update_from_arrays(
      self.xpos,
      self.xmat,
      env_idx=sel,
      qpos=self.qpos,
      qvel=np.zeros((len(self.env_ids), self.mj_model.nv)),
    )
    self.info.content = self._info_html()
    self.shown = (sel, k, show_all, spread)

  def run(self) -> None:
    print("[INFO] open the viser URL above; Ctrl+C exits")
    last = time.monotonic()
    advance = 0.0
    try:
      while True:
        now = time.monotonic()
        if self.playing:
          advance += (now - last) * self.speed.value / self.step_dt
          steps = int(advance)
          advance -= steps
          if steps:
            end = int(self.slider.max)
            frame = self.frame + steps
            if frame > end:
              frame = 0 if self.loop.value else end
              self.playing = self.loop.value
            self.slider.value = frame
            self.frame = frame
        else:
          advance = 0.0
        last = now
        # Follow env selection / "Hide others" changes made in the Scene GUI.
        if self.env_dropdown.value != self.labels[self.trace]:
          self.env_dropdown.value = self.labels[self.trace]
        end = self._end_frame()
        if int(self.slider.max) != end:
          self.slider.max = end
          if self.frame > end:
            self.slider.value = self.frame = end
        spread = float(self.spread.value) if self.show_all else 0.0
        if self.shown != (self.trace, self.frame, self.show_all, spread):
          self._show()
        elif self.scene.needs_update:
          self.scene.refresh_visualization()
        time.sleep(1 / 60)
    except KeyboardInterrupt:
      self.server.stop()


def run_replay(
  npz: tyro.conf.Positional[Path],
  env: int = 0,
  speed: float = 1.0,
  all_envs: bool = False,
  spread: float = 0.0,
  port: int = 8080,
) -> None:
  """Replay a validate .npz in viser.

  Args:
    npz: The .npz validate wrote (its .json summary must sit next to it).
    env: Index into the traced envs to select first (not the env id).
    speed: Playback speed relative to real time.
    all_envs: Show every traced env at once (toggle later with Scene >
      Environment > Hide others); the camera follows the selected one.
    spread: Lateral spacing [m] between the envs when all are shown; 0
      overlays them where they ran.
    port: viser server port.
  """
  # Populate the task registry, as validate does.
  import mjlab.tasks  # noqa: F401

  replay = ValidateReplay(npz, port)
  if not 0 <= env < len(replay.env_ids):
    raise ValueError(f"--env {env}: the file traces {len(replay.env_ids)} envs")
  replay.setup(env, speed, all_envs, spread)
  replay.run()


def main():
  tyro.cli(run_replay, description=__doc__, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
