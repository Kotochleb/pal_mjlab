"""Play a jump policy in Viser with its state machine, rewards and terminations.

Same CLI as ``mjlab.scripts.play`` (checkpoint, W&B run, zero/random agents),
restricted to tasks with a ``JumpHeightCommand`` and always using Viser. On top
of the stock Rewards and Metrics tabs, the viewer adds:

- Jump: the STANCE -> FLIGHT -> LANDED -> SETTLED machine with the current
  state highlighted, its counters and flags, the end of the last episode, and
  plots of every state signal.
- Terminations: which terms fired, how often since start, and their plots.

State is recorded right after the termination manager runs, before the
auto-reset, so the step that ends an episode shows the state it ended in.

  uv run play-jump Mjlab-Jump-Vertical-<variant> --wandb-run-path <run>
  uv run play-jump Mjlab-Jump-Vertical-<variant> --checkpoint-file model.pt
  uv run play-jump Mjlab-Jump-Vertical-<variant> --settle-stage 1 ...

Every episode's jump target is PLAY_TARGET_HEIGHT. Settling uses one stage of
the training settle curriculum (SETTLE_STAGES), the last one unless
--settle-stage picks another (Python indexing, so -1 is the last).
"""

from __future__ import annotations

import dataclasses
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import mjlab.scripts.play as mjlab_play
import tyro
import viser
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer.viser.term_plotter import ViserTermPlotter
from mjlab.viewer.viser.viewer import ViserPlayViewer
from pal_mjlab.tasks.jumping.env_cfgs import SETTLE_STAGES
from pal_mjlab.tasks.jumping.mdp.commands import (
  FLIGHT,
  LANDED,
  SETTLED,
  STANCE,
  JumpHeightCommand,
)
from typing_extensions import override

# World-frame base-link jump target (m) of every played episode.
PLAY_TARGET_HEIGHT = 1.275
# Tasks driven by a JumpHeightCommand (jumping RL and its tracking pretraining).
JUMP_TASK_PREFIXES = ("Mjlab-Jump-", "Mjlab-Tracking-Jump-")
PHASE_NAMES = {STANCE: "STANCE", FLIGHT: "FLIGHT", LANDED: "LANDED", SETTLED: "SETTLED"}
PHASE_COLORS = {
  STANCE: "#7f7f7f",
  FLIGHT: "#1f77b4",
  LANDED: "#ff7f0e",
  SETTLED: "#2ca02c",
}
# Plotted state signals, in display order; booleans plot as 0/1.
STATE_SIGNALS = (
  "phase",
  "height",
  "target_height",
  "peak_height",
  "command",
  "base_vz",
  "air_time",
  "landed_time",
  "settled_time",
  "contact_lost_time",
  "pose_error",
  "both_feet",
  "any_foot",
  "still",
  "settled",
)
FLAGS = (
  "seen_contact",
  "apex_reached",
  "just_takeoff",
  "just_apex",
  "just_touchdown",
)


@dataclass
class JumpSnapshot:
  """One environment's jump state and termination flags after one step."""

  step: int
  env_idx: int
  phase: int
  values: dict[str, float]
  flags: dict[str, bool]
  terminations: dict[str, bool]
  time_outs: dict[str, bool]

  @property
  def fired(self) -> list[str]:
    return [name for name, done in self.terminations.items() if done]


def find_jump_command(env: Any) -> tuple[str, JumpHeightCommand]:
  manager = env.command_manager
  for name in manager.active_terms:
    term = manager.get_term(name)
    if isinstance(term, JumpHeightCommand):
      return name, term
  raise ValueError("The task has no JumpHeightCommand; use mjlab's play instead.")


@dataclass
class JumpStateRecorder:
  """Snapshots the selected env each time the termination manager computes.

  Terminations run after the jump state update and before the auto-reset, so
  a snapshot taken there still holds the state an episode ended in.
  """

  env: Any
  env_idx: int = 0
  max_pending: int = 1000
  pending: deque[JumpSnapshot] = field(init=False)
  command_name: str = field(init=False)
  command: JumpHeightCommand = field(init=False)
  hold_time: float | None = field(init=False)

  def __post_init__(self) -> None:
    self.pending = deque(maxlen=self.max_pending)
    self.command_name, self.command = find_jump_command(self.env)
    manager = self.env.termination_manager
    self.hold_time = None
    if "jump_finished" in manager.active_terms:
      params = manager.get_term_cfg("jump_finished").params
      self.hold_time = params.get("hold_time") or self.command.settle_hold_time
    compute = manager.compute

    def compute_and_record():
      dones = compute()
      self.pending.append(self.snapshot())
      return dones

    manager.compute = compute_and_record

  def snapshot(self) -> JumpSnapshot:
    env, command, i = self.env, self.command, self.env_idx
    data = command.robot.data
    dt = env.step_dt
    found = env.scene[command.cfg.contact_sensor_name].data.found[i] > 0
    values = {
      "phase": command.phase[i],
      "height": command.height[i],
      "target_height": command.target_height[i],
      "peak_height": command.peak_height[i],
      "command": command.command[i, 0],
      "base_vz": data.root_link_lin_vel_w[i, 2],
      "air_time": command.air_time[i],
      "landed_time": command.landed_time[i],
      "settled_time": command.settled_steps[i] * dt,
      "contact_lost_time": command.contact_lost_steps[i] * dt,
      "pose_error": command.pose_error[i],
      "both_feet": found.all(),
      "any_foot": found.any(),
      "still": command.still[i],
      "settled": command.settled[i],
    }
    manager = env.termination_manager
    return JumpSnapshot(
      step=int(env.common_step_counter),
      env_idx=i,
      phase=int(command.phase[i]),
      values={k: float(v) for k, v in values.items()},
      flags={k: bool(getattr(command, k)[i]) for k in FLAGS},
      terminations={
        name: bool(manager.get_term(name)[i]) for name in manager.active_terms
      },
      time_outs={
        name: manager.get_term_cfg(name).time_out for name in manager.active_terms
      },
    )

  def drain(self) -> list[JumpSnapshot]:
    snapshots = list(self.pending)
    self.pending.clear()
    return snapshots


def state_machine_html(snapshot: JumpSnapshot, hold_time: float | None) -> str:
  """The four states as boxes, the current one filled, plus counters/flags."""
  boxes = []
  for phase, name in PHASE_NAMES.items():
    color = PHASE_COLORS[phase]
    active = phase == snapshot.phase
    style = (
      f"background:{color};color:white;font-weight:bold;"
      if active
      else f"color:{color};opacity:0.55;"
    )
    boxes.append(
      f'<span style="{style}border:2px solid {color};border-radius:4px;'
      f'padding:2px 6px;">{name}</span>'
    )
  arrow = '<span style="margin:0 3px;">&rarr;</span>'
  v = snapshot.values
  hold = f" / {hold_time:.2f}" if hold_time is not None else ""
  grace = snapshot.values["contact_lost_time"]
  rows = [
    ("Step", f"{snapshot.step} (env #{snapshot.env_idx})"),
    ("Height / target", f"{v['height']:.3f} / {v['target_height']:.3f} m"),
    (
      "Peak (apex err)",
      f"{v['peak_height']:.3f} m ({v['peak_height'] - v['target_height']:+.3f})",
    ),
    ("Command obs", f"{v['command']:.3f}"),
    ("Base vz", f"{v['base_vz']:+.2f} m/s"),
    ("Air / landed", f"{v['air_time']:.2f} / {v['landed_time']:.2f} s"),
    ("Settled hold", f"{v['settled_time']:.2f}{hold} s"),
    ("One foot off", f"{grace:.2f} s"),
    ("Pose error", f"{v['pose_error']:.2f}"),
  ]
  checks = {
    "both feet": v["both_feet"] > 0,
    "still": v["still"] > 0,
    "settled": v["settled"] > 0,
    **snapshot.flags,
  }
  table = "".join(
    f"<tr><td style='padding-right:8px;'><strong>{k}</strong></td><td>{val}</td></tr>"
    for k, val in rows
  )
  ticks = " ".join(
    f'<span style="color:{"#2ca02c" if on else "#aaa"};">'
    f"{'&#9679;' if on else '&#9675;'} {name}</span>"
    for name, on in checks.items()
  )
  return (
    '<div style="font-size:0.85em;line-height:1.5;padding:0 1em 0.5em 1em;">'
    f"<div>{arrow.join(boxes)}</div>"
    f'<table style="margin-top:6px;">{table}</table>'
    f'<div style="margin-top:4px;">{ticks}</div></div>'
  )


def episode_end_html(snapshot: JumpSnapshot | None) -> str:
  if snapshot is None:
    return '<div style="font-size:0.85em;padding:0 1em;"><em>No episode ended yet.</em></div>'
  kinds = ", ".join(
    f"{name} ({'time-out' if snapshot.time_outs[name] else 'failure'})"
    for name in snapshot.fired
  )
  v = snapshot.values
  return (
    '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
    f"<strong>Step {snapshot.step}:</strong> {kinds}<br/>"
    f"in <strong>{PHASE_NAMES[snapshot.phase]}</strong>, apex "
    f"{v['peak_height']:.3f} m for target {v['target_height']:.3f} m</div>"
  )


def termination_html(
  snapshot: JumpSnapshot | None, counts: Counter[str], episodes: int
) -> str:
  if snapshot is None:
    return ""
  rows = []
  for name, done in snapshot.terminations.items():
    kind = "time-out" if snapshot.time_outs[name] else "failure"
    color = "#2ca02c" if snapshot.time_outs[name] else "#d62728"
    mark = f'<span style="color:{color};">&#9679;</span>' if done else "&#9675;"
    share = f"{100 * counts[name] / episodes:.0f}%" if episodes else "-"
    rows.append(
      f"<tr><td>{mark} {name}</td><td style='padding:0 8px;'>{kind}</td>"
      f"<td>{counts[name]}</td><td style='padding-left:8px;'>{share}</td></tr>"
    )
  return (
    '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
    f"Episodes ended: {episodes}<table>"
    "<tr><th align=left>term</th><th align=left>kind</th><th>count</th>"
    "<th>share</th></tr>" + "".join(rows) + "</table></div>"
  )


def _param_text(value: Any) -> str:
  # SceneEntityCfg and similar configs: show the entity and what it selects.
  if hasattr(value, "name") and not isinstance(value, str):
    selected = [
      f"{key}={list(getattr(value, key))}"
      for key in ("body_names", "joint_names", "geom_names", "site_names")
      if getattr(value, key, None)
    ]
    return value.name + (f" ({', '.join(selected)})" if selected else "")
  return str(value)


def enabled_terminations_html(manager: Any) -> str:
  """Every configured termination term: function, kind and parameters."""
  rows = []
  for name in manager.active_terms:
    cfg = manager.get_term_cfg(name)
    kind = "time-out" if cfg.time_out else "failure"
    color = "#2ca02c" if cfg.time_out else "#d62728"
    func = getattr(cfg.func, "__name__", type(cfg.func).__name__)
    params = "<br/>".join(
      f"{key} = {_param_text(value)}"
      for key, value in cfg.params.items()
      if key != "command_name"
    )
    rows.append(
      f'<tr style="vertical-align:top;"><td><strong>{name}</strong><br/>'
      f"<small>{func}()</small></td>"
      f'<td style="padding:0 8px;color:{color};">{kind}</td>'
      f"<td><small>{params or '-'}</small></td></tr>"
    )
  return (
    '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
    f"{len(rows)} enabled<table>" + "".join(rows) + "</table></div>"
  )


class JumpViserPlayViewer(ViserPlayViewer):
  """ViserPlayViewer with Jump state-machine and Terminations tabs."""

  @override
  def setup(self) -> None:
    # The stock setup keeps its tab group local; capture it to add our tabs.
    gui = self._server.gui
    add_tab_group = gui.add_tab_group
    tabs_box: list[Any] = []

    def capture(*args, **kwargs):
      tabs = add_tab_group(*args, **kwargs)
      tabs_box.append(tabs)
      return tabs

    gui.add_tab_group = capture
    try:
      super().setup()
    finally:
      gui.add_tab_group = add_tab_group
    tabs = tabs_box[0]

    env = self.env.unwrapped
    self._recorder = JumpStateRecorder(env, env_idx=self._scene.env_idx)
    self._last_snapshot: JumpSnapshot | None = None
    self._last_end: JumpSnapshot | None = None
    self._end_counts: Counter[str] = Counter()
    self._episodes = 0
    self._jump_env_idx = self._scene.env_idx

    with tabs.add_tab("Jump", icon=viser.Icon.ARROW_BIG_UP):
      with gui.add_folder("State machine"):
        self._state_html = gui.add_html("")
      with gui.add_folder("Last episode end"):
        self._end_html = gui.add_html(episode_end_html(None))
      self._state_plotter = ViserTermPlotter(
        self._server, list(STATE_SIGNALS), name="State", env_idx=self._jump_env_idx
      )
    with tabs.add_tab("Terminations", icon=viser.Icon.PLAYER_STOP):
      with gui.add_folder("Enabled terminations"):
        gui.add_html(enabled_terminations_html(env.termination_manager))
      with gui.add_folder("Fired"):
        self._termination_html = gui.add_html("")
      self._termination_plotter = ViserTermPlotter(
        self._server,
        list(env.termination_manager.active_terms),
        name="Termination",
        env_idx=self._jump_env_idx,
      )

  def _clear_jump_panels(self) -> None:
    self._recorder.pending.clear()
    self._last_end = None
    self._end_counts.clear()
    self._episodes = 0
    self._end_html.content = episode_end_html(None)
    for plotter in (self._state_plotter, self._termination_plotter):
      plotter.clear_histories()
      plotter.update_env_idx(self._scene.env_idx)

  def _update_jump_panels(self) -> None:
    if self._scene.env_idx != self._jump_env_idx:
      self._jump_env_idx = self._scene.env_idx
      self._recorder.env_idx = self._jump_env_idx
      self._clear_jump_panels()
    snapshots = self._recorder.drain()
    for snapshot in snapshots:
      self._state_plotter.update(
        [(name, [snapshot.values[name]]) for name in STATE_SIGNALS]
      )
      self._termination_plotter.update(
        [(name, [float(done)]) for name, done in snapshot.terminations.items()]
      )
      if snapshot.fired:
        self._episodes += 1
        self._end_counts.update(snapshot.fired)
        self._last_end = snapshot
        self._end_html.content = episode_end_html(snapshot)
    if snapshots:
      self._last_snapshot = snapshots[-1]
    if self._last_snapshot is not None:
      self._state_html.content = state_machine_html(
        self._last_snapshot, self._recorder.hold_time
      )
      self._termination_html.content = termination_html(
        self._last_snapshot, self._end_counts, self._episodes
      )

  @override
  def _update_env_dependent_plots(self) -> None:
    super()._update_env_dependent_plots()
    self._update_jump_panels()

  @override
  def reset_environment(self) -> None:
    super().reset_environment()
    self._clear_jump_panels()


@dataclass(frozen=True)
class PlayJumpConfig(PlayConfig):
  settle_stage: int = -1
  """Index into SETTLE_STAGES (joint speed limit, hold time) used to settle."""


def apply_settle_stage(cfg: Any, stage: int) -> tuple[float, float]:
  """Set the jump command's settle speed limit and hold time to SETTLE_STAGES[stage]."""
  if not -len(SETTLE_STAGES) <= stage < len(SETTLE_STAGES):
    raise ValueError(
      f"settle_stage must index the {len(SETTLE_STAGES)} SETTLE_STAGES, got {stage}."
    )
  _, speed, hold_time = SETTLE_STAGES[stage]
  command = cfg.commands["jump_height"]
  command.settle_joint_speed = speed
  command.settle_hold_time = hold_time
  return speed, hold_time


def main() -> None:
  maybe_print_top_level_help("play-jump")
  import mjlab.tasks  # noqa: F401

  jump_tasks = [task for task in list_tasks() if task.startswith(JUMP_TASK_PREFIXES)]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(jump_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayJumpConfig,
    args=remaining_args,
    default=PlayJumpConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  args = dataclasses.replace(args, viewer="viser")

  def load_play_env_cfg(task_id: str, play: bool = False):
    cfg = load_env_cfg(task_id, play=play)
    cfg.commands["jump_height"].height_range = (PLAY_TARGET_HEIGHT,) * 2
    speed, hold_time = apply_settle_stage(cfg, args.settle_stage)
    print(
      f"[play-jump] settle stage {args.settle_stage}: joint speed < {speed} rad/s"
      f" held {hold_time} s"
    )
    return cfg

  # run_play loads the env config by module-global name.
  mjlab_play.load_env_cfg = load_play_env_cfg
  # run_play builds mjlab_play.ViserPlayViewer by module-global name.
  mjlab_play.ViserPlayViewer = JumpViserPlayViewer
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
