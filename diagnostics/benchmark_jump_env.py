"""Time env.step of a jumping task, optionally per manager and with pieces disabled.

Builds the task's training env (as `train` does, without the learner), steps it
with zero or random actions and reports the mean wall time per env.step, env
steps per second, and the implied rollout time of one PPO iteration
(num_steps_per_env steps of the task's rl config). Learning time is not
included: compare it with the "collection" time rsl_rl prints per iteration.

--breakdown also times each part of env.step separately (physics, the
managers, resets). On CUDA every timed part is wrapped in
torch.cuda.synchronize(), which removes the overlap between the CPU and GPU, so
the breakdown is slower in total than the plain run; use it to find which part
grew, and the plain run for the absolute number. Parts are nested: "reset"
includes the reset-mode events and every manager's reset.

--disable drops new jump-env pieces to compare against: the replay event
(jump_state_replay), the height curriculum (jump_height), the settle
curriculum (settle) or the pose reward (pose).

Examples (on a GPU node, matching slurm_pfc_jump_from_wandb.bash):
  uv run python diagnostics/benchmark_jump_env.py --device cuda:0 \\
    --num-envs 4096 --timestep 0.005 --decimation 4 --graph-conditional false
  uv run python diagnostics/benchmark_jump_env.py --device cuda:0 \\
    --num-envs 4096 --breakdown --disable replay height_curriculum
Locally on CPU:
  uv run python diagnostics/benchmark_jump_env.py --num-envs 64 --steps 50
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import pal_mjlab.tasks  # noqa: F401  (registers every task)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

DISABLE = {
  "replay": ("events", "jump_state_replay"),
  "height_curriculum": ("curriculum", "jump_height"),
  "settle_curriculum": ("curriculum", "settle"),
  "pose": ("rewards", "pose"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--task", default="Mjlab-Jump-Vertical-Pal-Kangaroo-Full")
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=200, help="timed env steps")
  parser.add_argument("--warmup", type=int, default=50, help="untimed env steps")
  parser.add_argument(
    "--actions",
    choices=("zero", "random"),
    default="zero",
    help="zero: stand still; random: uniform in [-1, 1] (falls, many resets)",
  )
  parser.add_argument("--timestep", type=float, help="override sim timestep (s)")
  parser.add_argument("--decimation", type=int, help="override decimation")
  parser.add_argument(
    "--graph-conditional",
    choices=("true", "false"),
    help="override env.sim.graph_conditional",
  )
  parser.add_argument("--disable", nargs="*", default=[], choices=sorted(DISABLE))
  parser.add_argument(
    "--breakdown", action="store_true", help="also time each part of env.step"
  )
  return parser.parse_args()


def build_env(args: argparse.Namespace) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  if args.timestep is not None:
    cfg.sim.mujoco.timestep = args.timestep
  if args.decimation is not None:
    cfg.decimation = args.decimation
  if args.graph_conditional is not None:
    cfg.sim.graph_conditional = args.graph_conditional == "true"
  for name in args.disable:
    group, key = DISABLE[name]
    if getattr(cfg, group).pop(key, None) is None:
      raise SystemExit(f"--disable {name}: {args.task} has no {group}[{key!r}].")
  return ManagerBasedRlEnv(cfg, device=args.device)


class Timers:
  """Wraps bound methods so every call adds its (synchronized) wall time."""

  def __init__(self, device: str):
    self.cuda = device.startswith("cuda")
    self.total: dict[str, float] = defaultdict(float)
    self.calls: dict[str, int] = defaultdict(int)
    self.enabled = False

  @contextmanager
  def section(self, name: str):
    if not self.enabled:
      yield
      return
    if self.cuda:
      torch.cuda.synchronize()
    start = time.perf_counter()
    try:
      yield
    finally:
      if self.cuda:
        torch.cuda.synchronize()
      self.total[name] += time.perf_counter() - start
      self.calls[name] += 1

  def wrap(self, owner, method: str, name: str, key_arg: str | None = None) -> None:
    original = getattr(owner, method)

    def timed(*args, **kwargs):
      label = name if key_arg is None else f"{name}[{kwargs.get(key_arg)}]"
      with self.section(label):
        return original(*args, **kwargs)

    setattr(owner, method, timed)


def instrument(env: ManagerBasedRlEnv, timers: Timers) -> None:
  timers.wrap(env.action_manager, "process_action", "action process")
  timers.wrap(env.action_manager, "apply_action", "action apply (per substep)")
  timers.wrap(env.sim, "step", "physics step (per substep)")
  timers.wrap(env.sim, "forward", "physics forward")
  timers.wrap(env.sim, "sense", "sensors")
  timers.wrap(env.scene, "write_data_to_sim", "scene write to sim")
  timers.wrap(env.scene, "update", "scene update (per substep)")
  timers.wrap(env.metrics_manager, "compute_substep", "metrics (per substep)")
  timers.wrap(env.termination_manager, "compute", "terminations")
  timers.wrap(env.reward_manager, "compute", "rewards")
  timers.wrap(env.metrics_manager, "compute", "metrics")
  timers.wrap(env.event_manager, "apply", "events", key_arg="mode")
  timers.wrap(env, "_reset_idx", "reset (nested)")
  timers.wrap(env.command_manager, "compute", "commands")
  timers.wrap(env.observation_manager, "compute", "observations")


def run(env, args, steps: int, timers: Timers | None) -> tuple[float, int]:
  """Mean seconds per env.step over `steps`, and the number of episode ends."""
  dim = env.action_manager.total_action_dim
  generator = torch.Generator(device=env.device).manual_seed(0)
  sync = torch.cuda.synchronize if env.device.startswith("cuda") else lambda: None
  resets = 0
  sync()
  start = time.perf_counter()
  for _ in range(steps):
    if args.actions == "zero":
      actions = torch.zeros(env.num_envs, dim, device=env.device)
    else:
      actions = torch.rand(env.num_envs, dim, device=env.device, generator=generator)
      actions = 2 * actions - 1
    if timers is None:
      env.step(actions)
    else:
      with timers.section("env.step (total)"):
        env.step(actions)
    resets += int(env.reset_buf.sum())
  sync()
  return (time.perf_counter() - start) / steps, resets


def main() -> None:
  args = parse_args()
  env = build_env(args)
  try:
    steps_per_iteration = load_rl_cfg(args.task).num_steps_per_env
    print(
      f"{args.task} on {args.device}: {env.num_envs} envs, "
      f"physics dt {env.physics_dt:g} s x decimation {env.cfg.decimation}, "
      f"{args.actions} actions, disabled: {args.disable or 'nothing'}"
    )
    env.reset()
    run(env, args, args.warmup, None)
    per_step, resets = run(env, args, args.steps, None)
    print(
      f"env.step: {per_step * 1e3:.2f} ms  |  "
      f"{env.num_envs / per_step:,.0f} env steps/s  |  "
      f"rollout of one PPO iteration ({steps_per_iteration} steps): "
      f"{per_step * steps_per_iteration:.2f} s  |  "
      f"episode ends per step: {resets / args.steps:.1f}"
    )
    if args.breakdown:
      timers = Timers(args.device)
      instrument(env, timers)
      timers.enabled = True
      run(env, args, args.steps, timers)
      total = timers.total["env.step (total)"]
      print(f"\nBreakdown over {args.steps} synchronized steps (ms per env.step):")
      for name, seconds in sorted(timers.total.items(), key=lambda kv: -kv[1]):
        print(
          f"  {name:34s} {seconds / args.steps * 1e3:9.2f}"
          f"  {100 * seconds / total:5.1f}%  ({timers.calls[name]} calls)"
        )
  finally:
    env.close()


if __name__ == "__main__":
  main()
