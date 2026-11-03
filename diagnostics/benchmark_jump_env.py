"""Time env.step of a jumping task driven by a trained policy, optionally per
manager and with pieces disabled.

Builds the task's training env (as `train` does, without the learner) and steps
it with a policy loaded from a checkpoint (--checkpoint), or with zero/random
actions. It reports the mean wall time per env.step, env steps per second, the
implied rollout time of one PPO iteration (num_steps_per_env steps of the
task's rl config), the policy inference time, how episodes ended, and the
replay buffer size. Learning time is not included: compare the rollout time
with the "collection" time rsl_rl prints per iteration.

With a policy, the run covers whole jumps and a filled replay buffer:
  * Loading a jump checkpoint restores its step counter and the height
    curriculum's progress (JumpOnPolicyRunner), so targets match that point in
    training. An actor-only (tracking) checkpoint starts at the initial range.
  * Warm-up runs until the replay buffer holds --min-replay-states states
    (episodes that finished their jump), or for --max-warmup steps, so the
    timed steps include restarts from replayed states.
  * The timed window (--steps, default 1000 = 20 s at a 0.02 s policy step)
    spans several full episodes: takeoff, flight, landing, settling, reset.
Zero or random actions never jump, so the buffer stays empty.

--breakdown also times each part of env.step separately (physics, the
managers, resets, policy inference). On CUDA every timed part is wrapped in
torch.cuda.synchronize(), which removes the overlap between the CPU and GPU, so
the breakdown is slower in total than the plain run; use it to find which part
grew, and the plain run for the absolute number. Parts are nested: "reset"
includes the reset-mode events and every manager's reset.

--disable drops new jump-env pieces to compare against: the replay event
(jump_state_replay), the height curriculum (jump_height), the settle
curriculum (settle) or the pose reward (pose).

Examples (on a GPU node, matching the jump slurm scripts):
  uv run python diagnostics/benchmark_jump_env.py --device cuda:0 \\
    --num-envs 4096 --timestep 0.005 --decimation 4 --graph-conditional false \\
    --checkpoint path/to/model_5000.pt
  uv run python diagnostics/benchmark_jump_env.py --device cuda:0 \\
    --checkpoint path/to/model_5000.pt --breakdown --disable replay
Locally on CPU, without a policy:
  uv run python diagnostics/benchmark_jump_env.py --num-envs 64 --steps 50
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  (registers every task)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

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
  parser.add_argument(
    "--checkpoint", type=Path, help="policy checkpoint (.pt); else --actions"
  )
  parser.add_argument(
    "--actions",
    choices=("zero", "random"),
    default="zero",
    help="without --checkpoint: zero (stand still) or uniform in [-1, 1]",
  )
  parser.add_argument("--steps", type=int, default=1000, help="timed env steps")
  parser.add_argument(
    "--min-warmup", type=int, default=50, help="untimed env steps, at least"
  )
  parser.add_argument(
    "--max-warmup",
    type=int,
    default=1500,
    help="untimed env steps, at most, while the replay buffer fills",
  )
  parser.add_argument(
    "--min-replay-states",
    type=int,
    default=512,
    help="warm up until the replay buffer holds this many states",
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


def load_policy(args, vec_env: RslRlVecEnvWrapper, agent_cfg):
  """The checkpoint's inference policy, loaded as training resumes it.

  JumpOnPolicyRunner without a load_cfg: a jump checkpoint resumes completely
  (step counter and curriculum state included); a checkpoint of another task
  (e.g. the stage 1 tracking policy) loads only its actor and keeps this env's
  own step counter, so the curricula start at their first stage. (A load_cfg,
  as `play` passes, would restore the tracking run's step counter instead.)
  """
  if not args.checkpoint.exists():
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(str(args.checkpoint), map_location=args.device)
  return runner.get_inference_policy(device=args.device)


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


def replay_term(env: ManagerBasedRlEnv):
  if "jump_state_replay" not in env.event_manager.active_terms.get("step", []):
    return None
  return env.event_manager.get_term_cfg("jump_state_replay").func


def replay_size(env: ManagerBasedRlEnv) -> int | None:
  term = replay_term(env)
  return None if term is None else term.size


class Driver:
  """Steps the wrapped env with the policy (or zero/random actions)."""

  def __init__(self, args, vec_env: RslRlVecEnvWrapper, policy):
    self.vec_env = vec_env
    self.env: ManagerBasedRlEnv = vec_env.unwrapped
    self.policy = policy
    self.mode = "policy" if policy is not None else args.actions
    self.generator = torch.Generator(device=self.env.device).manual_seed(0)
    self.obs = vec_env.get_observations()
    terminations = self.env.termination_manager
    self.term_names = list(terminations.active_terms)
    self.ends = torch.zeros(len(self.term_names), device=self.env.device)
    self.sync = (
      torch.cuda.synchronize if self.env.device.startswith("cuda") else lambda: None
    )

  def actions(self, timers: Timers | None) -> torch.Tensor:
    env = self.env
    if self.policy is not None:
      if timers is None:
        return self.policy(self.obs)
      with timers.section("policy inference"):
        return self.policy(self.obs)
    shape = (env.num_envs, env.action_manager.total_action_dim)
    if self.mode == "zero":
      return torch.zeros(shape, device=env.device)
    return 2 * torch.rand(shape, device=env.device, generator=self.generator) - 1

  def step(self, timers: Timers | None) -> None:
    actions = self.actions(timers)
    if timers is None:
      self.obs, *_ = self.vec_env.step(actions)
    else:
      with timers.section("env.step (total)"):
        self.obs, *_ = self.vec_env.step(actions)
    terminations = self.env.termination_manager
    # Every term is computed every step (unlike the reset ids, which are only
    # the envs that actually ended); stays on the device, no sync per step.
    for i, name in enumerate(self.term_names):
      self.ends[i] += terminations.get_term(name).sum()

  def warm_up(self, args) -> int:
    """Untimed steps until the replay buffer holds min_replay_states states."""
    steps = 0
    while steps < args.max_warmup:
      self.step(None)
      steps += 1
      if steps < args.min_warmup:
        continue
      size = replay_size(self.env)
      if self.policy is None or size is None or size >= args.min_replay_states:
        break
    return steps

  @torch.no_grad()
  def run(self, steps: int, timers: Timers | None) -> float:
    """Mean seconds per (policy + env) step over `steps`."""
    self.ends.zero_()
    self.sync()
    start = time.perf_counter()
    for _ in range(steps):
      self.step(timers)
    self.sync()
    return (time.perf_counter() - start) / steps

  def episode_ends(self) -> str:
    counts = self.ends.tolist()
    total = sum(counts) or 1.0
    return ", ".join(
      f"{name} {count:.0f} ({100 * count / total:.0f}%)"
      for name, count in zip(self.term_names, counts, strict=True)
      if count
    )


def describe_curriculum(env: ManagerBasedRlEnv) -> str:
  command = env.command_manager.get_term("jump_height")
  low, high = command.height_range
  return (
    f"step counter {env.common_step_counter}, target apex {low:.3f}-{high:.3f} m, "
    f"settle {command.settle_joint_speed:.2f} rad/s for "
    f"{command.settle_hold_time:.2f} s"
  )


def main() -> None:
  args = parse_args()
  agent_cfg = load_rl_cfg(args.task)
  env = build_env(args)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    policy = load_policy(args, vec_env, agent_cfg) if args.checkpoint else None
    driver = Driver(args, vec_env, policy)
    source = f"policy {args.checkpoint}" if policy is not None else args.actions
    print(
      f"{args.task} on {args.device}: {env.num_envs} envs, "
      f"physics dt {env.physics_dt:g} s x decimation {env.cfg.decimation} "
      f"(policy dt {env.step_dt:g} s), actions: {source}, "
      f"disabled: {args.disable or 'nothing'}"
    )
    with torch.no_grad():
      warmup = driver.warm_up(args)
    size = replay_size(env)
    print(
      f"warm-up: {warmup} steps ({warmup * env.step_dt:.1f} s), replay buffer "
      f"{'disabled' if size is None else f'{size} states'}; {describe_curriculum(env)}"
    )
    if policy is not None and size is not None and size < args.min_replay_states:
      print(
        f"[WARN] replay buffer below --min-replay-states={args.min_replay_states} "
        f"after --max-warmup={args.max_warmup} steps; few jumps finish."
      )
    steps_per_iteration = agent_cfg.num_steps_per_env
    per_step = driver.run(args.steps, None)
    print(
      f"env.step: {per_step * 1e3:.2f} ms  |  "
      f"{env.num_envs / per_step:,.0f} env steps/s  |  "
      f"rollout of one PPO iteration ({steps_per_iteration} steps): "
      f"{per_step * steps_per_iteration:.2f} s  |  "
      f"timed {args.steps} steps ({args.steps * env.step_dt:.1f} s)"
    )
    print(f"episode ends: {driver.episode_ends() or 'none'}")
    size = replay_size(env)
    print(
      f"after: replay buffer {'disabled' if size is None else f'{size} states'}; "
      f"{describe_curriculum(env)}"
    )
    if args.breakdown:
      timers = Timers(args.device)
      instrument(env, timers)
      timers.enabled = True
      driver.run(args.steps, timers)
      total = timers.total["env.step (total)"] + timers.total["policy inference"]
      print(f"\nBreakdown over {args.steps} synchronized steps (ms per step):")
      for name, seconds in sorted(timers.total.items(), key=lambda kv: -kv[1]):
        print(
          f"  {name:34s} {seconds / args.steps * 1e3:9.2f}"
          f"  {100 * seconds / total:5.1f}%  ({timers.calls[name]} calls)"
        )
  finally:
    vec_env.close()


if __name__ == "__main__":
  main()
