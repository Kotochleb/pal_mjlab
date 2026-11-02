"""Train PAL Robotics KANGAROO FULL with its axes exposed as tyro flags.

Mirrors ``mjlab.scripts.train``'s two-stage CLI, except the task isn't
chosen from a list of ~400 statically registered
``Mjlab-Velocity-*-Pal-Kangaroo-Full-*`` ids: the axes that used to spell
out those ids (terrain, transmission, femur closure, mjcf variant, body,
ankle observation, actuator model) are ordinary flags on a single task,
e.g.::

  train-kangaroo-full --transmission actuator --lower-body True --agent.max-iterations 5000
  train-kangaroo-full --help              # axis choices
  train-kangaroo-full --lower-body True --help   # training/env options for that axis choice
"""

import sys

import mjlab
import tyro
from mjlab.scripts.train import TrainConfig, launch_training
from pal_mjlab.tasks.velocity.kangaroo_full import (
  KangarooFullAxes,
  register_kangaroo_full_task,
)


def _maybe_print_top_level_help(prog: str) -> None:
  if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
    tyro.cli(KangarooFullAxes, args=["--help"], config=mjlab.TYRO_FLAGS)


def main() -> None:
  _maybe_print_top_level_help("train-kangaroo-full")

  # Import tasks to populate the registry (other robots, plus the modules
  # this task's own env/rl cfgs depend on).
  import mjlab.tasks  # noqa: F401

  axes, remaining_args = tyro.cli(
    KangarooFullAxes,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  task_id = register_kangaroo_full_task(axes)

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(task_id),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=task_id, args=args)


if __name__ == "__main__":
  main()
