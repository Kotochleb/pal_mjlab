"""Play a PAL Robotics KANGAROO FULL policy with its axes exposed as tyro flags.

Mirrors ``mjlab.scripts.play``'s CLI; see ``train_kangaroo_full`` for why the
task is chosen via axis flags instead of one of the statically registered
``Mjlab-Velocity-*-Pal-Kangaroo-Full-*`` ids.
"""

import sys

import mjlab
import tyro
from mjlab.scripts.play import PlayConfig, run_play
from pal_mjlab.tasks.velocity.kangaroo_full import (
  KangarooFullAxes,
  register_kangaroo_full_task,
)


def _maybe_print_top_level_help(prog: str) -> None:
  if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
    tyro.cli(KangarooFullAxes, args=["--help"], config=mjlab.TYRO_FLAGS)


def main() -> None:
  _maybe_print_top_level_help("play-kangaroo-full")

  import mjlab.tasks  # noqa: F401

  axes, remaining_args = tyro.cli(
    KangarooFullAxes,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  task_id = register_kangaroo_full_task(axes)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_play(task_id, args)


if __name__ == "__main__":
  main()
