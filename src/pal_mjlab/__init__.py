from pathlib import Path

PAL_MJLAB_SRC_PATH: Path = Path(__file__).parent

# Anything this module exports must be defined ABOVE the `import mjlab` below.
#
# mjlab runs its "mjlab.tasks" entry points from inside its own __init__, and
# ours points at `pal_mjlab.tasks`, which imports `pal_mjlab.robots` and the
# rest of this package. So whichever pal_mjlab module first pulls in mjlab
# gets that whole tree imported on top of itself, mid-body -- and if that
# module was `pal_mjlab.robots` (e.g. `import pal_mjlab.robots`, or running a
# robot constants module with `python -m`), the re-entrant import finds it
# half-initialized and fails. mjlab swallows the ImportError as a `[WARN]`, so
# the symptom is silent: the process runs on with an empty task registry.
#
# Importing mjlab here, from the package root, makes that scan happen while
# only this file is on the import stack -- and this file is already done
# defining everything anyone reads from it. Every pal_mjlab submodule then
# starts with mjlab and the task tree fully imported, so import order stops
# mattering.
import mjlab  # noqa: F401,E402
