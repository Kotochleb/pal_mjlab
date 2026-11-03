from mjlab.utils.lab_api.tasks.importer import import_packages

from pal_mjlab import sim_warnings

# Hide mujoco-warp's solver/linesearch iteration-cap prints for every env
# (override with PAL_MJLAB_SUPPRESS_OVERFLOW, see pal_mjlab/sim_warnings.py).
sim_warnings.install()

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
