"""Persist LP-ACRL teacher state alongside the jumping policy checkpoint."""

from pal_mjlab.tasks.velocity.lp_acrl_runner import LpAcrlVelocityOnPolicyRunner


class LpAcrlJumpOnPolicyRunner(LpAcrlVelocityOnPolicyRunner):
  command_name = "jump_height"
