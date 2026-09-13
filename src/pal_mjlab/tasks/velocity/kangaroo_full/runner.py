"""Runner for kangaroo_full: hands the policy's action distribution to the env.

Curriculum terms only ever see the env, while the policy lives in the runner,
so a curriculum that wants to touch the policy -- the ``policy_std_range``
ramp in :mod:`.mdp.curriculums` -- needs the runner to leave a reference on
the env. That is the only thing this subclass adds.
"""

from __future__ import annotations

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

POLICY_DISTRIBUTION_ATTR = "policy_distribution"
"""Name of the env attribute the runner stores the actor's distribution under."""


class KangarooFullOnPolicyRunner(VelocityOnPolicyRunner):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    setattr(
      self.env.unwrapped,
      POLICY_DISTRIBUTION_ATTR,
      self.alg.get_policy().distribution,
    )
