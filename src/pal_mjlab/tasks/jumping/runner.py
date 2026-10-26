"""Vertical-jump runner that can start from a policy pretrained elsewhere."""

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class JumpOnPolicyRunner(VelocityOnPolicyRunner):
  """``VelocityOnPolicyRunner`` that warm-starts from foreign checkpoints.

  A checkpoint whose critic does not fit this env's critic is a policy trained
  on another task with the same actor observations and actions, e.g.
  ``Mjlab-Tracking-Jump-*``, whose critic observes the reference motion. Only
  its actor (weights, observation normalizer, action std) is loaded; training
  starts at iteration 0 with a fresh critic and optimizer and with this env's
  own step counter, so the height curriculum starts at its first stage. Any
  other checkpoint resumes as usual.
  """

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is not None or self._critic_fits(path, map_location):
      return super().load(path, load_cfg, strict, map_location)
    print(
      f"[INFO] {path}: critic does not fit this task; loading only its actor "
      "as a pretrained policy."
    )
    env = self.env.unwrapped
    step_counter = env.common_step_counter
    infos = super().load(path, {"actor": True}, strict, map_location)
    env.common_step_counter = step_counter
    return infos

  def _critic_fits(self, path: str, map_location: str | None) -> bool:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    critic = loaded.get("critic_state_dict")
    if critic is None:  # Legacy layout; the base loader migrates it.
      return True
    own = self.alg._raw_critic.state_dict()
    return critic.keys() == own.keys() and all(
      critic[k].shape == own[k].shape for k in own
    )
