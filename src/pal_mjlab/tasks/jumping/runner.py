"""Vertical-jump runner that can start from a policy pretrained elsewhere."""

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

# PPO updates, after loading only a pretrained actor, during which only the
# fresh critic learns (see JumpOnPolicyRunner).
CRITIC_WARMUP_ITERATIONS = 50


class JumpOnPolicyRunner(VelocityOnPolicyRunner):
  """``VelocityOnPolicyRunner`` that warm-starts from foreign checkpoints.

  A checkpoint whose critic does not fit this env's critic is a policy trained
  on another task with the same actor observations and actions, e.g.
  ``Mjlab-Tracking-Jump-*``, whose critic observes the reference motion. Only
  its actor (weights, observation normalizer, action std) is loaded; training
  starts at iteration 0 with a fresh critic and optimizer and with this env's
  own step counter, so the height curriculum starts from its initial range.
  Any other checkpoint resumes as usual, including the state of curriculum
  terms that keep one (``state_dict``/``load_state_dict``), such as the
  stair height curriculum's state (offset, stage, averages).

  After such an actor-only load, the first CRITIC_WARMUP_ITERATIONS PPO
  updates train the critic alone: the actor's weights, action std and
  observation normalizer stay as loaded (their parameters get no gradient,
  so Adam leaves them untouched), and the learning rate stays fixed instead of
  following the adaptive KL schedule (with an unchanged actor the KL is ~0 and
  would push it up). Advantages from a random critic would otherwise undo the
  pretrained jump. A resumed jump checkpoint has no warm-up.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.critic_warmup_left = 0
    self._ppo_update = self.alg.update
    self.alg.update = self._update

  def _update(self) -> dict[str, float]:
    if self.critic_warmup_left <= 0:
      return self._ppo_update()
    self.critic_warmup_left -= 1
    alg = self.alg
    actor = alg._raw_actor
    frozen = [p for p in actor.parameters() if p.requires_grad]
    schedule = alg.schedule
    for p in frozen:
      p.requires_grad_(False)
    actor.update_normalization = lambda obs: None
    alg.schedule = "fixed"
    try:
      return self._ppo_update()
    finally:
      for p in frozen:
        p.requires_grad_(True)
      del actor.update_normalization  # Back to the class's method.
      alg.schedule = schedule

  def _stateful_curricula(self) -> dict:
    manager = self.env.unwrapped.curriculum_manager
    terms = {}
    for name in getattr(manager, "active_terms", []):
      func = manager.get_term_cfg(name).func
      if callable(getattr(func, "state_dict", None)):
        terms[name] = func
    return terms

  def save(self, path: str, infos=None) -> None:
    state = {name: t.state_dict() for name, t in self._stateful_curricula().items()}
    super().save(path, {**(infos or {}), "curriculum_state": state})

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is not None or self._critic_fits(path, map_location):
      infos = super().load(path, load_cfg, strict, map_location)
      saved = (infos or {}).get("curriculum_state", {})
      for name, term in self._stateful_curricula().items():
        if name in saved:
          term.load_state_dict(saved[name])
      return infos
    print(
      f"[INFO] {path}: critic does not fit this task; loading only its actor "
      "as a pretrained policy."
    )
    env = self.env.unwrapped
    step_counter = env.common_step_counter
    infos = super().load(path, {"actor": True}, strict, map_location)
    env.common_step_counter = step_counter
    self.critic_warmup_left = CRITIC_WARMUP_ITERATIONS
    print(
      f"[INFO] Training only the critic for the first {CRITIC_WARMUP_ITERATIONS}"
      " PPO updates."
    )
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
