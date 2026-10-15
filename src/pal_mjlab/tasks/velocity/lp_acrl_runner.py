"""Persist LP-ACRL teacher state alongside the velocity policy checkpoint."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from pal_mjlab.tasks.velocity.mdp.lp_acrl import LpAcrlCommand


class LpAcrlVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  command_name = "twist"

  def _lp_acrl_command(self) -> LpAcrlCommand:
    command = self.env.unwrapped.command_manager.get_term(self.command_name)
    assert isinstance(command, LpAcrlCommand)
    return command

  def save(self, path: str, infos=None):
    infos = {**(infos or {}), "lp_acrl": self._lp_acrl_command().curriculum_state_dict()}
    super().save(path, infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "lp_acrl" in infos:
      self._lp_acrl_command().load_curriculum_state_dict(infos["lp_acrl"])
    return infos
