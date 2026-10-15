"""Persist LP-ACRL teacher state alongside the velocity policy checkpoint."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from pal_mjlab.tasks.velocity.mdp.lp_acrl import LpAcrlVelocityCommand


class LpAcrlVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    command = self.env.unwrapped.command_manager.get_term("twist")
    assert isinstance(command, LpAcrlVelocityCommand)
    infos = {**(infos or {}), "lp_acrl": command.curriculum_state_dict()}
    super().save(path, infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "lp_acrl" in infos:
      command = self.env.unwrapped.command_manager.get_term("twist")
      assert isinstance(command, LpAcrlVelocityCommand)
      command.load_curriculum_state_dict(infos["lp_acrl"])
    return infos
