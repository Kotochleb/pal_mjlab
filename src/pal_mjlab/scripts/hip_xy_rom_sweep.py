"""Map the hip_xy actuator space onto (joint 2, joint 3) range of motion.

Supports two model families that realise the same hip_xy mechanism differently:

* ``tendons``: ``pal_kangaroo_full/xmls/kangaroo_full_tendons.xml`` -- the two
  hip_xy screws are spatial tendons ``{side}_hip_xy_{l,r}_slider`` spanning
  ``leg_{side}_1_link`` -> ``leg_{side}_3_link``.  Their *length* is the
  actuator coordinate.
* ``sliders``: ``pal_kangaroo_full_full/xmls/kangaroo_full.xml`` -- the screws
  are real prismatic joints ``leg_{side}_{2,3}_actuator`` on a bracket/motor
  gimbal, closed onto ``leg_{side}_3_link`` through ``<connect>`` equalities.
  Their *qpos* is the actuator coordinate.

Both are on the same slider axis, offset by the tendon rest length
``HIP_XY_TENDON_OFFSET`` (tendon_length = slider_qpos + offset), and the l/r
pairing is identical: ``*_l_slider`` <-> ``leg_*_2_actuator``,
``*_r_slider`` <-> ``leg_*_3_actuator``.

For every model and leg the script sweeps an ``n x n`` grid of (l, r) commands
from the min to the max of each actuator's declared range, relaxes the
mechanism quasi-statically at every setpoint and records the resulting
``leg_{side}_2_joint`` / ``leg_{side}_3_joint`` angles.  Everything is stored
in ONE ``.npy`` file (a pickled dict, see ``--out``); load with::

    res = np.load("hip_xy_rom.npy", allow_pickle=True).item()
    res["tendons"]["left"]["joint2"]        # (n, n) array, [i_l, i_r]
    res["sliders"]["left"]["l_cmd_slider"]  # commands in common slider space

Quasi-static recipe (see the project's kinematic-validation notes): every
joint outside the hip_xy loop is *deleted* from the spec so its body is welded
at qpos0 -- no lock-fighting and a tiny DOF count; gravity is off; every
remaining joint gets damping + armature (one cross body weighs 1e-5 kg); the
two actuator coordinates are driven by temporary position actuators; each
setpoint is approached from qpos0 by ramping the command in small increments
and then relaxed until the free DOFs are quiescent (no warm start across grid
points: closure loops have mirror branches, and an unreachable corner would
otherwise flip the mechanism for the rest of the grid).

Run::

    python -m pal_mjlab.scripts.hip_xy_rom_sweep --n 15 --out hip_xy_rom.npy
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

_ROBOTS_DIR = Path(__file__).resolve().parents[1] / "robots"
TENDONS_XML = _ROBOTS_DIR / "pal_kangaroo_full" / "xmls" / "kangaroo_full_tendons.xml"
SLIDERS_XML = _ROBOTS_DIR / "pal_kangaroo_full_full" / "xmls" / "kangaroo_full.xml"

# Tendon length at qpos0 == slider qpos 0 (same for l/r, left/right). Subtract
# it from a tendon length to land in the slider models' / ground-truth LUTs'
# coordinate. NOT the hip_z value (0.09122257764) -- easy to confuse.
HIP_XY_TENDON_OFFSET = 0.09344327156

ModelKind = Literal["tendons", "sliders"]


@dataclass
class ModelSpec:
  """How one model family exposes the hip_xy mechanism."""

  kind: ModelKind
  xml: Path
  trntype: mujoco.mjtTrn
  # (l, r) actuator coordinate names per side -- tendon names or joint names.
  drivers: dict[str, tuple[str, str]]
  # Joints that must stay free for the mechanism to move; everything else is
  # deleted (welded at qpos0). Joints 2/3 are appended automatically.
  loop_joints: dict[str, tuple[str, ...]]
  offset: float  # actuator coordinate -> slider space offset

  def joint23(self, side: str) -> tuple[str, str]:
    return f"leg_{side}_2_joint", f"leg_{side}_3_joint"


MODELS: dict[ModelKind, ModelSpec] = {
  "tendons": ModelSpec(
    kind="tendons",
    xml=TENDONS_XML,
    trntype=mujoco.mjtTrn.mjTRN_TENDON,
    drivers={
      side: (f"{side}_hip_xy_l_slider", f"{side}_hip_xy_r_slider")
      for side in ("left", "right")
    },
    loop_joints={side: () for side in ("left", "right")},
    offset=HIP_XY_TENDON_OFFSET,
  ),
  "sliders": ModelSpec(
    kind="sliders",
    xml=SLIDERS_XML,
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
    drivers={
      side: (f"leg_{side}_2_actuator", f"leg_{side}_3_actuator")
      for side in ("left", "right")
    },
    loop_joints={
      side: (
        f"{side}_hip_xy_bracket_l",
        f"{side}_hip_xy_motor_l",
        f"leg_{side}_2_actuator",
        f"{side}_hip_xy_cross_l",
        f"{side}_hip_xy_bracket_r",
        f"{side}_hip_xy_motor_r",
        f"leg_{side}_3_actuator",
        f"{side}_hip_xy_cross_r",
      )
      for side in ("left", "right")
    },
    offset=0.0,
  ),
}


@dataclass
class SweepParams:
  n: int = 15
  kp: float = 2e5
  kv: float = 500.0  # actuator-space velocity gain
  damping: float = 30.0
  armature: float = 0.01
  # implicitfast integrates joint damping and the actuators' velocity feedback
  # implicitly, so 1e-3 is stable here and gives the same equilibria as
  # Euler at 1e-4 (checked to ~1e-4 rad) ten times faster.
  timestep: float = 1e-3
  # Settle criteria, in sim seconds (see relax()).
  qvel_tol: float = 1e-4
  pos_tol: float = 1e-5
  pos_window: float = 1.0  # must exceed the ~0.4 s ringing period of the leg
  min_time: float = 0.1
  max_time: float = 30.0
  # Bound on the temporary actuators' force. With gravity off a reachable
  # setpoint needs ~0 N at rest, so this only matters at unreachable corners,
  # where an unbounded PD would shove kN into the joint limits and break the
  # soft closure constraints.
  force_limit: float = 1000.0
  # Commands are ramped from rest to each setpoint in increments no larger
  # than this (native units, m) so the mechanism never jumps closure branches.
  max_cmd_step: float = 2e-3
  ramp_time: float = 0.02  # sim seconds per ramp increment
  unlimited: bool = False  # drop joint limits on joints 2/3
  # Optional overrides of the declared joint 2/3 ranges, applied to every model
  # (e.g. to give the sliders model the tendon model's +-0.471 hip roll limit).
  j2_range: tuple[float, float] | None = None
  j3_range: tuple[float, float] | None = None


@dataclass
class SideResult:
  """Sweep result for one (model, side); every grid array is indexed [i_l, i_r]."""

  driver_names: tuple[str, str]
  joint_names: tuple[str, str]
  driver_range: np.ndarray  # (2, 2): [l|r, min|max], native units
  joint_range: np.ndarray  # (2, 2): declared range of joints 2/3
  offset: float
  l_cmd: np.ndarray  # native (tendon length or slider qpos)
  r_cmd: np.ndarray
  l_meas: np.ndarray  # achieved native coordinate after relaxation
  r_meas: np.ndarray
  joint2: np.ndarray
  joint3: np.ndarray
  settled: np.ndarray  # bool: quiescence reached before max_time
  diverged: np.ndarray  # bool: MuJoCo hit BADQACC / auto-reset during relax
  steps: np.ndarray  # relaxation steps used
  extra: dict = field(default_factory=dict)

  def as_dict(self) -> dict:
    d = {k: v for k, v in self.__dict__.items() if k != "extra"}
    d["l_cmd_slider"] = self.l_cmd - self.offset
    d["r_cmd_slider"] = self.r_cmd - self.offset
    d["l_meas_slider"] = self.l_meas - self.offset
    d["r_meas_slider"] = self.r_meas - self.offset
    d.update(self.extra)
    return d


def build_model(
  ms: ModelSpec, side: str, p: SweepParams
) -> tuple[mujoco.MjModel, tuple[int, int]]:
  """Compile a reduced model: only this side's hip_xy loop is articulated.

  Returns the model and the (l, r) actuator ids.
  """
  spec = mujoco.MjSpec.from_file(str(ms.xml))
  keep = set(ms.loop_joints[side]) | set(ms.joint23(side))
  missing = keep - {j.name for j in spec.joints}
  if missing:
    raise KeyError(f"{ms.xml.name}: joints not found: {sorted(missing)}")

  # Joint equalities referencing a joint we are about to weld must go too.
  for eq in list(spec.equalities):
    if eq.type == mujoco.mjtEq.mjEQ_JOINT and (
      eq.name1 not in keep or (eq.name2 and eq.name2 not in keep)
    ):
      spec.delete(eq)
  for jnt in list(spec.joints):
    if jnt.name not in keep:
      spec.delete(jnt)

  j2_name, j3_name = ms.joint23(side)
  for jnt in spec.joints:
    # damping is a per-DOF vector in the spec bindings (hinge/slide use [0]).
    jnt.damping = np.full_like(np.asarray(jnt.damping, dtype=float), p.damping)
    jnt.armature = p.armature
    override = {j2_name: p.j2_range, j3_name: p.j3_range}.get(jnt.name)
    if override is not None:
      jnt.range[:] = override
      jnt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    if p.unlimited and jnt.name in (j2_name, j3_name):
      jnt.limited = mujoco.mjtLimited.mjLIMITED_FALSE

  _prune_rigid_equalities(spec)

  spec.option.gravity[:] = 0.0
  spec.option.timestep = p.timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  # No contacts: welded arms/legs may interpenetrate and would only add noise.
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

  for name in ms.drivers[side]:
    act = spec.add_actuator(
      name=f"{name}_sweep_pos",
      target=name,
      trntype=ms.trntype,
      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
    )
    act.gainprm[0] = p.kp
    act.biasprm[1] = -p.kp
    act.biasprm[2] = -p.kv
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange[:] = (-p.force_limit, p.force_limit)

  model = spec.compile()
  assert model.nu == 2, model.nu
  return model, (0, 1)  # actuators are appended in (l, r) order


def _prune_rigid_equalities(spec: mujoco.MjSpec) -> None:
  """Delete equalities whose two ends are rigidly connected after welding.

  Once most joints are gone, most closure constraints join bodies that move
  together anyway; they still cost solver rows every step. An equality is
  rigid when the set of remaining joints between each end and the world is the
  same (then the relative pose of the ends is constant). Decided on a throwaway
  compile so the body/site/tendon bookkeeping comes from MuJoCo itself.
  """
  model = spec.compile()

  def chain_joints(body: int) -> frozenset[str]:
    js = []
    while body:
      js += [
        model.joint(model.body_jntadr[body] + k).name
        for k in range(model.body_jntnum[body])
      ]
      body = model.body_parentid[body]
    return frozenset(js)

  def end_bodies(eq: int) -> tuple[int, int] | None:
    t, o1, o2 = model.eq_type[eq], model.eq_obj1id[eq], model.eq_obj2id[eq]
    if t in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
      if model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_SITE:
        return model.site_bodyid[o1], model.site_bodyid[o2]
      return o1, o2
    if t == mujoco.mjtEq.mjEQ_TENDON and model.tendon_num[o1] == 2:
      adr = model.tendon_adr[o1]
      return model.site_bodyid[model.wrap_objid[adr]], model.site_bodyid[
        model.wrap_objid[adr + 1]
      ]
    return None

  rigid = []
  for eq in range(model.neq):
    ends = end_bodies(eq)
    if ends is not None and chain_joints(ends[0]) == chain_joints(ends[1]):
      rigid.append(eq)
  # Spec equalities are in the same order as the compiled ones.
  spec_eqs = list(spec.equalities)
  assert len(spec_eqs) == model.neq
  for eq in rigid:
    spec.delete(spec_eqs[eq])


def driver_range(model: mujoco.MjModel, ms: ModelSpec, name: str) -> np.ndarray:
  if ms.trntype == mujoco.mjtTrn.mjTRN_TENDON:
    return model.tendon_range[model.tendon(name).id].copy()
  return model.jnt_range[model.joint(name).id].copy()


def driver_value(
  data: mujoco.MjData, model: mujoco.MjModel, ms: ModelSpec, name: str
) -> float:
  if ms.trntype == mujoco.mjtTrn.mjTRN_TENDON:
    return float(data.ten_length[model.tendon(name).id])
  return float(data.qpos[model.jnt_qposadr[model.joint(name).id]])


def relax(
  model: mujoco.MjModel, data: mujoco.MjData, p: SweepParams
) -> tuple[bool, bool, int]:
  """Step until quiescent. Returns (settled, diverged, steps).

  Settled means either every DOF is slower than ``qvel_tol`` or the pose has
  stopped changing (max |dq| below ``pos_tol`` over two consecutive
  ``pos_window`` seconds -- two, so a turning point of the slow leg ringing
  cannot pass) -- the latter catches setpoints pinned against a joint limit,
  where the saturated actuator keeps the limit constraint faintly chattering.
  """
  bad = mujoco.mjtWarning.mjWARN_BADQACC
  data.warning[bad].number = 0
  diverged = False
  chunk = 10
  min_steps = int(p.min_time / p.timestep)
  max_steps = int(p.max_time / p.timestep)
  window = int(p.pos_window / p.timestep)
  q_prev, k_prev, still_windows = data.qpos.copy(), 0, 0
  for k in range(chunk, max_steps + 1, chunk):
    mujoco.mj_step(model, data, nstep=chunk)
    if data.warning[bad].number:
      # MuJoCo auto-resets to qpos0 on a diverged step and *clears* the
      # counter first, so any non-zero reading means "at least once".
      diverged = True
      data.warning[bad].number = 0
    if k < min_steps:
      continue
    if np.max(np.abs(data.qvel)) < p.qvel_tol:
      return True, diverged, k
    if k - k_prev >= window:
      still_windows = (
        still_windows + 1 if np.max(np.abs(data.qpos - q_prev)) < p.pos_tol else 0
      )
      if still_windows >= 2:
        return True, diverged, k
      q_prev, k_prev = data.qpos.copy(), k
  return False, diverged, max_steps


def move_to(
  model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, p: SweepParams
) -> tuple[bool, bool, int]:
  """Ramp ctrl to ``target`` in small increments, then relax fully there."""
  start = data.ctrl[:2].copy()
  n_inc = int(np.ceil(np.max(np.abs(target - start)) / p.max_cmd_step))
  for a in np.linspace(0.0, 1.0, n_inc + 1)[1:-1]:
    data.ctrl[:2] = start + a * (target - start)
    mujoco.mj_step(model, data, nstep=max(1, int(p.ramp_time / p.timestep)))
  data.ctrl[:2] = target
  return relax(model, data, p)


def sweep_side(
  ms: ModelSpec, side: str, p: SweepParams, verbose: bool = True
) -> SideResult:
  model, (ia_l, ia_r) = build_model(ms, side, p)
  data = mujoco.MjData(model)
  l_name, r_name = ms.drivers[side]
  j2_name, j3_name = ms.joint23(side)
  q2 = model.jnt_qposadr[model.joint(j2_name).id]
  q3 = model.jnt_qposadr[model.joint(j3_name).id]

  rng = np.stack([driver_range(model, ms, l_name), driver_range(model, ms, r_name)])
  jrng = np.stack(
    [model.jnt_range[model.joint(j2_name).id], model.jnt_range[model.joint(j3_name).id]]
  )
  l_vals = np.linspace(rng[0, 0], rng[0, 1], p.n)
  r_vals = np.linspace(rng[1, 0], rng[1, 1], p.n)
  L, R = np.meshgrid(l_vals, r_vals, indexing="ij")

  shape = (p.n, p.n)
  out = SideResult(
    driver_names=(l_name, r_name),
    joint_names=(j2_name, j3_name),
    driver_range=rng,
    joint_range=jrng,
    offset=ms.offset,
    l_cmd=L,
    r_cmd=R,
    l_meas=np.full(shape, np.nan),
    r_meas=np.full(shape, np.nan),
    joint2=np.full(shape, np.nan),
    joint3=np.full(shape, np.nan),
    settled=np.zeros(shape, dtype=bool),
    diverged=np.zeros(shape, dtype=bool),
    steps=np.zeros(shape, dtype=np.int32),
  )

  def reset_to_rest() -> None:
    # ctrl == current coordinate, so the following move is a ramp from qpos0.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.ctrl[ia_l] = driver_value(data, model, ms, l_name)
    data.ctrl[ia_r] = driver_value(data, model, ms, r_name)

  t0 = time.time()
  n_done = 0
  for i in range(p.n):
    for j in range(p.n):
      # Every setpoint is approached from qpos0 rather than warm-started from
      # its neighbour: closure loops have mirror branches, and once a corner
      # that cannot be reached pushes the mechanism through a singularity a
      # warm start stays on the wrong branch for the rest of the grid.
      reset_to_rest()
      settled, diverged, k = move_to(model, data, np.array([L[i, j], R[i, j]]), p)
      out.l_meas[i, j] = driver_value(data, model, ms, l_name)
      out.r_meas[i, j] = driver_value(data, model, ms, r_name)
      out.joint2[i, j] = data.qpos[q2]
      out.joint3[i, j] = data.qpos[q3]
      out.settled[i, j] = settled
      out.diverged[i, j] = diverged
      out.steps[i, j] = k
      n_done += 1
    if verbose:
      print(
        f"  [{ms.kind}/{side}] row {i + 1}/{p.n}  "
        f"j2 {np.nanmin(out.joint2):+.3f}..{np.nanmax(out.joint2):+.3f}  "
        f"j3 {np.nanmin(out.joint3):+.3f}..{np.nanmax(out.joint3):+.3f}  "
        f"unsettled {int((~out.settled).sum()) - (p.n * p.n - n_done)}  "
        f"({time.time() - t0:.0f}s)",
        flush=True,
      )

  out.extra = {
    "xml": str(ms.xml),
    "params": p.__dict__.copy(),
    "nq_reduced": model.nq,
  }
  return out


def summarize(res: dict) -> None:
  for kind, sides in res.items():
    if not isinstance(sides, dict) or kind.startswith("_"):
      continue
    for side, r in sides.items():
      err_l = np.abs(r["l_meas"] - r["l_cmd"])
      err_r = np.abs(r["r_meas"] - r["r_cmd"])
      print(
        f"{kind:8s} {side:5s}: joint2 [{r['joint2'].min():+.4f}, {r['joint2'].max():+.4f}]  "
        f"joint3 [{r['joint3'].min():+.4f}, {r['joint3'].max():+.4f}]  "
        f"tracking err max l/r {err_l.max() * 1e3:.2f}/{err_r.max() * 1e3:.2f} mm  "
        f"settled {r['settled'].sum()}/{r['settled'].size}  diverged {r['diverged'].sum()}"
      )


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
  ap.add_argument(
    "--sides", nargs="+", choices=["left", "right"], default=["left", "right"]
  )
  ap.add_argument("--tendons-xml", type=Path, default=TENDONS_XML)
  ap.add_argument("--sliders-xml", type=Path, default=SLIDERS_XML)
  ap.add_argument("--out", type=Path, default=Path("hip_xy_rom.npy"))
  ap.add_argument(
    "--n", type=int, default=SweepParams.n, help="grid points per actuator"
  )
  ap.add_argument("--kp", type=float, default=SweepParams.kp)
  ap.add_argument("--kv", type=float, default=SweepParams.kv)
  ap.add_argument("--damping", type=float, default=SweepParams.damping)
  ap.add_argument("--armature", type=float, default=SweepParams.armature)
  ap.add_argument("--timestep", type=float, default=SweepParams.timestep)
  ap.add_argument("--qvel-tol", type=float, default=SweepParams.qvel_tol)
  ap.add_argument(
    "--max-time",
    type=float,
    default=SweepParams.max_time,
    help="sim seconds per setpoint",
  )
  ap.add_argument("--force-limit", type=float, default=SweepParams.force_limit)
  ap.add_argument("--max-cmd-step", type=float, default=SweepParams.max_cmd_step)
  ap.add_argument(
    "--unlimited",
    action="store_true",
    help="remove the declared joint limits on joints 2/3",
  )
  ap.add_argument(
    "--j2-range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    help="override joint 2's range in every model",
  )
  ap.add_argument(
    "--j3-range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    help="override joint 3's range in every model (e.g. -0.471239 0.471239)",
  )
  ap.add_argument("-q", "--quiet", action="store_true")
  args = ap.parse_args()

  MODELS["tendons"].xml = args.tendons_xml
  MODELS["sliders"].xml = args.sliders_xml
  p = SweepParams(
    n=args.n,
    kp=args.kp,
    kv=args.kv,
    damping=args.damping,
    armature=args.armature,
    timestep=args.timestep,
    qvel_tol=args.qvel_tol,
    max_time=args.max_time,
    force_limit=args.force_limit,
    max_cmd_step=args.max_cmd_step,
    unlimited=args.unlimited,
    j2_range=tuple(args.j2_range) if args.j2_range else None,
    j3_range=tuple(args.j3_range) if args.j3_range else None,
  )

  results: dict = {
    "_meta": {"params": p.__dict__.copy(), "offset_tendons": HIP_XY_TENDON_OFFSET}
  }
  for kind in args.models:
    ms = MODELS[kind]
    results[kind] = {}
    for side in args.sides:
      if not args.quiet:
        print(f"== {kind} ({ms.xml.name}) / {side}", flush=True)
      results[kind][side] = sweep_side(ms, side, p, verbose=not args.quiet).as_dict()

  np.save(args.out, results, allow_pickle=True)
  summarize(results)
  print(f"saved -> {args.out.resolve()}")


if __name__ == "__main__":
  main()
