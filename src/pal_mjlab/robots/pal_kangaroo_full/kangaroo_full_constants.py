"""Pal Robotics KANGAROO FULL constants."""

import re
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.string import resolve_expr
from pal_mjlab import PAL_MJLAB_SRC_PATH
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  FULL_COLLISION,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  _calc_leg_params,
)
from pal_mjlab.robots.pal_kangaroo_full.actuator import (
  TransmitedIdealPdActuatorCfg,
  load_transmission_table,
)


REGEX_SIMPLE_MODEL_OBSERVABLE_JOINTS_ONLY = r"^(?!leg_.*_length_actuator$).*$"
REGEX_SIMPLE_MODEL_ACTUATED_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee)_joint$|leg_.*_length_actuator$).*$"
)
REGEX_ACTUATED_JOINTS_ONLY = r"^(?!leg_.*_(femur|knee|length)_joint$).*$"
REGEX_ALL_OBSERVABLE_JOINTS = r".*"


REGEX_POSE_REVEOLUTE_JOINTS_ONLY = (
  r"^(?!leg_.*_(femur|knee|length)_joint$|leg_.*_length_actuator$).*$"
)
REGEX_LINEAR_JOINTS_ONLY = r"leg_.*_length_actuator"

KANGAROO_FULL_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls"
KANGAROO_FULL_XML = KANGAROO_FULL_PATH / "kangaroo_tendons.xml"
KANGAROO_FULL_TENDON_HIP_XML = KANGAROO_FULL_PATH / "kangaroo_tendons_hip.xml"
KANGAROO_FULL_TENDON_HIP_CL_XML = KANGAROO_FULL_PATH / "kangaroo_tendons_cl.xml"

LEG_LENGHT_TRAMISSION_LOOKUP_PATH = (
  KANGAROO_FULL_PATH.parent / "transmission" / "leg_length.csv"
)

for p in [
  KANGAROO_FULL_XML,
  KANGAROO_FULL_TENDON_HIP_XML,
  KANGAROO_FULL_TENDON_HIP_CL_XML,
  LEG_LENGHT_TRAMISSION_LOOKUP_PATH,
]:
  assert p.exists(), f"Missing: {p}"


KANGAROO_TENDON_LENGTHS: dict[str, float] = {
  r"(left|right)_hip_xy_link": 0.09,
  r"(left|right)_knee_rods": 0.215,
  r"(left|right)_femur_rod": 0.40427,
  r"(left|right)_ankle_(femur|tibia)_bar_(l|r)": 0.38,
}


def _enforce_tendon_lengths(
  model: mujoco.MjModel, data: mujoco.MjData, lengths: dict[str, float]
) -> None:
  mujoco.mj_forward(model, data)

  for i in range(model.neq):
    eq = model.eq(i)
    if eq.type != mujoco.mjtEq.mjEQ_TENDON:
      continue
    tendon_id = int(eq.obj1id.item())
    tendon_name = model.tendon(tendon_id).name
    matches = [
      (pattern, length)
      for pattern, length in lengths.items()
      if re.fullmatch(pattern, tendon_name)
    ]
    if not matches:
      continue
    if len(matches) > 1:
      raise ValueError(
        f"Tendon '{tendon_name}' matches multiple length patterns: "
        f"{[pattern for pattern, _ in matches]}"
      )
    print(tendon_name, data.ten_length[tendon_id])
    target_length = matches[0][1]
    length_at_qpos0 = data.ten_length[tendon_id]
    eq.data[0] = target_length - length_at_qpos0


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_kangaroo_full_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_FULL_XML)


def get_kangaroo_full_tendon_hip_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_FULL_TENDON_HIP_XML)


def get_kangaroo_full_tendon_hip_cl_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_FULL_TENDON_HIP_CL_XML)


##
# Actuator config.
##


KANGAROO_SERIAL_JOINTS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_1_joint",), **_calc_leg_params(100.0, 80.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_2_joint",), **_calc_leg_params(100.0, 230.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_3_joint",), **_calc_leg_params(100.0, 139.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
  ),
)

KANGAROO_LEG_ACTUATORS = KANGAROO_SERIAL_JOINTS + (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    # saturation_effort=5000.0,
    # velocity_limit=0.625,
    **_calc_leg_params(6000.0, 5000.0),
  ),
)

KANGAROO_LEG_ACTUATORS_LOW = KANGAROO_SERIAL_JOINTS + (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    **_calc_leg_params(5700.0, 5000.0),
  ),
)


KANGAROO_LEG_ACTUATORS_SEMI_SERIAL = KANGAROO_SERIAL_JOINTS + (
  TransmitedIdealPdActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_joint$",),
    joint_to_actuator_map={
      "leg_right_length_joint": "leg_right_length_actuator",
      "leg_left_length_joint": "leg_left_length_actuator",
    },
    transmission=load_transmission_table(LEG_LENGHT_TRAMISSION_LOOKUP_PATH),
    actuator_effort_limit=5000.0,
    **_calc_leg_params(1600.0, 1100.0),
  ),
)


KANGAROO_LEG_ACTUATORS_TENDON_HIPS = (
  BuiltinPositionActuatorCfg(
    transmission_type=TransmissionType.TENDON,
    target_names_expr=(r"(left|right)_hip_z_slider$",),
    **_calc_leg_params(2500.0, 2000.0),
    # saturation_effort=2000.0,
    # velocity_limit=0.4,
  ),
  BuiltinPositionActuatorCfg(
    transmission_type=TransmissionType.TENDON,
    target_names_expr=(r"(left|right)_hip_xy_(l|r)_slider$",),
    **_calc_leg_params(10000000.0, 20000.0),
    # saturation_effort=2000.0,
    # velocity_limit=0.4,
  ),
  # BuiltinPositionActuatorCfg(
  #   target_names_expr=("leg_.*_2_joint",), **_calc_leg_params(100.0, 230.0)
  # ),
  # BuiltinPositionActuatorCfg(
  #   target_names_expr=("leg_.*_3_joint",), **_calc_leg_params(100.0, 139.0)
  # ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
  ),
  # BuiltinPositionActuatorCfg(
  #   transmission_type=TransmissionType.TENDON,
  #   target_names_expr=(r"(left|right)_ankle_(l|r)_slider$",),
  #   saturation_effort=630,
  #   velocity_limit=0.4,
  #   **_calc_leg_params(630.0, 2000.0),
  # ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    # saturation_effort=5000.0,
    # velocity_limit=0.625,
    **_calc_leg_params(6000.0, 5000.0),
  ),
)

KANGAROO_LEG_ACTUATORS_TENDON_HIPS_CL = (
  BuiltinPositionActuatorCfg(
    transmission_type=TransmissionType.JOINT,
    target_names_expr=(r"leg_(left|right)_1_actuator$",),
    **_calc_leg_params(2500.0, 2000.0),
    # saturation_effort=2000.0,
    # velocity_limit=0.4,
  ),
  BuiltinPositionActuatorCfg(
    transmission_type=TransmissionType.JOINT,
    target_names_expr=(r"(left|right)_hip_xy_(l|r)_slider$",),
    **_calc_leg_params(730.0, 2000.0),
    # saturation_effort=2000.0,
    # velocity_limit=0.4,
  ),
  # BuiltinPositionActuatorCfg(
  #   target_names_expr=("leg_.*_2_joint",), **_calc_leg_params(100.0, 230.0)
  # ),
  # BuiltinPositionActuatorCfg(
  #   target_names_expr=("leg_.*_3_joint",), **_calc_leg_params(100.0, 139.0)
  # ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30.0, 140.0)
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30.0, 82.0)
  ),
  # BuiltinPositionActuatorCfg(
  #   transmission_type=TransmissionType.TENDON,
  #   target_names_expr=(r"(left|right)_ankle_(l|r)_slider$",),
  #   saturation_effort=630,
  #   velocity_limit=0.4,
  #   **_calc_leg_params(630.0, 2000.0),
  # ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_(left|right)_length_actuator$",),
    # saturation_effort=5000.0,
    # velocity_limit=0.625,
    **_calc_leg_params(6000.0, 5000.0),
  ),
)

COMMON_ACTUATORS = KANGAROO_LEG_ACTUATORS + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
COMMON_ACTUATORS_LOW = KANGAROO_LEG_ACTUATORS_LOW + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
COMMON_ACTUATORS_SEMI_SERIAL = KANGAROO_LEG_ACTUATORS_SEMI_SERIAL + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)

COMMON_ACTUATORS_TENDON_HIPS = KANGAROO_LEG_ACTUATORS_TENDON_HIPS + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
COMMON_ACTUATORS_TENDON_HIPS_CL = KANGAROO_LEG_ACTUATORS_TENDON_HIPS_CL + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)
##
# Keyframes.
##

_INIT_STATE_JOINT_POS_BASE: dict[str, float] = {
  "leg_left_1_joint": -0.012,
  "leg_right_1_joint": 0.012,
  "leg_.*_2_joint": 0.054,
  "leg_left_3_joint": 0.04,
  "leg_right_3_joint": -0.04,
  "leg_.*_length_joint": 0.6,
  "leg_.*_length_actuator": 0.0284,
  "leg_.*_4_joint": -0.053,
  "leg_.*_5_joint": 0.0,
  "leg_.*_femur_joint": 0.9,
  "leg_.*_knee_joint": 1.8,
  "arm_left_1_joint": 0.24,
  "arm_right_1_joint": -0.24,
  "arm_.*_2_joint": 1.32,
  "arm_left_3_joint": 1.57,
  "arm_right_3_joint": -1.57,
  "arm_.*_4_joint": 0.8,
  "pelvis_1_joint": 0.0,
  "pelvis_2_joint": 0.0,
}

_INIT_STATE_BASE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.95),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=_INIT_STATE_JOINT_POS_BASE,
  joint_vel={".*": 0.0},
)


def _compute_default_cl_actuator_pos(
  spec_fn,
  init_state: EntityCfg.InitialStateCfg,
  relax_pattern: str,
  report_pattern: str,
  n_settle_steps: int = 200,
) -> dict[str, float]:
  """Settle a CL closed-loop sub-mechanism (motor hinge(s) + slide joint) into
  the position that satisfies its ``<connect>`` equality constraint, given
  every other joint held at its ``init_state`` value.

  A CL mechanism has no closed-form length like a spatial tendon does -- its
  rest position is only defined by where the compliant ``<connect>``
  constraint pulls it once the rest of the leg is posed. Leaving it unset in
  ``INIT_STATE.joint_pos`` (falling back to 0.0) means the model starts with
  real constraint violation at reset and needs several simulation steps to
  settle -- unlike :func:`_compute_tendon_length_at_default_pose`, which
  works because a tendon's length *is* a closed-form function of the other
  joints. This instead relaxes only the joints matching ``relax_pattern``
  (holding everything else, including the free/root joint, clamped to its
  init_state value each step) until the constraint is satisfied, then reads
  back the settled position of the joints matching ``report_pattern`` (the
  real actuator DOF(s), not the mechanism's passive hinges) -- the direct
  equivalent of a real joint's numeric ``INIT_STATE`` default, just solved
  for instead of hand-measured.

  Args:
    relax_pattern: fullmatch regex selecting every joint in the sub-mechanism
      that should be left free to settle (actuator slide joint(s) plus any
      passive hinges it rides on).
    report_pattern: fullmatch regex, a subset of ``relax_pattern``, selecting
      which settled joints to actually return (the real actuator DOF(s)).
  """
  spec = spec_fn()
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  data = mujoco.MjData(model)

  relax_re = re.compile(relax_pattern)
  relaxed_joints, other_joints = [], []
  for j in range(model.njnt):
    name = model.joint(j).name
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    (relaxed_joints if relax_re.fullmatch(name) else other_joints).append(name)
  root_joints = tuple(
    model.joint(j).name
    for j in range(model.njnt)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  )

  other_values = dict(
    zip(other_joints, resolve_expr(init_state.joint_pos, tuple(other_joints), 0.0))
  )

  def _clamp_fixed_state() -> None:
    for name, value in other_values.items():
      jid = model.joint(name).id
      data.qpos[model.jnt_qposadr[jid]] = value
      data.qvel[model.jnt_dofadr[jid]] = 0.0
    for name in root_joints:
      jid = model.joint(name).id
      qadr = model.jnt_qposadr[jid]
      data.qpos[qadr : qadr + 3] = init_state.pos
      data.qpos[qadr + 3 : qadr + 7] = init_state.rot
      dadr = model.jnt_dofadr[jid]
      data.qvel[dadr : dadr + 6] = 0.0

  _clamp_fixed_state()
  mujoco.mj_forward(model, data)
  for _ in range(n_settle_steps):
    mujoco.mj_step(model, data)
    _clamp_fixed_state()
    mujoco.mj_forward(model, data)

  report_re = re.compile(report_pattern)
  return {
    name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]])
    for name in relaxed_joints
    if report_re.fullmatch(name)
  }


_CL_ACTUATOR_DEFAULTS: dict[str, float] = {}
for _relax, _report in (
  (
    r"leg_(left|right)_1_actuator|(left|right)_hip_z_motor",
    r"leg_(left|right)_1_actuator",
  ),
  (
    r"(left|right)_hip_xy_bracket_(l|r)"
    r"|(left|right)_hip_xy_motor_(l|r)"
    r"|(left|right)_hip_xy_(l|r)_slider",
    r"(left|right)_hip_xy_(l|r)_slider",
  ),
):
  _CL_ACTUATOR_DEFAULTS.update(
    _compute_default_cl_actuator_pos(
      get_kangaroo_full_tendon_hip_cl_spec, _INIT_STATE_BASE, _relax, _report
    )
  )

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=_INIT_STATE_BASE.pos,
  rot=_INIT_STATE_BASE.rot,
  joint_pos={**_INIT_STATE_BASE.joint_pos, **_CL_ACTUATOR_DEFAULTS},
  joint_vel=_INIT_STATE_BASE.joint_vel,
)


def _compute_tendon_length_at_default_pose(
  spec_fn, init_state: EntityCfg.InitialStateCfg, tendon_pattern: str
) -> dict[str, float]:
  """Tendon length(s) at the compiled default/keyframe pose.

  This is the TENDON-transmission analogue of what ``use_default_offset=True``
  gives JOINT actuators for free: ``JointPositionAction`` reads
  ``entity.data.default_joint_pos``, the joint's own value at the compiled
  default pose, so a raw action of 0 holds the default pose exactly.
  ``TendonLengthActionCfg`` has no ``use_default_offset`` option, so this
  computes the equivalent directly -- the tendon's actual length once the
  model is posed at ``init_state`` -- for use as the action offset, instead of
  a hand-maintained constant that can silently drift out of sync with the
  model (as it previously did: the manually-measured offset was off by
  ~0.3-0.5 mm from the true default-pose length).
  """
  spec = spec_fn()
  # The raw XML references a `terrain` body for foot-collision excludes that
  # only resolves once attached into a full scene; add a placeholder so this
  # compiles standalone, matching what Scene assembly would provide.
  spec.worldbody.add_body(name="terrain")
  model = spec.compile()
  data = mujoco.MjData(model)

  joint_names = tuple(
    model.joint(j).name
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  )
  values = resolve_expr(init_state.joint_pos, joint_names, 0.0)
  for name, value in zip(joint_names, values):
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  mujoco.mj_forward(model, data)

  return {
    model.tendon(i).name: float(data.ten_length[i])
    for i in range(model.ntendon)
    if re.fullmatch(tendon_pattern, model.tendon(i).name)
  }


# Fixed, physically-meaningful rod-length constants -- these are the tendon
# length at the mechanism's own geometric zero reference, independently
# verified against the true-kinematics lookup tables (hip_z_trans*.csv,
# joint_to_actuator_*.npy): 0.09122257764 (hip_z) and 0.09344327156 (hip_xy)
# both matched MuJoCo's tendon length to within ~0.001-0.003 rad-equivalent
# once combined with the tendon's actual length at a given pose. Same values,
# same structure as commit d564e7ca (`Update tendon offset lengths`).
KANGAROO_TENDON_OFFSETS: dict[str, float] = {
  r"(left|right)_hip_z_slider$": 0.09122257764,
  r"right_hip_xy_r_slider$": 0.09344327156,
  r"right_hip_xy_l_slider$": 0.09344327156,
  r"left_hip_xy_r_slider$": 0.09344327156,
  r"left_hip_xy_l_slider$": 0.09344327156,
}


def _match_tendon_offset(name: str) -> float:
  for pattern, value in KANGAROO_TENDON_OFFSETS.items():
    if re.fullmatch(pattern, name):
      return value
  raise KeyError(f"No KANGAROO_TENDON_OFFSETS pattern matches tendon {name!r}")


# Unlike commit d564e7ca's hand-typed INIT_STATE_TENDONS deltas (which drifted
# ~0.3-0.5 mm out of sync with the model over time), this delta is solved for
# directly from the compiled model at INIT_STATE, so it can never go stale.
_KANGAROO_TENDON_LENGTH_AT_INIT_STATE = _compute_tendon_length_at_default_pose(
  get_kangaroo_full_tendon_hip_spec,
  INIT_STATE,
  r"(left|right)_hip_z_slider$|(left|right)_hip_xy_(l|r)_slider$",
)
INIT_STATE_TENDONS: dict[str, float] = {
  name: length - _match_tendon_offset(name)
  for name, length in _KANGAROO_TENDON_LENGTH_AT_INIT_STATE.items()
}

KANGAROO_INIT_STATE_TENDONS_OFFSETS: dict[str, float] = {
  name: _match_tendon_offset(name) + delta for name, delta in INIT_STATE_TENDONS.items()
}

KANGAROO_INIT_STATE_SIMPLE_TO_FULL_JACOBIAN = {
  r"leg_.*_length_actuator": 0.2632150548042582,
  # r"(left|right)_hip_z_slider$": 0.2632150548042582,
  # r"(left|right)_hip_xy_(l|r)_slider$": 0.2632150548042582,
}


KANGAROO_FULL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS),
  soft_joint_pos_limit_factor=0.99,
)
KANGAROO_FULL_ARTICULATION_LOW = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS_LOW),
  soft_joint_pos_limit_factor=0.99,
)

KANGAROO_FULL_ARTICULATION_SEMI_SERIAL = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS_SEMI_SERIAL),
  soft_joint_pos_limit_factor=0.99,
)

KANGAROO_FULL_ARTICULATION_TENDON_HIPS = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS_TENDON_HIPS),
  soft_joint_pos_limit_factor=0.99,
)

KANGAROO_FULL_ARTICULATION_TENDON_HIPS_CL = EntityArticulationInfoCfg(
  actuators=(COMMON_ACTUATORS_TENDON_HIPS_CL),
  soft_joint_pos_limit_factor=0.99,
)

_ROBOT_CONFIGS = {
  "kangaroo_full": (
    get_kangaroo_full_spec,
    KANGAROO_FULL_ARTICULATION,
    FULL_COLLISION,
  ),
  "kangaroo_full_low_pd": (
    get_kangaroo_full_spec,
    KANGAROO_FULL_ARTICULATION_LOW,
    FULL_COLLISION,
  ),
  "kangaroo_full_semi_serial": (
    get_kangaroo_full_spec,
    KANGAROO_FULL_ARTICULATION_SEMI_SERIAL,
    FULL_COLLISION,
  ),
  "kangaroo_full_tendon_hips": (
    get_kangaroo_full_tendon_hip_spec,
    KANGAROO_FULL_ARTICULATION_TENDON_HIPS,
    FULL_COLLISION,
  ),
  "kangaroo_full_tendon_hips_cl": (
    get_kangaroo_full_tendon_hip_cl_spec,
    KANGAROO_FULL_ARTICULATION_TENDON_HIPS_CL,
    FULL_COLLISION,
  ),
}


def _make_robot_cfg(variant: str) -> EntityCfg:
  spec_fn, articulation, collision = _ROBOT_CONFIGS[variant]
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
  )


def get_kangaroo_full_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full")


def get_kangaroo_full_robot_low_pd_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full_low_pd")


def get_kangaroo_full_robot_semi_serial_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full_semi_serial")


def get_kangaroo_full_robot_tendon_hips_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full_tendon_hips")


def get_kangaroo_full_robot_tendon_hips_cl_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo_full_tendon_hips_cl")


##
# Final config.
##


def _build_action_scales(
  articulation: EntityArticulationInfoCfg,
  transmission_type: TransmissionType = TransmissionType.JOINT,
  exclude: set = frozenset(),
) -> tuple[dict, tuple]:
  """Build action scale dict and actuator names from articulation config."""
  scales, names = {}, []
  for a in articulation.actuators:
    if a.transmission_type != transmission_type:
      continue
    e = (
      a.effort_limit
      if isinstance(a.effort_limit, dict)
      else {n: a.effort_limit for n in a.target_names_expr}
    )
    s = (
      a.stiffness
      if isinstance(a.stiffness, dict)
      else {n: a.stiffness for n in a.target_names_expr}
    )
    for n in a.target_names_expr:
      if n in e and n in s and s[n] and n not in exclude:
        scales[n] = 0.25 * e[n] / s[n]
        names.append(n)
  return scales, tuple(names)


(
  KANGAROO_FULL_TENDON_HIPS_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES,
) = _build_action_scales(KANGAROO_FULL_ARTICULATION_TENDON_HIPS, TransmissionType.JOINT)

(
  KANGAROO_FULL_TENDON_HIPS_TENDON_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES,
) = _build_action_scales(
  KANGAROO_FULL_ARTICULATION_TENDON_HIPS, TransmissionType.TENDON
)
KANGAROO_FULL_TENDON_HIPS_ACTUATOR_NAMES = (
  KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES
  + KANGAROO_FULL_TENDON_HIPS_ACTUATED_TENDONS_NAMES
)

# Per-term scale/offset dicts, split by hip_z vs. hip_xy, mirroring the CL
# split above: each TendonLengthActionCfg's `scale`/`offset` must only
# contain keys matching *its own* term's targets, since resolve_matching_
# names_values errors if a key matches none of them.
KANGAROO_FULL_TENDON_HIPS_HIPZ_TENDON_ACTION_SCALE = {
  k: v for k, v in KANGAROO_FULL_TENDON_HIPS_TENDON_ACTION_SCALE.items() if "hip_z" in k
}
KANGAROO_FULL_TENDON_HIPS_HIPXY_TENDON_ACTION_SCALE = {
  k: v
  for k, v in KANGAROO_FULL_TENDON_HIPS_TENDON_ACTION_SCALE.items()
  if "hip_xy" in k
}

KANGAROO_INIT_STATE_HIPZ_TENDONS_OFFSETS = {
  k: v for k, v in KANGAROO_INIT_STATE_TENDONS_OFFSETS.items() if "hip_z" in k
}
KANGAROO_INIT_STATE_HIPXY_TENDONS_OFFSETS = {
  k: v for k, v in KANGAROO_INIT_STATE_TENDONS_OFFSETS.items() if "hip_xy" in k
}

(
  KANGAROO_FULL_TENDON_HIPS_CL_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_ACTUATED_JOINTS_NAMES,
) = _build_action_scales(
  KANGAROO_FULL_ARTICULATION_TENDON_HIPS_CL, TransmissionType.JOINT
)

# Split the CL joint actuators into "everything but the closed-loop hip
# actuators" and "hip_z only" / "hip_xy only" so the action vector layout
# matches the tendon-hips variant exactly: a main JOINT term covering the same
# joint set/order as KANGAROO_FULL_TENDON_HIPS_ACTUATED_JOINTS_NAMES, followed
# by the hip_z and hip_xy terms -- mirroring the tendon variant's
# joint_pos (main) + tendon_pos (hip_z) + hip_xy_pos (hip_xy) split, just with
# JOINT instead of TENDON transmission for the closed-loop pairs. This keeps
# both variants' policies observation- and action-compatible: same dims, same
# ordering.
_CL_HIPZ_PATTERN = r"leg_(left|right)_1_actuator$"
_CL_HIPXY_PATTERN = r"(left|right)_hip_xy_(l|r)_slider$"

(
  KANGAROO_FULL_TENDON_HIPS_CL_MAIN_JOINT_ACTION_SCALE,
  KANGAROO_FULL_TENDON_HIPS_CL_MAIN_ACTUATED_JOINTS_NAMES,
) = _build_action_scales(
  KANGAROO_FULL_ARTICULATION_TENDON_HIPS_CL,
  TransmissionType.JOINT,
  exclude={_CL_HIPZ_PATTERN, _CL_HIPXY_PATTERN},
)

# Derived as the set difference of the full vs. main dicts/tuples above, so the
# hip_z/hip_xy scales always stay in sync with whatever
# KANGAROO_LEG_ACTUATORS_TENDON_HIPS_CL actually specifies -- no duplicated/
# hardcoded stiffness or effort_limit values.
_cl_split_off = {
  k: v
  for k, v in KANGAROO_FULL_TENDON_HIPS_CL_JOINT_ACTION_SCALE.items()
  if k not in KANGAROO_FULL_TENDON_HIPS_CL_MAIN_JOINT_ACTION_SCALE
}
_cl_split_off_names = tuple(
  n
  for n in KANGAROO_FULL_TENDON_HIPS_CL_ACTUATED_JOINTS_NAMES
  if n not in KANGAROO_FULL_TENDON_HIPS_CL_MAIN_ACTUATED_JOINTS_NAMES
)

KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_JOINT_ACTION_SCALE = {
  k: v for k, v in _cl_split_off.items() if k == _CL_HIPZ_PATTERN
}
KANGAROO_FULL_TENDON_HIPS_CL_HIPZ_ACTUATED_JOINTS_NAMES = (_CL_HIPZ_PATTERN,)

KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_JOINT_ACTION_SCALE = {
  k: v for k, v in _cl_split_off.items() if k == _CL_HIPXY_PATTERN
}
KANGAROO_FULL_TENDON_HIPS_CL_HIPXY_ACTUATED_JOINTS_NAMES = (_CL_HIPXY_PATTERN,)


(
  KANGAROO_FULL_JOINT_ACTION_SCALE,
  KANGAROO_FULL_ACTUATED_JOINTS_NAMES,
) = _build_action_scales(KANGAROO_FULL_ARTICULATION, TransmissionType.JOINT)
KANGAROO_FULL_JOINT_ACTION_SCALE_LOW, KANGAROO_FULL_JOINT_NAMES = _build_action_scales(
  KANGAROO_FULL_ARTICULATION_LOW, TransmissionType.JOINT
)
(
  KANGAROO_FULL_JOINT_ACTION_SCALE_SEMI_SERIAL,
  KANGAROO_FULL_ACTUATED_JOINTS_NAMES_SEMI_SERIAL,
) = _build_action_scales(KANGAROO_FULL_ARTICULATION_SEMI_SERIAL, TransmissionType.JOINT)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_full_robot_tendon_hips_cl_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  _enforce_tendon_lengths(model, data, KANGAROO_TENDON_LENGTHS)
  viewer.launch(model, data)
