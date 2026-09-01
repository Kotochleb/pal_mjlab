"""Sweep a stiff position actuator on one joint, plot the response of another.

Loads a MuJoCo model, disables gravity, locks every joint except a
user-specified subset, drives one of the free joints with a very stiff
position actuator across a user-specified range, and plots the resulting
position of a second, purely-observed joint against the commanded position
of the driven joint.

Useful for e.g. checking how a tendon/mechanism couples one joint's motion
to another's.

All user-facing settings are the module-level variables below.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# User-configurable variables
# ---------------------------------------------------------------------------

MODEL_PATH = "path/to/model.xml"

# Joints that stay free to move. Every other joint in the model is locked:
# held at its initial qpos with zero qvel for the whole sim.
FREE_JOINTS = ["joint_a", "joint_b"]

# Which of the joints listed above is driven by the stiff position actuator,
# and which one is only observed/logged (must both be in FREE_JOINTS).
ACTUATED_JOINT = "joint_a"
OBSERVED_JOINT = "joint_b"

# Range swept by the position actuator's setpoint, in the joint's native
# units (radians for hinge, meters for slide).
JOINT_RANGE = (-0.3, 0.3)
N_SETPOINTS = 100

# Simulation substeps run at each setpoint before recording, to let the
# mechanism settle onto its new equilibrium. With SIM_TIMESTEP = 1e-4 this
# is 0.2s of simulated settling time.
SETTLE_STEPS = 2000

# Position-actuator gains. KP is deliberately huge so the actuator tracks
# its setpoint almost rigidly; KV adds damping so it settles instead of
# ringing.
ACTUATOR_KP = 1.0e6
ACTUATOR_KV = 2.0 * np.sqrt(ACTUATOR_KP)
ACTUATOR_FORCERANGE = 1.0e6  # N or N*m, applied symmetrically.

# Simulation timestep used for the sweep. MuJoCo's implicit integrators only
# treat *velocity*-dependent (damping-like) terms implicitly, not the
# position-feedback (kp) term of a stiff position actuator, so a very large
# KP still needs a small timestep to stay stable -- there's no real-time
# requirement here, so err small. Tighten further (e.g. 1e-5) if you see
# "Nan, Inf or huge value" warnings; loosen (e.g. the model's own default)
# if KP is modest and you want the sweep to run faster.
SIM_TIMESTEP = 1.0e-4

# Implicit integrators handle joint/tendon damping and actuator damping
# implicitly, which helps stability; implicitfast is a good default here.
SIM_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

# Where to save the resulting plot. Set to None to just show it instead.
PLOT_PATH = "joint_coupling_sweep.png"

# ---------------------------------------------------------------------------


def build_model() -> mujoco.MjModel:
    """Load the model, disable gravity, and attach the sweep actuator."""
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = SIM_INTEGRATOR
    spec.option.timestep = SIM_TIMESTEP

    spec.add_actuator(
        name="sweep_position_actuator",
        target=ACTUATED_JOINT,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[ACTUATOR_KP, 0.0, 0.0] + [0.0] * 7,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -ACTUATOR_KP, -ACTUATOR_KV] + [0.0] * 7,
        ctrllimited=True,
        ctrlrange=[min(JOINT_RANGE), max(JOINT_RANGE)],
        forcelimited=True,
        forcerange=[-ACTUATOR_FORCERANGE, ACTUATOR_FORCERANGE],
    )

    return spec.compile()


def locked_joint_slices(
    model: mujoco.MjModel, free_joints: list[str]
) -> tuple[list[slice], list[slice]]:
    """qpos/qvel slices for every joint NOT in `free_joints`."""
    qpos_slices, dof_slices = [], []
    ndof_per_type = {
        mujoco.mjtJoint.mjJNT_FREE: (7, 6),
        mujoco.mjtJoint.mjJNT_BALL: (4, 3),
        mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
        mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
    }
    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if name in free_joints:
            continue
        nq, nv = ndof_per_type[mujoco.mjtJoint(model.jnt_type[jnt_id])]
        qpos_adr = model.jnt_qposadr[jnt_id]
        dof_adr = model.jnt_dofadr[jnt_id]
        qpos_slices.append(slice(qpos_adr, qpos_adr + nq))
        dof_slices.append(slice(dof_adr, dof_adr + nv))
    return qpos_slices, dof_slices


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for joint_name in (ACTUATED_JOINT, OBSERVED_JOINT):
        if joint_name not in FREE_JOINTS:
            raise ValueError(f"{joint_name!r} must be listed in FREE_JOINTS")

    locked_qpos_slices, locked_dof_slices = locked_joint_slices(model, FREE_JOINTS)
    locked_qpos0 = [data.qpos[s].copy() for s in locked_qpos_slices]

    def enforce_locks() -> None:
        for s, q0 in zip(locked_qpos_slices, locked_qpos0):
            data.qpos[s] = q0
        for s in locked_dof_slices:
            data.qvel[s] = 0.0
        mujoco.mj_forward(model, data)

    actuator_id = model.actuator("sweep_position_actuator").id
    observed_jnt_id = model.joint(OBSERVED_JOINT).id
    observed_qpos_adr = model.jnt_qposadr[observed_jnt_id]
    actuated_jnt_id = model.joint(ACTUATED_JOINT).id
    actuated_qpos_adr = model.jnt_qposadr[actuated_jnt_id]

    setpoints = np.linspace(JOINT_RANGE[0], JOINT_RANGE[1], N_SETPOINTS)
    actuated_positions = np.zeros(N_SETPOINTS)
    observed_positions = np.zeros(N_SETPOINTS)

    enforce_locks()
    for i, setpoint in enumerate(setpoints):
        data.ctrl[actuator_id] = setpoint
        for _ in range(SETTLE_STEPS):
            enforce_locks()
            mujoco.mj_step(model, data)
        actuated_positions[i] = data.qpos[actuated_qpos_adr]
        observed_positions[i] = data.qpos[observed_qpos_adr]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(actuated_positions, observed_positions, marker="o", markersize=3)
    ax.set_xlabel(f"{ACTUATED_JOINT} position")
    ax.set_ylabel(f"{OBSERVED_JOINT} position")
    ax.set_title(f"{OBSERVED_JOINT} vs {ACTUATED_JOINT}")
    ax.grid(True)
    fig.tight_layout()

    if PLOT_PATH is not None:
        fig.savefig(PLOT_PATH, dpi=150)
        print(f"Saved plot to {PLOT_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
