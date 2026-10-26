# Kangaroo Full vertical jumping

This separate task family adapts Section V-B / Table IV of
[Olsen et al., arXiv:2510.24584](https://arxiv.org/pdf/2510.24584).
The authors' implementation is
[NTNU ARL Olympus Lab](https://github.com/ntnu-arl/olympus_lab), inspected at
commit `f863c4b52b167c53e4e559017f2a2d7d7d2feaa5`. In particular:

- [Vertical environment and rewards](https://github.com/ntnu-arl/olympus_lab/blob/f863c4b52b167c53e4e559017f2a2d7d7d2feaa5/envs/vertical_jump/vertical_jump_env.py)
- [Jump state machine](https://github.com/ntnu-arl/olympus_lab/blob/f863c4b52b167c53e4e559017f2a2d7d7d2feaa5/envs/vertical_jump/jump_state_machine.py)
- [Ballistic height estimate](https://github.com/ntnu-arl/olympus_lab/blob/f863c4b52b167c53e4e559017f2a2d7d7d2feaa5/utilities/projectile_motion.py)
- [Reference-state initialization and curriculum](https://github.com/ntnu-arl/olympus_lab/blob/f863c4b52b167c53e4e559017f2a2d7d7d2feaa5/envs/vertical_jump/curriculum.py)

The implementation uses mjlab managers and the existing Kangaroo Full robot,
actions, sensors, joint mappings, randomization, and PPO configuration. It does
not depend on Isaac Lab or the reference repository.

## Run

```bash
uv run train Mjlab-Jump-Vertical-Pal-Kangaroo-Full --env.scene.num-envs 4096
uv run play Mjlab-Jump-Vertical-Pal-Kangaroo-Full --agent zero
uv run play Mjlab-Jump-Vertical-Pal-Kangaroo-Full --checkpoint-file /path/to/model.pt
```

The short ID uses the Full robot, actuator transmission, prismatic femur,
tendon MJCF, raw ankle observations, and builtin actuators. All 46 supported
Kangaroo **Full** flat-model combinations also have explicit IDs: replace
`Mjlab-Velocity-Flat-` with `Mjlab-Jump-Vertical-` in their existing IDs (omit
velocity curriculum suffixes). For example:

```bash
uv run train Mjlab-Jump-Vertical-Pal-Kangaroo-Full-Transmission-Lut-Femur-Linkage-Mjcf-Tendons-Body-LowerBody-AnkleObs-AnkleNormalized-ActuatorModel-DcMotor --env.scene.num-envs 4096
```

## Pretraining by tracking the jump motion

`Mjlab-Tracking-Jump-<variant>` (`tasks/tracking/kangaroo_full_jump/`) tracks
`motions/jump_ocp_solution_mjlab.npz` with the full-model tracking rewards and
critic, but its actor observes this task's `jump_height` command instead of the
reference joints. The target is the motion's world-frame base apex (1.278 m) +
U(−0.1, 0.1) m per episode, zeroed once both feet land, from the same
`JumpHeightCommand` state machine. Every episode is one pass through the motion
from its first frame. Actor observations, actions, and actor network equal
`Mjlab-Jump-Vertical-<variant>`'s. The motion is lower body, so only
`-Body-LowerBody-` variants exist.

```bash
uv run train Mjlab-Tracking-Jump-Pal-Kangaroo-Full-Transmission-Actuator-Femur-Prismatic-Mjcf-Tendons-Body-LowerBody-AnkleObs-AnkleRaw-ActuatorModel-Builtin --env.scene.num-envs 4096
uv run train Mjlab-Jump-Vertical-Pal-Kangaroo-Full-Transmission-Actuator-Femur-Prismatic-Mjcf-Tendons-Body-LowerBody-AnkleObs-AnkleRaw-ActuatorModel-Builtin --env.scene.num-envs 4096 --agent.resume True --agent.load-run <tracking run> --agent.load-checkpoint <model_N.pt>
```

The jump tasks' `JumpOnPolicyRunner` recognises a checkpoint whose critic does
not fit (the tracking critic observes the motion) and loads only its actor
(weights, observation normalizer, action std). Training then starts at
iteration 0 with a fresh critic and optimizer and the jump task's own step
counter, so the height curriculum starts at its first stage. Jump checkpoints
resume as usual. `--agent.load-run` is resolved under the jump experiment's
log directory, so copy the tracking run there (or pass `--wandb-run-path`).

## Commands, observations, and episodes

Jump height means the **world-frame z of the base link at its apex**, in
metres; the plane terrain is the world z = 0 plane. Kangaroo Full stands with its
base at 0.93 m, so a 1.278 m command asks for a 0.348 m rise. It does not mean
foot clearance or a rise above standing. Targets, the measured apex
(`peak_height`, the base's maximum world z in flight), the ballistic estimate,
the posture blend, and the touchdown checks all use this world-frame height.

The actor observes exactly what the full-model tracking actor does
(`full_model_tracking_actor_observations` in
`tasks/tracking/kangaroo_full/env_cfgs.py`, shared by both tasks): `command`,
`base_ang_vel`, `joint_pos`, `joint_vel`, `actions`, `imu_projected_gravity`,
`base_lin_acc`, with the tracking noise, ordering, and no history. Only the
`command` term differs: it is a single scalar containing the sampled world-frame
apex until
touchdown, then zero to request settling, so no extra phase observation is
needed. The target used for rewards remains fixed for the episode. The critic
keeps the flat-velocity observation terms with the same scalar command, plus
the privileged `base_height` (world-frame base-link z, the frame targets and
apex are measured in), and stacks the last `OBSERVATION_HISTORY = 10` frames of
every term, flattened.

Touchdown (FLIGHT → LANDED) requires **both** feet in contact; a single-foot
touch keeps the jump in flight. An episode contains one jump and ends once the
landed robot has stayed
**settled** for `LANDING_HOLD_TIME = 0.5` s, or at 8 s. Settled
(`JumpHeightCommand.settled`) means both feet in contact, posture close to the
default pose — `exp(-mean((q-q_default)^2/std^2)) ≥ exp(-1)`, i.e. RMS error
within one `std_standing` (0.05) over the posture reward's actuated joints — every
such joint slower than 0.1 rad/s, and the base slower than 0.05 m/s and
0.1 rad/s (`settle_*` fields of `JumpHeightCommandCfg`). The hold counter
(`settled_steps`) restarts whenever any condition breaks. The state machine is
STANCE → FLIGHT → LANDED → SETTLED: SETTLED, the last state, is entered from
LANDED while settled and falls back to LANDED when settling breaks.
`landing_completed` pays on every step in SETTLED, and `jump_finished` fires
after 0.5 s in it.
Initial spawn drops cannot
count as takeoff: ordinary episodes must first establish contact, then lose all
foot contacts with positive vertical speed. Apex rewards require a minimum
flight duration and a minimum world-frame apex (`min_jump_height`, 0.96 m:
standing height + `MIN_JUMP_RISE`), and cannot be collected again by bouncing. External
push events are removed.

Every episode, in training and play, starts standing still at the nominal
root pose, heading, and joint configuration: there are no flight starts and
`reset_base` has empty pose and velocity ranges. Play samples the full height
range.

## Terminations

The selected conditions follow the authors' `_get_dones()`, with the touchdown
apex tolerance widened to 0.30 m. Parameters live in `cfg.terminations`.

| Term | Trigger |
| --- | --- |
| `bad_touchdown` | At touchdown only: apex error above 0.30 m, or base height below 0.20 m above the ground |
| `jumped_again` | A new takeoff after landing |
| `sideways_takeoff` | Horizontal base speed above 0.75 m/s at takeoff |
| `walking_before_jump` | Horizontal displacement from the sampled reset position above 0.20 m while in stance |
| `idle_before_jump` | Still in stance after 3 s, or more than 1 s cumulative idle time at base speed below 0.10 m/s |
| `excessive_attitude` | Absolute roll or pitch, or wrapped yaw change from the reset heading, above 25 degrees |
| `horizontal_drift` | Either horizontal axis more than 0.20 m from the environment origin, in any phase |
| `jump_finished` | In SETTLED for 0.5 s (see above); a timeout rather than failure |
| `time_out` | 8 s episode limit |

The existing illegal-ground-contact termination remains. The attitude check
replaces the velocity task's fall-angle check. Yaw is relative to the reset
heading (the nominal heading, since reset yaw is no longer randomized).
Walking is measured from the reset position and horizontal drift from the
environment origin, which now coincide. Heights are world-frame base-link z; apex
error compares the measured and commanded world-frame apex. Takeoff detection uses the reference's 0.75 m/s upward
speed threshold. Idle time accumulates across separate pauses in stance and
clears on takeoff, matching the reference. All counters and event flags reset
per environment.

Bad touchdown and horizontal drift are failure conditions, with no separate
landing-drift cost. The achieved and estimated jump-height rewards remain.

## Height sampling curriculum

Targets are uniform within the current range and are sampled only on reset.
`HEIGHT_STAGES` in `env_cfgs.py` defines a time-based expansion:

| Environment steps | Height range (m) | PPO iteration with 24-step rollouts |
| --- | --- | --- |
| 0 | 1.230–1.330 | 0 |
| 24,000 | 1.130–1.430 | 1,000 |
| 48,000 | 1.205–1.480 | 2,000 |
| 72,000 | 1.280–1.530 | 3,000 |
| 96,000 | 1.355–1.580 | 4,000 |
| 120,000 | 1.430–1.630 | 5,000 |
| 144,000 | 1.505–1.680 | 6,000 |
| 168,000 | 1.580–1.730 | 7,000 |
| 192,000 | 1.655–1.780 | 8,000 |
| 216,000 | 1.730–1.830 | 9,000 |
| 240,000 | 1.805–1.880 | 10,000 |
| 264,000 | 1.855–1.930 | 11,000 |
| 288,000 | 1.905–1.980 | 12,000 |
| 312,000 | 1.925–2.000 | 13,000 |

Ranges are world-frame base apex heights. The first two stages are the
world-frame apex of the reference motion `motions/jump_ocp_solution_mjlab.npz`
(`base_link` peaks at 1.278 m), rounded to 1.280 m, ± 5 cm, then ± 15 cm. Each later stage raises the
upper bound 5 cm and narrows the range 2.5 cm, so the lower bound rises 7.5 cm,
until the range is 7.5 cm wide at iteration 10,000. That 7.5 cm range then keeps
rising 5 cm per stage, with the last stage capped at a 2 m upper bound.
Play samples the last stage.

Tune `cfg.curriculum["jump_height"].params["stages"]` for training and
`cfg.commands["jump_height"].height_range` for play. The schedule uses global
environment steps, independent of environment count. The existing runner saves
and restores that counter; the curriculum resumes at the corresponding stage.
Unlike the reference implementation's success-based initialization curriculum,
this schedule directly expands command height while retaining easy commands.

## Rewards

The only retained velocity rewards are `body_ang_vel`, `angular_momentum`,
`dof_pos_limits`, and `action_rate_l2`, with their existing weights. There is
no orientation (`upright`) or foot-slip term; attitude is bounded by the
`excessive_attitude` termination and the angular-velocity/momentum costs.

The additional rewards are:

| Term | Weight | Behavior |
| --- | --- | --- |
| `jump_height` | 20 | Once at measured apex: Gaussian(error, 0.10 m) + 3 × Laplacian(error, 0.05 m) |
| `estimated_jump_height` | 20 | Same kernel during ascent, using `h + max(vz, 0)^2 / (2g)` |
| `pose` | 1 | Height-dependent per-joint posture reward |
| `horizontal_velocity` | 1 | Suppress horizontal motion, inspired by the authors' implementation |
| `ground_force_l2` | −0.05 | Squared foot-ground forces, normalized by 500 N |
| `joint_symmetry` | −1 | Σ (q_left − s·q_right)² over leg joints 1–5 and knee: s = +1 for 2, 4, knee (equal), s = −1 for 1, 3, 5 (mirrored) |
| `landing_completed` | 50 | Every step in the last state, SETTLED: `exp(-mean((q-q_default)^2/std_standing^2))`, from e⁻¹ at the pose tolerance to 1 at the reference |

The apex reward compensates for mjlab's `dt` scaling so its payout is independent
of policy frequency; all continuous rewards remain rates. The ballistic estimate
uses the configured simulation gravity and is disabled on descent.

`pose` is `exp(-mean((q-q_default)^2 / std(h)^2))`. Each joint's std interpolates
linearly from `std_ground` to `std_airborne` across `height_range`, the
world-frame base heights 0.93–0.98 m (standing height + `POSTURE_BLEND_RISE`),
clamped outside that interval. The defaults reuse the flat task's looser running and
tighter walking tolerances, respectively. Tune those dictionaries and
`height_range` in `cfg.rewards["pose"].params`. Joint order is resolved before
building the std tensors; passive linkage/screw joints follow the existing
posture selection.

`joint_symmetry` builds its left/right index pairs and mirror mask once at
init from joint names (`_left_` → `_right_`); `mirrored_joints` selects the
sign-flipped ones. Both legs share joint axes, so hip yaw/roll and ankle roll
mirror while hip/ankle pitch and knee match. The default pose is symmetric to
≤1.6 mrad under this convention. The former velocity
tracking, air-time, clearance, swing-height, collision-cost, convex-hull, and
joint-velocity-limit rewards are absent as requested. Reward weights and height
ranges are starting settings for Kangaroo; learning performance requires a full
training run.
