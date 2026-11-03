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

Jump height means the **world-frame z of the jump frame at its apex**, in
metres; the plane terrain is the world z = 0 plane. The jump frame is the
`torso_up_cameras_frame` site (on `pelvis_2_link`, between the front and rear
upper RGBD cameras in x, at y = 0 and at their height). Targets, the measured
apex (`peak_height`, the frame's maximum world z in flight), the ballistic
estimate and the touchdown checks all use it. The frame sits
0.4849 m above the base link with both pelvis joints at zero, so Kangaroo Full
stands with it at about 1.415 m; the hardcoded heights (`HEIGHT_INITIAL_RANGE`,
`HEIGHT_FINAL_RANGE`, `MIN_JUMP_HEIGHT`), formerly base-link heights, are
0.485 m higher, rounded to 5 mm. Jump tracking targets the frame's apex in the
reference motion, from its tracked `pelvis_2_link` poses.

The actor observes what the full-model tracking actor does
(`full_model_tracking_actor_observations` in
`tasks/tracking/kangaroo_full/env_cfgs.py`): `command`, `base_ang_vel`,
`joint_pos`, `joint_vel`, `actions`, `imu_projected_gravity`, `base_lin_acc`,
with the tracking noise and ordering, followed by the velocity-env
`base_lin_vel` (IMU velocimeter) and the world-frame `base_height_world`, each
with ±3 cm (cm/s) noise. The `command` term is a single scalar containing the
sampled world-frame apex until touchdown, then zero to request settling, so no
extra phase observation is needed. The target used for rewards remains fixed
for the episode. The critic keeps the flat-velocity observation terms with the
same scalar command, plus the privileged `jump_frame_height` (the jump frame's
world z, which targets and apex use) and a clean `base_height_world`. Actor and
critic both stack the last `OBSERVATION_HISTORY = 10` frames of every term,
flattened, in this task and in jump tracking (`configure_jump_observations`).

Touchdown (FLIGHT → LANDED) requires **both** feet in contact; a single-foot
touch keeps the jump in flight. An episode contains one jump and ends once the
landed robot has stayed
**settled** for the settle hold time, or at the episode limit. The settle
curriculum (`SETTLE_STAGES`) holds a loose terminal state (joints slower than
0.5 rad/s, no hold) for 150 iterations, then by iteration 800 tightens it only
halfway to full tightness (0.1 rad/s, `LANDING_HOLD_TIME = 0.2` s): a
0.3 rad/s limit and a 0.1 s hold, which play also uses. Settled
(`JumpHeightCommand.settled`) means both feet in contact, posture close to the
default pose — `exp(-mean((q-q_default)^2/std^2)) ≥ exp(-1)`, i.e. RMS error
within twice `std_standing` (2 × 0.05 = 0.1, `SETTLE_POSE_STD_SCALE`) over the posture reward's actuated joints — every
such joint slower than 0.2 rad/s (`SETTLE_JOINT_SPEED`), and the base slower than 0.05 m/s and
0.1 rad/s (`settle_*` fields of `JumpHeightCommandCfg`). The hold counter
(`settled_steps`) restarts whenever any condition breaks. The state machine is
STANCE → FLIGHT → LANDED → SETTLED: SETTLED, the last state, is entered from
LANDED while settled and falls back to LANDED when settling breaks.
`jump_finished` fires after the hold time in it, and `landing_completed` pays once, on
that step (sparse; see Rewards).
Initial spawn drops cannot
count as takeoff: ordinary episodes must first establish contact, then lose all
foot contacts with positive vertical speed. Apex rewards require a minimum
flight duration and a minimum world-frame apex (`min_jump_height`,
`MIN_JUMP_HEIGHT = 1.445` m: 3 cm above the standing jump frame), and cannot be collected again by bouncing. External
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
| `sideways_takeoff` | Horizontal base speed above 0.75 m/s at takeoff |
| `horizontal_drift` | Either horizontal axis more than 0.60 m from the environment origin, in any phase |
| `jump_finished` | In SETTLED for 0.5 s (see above); a timeout rather than failure |
| `time_out` | 8 s episode limit |

The existing illegal-ground-contact termination remains; the velocity task's
fall-angle check is removed. A second takeoff after landing, attitude, and
time spent in stance are not terminal (`walking_before_jump`,
`idle_before_jump` and `excessive_attitude` remain in `mdp` but are not
configured). Horizontal drift is measured from the environment origin. Apex error compares the measured and commanded
world-frame apex of the jump frame. Takeoff detection uses the reference's 0.75 m/s upward
speed threshold. Idle time accumulates across separate pauses in stance and
clears on takeoff, matching the reference. All counters and event flags reset
per environment.

Bad touchdown and horizontal drift are failure conditions, with no separate
landing-drift cost. The achieved and estimated jump-height rewards remain.

## Replaying states of the best finished jumps

In training (not play), `jump_state_replay` (a `step`-mode event) restarts
`REPLAY_PROBABILITY = 0.2` of episodes from a stored state of an earlier jump
that ended on `jump_finished`. Every env records at most one state per
episode: on reset it draws one of the jump's state transitions uniformly
(takeoff, apex, touchdown with both feet down, or settling) and a uniform
offset of 0 to `REPLAY_MAX_OFFSET = 0.3` s (0–15 steps), and records the state
that many steps after the first time that transition happens. A state that is
not full is discarded: one whose observation histories still hold backfilled
frames, or one taken on the step its episode ends (a restart would end at
once). When an episode finishes the jump, its recorded state joins the buffer
of `REPLAY_CAPACITY = 4096` states, scored with the episode's return (the sum
of every reward term's episode sum). Episodes that end any other way, that end
before their recording step, or that never reach their drawn transition add
nothing. Once the buffer is full, a new state
replaces the lowest-scoring one only if it scores higher. Restarts draw
uniformly among the best `REPLAY_TOP_FRACTION = 0.25` of the buffer by score.
Episodes that were themselves restarted from the buffer are never stored,
since their return only covers the part after the restart.

A state holds qpos/qvel (root xy relative to the env origin), the action
history, the jump command's state machine, the contact sensors' air/contact
timers, and the history of every actor and critic observation term, so the
restarted episode continues as the original did. Its target apex is the stored
one plus U(-5 mm, 5 mm) (`REPLAY_TARGET_NOISE`), also shifted in the observed
command's history. Replayed episodes inherit their apex, so they are left out
of `apex_height_error`. The buffer starts empty and lives in memory only; it is
not checkpointed.

## Height sampling curriculum

Targets are uniform within the current range and are sampled only on reset.
`jump_height_curriculum` moves both bounds linearly from
`HEIGHT_INITIAL_RANGE = (1.715, 1.815)` to `HEIGHT_FINAL_RANGE = (2.800, 3.000)`
(world-frame jump-frame apex heights):

- **Hold.** The initial range is the apex of the reference motion
  `motions/jump_ocp_solution_mjlab.npz` (`base_link` peaks at 1.278 m, the jump
  frame about 0.485 m higher), rounded to 1.765 m, ± 5 cm. It is held for
  `HEIGHT_HOLD_STEPS` (1,500 PPO iterations of 24 steps).
- **Ramp.** After that, progress rises by `1 / (HEIGHT_FINAL_STEP -
  HEIGHT_HOLD_STEPS)` per environment step, so with no setbacks the final range
  is reached at iteration 10,000 (`HEIGHT_FINAL_STEP`).
- **Success gate.** While the smoothed success rate (the share of episodes
  ending on `jump_finished`) is below `HEIGHT_SUCCESS_THRESHOLD = 0.5`,
  progress instead falls at `HEIGHT_BACKOFF_RATIO = 0.5` times the ramp rate,
  never below the initial range. It rises again once the rate is back at 50 %.
  The rate is an exponential moving average with a per-episode weight of
  `1 / HEIGHT_SUCCESS_WINDOW` (4,096 episodes). Replayed episodes start after
  takeoff and are left out.

Play samples the final range. The curriculum logs `min_height`, `max_height`,
`progress` and `success_rate` under `Curriculum/jump_height/`. Progress is not
a function of the step counter, so `JumpOnPolicyRunner` saves it (with the
success rate) in the checkpoint's `infos["curriculum_state"]` and restores it
on resume. Loading only the actor from a tracking checkpoint starts from the
initial range.

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
| `pose` | 0.1 | Per-joint posture reward, only while a foot touches the ground |
| `horizontal_velocity` | 1 | Suppress horizontal motion, inspired by the authors' implementation |
| `ground_force_l2` | −0.05 | Squared foot-ground forces, normalized by 500 N |
| `joint_symmetry` | −1 | Σ (q_left − s·q_right)² over leg joints 1–5 and knee: s = +1 for 2, 4, knee (equal), s = −1 for 1, 3, 5 (mirrored) |
| `landing_completed` | 20 | Once, on the `jump_finished` step (SETTLED held 0.5 s), an impulse (divided by step_dt): `exp(-mean((q-q_default)^2/(2·std_standing)^2))`, from e⁻¹ at the pose tolerance to 1 at the reference |

The apex reward compensates for mjlab's `dt` scaling so its payout is independent
of policy frequency; all continuous rewards remain rates. The ballistic estimate
uses the configured simulation gravity and is disabled on descent.

`pose` (`grounded_posture`) is `exp(-mean((q-q_default)^2 / std^2))` with the
flat task's per-joint `std_standing` tolerances, paid only on steps where at
least one foot is in contact (the jump command's `feet_ground_contact`
sensor). Airborne steps get no posture reward at all, so flight carries no
posture signal. Tune `std` in `cfg.rewards["pose"].params`. Joint order is
resolved before building the std tensor; passive linkage/screw joints follow
the existing posture selection.

`joint_symmetry` builds its left/right index pairs and mirror mask once at
init from joint names (`_left_` → `_right_`); `mirrored_joints` selects the
sign-flipped ones. Both legs share joint axes, so hip yaw/roll and ankle roll
mirror while hip/ankle pitch and knee match. The default pose is symmetric to
≤1.6 mrad under this convention. The former velocity
tracking, air-time, clearance, swing-height, collision-cost, convex-hull, and
joint-velocity-limit rewards are absent as requested. Reward weights and height
ranges are starting settings for Kangaroo; learning performance requires a full
training run.
