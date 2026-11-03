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
`Mjlab-Jump-Vertical-<variant>`'s. `-Body-LowerBody-` variants track the
lower-body motion; `-Body-FullBody-` variants track its copy
`motions/jump_ocp_solution_mjlab_arms.npz`, whose arm joints stay at their
Kangaroo Full default positions in every frame, so imitation keeps the arms
still. That copy is written by

```bash
uv run python -m pal_mjlab.scripts.add_default_arms_to_motion \
  --input-file motions/jump_ocp_solution_mjlab.npz \
  --output-file motions/jump_ocp_solution_mjlab_arms.npz
```

(joints in `SIMPLE_MODEL_JOINT_ORDER`, bodies re-computed by forward kinematics
of the simple full-body Kangaroo; torques and contact wrenches copied as-is).

In the free jump the arms are the policy's to use: on `-Body-FullBody-`
variants the grounded posture reward gives every arm joint a wide
`ARM_POSE_STD` (0.5 rad, legs and waist keep 0.05 rad), `joint_symmetry`
(currently commented out of the config) also pairs the arms (`arm_*_1`/`_3` equal and opposite, `_2`/`_4` equal), and the
velocity task's `self_collisions` cost is kept. The landing pose kernel
(`landing_completed`, settling) is unchanged, so landings still pay most with
the arms back at their default. `slurm_h100_tracking_jump_arms.bash` runs both
stages for a full-body variant.

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
**settled** for the settle hold time, or at the episode limit. Settled
(`JumpHeightCommand.settled`) means both feet in contact and every one of the
posture reward's actuated joints slower than 0.5 rad/s (`SETTLE_JOINT_SPEED`),
in any pose and at any base speed. The hold time is 0 s (`SETTLE_HOLD_TIME`), so
the first settled step ends the episode. There is no settle curriculum: these
bounds stay the same throughout training and in play. The hold counter
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
| `horizontal_drift` | Base more than 1 m (`MAX_JUMP_DRIFT`) from the environment origin before landing (stance, flight), more than 3 m (`MAX_LANDED_DRIFT`) once both feet are down |
| `excessive_rotation` | Base turned more than 30° (`MAX_ROTATION_DEG`), about any axis, from its orientation at the episode start (`reset_quat`, restored with replayed states) |
| `jumped_again` | A second takeoff after landing (LANDED or SETTLED): one jump per episode |
| `idle_before_jump` | Only before takeoff (STANCE): more than 2.5 s in stance without taking off (`MAX_STANCE_TIME`), or more than 0.5 s in total standing still there (`MAX_IDLE_TIME`, base slower than 0.1 m/s) |
| `jump_finished` | In SETTLED for the settle hold time (see above): the task's success, a true terminal state for PPO (`time_out=False`, no value bootstrapped after it); `play-jump` shows it as a success |
| `time_out` | 8 s episode limit |

The existing illegal-ground-contact termination remains; the velocity task's
fall-angle check is removed. Per-axis attitude and walking in stance are not
terminal (`walking_before_jump` and `excessive_attitude` remain in `mdp` but
are not configured). Horizontal drift is measured from the environment origin. Apex error compares the measured and commanded
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
Every state is tagged with the height curriculum stage (stair) it was stored
in. While the curriculum is not failing (stage success rate at least 50 %, or
not known yet), at least `REPLAY_KEEP_PREVIOUS_FRACTION = 0.5` of the capacity
stays with states of earlier stages, as long as there are that many: new states
then only replace earlier-stage states above that share, otherwise
lower-scoring current-stage ones. While it fails, new states may replace any.
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
`jump_height_curriculum` shifts the whole 10 cm range of world-frame
jump-frame apex heights in stairs, starting from
`HEIGHT_INITIAL_RANGE = (1.715, 1.815)`:

- **Hold.** The initial range is the apex of the reference motion
  `motions/jump_ocp_solution_mjlab.npz` (`base_link` peaks at 1.278 m, the jump
  frame about 0.485 m higher), rounded to 1.765 m, ± 5 cm. It is held for
  `HEIGHT_HOLD_STEPS` (500 PPO iterations of `ITERATION_STEPS = 64` steps).
- **Stairs.** From then on the curriculum decides once every
  `HEIGHT_STAIR_STEPS`, the first decision at the end of the hold. While jumps
  succeed it climbs `HEIGHT_STEP_UP = 5 cm`, until the upper bound reaches
  3.0 m (`HEIGHT_FINAL_RANGE`'s top): `HEIGHT_STAIRS = 24` stairs, the last
  one capped. The interval spreads them so that with no setbacks the top stair
  (2.900–3.000 m) comes by iteration 20,000 (`HEIGHT_FINAL_STEP`), about 848
  iterations apart.
- **Success gate.** While the smoothed success rate (the share of episodes
  ending on `jump_finished`) is below `HEIGHT_SUCCESS_THRESHOLD = 0.5`, each
  decision instead moves back `HEIGHT_STEP_DOWN = 1 cm`, never below the
  initial range. The rate is an exponential moving average with a per-episode
  weight of `1 / HEIGHT_SUCCESS_WINDOW` (4,096 episodes). Replayed episodes
  start after takeoff and are left out.
- **Apex-error gate.** While success is at or above the threshold but the
  smoothed apex error (`|apex - target|` of finished, non-replayed episodes
  that reached their apex, same moving average per jump) exceeds
  `HEIGHT_MAX_APEX_ERROR = 0.05` m, the range stays.

Every move starts a new stage (`stage` counts them, the replay buffer tags its
states with it) and restarts both averages, so each decision judges only the
current stair. Play samples `HEIGHT_FINAL_RANGE = (2.800, 3.000)`. The
curriculum logs `min_height`, `max_height`, `offset`, `stage`, `success_rate`
and `apex_error` under `Curriculum/jump_height/`. Its state is not a function
of the step counter, so `JumpOnPolicyRunner` saves it in the checkpoint's
`infos["curriculum_state"]` and restores it on resume. Loading only the actor
from a tracking checkpoint starts from the initial range.

## Metrics

On every episode end the jump command logs, besides `apex_height_error`
(per episode that reached its apex), `Metrics/jump_height/full_jump`: per
ended episode that started standing (replayed restarts and the initial reset
are left out), 1 if it went through the whole jump (stance, takeoff, flight,
both feet down, settled and held for the settle hold time, i.e. ended on
`jump_finished`), else 0. Its average is the share of episodes doing the full
jump.

## PPO

`rl_cfg.py` keeps the full-model velocity PPO except for the credit horizon,
since the task's value comes in sparse payouts (apex, then landing) seconds
apart:

- `gamma = 1 - 1/(1.5 · 215) ≈ 0.9969` (`JUMP_HORIZON_SCALE` ×
  `JUMP_MOTION_FRAMES`): the effective horizon `1 / (1 - gamma)` is 1.5 times
  the reference jump npz's length, 322.5 policy steps (6.45 s), instead of
  0.99's 100 steps.
- `lam = 0.98` (`JUMP_LAM`), from 0.95.
- `num_steps_per_env = ITERATION_STEPS = 64` (1.28 s rollouts), from 24. The
  curricula count PPO iterations of this length.

After loading only a pretrained actor (from a tracking checkpoint),
`JumpOnPolicyRunner` trains only the critic for the first
`CRITIC_WARMUP_ITERATIONS = 50` PPO updates: the actor's weights, action std
and observation normalizer stay as loaded, and the learning rate stays fixed
rather than adapting to the (zero) KL. Resumed jump checkpoints skip it.

## Rewards

The only retained velocity rewards are `body_ang_vel`, `angular_momentum`,
`dof_pos_limits`, and `action_rate_l2`, with their existing weights. There is
no orientation (`upright`) or foot-slip term; attitude is bounded by the
`excessive_attitude` termination and the angular-velocity/momentum costs.

The additional rewards are:

| Term | Weight | Behavior |
| --- | --- | --- |
| `jump_height` | 45 | Once at measured apex: Gaussian(error, 0.10 m) + 3 × Laplacian(error, 0.05 m) |
| `estimated_jump_height` | 20 | Same kernel during ascent, using `h + max(vz, 0)^2 / (2g)` |
| `pose` | 0.1 | Per-joint posture reward, only while a foot touches the ground |
| `start_orientation` | 1 | `exp(-(angle / 0.25)^2)`, angle = the base's rotation about any axis from its orientation at the episode start (`reset_quat`), in every phase |
| `horizontal_velocity` | 2 | Suppress horizontal motion (stay in place), inspired by the authors' implementation |
| `joint_symmetry` | −7, **commented out for now** | Σ (q_left − s·q_right)² over leg joints 1–5 and knee (and arm joints 1–4 with arms): s = +1 for legs 2, 4, knee and arms 2, 4 (equal), s = −1 for legs 1, 3, 5 and arms 1, 3 (mirrored) |
| `catch_landing` | 3 | For 0.3 s after both feet touch down (LANDED or SETTLED): `clip(vz / vz_touchdown, 0, 1)`, 1 while the base keeps sinking as fast as it touched down, less as it brakes, 0 once it stops or rises |
| `landing_completed` | 120 | Once, on the `jump_finished` step, an impulse (divided by step_dt): `exp(-mean((q-q_default)^2/(2·std_standing)^2))`, from e⁻¹ at the pose tolerance to 1 at the reference |

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
