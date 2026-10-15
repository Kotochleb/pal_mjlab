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

## Commands, observations, and episodes

Jump height means **root/base apex rise above nominal standing root height**, in
metres, relative to each environment's origin. A 0.20 m command targets a root
apex 0.20 m above the model's nominal 0.91 m standing root height. It does not
mean foot clearance or an absolute root height of 0.20 m.

Actor and critic retain their respective flat-velocity observation terms,
noise, ordering, and joint mappings. The existing `command` term becomes a
single scalar. It contains the sampled rise until touchdown, then zero to
request settling, so no extra phase observation is needed. The target used for
rewards remains fixed for the episode. No additional base-height or velocity
measurement is added to the actor. Default dimensions are actor 84 / critic 102.

An episode contains one jump and ends after 0.5 s of stable two-foot contact,
or at 5 s. Existing fall/contact terminations remain. Initial spawn drops cannot
count as takeoff: ordinary episodes must first establish contact, then lose all
foot contacts with positive vertical speed. Apex rewards require a minimum
flight duration and rise, and cannot be collected again by bouncing. External
push events are removed.

Training initializes 20% of episodes along a ballistic ascent/descent with the
model's nominal joint configuration. This simplified reference-state
initialization teaches flight and landing without quadruped-specific IK. These
episodes cannot receive the achieved-apex reward. Set
`cfg.commands["jump_height"].flight_reset_probability = 0` to disable it.
Play always starts on the ground and samples the full height range.

## Height sampling curriculum

Targets are uniform within the current range and are sampled only on reset.
`HEIGHT_STAGES` in `env_cfgs.py` defines a time-based expansion:

| Environment steps | Height range (m) | PPO iteration with 24-step rollouts |
| --- | --- | --- |
| 0 | 0.05–0.15 | 0 |
| 48,000 | 0.05–0.25 | 2,000 |
| 120,000 | 0.05–0.40 | 5,000 |
| 240,000 | 0.05–0.60 | 10,000 |

Tune `cfg.curriculum["jump_height"].params["stages"]` for training and
`cfg.commands["jump_height"].height_range` for play. The schedule uses global
environment steps, independent of environment count. The existing runner saves
and restores that counter; the curriculum resumes at the corresponding stage.
Unlike the reference implementation's success-based initialization curriculum,
this schedule directly expands command height while retaining easy commands.

### LP-ACRL alternative

Every jump ID, including the short one, also has a `-Curriculum-LpAcrl`
variant. It replaces the time-based schedule with the learning-progress
teacher used by the velocity tasks (`pal_mjlab.tasks.velocity.mdp.lp_acrl`):
the final 0.05–0.60 m range is split into `HEIGHT_BINS = 15` equal bins
(≈3.7 cm each), and each one is a task. At reset, the finished episode's total
return is credited to its bin. Every 1200 environment steps, bins are reweighted
by softmax(Δmean return / β), with 5% uniform exploration. The new episode then
draws a bin and samples its target uniformly within that bin. Play uses the
same range with uniform bin weights. `LpAcrlJumpOnPolicyRunner` saves the
teacher state in checkpoints. Tune it through
`cfg.commands["jump_height"].height_bins` and `.lp_acrl`. Synthetic flight
starts are kept, but cannot earn the apex reward, so they lower every bin's
mean return equally.

```bash
uv run train Mjlab-Jump-Vertical-Pal-Kangaroo-Full-Curriculum-LpAcrl --env.scene.num-envs 4096
```

## Rewards

The only retained velocity rewards are `upright`, `body_ang_vel`,
`angular_momentum`, `dof_pos_limits`, `action_rate_l2`, and `foot_slip`,
with their existing weights. `upright` replaces the paper's orientation-error
term. Angular-velocity and momentum regularization use the existing velocity
implementation. The slip cost is active in every phase, including after the
command becomes zero.

The additional rewards are:

| Term | Weight | Behavior |
| --- | --- | --- |
| `jump_height` | 20 | Once at measured apex: Gaussian(error, 0.10 m) + 3 × Laplacian(error, 0.05 m) |
| `estimated_jump_height` | 20 | Same kernel during ascent, using `h + max(vz, 0)^2 / (2g)` |
| `pose` | 1 | Height-dependent per-joint posture reward |
| `horizontal_velocity` | 1 | Suppress horizontal motion, inspired by the authors' implementation |
| `ground_force_l2` | −0.05 | Squared foot-ground forces, normalized by 500 N |
| `soft_impact` | 1 | Limit deceleration during the first 0.3 s after touchdown |
| `landing_drift` | −10 | After takeoff (flight and landed): squared horizontal distance of the whole-body COM from its takeoff position |

The apex reward compensates for mjlab's `dt` scaling so its payout is independent
of policy frequency; all continuous rewards remain rates. The ballistic estimate
uses the configured simulation gravity and is disabled on descent.

`pose` is `exp(-mean((q-q_default)^2 / std(h)^2))`. Each joint's std interpolates
linearly from `std_ground` at 0 m rise to `std_airborne` at 0.25 m rise, clamped
outside that interval. The defaults reuse the flat task's looser running and
tighter walking tolerances, respectively. Tune those dictionaries and
`height_range` in `cfg.rewards["pose"].params`. Joint order is resolved before
building the std tensors; passive linkage/screw joints follow the existing
posture selection.

Joint symmetry, catch landing, and damp landing are absent. The former velocity
tracking, air-time, clearance, swing-height, collision-cost, convex-hull, and
joint-velocity-limit rewards are absent as requested. Reward weights and height
ranges are starting settings for Kangaroo; learning performance requires a full
training run.
