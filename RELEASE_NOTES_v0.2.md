# humanoid-fall-safety-bench v0.2.0

v0.2 adds one new scenario and one new baseline on top of v0.1.0. Nothing from v0.1 was removed or
changed: `python run_bench.py --suite v0.1 --seed 42` reproduces the v0.1.0 runs SHA-256
(`8181dba2…`), its summary SHA-256 and all 240 trajectory hashes exactly. License: Apache-2.0.

## Reproduce

```bash
pip install -r requirements.txt
python tests/check_install.py
python run_bench.py --all --seed 42 --export-gif     # both suites, about 100 s of simulation on 2 cores
python run_bench.py --suite single_leg --seed 42     # v0.2 suite only
```

## What's new

- **Single-leg stance scenario** (`--suite single_leg`). The robot runs a scripted transition: it
  shifts weight onto the right foot, lifts the left foot about 10 cm and holds single-leg stance.
  The three v0.1 faults (F1–F3) are then injected. There are two tiers: quiet, and a gentle push of
  15–35 N. The push bound was set from the no-fault control only.
  - Acceptance gate: the no-fault control stays upright in **16/16** runs.
  - Post-fault horizon: 5 s.
- **`realistic_brake_lock` baseline.** It models spring-applied electromechanical brakes:
  - a 40 ms dead time, then a 10 ms ramp;
  - a finite holding torque of 1.0 x the rated joint torque, as a stiff dry-friction clamp that
    slips above its limit and does not creep below it;
  - partial braking: leg roll and pitch joints and the waist are braked, while hip yaw and the arms
    are not.

  All parameters are in `BrakeConfig` and were fixed before any v0.2 result was computed.
- **`realistic_brake_all_joints` diagnostic.** The same delay and holding torque, with every joint
  braked. It is reported separately, not as a baseline.
- **CLI.** New flags `--suite {v0.1,single_leg,all}` and `--no-diagnostic`. `--all` runs both
  suites.
- **New outputs.**
  - `results/v0.2_single_leg_summary.md` and `results/v0.2_single_leg_metrics.json`, with the same
    oracle and SHA-256 provenance as v0.1;
  - `results/v0.2_single_leg_plot.png` and `results/v0.2_single_leg_grid.gif`.

## Results (seed 42, single-leg suite, 48 runs per arm)

| | Falls | Head strikes | Max head impact |
|---|---|---|---|
| No containment (reference) | 0/48 | 0/48 | - |
| Naive STO | **48/48** | 20/48 | 4.43 m/s |
| Naive brake lock (ideal) | 0/48 | 0/48 | - |
| **Realistic brake lock** | **47/48** | **45/48** | 3.25 m/s |
| Diagnostic: realistic brake, all joints braked | 0/48 | 0/48 | - |

The single-leg runs SHA-256 is `ede1f509e3d7da9a3871b5aaa34eabdad660c541962c0b1b2767077ba60fadfb`
(MuJoCo 4.0.1). Serial and parallel runs give identical hashes.

**What the results show:**
- **Instant torque cut-off still causes every fall**, now in single-leg stance too.
- **The ideal instant all-joint lock passes single-leg stance in these tiers.** We report this as
  measured physics, not as a safety argument.
- **Realistic brakes fail almost always, and almost always head-first.** They also fail slowly:
  the median fall comes 2.7 s after the fault, and 11 falls come after 3 s.
- **The cause is partial braking.** The same delay and holding torque with every joint braked
  gives 0/48 falls. Whether brake lock is safe on a balancing robot depends on which joints carry
  brakes.

## Engineering notes

- The friction clamp of the realistic brakes uses solref 5 ms and solimp 0.999/0.9999. MuJoCo's
  default soft friction let loaded joints creep about 0.05 rad/s below their holding torque, which
  is a simulation artifact. With the stiffened settings, measured drift is < 1 mrad over 2 s.
- The single-leg horizon is 5 s, longer than v0.1's 3 s, because realistic-brake topples are slow.
  We chose this after seeing falls near 3 s, and the summary reports falls after 3 s explicitly.

## Known limitations

- The single-leg stance is a static hold. Walking with gait-phase faults is not yet included.
- There is one realistic brake configuration, with no sweep over delay, holding torque or braked
  joints yet.
- Simulation only. The bench is not hardware certification and makes no claim of conformity with
  any standard.

## Next (v0.3)

- Walking with faults injected at defined gait phases.
- Brake parameter sweeps.
- Sensor latency and dropout, combined faults and policy oscillation.
- At least 20 seeds per cell across at least two open humanoid morphologies.
- A CI-verified v1.0 with a Zenodo DOI.
