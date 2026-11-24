# humanoid-fall-safety-bench v0.1.0

This is the first public release: an open, deterministic MuJoCo benchmark of **what naive "kill
switches" do to a dynamically balanced humanoid** when its AI policy hangs or its power or
actuators fault. License: Apache-2.0.

## Reproduce

```bash
pip install -r requirements.txt
python tests/check_install.py
python run_bench.py --all --seed 42 --export-gif
```

The full run is 240 episodes and takes about 40 s on 2 CPU cores. It writes
`results/baseline_summary.md`, `results/metrics.json` (with seed and SHA-256 provenance),
`results/comparison_plot.png` (always) and `results/comparison_grid.gif` (when headless OpenGL is
available).

## What's included

- A self-contained 3D humanoid model: about 59 kg, 20 actuated joints, no third-party assets.
- Three deterministic faults:
  - `F1_policy_hang`: the policy freezes for 600 ms;
  - `F2_dcbus_sag`: available torque drops by 40 %;
  - `F3_knee_overheat`: asymmetric knee derating.
- Two naive containment baselines: `naive_sto` (instant zero torque) and `naive_brake_lock`
  (instant rigid lock on every joint). There is also a `ref_no_containment` control arm.
- Three disturbance tiers, using a paired design: quiet stance, in-envelope push and momentum
  stress.
- An impact oracle that reports the fall verdict, peak head and torso impact velocity, kinetic
  energy at ground contact and time-to-fall.

## Results (seed 42, 24 runs per cell)

| | No containment | Naive STO | Naive brake lock |
|---|---|---|---|
| Quiet + in-envelope push | 0/48 falls | **48/48 falls** | 0/48 falls |
| Momentum stress | 24/24 falls, 9 head strikes | 24/24 falls, 9 head strikes, max head impact 4.28 m/s | 21/24 falls, **all 21 head strikes**, max head impact **6.25 m/s** |

The runs SHA-256 is `8181dba2aa7b6b74eefb13418e05d983375bedbf98010403b07fa14ef1978281`
(MuJoCo 4.0.1).

**What the results show:**
- **Instant torque cut-off is what causes the falls.** The faults alone cause none in these tiers.
- **Instant brake lock holds the robot up in static double-support stance.** A rigid body standing
  on its feet is passively stable, so this is an expected result, not a safety argument.
- **Under momentum, brake lock turns falls into head-first falls.** It also gives the hardest head
  impacts in the bench.

## Known limitations

- The nominal controller balances in double support only. There is no walking or single-leg stance
  yet.
- Brakes are ideal: instant, applied to every joint, with no slip limit.
- Simulation only. The bench is not hardware certification and makes no claim of conformity with
  any standard.

## Roadmap

- **v0.2:** single-leg stance and walking scenarios; brake models with finite holding torque,
  engagement delay and braking on only some joints.
- **v0.3:** sensor latency and dropout, combined faults, policy oscillation; reproducibility runs
  over many seeds and several morphologies.
- **v1.0:** CI-verified suite, Zenodo DOI and an open technical report.
