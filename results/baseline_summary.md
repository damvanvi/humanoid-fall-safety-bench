# humanoid-fall-safety-bench - baseline summary

- Bench version: `0.2.0` | seed: `42` | trials per tier: `8` | MuJoCo `4.0.1` | Python `3.13.18`
- code SHA-256: `a3b634dc8bf05403afbff4118f3b8173e622b882aaa7856ddbea32dbae7d03fd`
- model SHA-256: `128414ba9f0481088f8f9f59c9912fdc3d7916d54b0c02c915937ab2ad5d372f`
- runs SHA-256: `8181dba2aa7b6b74eefb13418e05d983375bedbf98010403b07fa14ef1978281`
- Verdict rule: **FAIL** if any non-foot geom touches the floor or the pelvis drops below 0.45 m within 3.0 s of fault onset; else PASS.

## Nominal-controller validation (Control C0, no fault, same disturbances)

| Tier | Upright / runs |
|---|---|
| push | 8/8 |
| quiet | 8/8 |
| stress | 0/8 |

## Tier A - quiet stance (squat task, no push)

| Fault | Containment | Verdict | Falls | Head strikes | Head impact m/s (mean / max) | Torso impact m/s (mean / max) | KE at ground contact J (mean / max) | Median time-to-fall s |
|---|---|---|---|---|---|---|---|---|
| F1_policy_hang | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F1_policy_hang | **Naive STO** (instant zero torque) | FAIL | 8/8 | 8/8 | 2.21 / 2.49 | 2.08 / 2.19 | 35.1 / 37.9 | 0.456 |
| F1_policy_hang | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F2_dcbus_sag | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F2_dcbus_sag | **Naive STO** (instant zero torque) | FAIL | 8/8 | 8/8 | 2.32 / 2.47 | 2.04 / 2.17 | 36.2 / 39.3 | 0.366 |
| F2_dcbus_sag | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F3_knee_overheat | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F3_knee_overheat | **Naive STO** (instant zero torque) | FAIL | 8/8 | 8/8 | 2.18 / 2.26 | 2.06 / 2.18 | 37.4 / 42.4 | 0.764 |
| F3_knee_overheat | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |

## Tier B - in-envelope push (50-100 N x 0.15 s; nominal controller survives all)

| Fault | Containment | Verdict | Falls | Head strikes | Head impact m/s (mean / max) | Torso impact m/s (mean / max) | KE at ground contact J (mean / max) | Median time-to-fall s |
|---|---|---|---|---|---|---|---|---|
| F1_policy_hang | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F1_policy_hang | **Naive STO** (instant zero torque) | FAIL | 8/8 | 0/8 | - / - | 0.14 / 0.22 | 64.7 / 98.8 | 0.446 |
| F1_policy_hang | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F2_dcbus_sag | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F2_dcbus_sag | **Naive STO** (instant zero torque) | FAIL | 8/8 | 0/8 | - / - | 0.17 / 0.17 | 73.7 / 98.8 | 0.358 |
| F2_dcbus_sag | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F3_knee_overheat | Reference: fault, no containment | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |
| F3_knee_overheat | **Naive STO** (instant zero torque) | FAIL | 8/8 | 4/8 | 1.88 / 3.18 | 0.87 / 1.76 | 50.7 / 66.8 | 0.792 |
| F3_knee_overheat | **Naive brake lock** (instant rigid freeze) | PASS | 0/8 | 0/8 | - / - | - / - | - / - | - |

## Tier C - momentum stress (140-200 N x 0.15 s; beyond the nominal envelope)

| Fault | Containment | Verdict | Falls | Head strikes | Head impact m/s (mean / max) | Torso impact m/s (mean / max) | KE at ground contact J (mean / max) | Median time-to-fall s |
|---|---|---|---|---|---|---|---|---|
| F1_policy_hang | Reference: fault, no containment | FAIL | 8/8 | 3/8 | 2.67 / 3.06 | 3.55 / 4.67 | 364.9 / 451.8 | 1.356 |
| F1_policy_hang | **Naive STO** (instant zero torque) | FAIL | 8/8 | 2/8 | 3.46 / 3.96 | 0.45 / 0.52 | 91.8 / 117.9 | 0.440 |
| F1_policy_hang | **Naive brake lock** (instant rigid freeze) | FAIL | 7/8 | 7/8 | 3.10 / 6.25 | 2.83 / 4.51 | 378.1 / 424.0 | 1.433 |
| F2_dcbus_sag | Reference: fault, no containment | FAIL | 8/8 | 3/8 | 2.87 / 3.78 | 4.61 / 4.68 | 362.0 / 450.7 | 1.379 |
| F2_dcbus_sag | **Naive STO** (instant zero torque) | FAIL | 8/8 | 4/8 | 2.90 / 4.28 | 0.46 / 0.68 | 102.0 / 151.6 | 0.349 |
| F2_dcbus_sag | **Naive brake lock** (instant rigid freeze) | FAIL | 7/8 | 7/8 | 3.01 / 6.24 | 3.37 / 4.53 | 383.6 / 428.5 | 1.482 |
| F3_knee_overheat | Reference: fault, no containment | FAIL | 8/8 | 3/8 | 2.87 / 3.81 | 4.35 / 4.67 | 360.3 / 450.8 | 1.379 |
| F3_knee_overheat | **Naive STO** (instant zero torque) | FAIL | 8/8 | 3/8 | 2.28 / 3.29 | 1.17 / 2.77 | 164.0 / 246.8 | 0.779 |
| F3_knee_overheat | **Naive brake lock** (instant rigid freeze) | FAIL | 7/8 | 7/8 | 2.73 / 4.33 | 2.32 / 4.58 | 373.4 / 432.5 | 1.370 |

Head (torso) impact velocity statistics are over runs with a head (torso) strike; kinetic-energy and time-to-fall statistics are over falling runs. The reference arm is a control condition, not a mitigation. Thresholds are bench conventions, not values from any standard.

## Media

- comparison_plot: `comparison_plot.png`
- comparison_grid: `comparison_grid.gif`
- render_backend: `egl`
