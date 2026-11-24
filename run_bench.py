#!/usr/bin/env python3
# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""humanoid-fall-safety-bench v0.2 -- one-command CLI.

    python run_bench.py --all --seed 42 --export-gif          # both suites
    python run_bench.py --suite v0.1 --seed 42                # v0.1 double-support suite only
    python run_bench.py --suite single_leg --seed 42          # v0.2 single-leg stance suite only

Suite v0.1 (unchanged from v0.1.0, reproduces its seed-42 runs SHA-256): every fault
(F1-F3) x {no-containment reference, naive STO, naive brake lock} across three
double-support tiers, with a no-fault control on identical disturbances. Writes
results/metrics.json, results/baseline_summary.md, results/comparison_plot.png and,
if offscreen OpenGL works, results/comparison_grid.gif.

Suite single_leg (v0.2): the same faults during single-leg stance, adding the
realistic brake lock baseline and one diagnostic ablation. Writes
results/v0.2_single_leg_metrics.json, results/v0.2_single_leg_summary.md,
results/v0.2_single_leg_plot.png and, if OpenGL works, results/v0.2_single_leg_grid.gif.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from env.fault_injector import FAULT_IDS  # noqa: E402
from baselines import BASELINES, BASELINES_V02, DIAGNOSTIC_ARMS, REFERENCE_ARMS  # noqa: E402

TIERS = ("quiet", "push", "stress")
PUSH_RANGE_N = {"push": (50.0, 100.0), "stress": (140.0, 200.0)}
PUSH_DUR_S = 0.15
FRAME_DT = 0.05   # replay frame period for the GIF (20 fps)

# ---- v0.2 single-leg stance suite
SL_TIERS = ("quiet", "push")
# Push upper bound fixed from the no-fault control only: C0 survives 40 N x 0.15 s in all
# 8 tested directions and fails at 60 N toward the swing side; tier B uses 15-35 N.
SL_PUSH_RANGE_N = (15.0, 35.0)
SL_POST_FAULT_HORIZON_S = 5.0
SL_MAIN_ARMS = ("ref_no_containment", "naive_sto", "naive_brake_lock", "realistic_brake_lock")


def make_scenario_sl(seed: int, tier: str, trial: int) -> dict:
    """Deterministic single-leg scenario (separate RNG stream from v0.1; paired across faults/arms)."""
    rng = np.random.default_rng([seed, 100 + SL_TIERS.index(tier), trial])
    base = {"suite": "single_leg", "tier": tier, "trial": trial,
            "post_fault_horizon_s": SL_POST_FAULT_HORIZON_S}
    if tier == "quiet":
        return {**base, "t_fault": round(float(rng.uniform(4.0, 6.0)), 3), "push": None}
    t_push = round(float(rng.uniform(4.0, 5.0)), 3)
    mag = float(rng.uniform(*SL_PUSH_RANGE_N))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    t_fault = round(t_push + float(rng.uniform(0.0, 0.25)), 3)
    return {**base, "t_fault": t_fault,
            "push": {"t": t_push, "dur": PUSH_DUR_S,
                     "force_xy": [round(mag * np.cos(ang), 4), round(mag * np.sin(ang), 4)],
                     "magnitude_N": round(mag, 3), "angle_deg": round(np.degrees(ang), 2)}}


def make_scenario(seed: int, tier: str, trial: int) -> dict:
    """Deterministic disturbance + fault timing. Identical for every fault/arm of a trial (paired design)."""
    rng = np.random.default_rng([seed, TIERS.index(tier), trial])
    if tier == "quiet":
        return {"tier": tier, "trial": trial, "t_fault": round(float(rng.uniform(2.0, 4.0)), 3), "push": None}
    t_push = round(float(rng.uniform(2.0, 3.5)), 3)
    lo, hi = PUSH_RANGE_N[tier]
    mag = float(rng.uniform(lo, hi))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    t_fault = round(t_push + float(rng.uniform(0.0, 0.25)), 3)
    return {"tier": tier, "trial": trial, "t_fault": t_fault,
            "push": {"t": t_push, "dur": PUSH_DUR_S, "force_xy": [round(mag * np.cos(ang), 4), round(mag * np.sin(ang), 4)],
                     "magnitude_N": round(mag, 3), "angle_deg": round(np.degrees(ang), 2)}}


def _job(args):
    fault, arm, scen, keep_frames = args
    from env.episode import run_episode
    out = run_episode(fault, arm, scen, frame_dt=FRAME_DT if keep_frames else None)
    res = out["result"]
    res["tier"] = scen["tier"]
    res["trial"] = scen["trial"]
    rep = None
    if keep_frames:
        rep = {"trace": out["trace"], "frames": np.asarray(out["frames"]), "frame_t": np.asarray(out["frame_t"]),
               "onset": scen["t_fault"]}
    return res, rep


def try_gif(frames_npz: str, gif_path: str, backends=None, timeout: int = 900, t_end: float = 2.6) -> tuple:
    """Try each headless GL backend in an isolated subprocess. Returns (ok, backend, log)."""
    if backends:
        candidates = list(backends)
    else:
        candidates = []
        if os.environ.get("MUJOCO_GL"):
            candidates.append(os.environ["MUJOCO_GL"])
        candidates += [b for b in ("egl", "osmesa", "glfw") if b not in candidates]
    log = []
    for backend in candidates:
        env = dict(os.environ, MUJOCO_GL=backend)
        if backend == "osmesa":
            env.setdefault("PYOPENGL_PLATFORM", "osmesa")
        elif backend == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            probe = subprocess.run([sys.executable, "-m", "metrics.media", "probe"], cwd=REPO, env=env,
                                   capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            log.append(f"{backend}: probe timed out")
            continue
        if probe.returncode != 0:
            tail = (probe.stderr or probe.stdout).strip().splitlines()[-1:] or ["(no output)"]
            log.append(f"{backend}: unavailable ({tail[0][:160]})")
            continue
        try:
            run = subprocess.run([sys.executable, "-m", "metrics.media", "gif", "--frames", frames_npz,
                                  "--out", gif_path, "--t-end", str(t_end)], cwd=REPO, env=env,
                                 capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            log.append(f"{backend}: GIF render timed out")
            continue
        if run.returncode == 0 and os.path.exists(gif_path):
            log.append(f"{backend}: GIF written")
            return True, backend, log
        tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["(no output)"]
        log.append(f"{backend}: render failed ({tail[0][:160]})")
    return False, None, log


def check_determinism(scen: dict, arms=("naive_sto", "naive_brake_lock")) -> dict:
    from env.episode import run_episode
    out = {}
    for arm in arms:
        h = [run_episode("F1_policy_hang", arm, scen)["result"]["trajectory_sha256"] for _ in range(2)]
        out[arm] = {"sha256": h[0], "repeat_identical": h[0] == h[1]}
    return out


def run_jobs(jobs: list, n_jobs: int) -> tuple:
    t0 = time.time()
    results, reps = [], {}
    if n_jobs == 1:
        it = map(_job, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        it = pool.map(_job, jobs, chunksize=2)
    for i, (job, (res, rep)) in enumerate(zip(jobs, it), 1):
        results.append(res)
        if rep is not None:
            reps[(res["tier"], res["fault_id"], res["arm"])] = rep
        if i % 10 == 0 or i == len(jobs):
            print(f"  [{i:4d}/{len(jobs)}] {time.time() - t0:6.1f} s", flush=True)
    if pool is not None:
        pool.shutdown()
    print(f"simulated {len(results)} episodes in {time.time() - t0:.1f} s")
    return results, reps


def export_gif(a, reps, results, row_keys, cols, frames_name, gif_name, t_end) -> dict:
    rows, grid = [], {}
    for r, (tier, fault) in enumerate(row_keys):
        rows.append(f"{fault.split('_')[0]} {fault.split('_', 1)[1]} | {tier}")
        for c, arm in enumerate(cols):
            rp = reps.get((tier, fault, arm))
            if rp is None:
                continue
            verdict = next(x["verdict"] for x in results if x["tier"] == tier and x["fault_id"] == fault
                           and x["arm"] == arm and x["trial"] == 0)
            grid[(r, c)] = {"qpos": rp["frames"], "t": rp["frame_t"], "onset": rp["onset"], "verdict": verdict}
    frames_npz = os.path.join(a.out, frames_name)
    np.savez_compressed(frames_npz, rows=np.array(rows, dtype=object), cols=np.array(cols, dtype=object),
                        grid=np.array(grid, dtype=object))
    gif = os.path.join(a.out, gif_name)
    backends = None if a.render_backends == "auto" else [b.strip() for b in a.render_backends.split(",") if b.strip()]
    if os.path.exists(gif):
        os.remove(gif)
    ok, backend, log = try_gif(frames_npz, gif, backends, t_end=t_end)
    for line in log:
        print(f"  render: {line}")
    if ok:
        print(f"wrote {gif} (MUJOCO_GL={backend})")
        return {gif_name.rsplit(".", 1)[0]: os.path.basename(gif), "render_backend": backend}
    print("headless OpenGL unavailable -> GIF skipped; the PNG plot is the substitute preview")
    return {gif_name.rsplit(".", 1)[0]: "not rendered (no working headless OpenGL); see the PNG plot"}


def run_suite_v01(a, argv, n_jobs) -> int:
    """Suite v0.1: identical job list, records and outputs to v0.1.0."""
    tiers = [t for t in a.tiers if t in TIERS]
    if not tiers:
        return 0
    faults = list(FAULT_IDS) if (a.all or not a.fault) else a.fault
    arms = list(BASELINES) if (a.all or not a.baseline) else [b for b in a.baseline if b in BASELINES]
    if not a.no_reference:
        arms = list(REFERENCE_ARMS) + arms

    import mujoco
    print(f"humanoid-fall-safety-bench v0.2 | suite v0.1 (double support) | MuJoCo {mujoco.__version__} | "
          f"seed {a.seed} | trials/tier {a.trials} | tiers {','.join(tiers)}")

    rep_tier = "push" if "push" in tiers else tiers[0]
    jobs = []
    for tier in tiers:
        for k in range(a.trials):
            scen = make_scenario(a.seed, tier, k)
            keep = (tier == rep_tier and k == 0)
            jobs.append(("C0_no_fault", "nominal_control", scen, keep))
            for fault in faults:
                for arm in arms:
                    jobs.append((fault, arm, scen, keep))
    keep_stress = "stress" in tiers and rep_tier != "stress"
    if keep_stress:  # one extra replayed row for the GIF: F1 under momentum stress
        jobs = [(f, ar, s, kp or (s["tier"] == "stress" and s["trial"] == 0 and f == "F1_policy_hang"))
                for (f, ar, s, kp) in jobs]
    results, reps = run_jobs(jobs, n_jobs)

    from metrics.media import plot_comparison
    media = {}
    rep_plot = {(f, ar): v for (t, f, ar), v in reps.items() if t == rep_tier}
    sc0 = make_scenario(a.seed, rep_tier, 0)
    lbl = f"tier '{rep_tier}', trial 0"
    if sc0.get("push"):
        lbl += f", push {sc0['push']['magnitude_N']:.0f} N @ {sc0['push']['angle_deg']:.0f} deg"
    png = plot_comparison(rep_plot, results, os.path.join(a.out, "comparison_plot.png"), lbl)
    media["comparison_plot"] = os.path.basename(png)
    print(f"wrote {png}")
    if a.export_gif:
        cols = [ar for ar in ("ref_no_containment", "naive_sto", "naive_brake_lock") if ar in arms]
        row_keys = [(rep_tier, f) for f in faults] + ([("stress", "F1_policy_hang")] if keep_stress else [])
        m = export_gif(a, reps, results, row_keys, cols, "replay_frames.npz", "comparison_grid.gif", 2.6)
        if "comparison_grid" in m:
            media.update(m)

    config = {
        "suite": "v0.1",
        "seed": a.seed, "trials": a.trials, "tiers": list(tiers), "faults": faults, "arms": arms,
        "push_range_N": PUSH_RANGE_N, "push_duration_s": PUSH_DUR_S,
        "post_fault_horizon_s": 3.0, "physics_timestep_s": 0.001, "policy_rate_hz": 100,
        "command": " ".join(["python", "run_bench.py"] + (argv if argv is not None else sys.argv[1:])),
    }
    if a.check_determinism:
        det = check_determinism(make_scenario(a.seed, rep_tier, 0))
        config["determinism_check"] = det
        print("determinism (suite v0.1):", {k: v["repeat_identical"] for k, v in det.items()})

    from metrics.report_builder import build_reports
    doc = build_reports(results, config, a.out, media)
    print(f"wrote {os.path.join(a.out, 'metrics.json')}  (runs SHA-256 {doc['provenance']['runs_sha256'][:16]}...)")
    print(f"wrote {os.path.join(a.out, 'baseline_summary.md')}")

    ctrl_fail = [r for r in results if r["fault_id"] == "C0_no_fault" and r["tier"] != "stress" and r["fell"]]
    if ctrl_fail:
        print(f"WARNING: nominal controller fell in {len(ctrl_fail)} no-fault control run(s) outside the "
              f"stress tier -> scenario envelope is invalid for this build.")
        return 2
    return 0


def run_suite_single_leg(a, argv, n_jobs) -> int:
    """Suite single_leg (v0.2): faults during single-leg stance, 4 main arms + 1 diagnostic."""
    tiers = [t for t in a.tiers if t in SL_TIERS]
    if not tiers:
        return 0
    faults = list(FAULT_IDS) if (a.all or not a.fault) else a.fault
    pool_arms = {**BASELINES_V02, **({} if a.no_diagnostic else DIAGNOSTIC_ARMS)}
    arms = list(pool_arms) if (a.all or not a.baseline) else [b for b in a.baseline if b in pool_arms]
    if not a.no_reference:
        arms = list(REFERENCE_ARMS) + arms

    import mujoco
    print(f"humanoid-fall-safety-bench v0.2 | suite single_leg | MuJoCo {mujoco.__version__} | seed {a.seed} | "
          f"trials/tier {a.trials} | tiers {','.join(tiers)}")

    rep_tier = "quiet" if "quiet" in tiers else tiers[0]   # pre-specified: quiet tier, trial 0
    jobs = []
    for tier in tiers:
        for k in range(a.trials):
            scen = make_scenario_sl(a.seed, tier, k)
            keep = (tier == rep_tier and k == 0)
            jobs.append(("C0_no_fault", "nominal_control", scen, keep))
            for fault in faults:
                for arm in arms:
                    jobs.append((fault, arm, scen, keep and arm in SL_MAIN_ARMS))
    results, reps = run_jobs(jobs, n_jobs)

    from metrics.media import plot_comparison
    main_arms = [x for x in SL_MAIN_ARMS if x in arms]
    media = {}
    rep_plot = {(f, ar): v for (t, f, ar), v in reps.items() if t == rep_tier}
    png = plot_comparison(rep_plot, [r for r in results if r["arm"] in main_arms or r["arm"] == "nominal_control"],
                          os.path.join(a.out, "v0.2_single_leg_plot.png"),
                          f"single-leg stance, tier '{rep_tier}', trial 0", arms=main_arms,
                          title="humanoid-fall-safety-bench v0.2 - single-leg stance: containment vs. fault",
                          tier_mark={"quiet": "o", "push": "s"}, xlim=(-0.5, SL_POST_FAULT_HORIZON_S + 0.4))
    media["v0.2_single_leg_plot"] = os.path.basename(png)
    print(f"wrote {png}")
    if a.export_gif:
        media.update(export_gif(a, reps, results, [(rep_tier, f) for f in faults], main_arms,
                                "v0.2_single_leg_replay_frames.npz", "v0.2_single_leg_grid.gif",
                                SL_POST_FAULT_HORIZON_S + 0.1))

    from baselines.realistic_brake_lock import RealisticBrakeLock
    from env import single_leg_controller as slc
    config = {
        "suite": "single_leg",
        "seed": a.seed, "trials": a.trials, "tiers": list(tiers), "faults": faults, "arms": arms,
        "push_range_N": {"push": list(SL_PUSH_RANGE_N)}, "push_duration_s": PUSH_DUR_S,
        "post_fault_horizon_s": SL_POST_FAULT_HORIZON_S, "physics_timestep_s": 0.001, "policy_rate_hz": 100,
        "single_leg": {"stance_leg": "right", "transition_s": [slc.T_SETTLE, slc.T_SHIFT_END, slc.T_LIFT_END],
                       "fault_window_quiet_s": [4.0, 6.0], "push_window_s": [4.0, 5.0]},
        "realistic_brake_lock": RealisticBrakeLock.describe(),
        "notes": [
            "Stance leg = right (fixed before any baseline run): F3 derates the right knee hardest, "
            "and the loaded knee is the one that heats up.",
            f"Post-fault horizon is {SL_POST_FAULT_HORIZON_S} s (v0.1 uses 3 s); the 'Falls after 3 s' "
            "column shows which falls a 3 s horizon would have missed.",
            "Push tier upper bound (35 N) was set from the no-fault control alone (it survives 40 N x 0.15 s "
            "in all 8 tested directions, fails at 60 N toward the swing side).",
            "realistic_brake_all_joints is a diagnostic ablation, not a baseline: same delay and finite "
            "holding torque, every joint braked.",
        ],
        "command": " ".join(["python", "run_bench.py"] + (argv if argv is not None else sys.argv[1:])),
    }
    if a.check_determinism:
        det = check_determinism(make_scenario_sl(a.seed, rep_tier, 0),
                                arms=("naive_sto", "naive_brake_lock", "realistic_brake_lock"))
        config["determinism_check"] = det
        print("determinism (suite single_leg):", {k: v["repeat_identical"] for k, v in det.items()})

    from metrics.report_builder import ARM_ORDER_SL, TIER_TITLE_SL, build_reports
    doc = build_reports(results, config, a.out, media, prefix="v0.2_single_leg_",
                        title="v0.2 single-leg stance summary", arm_order=ARM_ORDER_SL,
                        tier_title=TIER_TITLE_SL, tiers=SL_TIERS, show_late=True)
    print(f"wrote {os.path.join(a.out, 'v0.2_single_leg_metrics.json')}  "
          f"(runs SHA-256 {doc['provenance']['runs_sha256'][:16]}...)")
    print(f"wrote {os.path.join(a.out, 'v0.2_single_leg_summary.md')}")

    ctrl = [r for r in results if r["fault_id"] == "C0_no_fault"]
    ctrl_fail = [r for r in ctrl if r["fell"]]
    if ctrl_fail:
        print(f"WARNING: single-leg acceptance gate FAILED: no-fault control fell in {len(ctrl_fail)}/{len(ctrl)} runs.")
        return 2
    print(f"single-leg acceptance gate: no-fault control upright in {len(ctrl)}/{len(ctrl)} runs")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every fault x containment x tier (both suites "
                    "unless --suite is given)")
    ap.add_argument("--suite", choices=("v0.1", "single_leg", "all"), default=None,
                    help="v0.1 = double-support suite (reproduces v0.1.0); single_leg = v0.2 suite")
    ap.add_argument("--fault", nargs="+", choices=FAULT_IDS, help="subset of faults")
    ap.add_argument("--baseline", nargs="+", choices=list(BASELINES_V02) + list(DIAGNOSTIC_ARMS),
                    help="subset of baselines")
    ap.add_argument("--tiers", nargs="+", choices=sorted(set(TIERS) | set(SL_TIERS)), default=list(TIERS))
    ap.add_argument("--no-reference", action="store_true", help="skip the no-containment reference arm")
    ap.add_argument("--no-diagnostic", action="store_true", help="skip the diagnostic brake ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=8, help="trials per tier (default 8)")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all CPUs, 1 = serial)")
    ap.add_argument("--export-gif", action="store_true", help="also try to render the comparison GIF(s)")
    ap.add_argument("--render-backends", default="auto",
                    help="comma list of MUJOCO_GL backends to try for the GIF, e.g. 'egl,osmesa' "
                         "(default auto: $MUJOCO_GL, egl, osmesa, glfw)")
    ap.add_argument("--out", default=os.path.join(REPO, "results"))
    ap.add_argument("--check-determinism", action="store_true", help="re-run one episode per baseline twice "
                    "and compare trajectory SHA-256")
    a = ap.parse_args(argv)

    if not a.all and not a.fault and not a.baseline and not a.suite:
        ap.error("choose --all, --suite, or a subset with --fault/--baseline")
    suite = a.suite or ("all" if a.all else "v0.1")
    os.makedirs(a.out, exist_ok=True)
    n_jobs = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)

    rc = 0
    if suite in ("v0.1", "all"):
        rc = max(rc, run_suite_v01(a, argv, n_jobs))
    if suite in ("single_leg", "all"):
        if suite == "single_leg" and a.tiers == list(TIERS):
            a.tiers = list(SL_TIERS)
        rc = max(rc, run_suite_single_leg(a, argv, n_jobs))
    return rc


if __name__ == "__main__":
    sys.exit(main())
