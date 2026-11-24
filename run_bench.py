#!/usr/bin/env python3
# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""humanoid-fall-safety-bench v0.1 -- one-command CLI.

    python run_bench.py --all --seed 42 --export-gif

Runs every fault (F1-F3) x every containment arm (naive STO, naive brake lock,
plus the no-containment reference) across three disturbance tiers, with a
no-fault control on identical disturbances, then writes:

    results/metrics.json          per-run records + summary + SHA-256 provenance
    results/baseline_summary.md   human-readable tables
    results/comparison_plot.png   trajectory + impact-energy plots (always; no OpenGL needed)
    results/comparison_grid.gif   side-by-side replay (only if offscreen OpenGL works)
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
from baselines import BASELINES, REFERENCE_ARMS  # noqa: E402

TIERS = ("quiet", "push", "stress")
PUSH_RANGE_N = {"push": (50.0, 100.0), "stress": (140.0, 200.0)}
PUSH_DUR_S = 0.15
FRAME_DT = 0.05   # replay frame period for the GIF (20 fps)


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


def try_gif(frames_npz: str, gif_path: str, backends=None, timeout: int = 900) -> tuple:
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
                                  "--out", gif_path], cwd=REPO, env=env, capture_output=True, text=True,
                                 timeout=timeout)
        except subprocess.TimeoutExpired:
            log.append(f"{backend}: GIF render timed out")
            continue
        if run.returncode == 0 and os.path.exists(gif_path):
            log.append(f"{backend}: GIF written")
            return True, backend, log
        tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["(no output)"]
        log.append(f"{backend}: render failed ({tail[0][:160]})")
    return False, None, log


def check_determinism(scen: dict) -> dict:
    from env.episode import run_episode
    out = {}
    for arm in ("naive_sto", "naive_brake_lock"):
        h = [run_episode("F1_policy_hang", arm, scen)["result"]["trajectory_sha256"] for _ in range(2)]
        out[arm] = {"sha256": h[0], "repeat_identical": h[0] == h[1]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every fault x containment x tier")
    ap.add_argument("--fault", nargs="+", choices=FAULT_IDS, help="subset of faults")
    ap.add_argument("--baseline", nargs="+", choices=list(BASELINES), help="subset of baselines")
    ap.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    ap.add_argument("--no-reference", action="store_true", help="skip the no-containment reference arm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=8, help="trials per tier (default 8)")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all CPUs, 1 = serial)")
    ap.add_argument("--export-gif", action="store_true", help="also try to render results/comparison_grid.gif")
    ap.add_argument("--render-backends", default="auto",
                    help="comma list of MUJOCO_GL backends to try for the GIF, e.g. 'egl,osmesa' "
                         "(default auto: $MUJOCO_GL, egl, osmesa, glfw)")
    ap.add_argument("--out", default=os.path.join(REPO, "results"))
    ap.add_argument("--check-determinism", action="store_true", help="re-run one episode per baseline twice "
                    "and compare trajectory SHA-256")
    a = ap.parse_args(argv)

    if not a.all and not a.fault and not a.baseline:
        ap.error("choose --all, or a subset with --fault/--baseline")
    faults = list(FAULT_IDS) if (a.all or not a.fault) else a.fault
    arms = list(BASELINES) if (a.all or not a.baseline) else a.baseline
    if not a.no_reference:
        arms = list(REFERENCE_ARMS) + arms
    os.makedirs(a.out, exist_ok=True)

    import mujoco
    print(f"humanoid-fall-safety-bench v0.1 | MuJoCo {mujoco.__version__} | seed {a.seed} | "
          f"trials/tier {a.trials} | tiers {','.join(a.tiers)}")

    rep_tier = "push" if "push" in a.tiers else a.tiers[0]
    jobs = []
    for tier in a.tiers:
        for k in range(a.trials):
            scen = make_scenario(a.seed, tier, k)
            keep = (tier == rep_tier and k == 0)
            jobs.append(("C0_no_fault", "nominal_control", scen, keep))
            for fault in faults:
                for arm in arms:
                    jobs.append((fault, arm, scen, keep))
    keep_stress = "stress" in a.tiers and rep_tier != "stress"
    if keep_stress:  # one extra replayed row for the GIF: F1 under momentum stress
        jobs = [(f, ar, s, kp or (s["tier"] == "stress" and s["trial"] == 0 and f == "F1_policy_hang"))
                for (f, ar, s, kp) in jobs]

    n_jobs = a.jobs if a.jobs > 0 else (os.cpu_count() or 1)
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

    # ---- media
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
        rows, grid = [], {}
        cols = [ar for ar in ("ref_no_containment", "naive_sto", "naive_brake_lock") if ar in arms]
        row_keys = [(rep_tier, f) for f in faults]
        if keep_stress:
            row_keys.append(("stress", "F1_policy_hang"))
        for r, (tier, fault) in enumerate(row_keys):
            rows.append(f"{fault.split('_')[0]} {fault.split('_', 1)[1]} | {tier}")
            for c, arm in enumerate(cols):
                rp = reps.get((tier, fault, arm))
                if rp is None:
                    continue
                verdict = next(x["verdict"] for x in results if x["tier"] == tier and x["fault_id"] == fault
                               and x["arm"] == arm and x["trial"] == 0)
                grid[(r, c)] = {"qpos": rp["frames"], "t": rp["frame_t"], "onset": rp["onset"], "verdict": verdict}
        frames_npz = os.path.join(a.out, "replay_frames.npz")
        np.savez_compressed(frames_npz, rows=np.array(rows, dtype=object), cols=np.array(cols, dtype=object),
                            grid=np.array(grid, dtype=object))
        gif = os.path.join(a.out, "comparison_grid.gif")
        backends = None if a.render_backends == "auto" else [b.strip() for b in a.render_backends.split(",") if b.strip()]
        if os.path.exists(gif):
            os.remove(gif)
        ok, backend, log = try_gif(frames_npz, gif, backends)
        for line in log:
            print(f"  render: {line}")
        if ok:
            media["comparison_grid"] = os.path.basename(gif)
            media["render_backend"] = backend
            print(f"wrote {gif} (MUJOCO_GL={backend})")
        else:
            media["comparison_grid"] = "not rendered (no working headless OpenGL); see comparison_plot.png"
            print("headless OpenGL unavailable -> GIF skipped; comparison_plot.png is the fallback preview")

    config = {
        "seed": a.seed, "trials": a.trials, "tiers": list(a.tiers), "faults": faults, "arms": arms,
        "push_range_N": PUSH_RANGE_N, "push_duration_s": PUSH_DUR_S,
        "post_fault_horizon_s": 3.0, "physics_timestep_s": 0.001, "policy_rate_hz": 100,
        "command": " ".join(["python", "run_bench.py"] + (argv if argv is not None else sys.argv[1:])),
    }
    if a.check_determinism:
        det = check_determinism(make_scenario(a.seed, rep_tier, 0))
        config["determinism_check"] = det
        print("determinism:", {k: v["repeat_identical"] for k, v in det.items()})

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


if __name__ == "__main__":
    sys.exit(main())
