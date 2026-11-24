# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Builds results/metrics.json and results/baseline_summary.md with seed + SHA-256 provenance."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_VERSION = "0.1.0"
SOURCE_GLOBS = ("run_bench.py", "env", "baselines", "metrics")

ARM_ORDER = ("ref_no_containment", "naive_sto", "naive_brake_lock")
ARM_LABEL = {
    "ref_no_containment": "Reference: fault, no containment",
    "naive_sto": "**Naive STO** (instant zero torque)",
    "naive_brake_lock": "**Naive brake lock** (instant rigid freeze)",
    "nominal_control": "Control C0: no fault",
}
TIER_TITLE = {
    "quiet": "Tier A - quiet stance (squat task, no push)",
    "push": "Tier B - in-envelope push (50-100 N x 0.15 s; nominal controller survives all)",
    "stress": "Tier C - momentum stress (140-200 N x 0.15 s; beyond the nominal envelope)",
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def source_provenance() -> dict:
    files = []
    for entry in SOURCE_GLOBS:
        p = os.path.join(REPO_ROOT, entry)
        if os.path.isfile(p):
            files.append(p)
        else:
            for dp, _, fns in os.walk(p):
                files += [os.path.join(dp, fn) for fn in fns if fn.endswith((".py", ".xml"))]
    files = sorted(set(files))
    per_file = {os.path.relpath(f, REPO_ROOT): _sha256_file(f) for f in files}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per_file.items()).encode()).hexdigest()
    return {"code_sha256": combined, "files": per_file}


def environment_info() -> dict:
    import mujoco
    return {
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.max(v))


def summarize(runs: list) -> list:
    cells = []
    keys = sorted({(r["tier"], r["fault_id"], r["arm"]) for r in runs})
    for tier, fault, arm in keys:
        rs = [r for r in runs if (r["tier"], r["fault_id"], r["arm"]) == (tier, fault, arm)]
        falls = [r for r in rs if r["fell"]]
        hv_mean, hv_max = _stats([r["peak_head_impact_velocity_mps"] for r in rs if r["head_contact"]])
        tv_mean, tv_max = _stats([r["peak_torso_impact_velocity_mps"] for r in rs if r["torso_contact"]])
        ke_mean, ke_max = _stats([r["ke_at_ground_contact_J"] for r in falls])
        ttf = [r["time_to_fall_s"] for r in falls if r["time_to_fall_s"] is not None]
        cells.append({
            "tier": tier, "fault_id": fault, "arm": arm, "n": len(rs),
            "falls": len(falls), "pass": len(rs) - len(falls),
            "head_strikes": sum(1 for r in rs if r["head_contact"]),
            "head_impact_mps_mean": hv_mean, "head_impact_mps_max": hv_max,
            "torso_impact_mps_mean": tv_mean, "torso_impact_mps_max": tv_max,
            "ke_contact_J_mean": ke_mean, "ke_contact_J_max": ke_max,
            "time_to_fall_s_median": float(np.median(ttf)) if ttf else None,
            "cell_verdict": "PASS" if not falls else "FAIL",
        })
    return cells


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def build_reports(runs: list, config: dict, out_dir: str, media: dict | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cells = summarize(runs)
    prov = source_provenance()
    runs_sha = canonical_sha256(runs)
    cells_sha = canonical_sha256(cells)
    doc = {
        "bench": "humanoid-fall-safety-bench",
        "bench_version": BENCH_VERSION,
        "license": "Apache-2.0",
        "config": config,
        "environment": environment_info(),
        "provenance": {
            "seed": config["seed"],
            "code_sha256": prov["code_sha256"],
            "model_sha256": prov["files"].get(os.path.join("env", "humanoid_model.xml")),
            "source_files_sha256": prov["files"],
            "runs_sha256": runs_sha,
            "summary_sha256": cells_sha,
            "note": ("runs_sha256 hashes the canonical JSON of all per-run records (each of which "
                     "carries the SHA-256 of its full 1 kHz qpos trajectory). Same seed + same code "
                     "+ same MuJoCo build => same hashes. Different CPUs/BLAS builds may differ in "
                     "the last float bits; compare verdicts and metrics, not hashes, across machines."),
        },
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "media": media or {},
        "summary": cells,
        "runs": runs,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)

    L = []
    L.append("# humanoid-fall-safety-bench - baseline summary\n")
    L.append(f"- Bench version: `{BENCH_VERSION}` | seed: `{config['seed']}` | trials per tier: "
             f"`{config['trials']}` | MuJoCo `{doc['environment']['mujoco']}` | Python "
             f"`{doc['environment']['python']}`")
    L.append(f"- code SHA-256: `{prov['code_sha256']}`")
    L.append(f"- model SHA-256: `{doc['provenance']['model_sha256']}`")
    L.append(f"- runs SHA-256: `{runs_sha}`")
    L.append(f"- Verdict rule: **FAIL** if any non-foot geom touches the floor or the pelvis drops "
             f"below 0.45 m within {config['post_fault_horizon_s']} s of fault onset; else PASS.\n")

    ctrl = [c for c in cells if c["fault_id"] == "C0_no_fault"]
    if ctrl:
        L.append("## Nominal-controller validation (Control C0, no fault, same disturbances)\n")
        L.append("| Tier | Upright / runs |")
        L.append("|---|---|")
        for c in ctrl:
            L.append(f"| {c['tier']} | {c['pass']}/{c['n']} |")
        L.append("")

    for tier in ("quiet", "push", "stress"):
        tc = [c for c in cells if c["tier"] == tier and c["fault_id"] != "C0_no_fault"]
        if not tc:
            continue
        L.append(f"## {TIER_TITLE[tier]}\n")
        L.append("| Fault | Containment | Verdict | Falls | Head strikes | Head impact m/s (mean / max) "
                 "| Torso impact m/s (mean / max) | KE at ground contact J (mean / max) | Median time-to-fall s |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for fault in sorted({c["fault_id"] for c in tc}):
            for arm in ARM_ORDER:
                c = next((x for x in tc if x["fault_id"] == fault and x["arm"] == arm), None)
                if c is None:
                    continue
                L.append(
                    f"| {fault} | {ARM_LABEL[arm]} | {c['cell_verdict']} | {c['falls']}/{c['n']} | "
                    f"{c['head_strikes']}/{c['n']} | {_f(c['head_impact_mps_mean'])} / {_f(c['head_impact_mps_max'])} | "
                    f"{_f(c['torso_impact_mps_mean'])} / {_f(c['torso_impact_mps_max'])} | "
                    f"{_f(c['ke_contact_J_mean'], 1)} / {_f(c['ke_contact_J_max'], 1)} | "
                    f"{_f(c['time_to_fall_s_median'], 3)} |")
        L.append("")
    L.append("Head (torso) impact velocity statistics are over runs with a head (torso) strike; "
             "kinetic-energy and time-to-fall statistics are over falling runs. The reference arm is a control "
             "condition, not a mitigation. Thresholds are bench conventions, not values from "
             "any standard.\n")
    if media:
        L.append("## Media\n")
        for k, v in media.items():
            L.append(f"- {k}: `{v}`")
    with open(os.path.join(out_dir, "baseline_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return doc
