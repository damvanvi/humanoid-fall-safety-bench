#!/usr/bin/env python3
# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Pre-flight check: byte-compile every source file, import every module, load the
model, run a 1-second smoke simulation, and probe headless rendering in a subprocess.

    python tests/check_install.py
"""
from __future__ import annotations

import importlib
import os
import py_compile
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MODULES = ["env.robot", "env.nominal_controller", "env.single_leg_controller", "env.fault_injector",
           "env.episode", "baselines", "baselines.naive_sto", "baselines.naive_brake_lock",
           "baselines.realistic_brake_lock", "baselines.reference",
           "metrics.impact_oracle", "metrics.report_builder", "metrics.media", "run_bench"]


def main() -> int:
    ok = True
    srcs = []
    for dp, _, fns in os.walk(REPO):
        if "results" in dp or "__pycache__" in dp:
            continue
        srcs += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    for f in sorted(srcs):
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            ok = False
            print("SYNTAX ERROR", e)
    print(f"[1/4] byte-compiled {len(srcs)} files: {'ok' if ok else 'FAILED'}")

    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"IMPORT ERROR {m}: {e!r}")
    import mujoco, numpy, matplotlib, imageio  # noqa: E401,F401
    print(f"[2/4] imported {len(MODULES)} modules: {'ok' if ok else 'FAILED'} | python {sys.version.split()[0]} "
          f"| mujoco {mujoco.__version__} | numpy {numpy.__version__} | matplotlib {matplotlib.__version__}")

    from env.robot import load_robot
    from env.episode import run_episode
    r = load_robot()
    assert r.model.nu == 20 and len(r.eq_ids) == 20, "unexpected actuator / brake count"
    res = run_episode("C0_no_fault", "nominal_control", {"tier": "smoke", "trial": 0, "t_fault": 0.5,
                                                         "push": None})["result"]
    stand = res["verdict"] == "PASS"
    ok &= stand
    print(f"[3/4] model loads ({r.model.nu} actuators, {r.total_mass:.1f} kg); nominal stance smoke run: "
          f"{'upright' if stand else 'FELL (unexpected)'}")

    found = None
    cands = [os.environ["MUJOCO_GL"]] if os.environ.get("MUJOCO_GL") else []
    cands += [b for b in ("egl", "osmesa", "glfw") if b not in cands]
    for backend in cands:
        env = dict(os.environ, MUJOCO_GL=backend)
        probe = subprocess.run([sys.executable, "-m", "metrics.media", "probe"], cwd=REPO, env=env,
                               capture_output=True, text=True)
        if probe.returncode == 0:
            found = backend
            break
    if found:
        print(f"[4/4] headless rendering: available with MUJOCO_GL={found} -> GIF export will work")
    else:
        print(f"[4/4] headless rendering: none of {cands} works here; run_bench.py will fall back to "
              "results/comparison_plot.png (not an error)")
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
