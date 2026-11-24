# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Minimal regression tests (pytest optional: `python -m pytest tests -q` or `python tests/test_smoke.py`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.episode import run_episode  # noqa: E402

QUIET = {"tier": "test", "trial": 0, "t_fault": 2.0, "push": None}


def test_nominal_stands():
    assert run_episode("C0_no_fault", "nominal_control", QUIET)["result"]["verdict"] == "PASS"


def test_sto_falls_on_policy_hang():
    r = run_episode("F1_policy_hang", "naive_sto", QUIET)["result"]
    assert r["verdict"] == "FAIL" and r["time_to_fall_s"] > 0.1   # trips at +100 ms, then falls


def test_brake_lock_engages_and_is_deterministic():
    a = run_episode("F2_dcbus_sag", "naive_brake_lock", QUIET)["result"]
    b = run_episode("F2_dcbus_sag", "naive_brake_lock", QUIET)["result"]
    assert a["containment_engaged_s"] is not None
    assert a["trajectory_sha256"] == b["trajectory_sha256"]


SINGLE_LEG = {"suite": "single_leg", "tier": "test", "trial": 0, "t_fault": 4.5, "push": None,
              "post_fault_horizon_s": 5.0}


def test_single_leg_nominal_stands_on_one_foot():
    out = run_episode("C0_no_fault", "nominal_control", SINGLE_LEG, frame_dt=0.5)
    assert out["result"]["verdict"] == "PASS"


def test_realistic_brake_engages_and_is_deterministic():
    a = run_episode("F2_dcbus_sag", "realistic_brake_lock", SINGLE_LEG)["result"]
    b = run_episode("F2_dcbus_sag", "realistic_brake_lock", SINGLE_LEG)["result"]
    assert a["containment_engaged_s"] is not None
    assert a["trajectory_sha256"] == b["trajectory_sha256"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
