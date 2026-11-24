# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""One deterministic episode: nominal stance -> fault -> trip -> containment -> oracle."""
from __future__ import annotations

import mujoco

from baselines import ALL_ARMS
from env.fault_injector import make_fault
from env.nominal_controller import HighLevelPolicy, JointDrives, initial_state, POLICY_HZ
from env.robot import load_robot
from metrics.impact_oracle import ImpactOracle

POST_FAULT_HORIZON_S = 3.0   # simulate at most this long after fault onset
POST_FALL_TAIL_S = 1.0       # keep simulating this long after a fall (bounces, slides)
NO_FAULT_HORIZON_S = 3.0     # C0 control: horizon after the would-be fault time


def run_episode(fault_id: str, arm: str, scenario: dict, frame_dt: float | None = None) -> dict:
    """Run one episode and return {"result", "trace", "frames", "frame_t"}.

    scenario keys: t_fault (s), push (None or {t, dur, force_xy}), plus free-form labels.
    """
    robot = load_robot()            # fresh model: baselines may edit model fields (brake locks)
    m = robot.model
    d = mujoco.MjData(m)
    initial_state(robot, d)

    t_fault = float(scenario["t_fault"])
    fault = make_fault(fault_id, robot, t_fault)
    policy = HighLevelPolicy(robot)
    drives = JointDrives(robot)
    reaction = ALL_ARMS[arm](robot) if fault_id != "C0_no_fault" else ALL_ARMS["ref_no_containment"](robot)
    oracle = ImpactOracle(robot, t_fault, frame_dt=frame_dt)

    dt = m.opt.timestep
    decim = int(round(1.0 / (POLICY_HZ * dt)))
    horizon = t_fault + (NO_FAULT_HORIZON_S if fault_id == "C0_no_fault" else POST_FAULT_HORIZON_S)
    n_steps = int(round(horizon / dt))
    push = scenario.get("push")
    torso = robot.body_ids["torso"]
    cmd = policy.compute(d, 0.0)
    engaged = False
    last_cmd_stamp = 0.0

    oracle.step(d, 0.0)
    for k in range(n_steps):
        t = k * dt
        if k % decim == 0 and fault.policy_alive(t):
            cmd = policy.compute(d, t)
            last_cmd_stamp = t
        tau = drives.torque(d, cmd, fault.torque_scale(t))
        if not engaged and t >= fault.detection_time:
            reaction.engage(d, t)
            oracle.containment_t = t
            engaged = True
        if engaged:
            tau = reaction.override(d, tau)
        d.ctrl[:] = tau / robot.gear

        d.xfrc_applied[torso, :3] = 0.0
        if push is not None and push["t"] <= t < push["t"] + push["dur"]:
            d.xfrc_applied[torso, 0:2] = push["force_xy"]

        mujoco.mj_step(m, d)
        oracle.step(d, t + dt)
        if oracle.ground_t is not None and t + dt > oracle.ground_t + POST_FALL_TAIL_S:
            break

    res = oracle.verdict()
    res.update({
        "fault_id": fault_id,
        "arm": arm if fault_id != "C0_no_fault" else "nominal_control",
        "fault": fault.describe(),
        "containment_engaged_s": oracle.containment_t,
        "last_policy_command_s": round(last_cmd_stamp, 4),
        "sim_end_s": round((k + 1) * dt, 4),
        "scenario": scenario,
    })
    return {"result": res, "trace": oracle.trace, "frames": oracle.frames, "frame_t": oracle.frame_t}
