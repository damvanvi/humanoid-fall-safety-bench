# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Nominal (pre-fault) controller: a deliberately simple PD standing/balance stack.

Purpose: hold the humanoid upright and performing a slow squat task *before* the
fault timestamp, so every run starts its fault from a clean, deterministic,
dynamically active state. It is NOT a safety controller and plays no role after
containment engages.

Two layers, mirroring a real humanoid stack:
  * HighLevelPolicy  -- the "AI policy" stand-in. Runs at 100 Hz and publishes a
                        command packet (joint targets + ankle balance torques).
                        Fault F1 freezes this layer.
  * JointDrives      -- per-joint PD servo loop at 1 kHz that tracks the most
                        recent command packet, clipped to the available torque.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from env.robot import Robot

POLICY_HZ = 100.0

# Nominal task: continuous slow squat (knee-bend parameter a, rad), ramped in:
#   a(t) = A0 + A1 * sin(2*pi*F*t)
# It keeps joints, COM and contact forces in motion at the fault timestamp, so
# containment is never tested on a perfectly static robot.
SQUAT_A0 = 0.25
SQUAT_A1 = 0.08
TASK_HZ = 0.5
RAMP_S = 1.0

# PD gains per joint-name suffix (N m/rad, N m s/rad).
_GAINS = {
    "abdomen_y": (500.0, 25.0), "abdomen_x": (500.0, 25.0),
    "shoulder_y": (80.0, 4.0), "shoulder_x": (80.0, 4.0), "elbow": (40.0, 2.0),
    "hip_x": (600.0, 30.0), "hip_z": (200.0, 10.0), "hip_y": (700.0, 35.0),
    "knee": (800.0, 40.0), "ankle_y": (450.0, 20.0), "ankle_x": (300.0, 15.0),
}

# Ankle-strategy COM feedback (whole-robot gains, split equally over both ankles).
BAL_KP_X, BAL_KD_X = 700.0, 160.0   # N m / m, N m s / m
BAL_KP_Y, BAL_KD_Y = 500.0, 120.0


def _suffix(name: str) -> str:
    for k in _GAINS:
        if name.endswith(k):
            return k
    raise KeyError(name)


def squat_param(t: float) -> float:
    ramp = min(1.0, max(0.0, t / RAMP_S))
    return SQUAT_A0 + ramp * SQUAT_A1 * np.sin(2.0 * np.pi * TASK_HZ * t)


def nominal_posture(robot: Robot, a: float) -> np.ndarray:
    """Joint targets (actuator order) for knee bend a: flat feet, upright trunk, relaxed arms."""
    q = np.zeros(len(robot.joint_names))
    for side in ("r", "l"):
        q[robot.idx(f"{side}_hip_y")] = -a
        q[robot.idx(f"{side}_knee")] = 2.0 * a
        q[robot.idx(f"{side}_ankle_y")] = -a
        q[robot.idx(f"{side}_elbow")] = -0.35
        q[robot.idx(f"{side}_shoulder_y")] = -0.10
    q[robot.idx("r_shoulder_x")] = -0.15
    q[robot.idx("l_shoulder_x")] = 0.15
    return q


def _root_height(a: float) -> float:
    """Pelvis height that puts both flat feet on the floor for knee bend a."""
    return 0.08 + 0.84 * np.cos(a) + 0.03 + 1e-3


def com_offset_table(robot: Robot, n: int = 21) -> tuple:
    """Static COM-x offset (COM minus ankle midpoint) of the nominal posture vs knee bend.

    Used as the balance reference so the ankle feedback follows the planned squat
    instead of fighting it. Pure kinematics, computed once at start-up.
    """
    m = robot.model
    d = mujoco.MjData(m)
    a_grid = np.linspace(SQUAT_A0 - SQUAT_A1, SQUAT_A0 + SQUAT_A1, n)
    offs = np.zeros(n)
    for i, a in enumerate(a_grid):
        mujoco.mj_resetData(m, d)
        d.qpos[robot.qpos_adr] = nominal_posture(robot, a)
        d.qpos[0:3] = [0.0, 0.0, _root_height(a)]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        ank = 0.5 * (d.xpos[robot.body_ids["r_foot"]] + d.xpos[robot.body_ids["l_foot"]])
        offs[i] = d.subtree_com[robot.body_ids["pelvis"]][0] - ank[0]
    return a_grid, offs


def initial_state(robot: Robot, data: mujoco.MjData) -> None:
    """Place the robot in the t=0 nominal posture with flat feet resting on the floor."""
    m = robot.model
    mujoco.mj_resetData(m, data)
    a = squat_param(0.0)
    data.qpos[robot.qpos_adr] = nominal_posture(robot, a)
    data.qpos[0:3] = [0.0, 0.0, _root_height(a)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(m, data)


@dataclass
class Command:
    """One command packet from the high-level policy to the joint drives."""
    q_ref: np.ndarray
    tau_ff: np.ndarray
    stamp: float


class HighLevelPolicy:
    """Stand-in for the high-level AI policy: slow squat task + ankle-strategy balance."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.period = 1.0 / POLICY_HZ
        self.ia_y = [robot.idx("r_ankle_y"), robot.idx("l_ankle_y")]
        self.ia_x = [robot.idx("r_ankle_x"), robot.idx("l_ankle_x")]
        self.com_a, self.com_off = com_offset_table(robot)

    def compute(self, data: mujoco.MjData, t: float) -> Command:
        r = self.robot
        a = squat_param(t)
        q_ref = nominal_posture(r, a)
        mujoco.mj_subtreeVel(r.model, data)
        pel = r.body_ids["pelvis"]
        com = data.subtree_com[pel]
        vcom = data.subtree_linvel[pel]
        ank = 0.5 * (data.xpos[r.body_ids["r_foot"]] + data.xpos[r.body_ids["l_foot"]])
        ex = com[0] - (ank[0] + np.interp(a, self.com_a, self.com_off))
        ey = com[1] - ank[1]
        tau = np.zeros(len(r.joint_names))
        tau[self.ia_y] = 0.5 * (BAL_KP_X * ex + BAL_KD_X * vcom[0])
        tau[self.ia_x] = -0.5 * (BAL_KP_Y * ey + BAL_KD_Y * vcom[1])
        return Command(q_ref=q_ref, tau_ff=tau, stamp=t)


class JointDrives:
    """1 kHz joint PD servo loop tracking the latest command packet."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.kp = np.array([_GAINS[_suffix(n)][0] for n in robot.joint_names])
        self.kd = np.array([_GAINS[_suffix(n)][1] for n in robot.joint_names])

    def torque(self, data: mujoco.MjData, cmd: Command, limit_scale: np.ndarray) -> np.ndarray:
        r = self.robot
        q = data.qpos[r.qpos_adr]
        qd = data.qvel[r.dof_adr]
        tau = self.kp * (cmd.q_ref - q) - self.kd * qd + cmd.tau_ff
        lim = r.gear * limit_scale
        return np.clip(tau, -lim, lim)
