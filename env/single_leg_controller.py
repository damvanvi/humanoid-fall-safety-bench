# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""v0.2 nominal (pre-fault) controller for the single-leg stance scenario.

Same two-layer structure as env/nominal_controller.py (100 Hz policy publishing
command packets, 1 kHz joint PD drives), so the same faults apply unchanged.
The policy runs a fixed, time-scripted transition:

  0.0 - 1.0 s   settle in double support (knee bend A0)
  1.0 - 2.5 s   lateral weight shift: both legs roll so the COM moves over the
                stance (right) foot
  2.5 - 3.0 s   lift the swing (left) foot about 0.12 m clear of the floor
  3.0 s ->      hold single-leg stance on the ~0.22 x 0.10 m stance foot

The stance leg is the RIGHT leg. This choice was fixed before any baseline was
run: F3_knee_overheat derates the right knee hardest, and in single support the
loaded knee is the one that heats up, so the derated knee is the stance knee.

Balance: ankle-strategy COM feedback. The reference is the COM of the commanded
posture (pure kinematics), measured relative to a support point that moves from
the ankle midpoint to the stance ankle during the shift. It is NOT a protective
function and plays no role after containment engages.
"""
from __future__ import annotations

import mujoco
import numpy as np

from env.nominal_controller import (POLICY_HZ, Command, JointDrives,  # noqa: F401  (re-export)
                                    nominal_posture)
from env.robot import Robot

A0 = 0.25            # knee-bend parameter held throughout (rad)
T_SETTLE = 1.0
T_SHIFT_END = 2.5
T_LIFT_END = 3.0
LIFT_H = 0.35        # extra swing-leg flexion parameter at full lift (rad) -> ~0.12 m foot clearance
STANCE = "r"
SWING = "l"

# Ankle-strategy COM feedback gains (whole robot), split by support weight.
BAL_KP_X, BAL_KD_X = 900.0, 200.0
BAL_KP_Y, BAL_KD_Y = 900.0, 300.0
# Posture-level COM servo: roll/pitch the stance chain to move the COM back over the
# stance foot (proportional + integral on the COM error, rad per m).
POS_KP_Y, POS_KI_Y = 2.0, 2.0
POS_KP_X, POS_KI_X = 1.0, 2.0
I_CLAMP = 0.05
# Stiffer roll servos for single support (hip roll carries the whole upper body).
SL_GAINS = {"hip_x": (1500.0, 60.0), "ankle_x": (600.0, 25.0)}


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def phase(t: float) -> tuple:
    """(w, lift): support weight on the stance foot (0..1) and swing-lift fraction (0..1)."""
    w = _smooth((t - T_SETTLE) / (T_SHIFT_END - T_SETTLE))
    lift = _smooth((t - T_SHIFT_END) / (T_LIFT_END - T_SHIFT_END))
    return w, lift


def single_leg_posture(robot: Robot, phi: float, lift: float) -> np.ndarray:
    """Double-support posture rolled by phi toward the stance side, with the swing leg lifted."""
    q = nominal_posture(robot, A0)
    # Rolling both legs by phi (hip_x = +phi, ankle_x = -phi) moves the pelvis toward -y
    # (over the right foot) while both feet stay flat.
    for side in ("r", "l"):
        q[robot.idx(f"{side}_hip_x")] = phi
        q[robot.idx(f"{side}_ankle_x")] = -phi
    h = LIFT_H * lift
    q[robot.idx(f"{SWING}_hip_y")] = -A0 - h
    q[robot.idx(f"{SWING}_knee")] = 2.0 * (A0 + h)
    q[robot.idx(f"{SWING}_ankle_y")] = -A0 - h
    return q


class _Kin:
    """Scratch MjData for kinematic COM of a commanded posture (root fixed upright at origin)."""

    def __init__(self, robot: Robot):
        self.r = robot
        self.d = mujoco.MjData(robot.model)

    def com_rel(self, q: np.ndarray, w: float) -> np.ndarray:
        r, d = self.r, self.d
        d.qpos[:] = r.model.qpos0
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[r.qpos_adr] = q
        mujoco.mj_kinematics(r.model, d)
        mujoco.mj_comPos(r.model, d)
        sup = _support_point(r, d, w)
        return d.subtree_com[r.body_ids["pelvis"]][:2] - sup


def _support_point(robot: Robot, data: mujoco.MjData, w: float) -> np.ndarray:
    st = data.xpos[robot.body_ids[f"{STANCE}_foot"]][:2]
    sw = data.xpos[robot.body_ids[f"{SWING}_foot"]][:2]
    mid = 0.5 * (st + sw)
    return (1.0 - w) * mid + w * st


def solve_phi(robot: Robot) -> float:
    """Roll angle that puts the COM of the lifted posture over the stance ankle (bisection)."""
    kin = _Kin(robot)

    def err(phi):
        return kin.com_rel(single_leg_posture(robot, phi, 1.0), 1.0)[1]

    lo, hi = 0.0, 0.4
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(err(mid)) == np.sign(err(lo)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class SingleLegDrives(JointDrives):
    """Same 1 kHz PD drives as v0.1, with stiffer hip/ankle roll gains for single support."""

    def __init__(self, robot: Robot):
        super().__init__(robot)
        for i, n in enumerate(robot.joint_names):
            for suf, (kp, kd) in SL_GAINS.items():
                if n.endswith(suf):
                    self.kp[i], self.kd[i] = kp, kd


class SingleLegPolicy:
    """Stand-in high-level policy: scripted transition into single-leg stance + COM balance."""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.kin = _Kin(robot)
        self.phi_star = solve_phi(robot)
        self.st_y, self.st_x = robot.idx(f"{STANCE}_ankle_y"), robot.idx(f"{STANCE}_ankle_x")
        self.sw_y, self.sw_x = robot.idx(f"{SWING}_ankle_y"), robot.idx(f"{SWING}_ankle_x")
        self.integ = np.zeros(2)
        self.dt = 1.0 / POLICY_HZ

    def compute(self, data: mujoco.MjData, t: float) -> Command:
        r = self.robot
        w, lift = phase(t)
        q_plan = single_leg_posture(r, self.phi_star * w, lift)
        ref = self.kin.com_rel(q_plan, w)
        mujoco.mj_subtreeVel(r.model, data)
        pel = r.body_ids["pelvis"]
        e = data.subtree_com[pel][:2] - _support_point(r, data, w) - ref
        v = data.subtree_linvel[pel][:2]
        self.integ = np.clip(self.integ + e * self.dt, -I_CLAMP, I_CLAMP)
        # Posture servo: COM too far toward -y -> roll legs back (+y); too far forward -> lean back.
        d_phi = POS_KP_Y * e[1] + POS_KI_Y * self.integ[1]
        d_pitch = POS_KP_X * e[0] + POS_KI_X * self.integ[0]
        q_ref = single_leg_posture(r, self.phi_star * w + d_phi, lift)
        for side in ("r", "l"):
            q_ref[r.idx(f"{side}_ankle_y")] += d_pitch
        tx = BAL_KP_X * e[0] + BAL_KD_X * v[0]
        ty = -(BAL_KP_Y * e[1] + BAL_KD_Y * v[1])
        share_st = 0.5 + 0.5 * w
        share_sw = 0.0 if lift > 0.0 else 1.0 - share_st
        tau = np.zeros(len(r.joint_names))
        tau[self.st_y], tau[self.st_x] = share_st * tx, share_st * ty
        tau[self.sw_y], tau[self.sw_x] = share_sw * tx, share_sw * ty
        return Command(q_ref=q_ref, tau_ff=tau, stamp=t)
