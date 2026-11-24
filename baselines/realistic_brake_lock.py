# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Baseline 3 (v0.2) -- realistic brake lock: spring-applied electromechanical brakes.

Same reflex as naive_brake_lock (cut motor torque, clamp the brakes), but with
the three properties that practical joint brakes have and the ideal model lacks:

1. Engagement delay. On the trip signal motor torque goes to zero at once (STO).
   The brake armature needs time to release and the spring to clamp, so no
   braking torque exists for DEAD_TIME_S; it then ramps to full over RAMP_S.
2. Finite holding torque. Each brake is a dry-friction clamp: it holds while the
   load on the joint stays below its holding torque, and slips at that torque
   otherwise. It is modelled with MuJoCo's joint friction loss (a Coulomb
   friction constraint), stiffened (solref 5 ms, solimp 0.999/0.9999) so that a
   braked joint does not creep below its holding torque: measured drift is
   < 1 mrad over 2 s at about 25 % of holding torque. Without this, MuJoCo's
   default soft friction lets loaded joints creep about 0.05 rad/s, which would
   be a simulation artifact rather than a brake property.
3. Partial braking. Brakes are fitted only to joints that carry significant
   gravity load in stance: both legs' hip roll, hip pitch, knee, ankle pitch,
   ankle roll, plus the waist (abdomen pitch and roll). Hip yaw (vertical axis,
   no gravity load in stance) and the arms are left unbraked and fall limp.

All parameters below were fixed before any v0.2 result was computed. HOLD_SCALE
multiplies each joint's rated continuous motor torque (its actuator gear).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class BrakeConfig:
    dead_time_s: float = 0.040
    ramp_s: float = 0.010
    hold_scale: float = 1.0
    braked_suffixes: tuple = ("hip_x", "hip_y", "knee", "ankle_y", "ankle_x", "abdomen_y", "abdomen_x")
    solref: tuple = (0.005, 1.0)
    solimp: tuple = (0.999, 0.9999, 0.001)


DEFAULT_CONFIG = BrakeConfig()


class RealisticBrakeLock:
    name = "realistic_brake_lock"
    label = "Realistic brake lock (40 ms delay, finite hold, partial)"
    config = DEFAULT_CONFIG

    def __init__(self, robot):
        self.robot = robot
        self.engaged_at = None
        cfg = self.config
        self.braked = np.array([any(n.endswith(s) for s in cfg.braked_suffixes) for n in robot.joint_names])
        self.dofs = robot.dof_adr[self.braked]
        self.hold = robot.gear[self.braked] * cfg.hold_scale

    def engage(self, data: mujoco.MjData, t: float) -> None:
        m = self.robot.model
        for dof in self.dofs:
            m.dof_solref[dof, :] = self.config.solref
            m.dof_solimp[dof, :3] = self.config.solimp
        self.engaged_at = t

    def override(self, data: mujoco.MjData, tau: np.ndarray) -> np.ndarray:
        cfg = self.config
        s = (data.time - self.engaged_at - cfg.dead_time_s) / cfg.ramp_s
        s = min(1.0, max(0.0, s))
        self.robot.model.dof_frictionloss[self.dofs] = self.hold * s
        return np.zeros_like(tau)

    @classmethod
    def describe(cls) -> dict:
        c = cls.config
        return {"dead_time_s": c.dead_time_s, "ramp_s": c.ramp_s, "hold_scale_x_rated": c.hold_scale,
                "braked_joint_suffixes": list(c.braked_suffixes), "unbraked": "hip_z (both), arms",
                "solref": list(c.solref), "solimp": list(c.solimp)}


class RealisticBrakeLockAllJoints(RealisticBrakeLock):
    """Diagnostic ablation: same delay and finite holding torque, but every joint braked.

    Not a separate baseline claim: it separates the effect of partial braking from
    the effect of engagement delay + finite holding torque.
    """

    name = "realistic_brake_all_joints"
    label = "Diagnostic: realistic brake on all joints"
    config = BrakeConfig(braked_suffixes=("",))   # "" matches every joint name
