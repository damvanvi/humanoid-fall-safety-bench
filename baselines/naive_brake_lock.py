# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Baseline 2 -- naive instant mechanical brake lock.

On the first trip signal motor torque is cut (STO) and every joint brake engages
at once, freezing each joint at its current angle. The brake is modelled as a
stiff, critically damped joint-equality constraint (solref = 5 ms, damping ratio
1), i.e. an ideal high-stiffness / high-damping lock with no slip limit. The
humanoid becomes a single rigid body standing on its feet -- the common
"brake and freeze" reflex.
"""
from __future__ import annotations

import mujoco
import numpy as np

LOCK_SOLREF = (0.005, 1.0)


class NaiveBrakeLock:
    name = "naive_brake_lock"
    label = "Naive brake lock (instant rigid freeze)"

    def __init__(self, robot):
        self.robot = robot
        self.engaged_at = None

    def engage(self, data: mujoco.MjData, t: float) -> None:
        r, m = self.robot, self.robot.model
        q_now = data.qpos[r.qpos_adr]
        for k, eq in enumerate(r.eq_ids):
            j = int(m.eq_obj1id[eq])
            # Joint equality with no joint2: qpos[j] - qpos0[j] == eq_data[0]
            m.eq_data[eq, :] = 0.0
            m.eq_data[eq, 0] = q_now[k] - m.qpos0[m.jnt_qposadr[j]]
            m.eq_solref[eq, :] = LOCK_SOLREF
            data.eq_active[eq] = 1
        self.engaged_at = t

    def override(self, data: mujoco.MjData, tau: np.ndarray) -> np.ndarray:
        return np.zeros_like(tau)
