# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Deterministic fault triggers.

Every fault is a pure function of simulation time (no hidden RNG): given the same
onset time and parameters it produces bit-identical effects. Randomness, where
used, lives only in the scenario sampler of run_bench.py and is seeded.

Each fault exposes three things to the simulation loop:
  policy_alive(t)      -> False while the high-level policy is frozen (F1)
  torque_scale(t)      -> per-actuator multiplier on available torque (F2, F3)
  detection_time       -> time at which the platform's fault monitor raises the
                          trip signal that the containment ("kill switch") acts on

  F1_policy_hang    High-level policy freezes for >= 500 ms (default 600 ms). The
                    joint drives keep servoing on the last, stale command packet,
                    so balance feedback is frozen. A heartbeat watchdog trips
                    after `watchdog_s` (default 100 ms).
  F2_dcbus_sag      Sudden -40 % drop in available actuator torque on every joint
                    (DC-bus sag / power-limit fold-back). Under-voltage comparator
                    trips after `detect_s` (default 5 ms).
  F3_knee_overheat  Asymmetric thermal derating: right-knee torque ramps linearly
                    to 25 % of rated over 0.8 s, left knee to 85 %. Over-temperature
                    trip fires when any joint has lost >= 40 % of rated torque.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.robot import Robot

FAULT_IDS = ("F1_policy_hang", "F2_dcbus_sag", "F3_knee_overheat")


@dataclass
class Fault:
    fault_id: str
    onset: float
    n_act: int
    params: dict = field(default_factory=dict)
    detection_time: float = float("inf")

    def policy_alive(self, t: float) -> bool:
        return True

    def torque_scale(self, t: float) -> np.ndarray:
        return np.ones(self.n_act)

    def describe(self) -> dict:
        return {"fault_id": self.fault_id, "onset_s": self.onset,
                "detection_s": self.detection_time, "params": dict(self.params)}


class NoFault(Fault):
    def __init__(self, robot: Robot):
        super().__init__("C0_no_fault", float("inf"), len(robot.joint_names))


class PolicyHang(Fault):
    def __init__(self, robot: Robot, onset: float, hang_s: float = 0.6, watchdog_s: float = 0.1):
        if hang_s < 0.5:
            raise ValueError("F1_policy_hang requires a freeze of >= 500 ms")
        super().__init__("F1_policy_hang", onset, len(robot.joint_names),
                         {"hang_s": hang_s, "watchdog_s": watchdog_s})
        self.detection_time = onset + watchdog_s

    def policy_alive(self, t: float) -> bool:
        return not (self.onset <= t < self.onset + self.params["hang_s"])


class DCBusSag(Fault):
    def __init__(self, robot: Robot, onset: float, drop: float = 0.40, detect_s: float = 0.005):
        super().__init__("F2_dcbus_sag", onset, len(robot.joint_names),
                         {"torque_drop": drop, "detect_s": detect_s})
        self.detection_time = onset + detect_s

    def torque_scale(self, t: float) -> np.ndarray:
        s = np.ones(self.n_act)
        if t >= self.onset:
            s *= 1.0 - self.params["torque_drop"]
        return s


class KneeOverheat(Fault):
    def __init__(self, robot: Robot, onset: float, ramp_s: float = 0.8,
                 right_final: float = 0.25, left_final: float = 0.85, trip_loss: float = 0.40):
        super().__init__("F3_knee_overheat", onset, len(robot.joint_names),
                         {"ramp_s": ramp_s, "r_knee_final": right_final,
                          "l_knee_final": left_final, "trip_loss": trip_loss})
        self.ir = robot.idx("r_knee")
        self.il = robot.idx("l_knee")
        # Right knee derates fastest; trip when it has lost `trip_loss` of rated torque.
        frac = trip_loss / (1.0 - right_final)
        self.detection_time = onset + ramp_s * min(1.0, frac)

    def torque_scale(self, t: float) -> np.ndarray:
        s = np.ones(self.n_act)
        if t >= self.onset:
            p = min(1.0, (t - self.onset) / self.params["ramp_s"])
            s[self.ir] = 1.0 - p * (1.0 - self.params["r_knee_final"])
            s[self.il] = 1.0 - p * (1.0 - self.params["l_knee_final"])
        return s


def make_fault(fault_id: str, robot: Robot, onset: float, **kw) -> Fault:
    if fault_id == "F1_policy_hang":
        return PolicyHang(robot, onset, **kw)
    if fault_id == "F2_dcbus_sag":
        return DCBusSag(robot, onset, **kw)
    if fault_id == "F3_knee_overheat":
        return KneeOverheat(robot, onset, **kw)
    if fault_id in ("C0_no_fault", "none"):
        return NoFault(robot)
    raise ValueError(f"unknown fault {fault_id!r}; choose from {FAULT_IDS}")
