# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Baseline 1 -- naive Safe Torque Off (STO).

On the first trip signal every actuator's torque is cut to exactly zero and stays
latched at zero for the rest of the episode (IEC 61800-5-2 style STO used as an
uncategorised "kill switch"). Only passive joint damping and joint limits remain.
This is the reflex used for fixed industrial arms; the bench measures what it
does to a robot whose stability depends on active control.
"""
from __future__ import annotations

import mujoco
import numpy as np


class NaiveSTO:
    name = "naive_sto"
    label = "Naive STO (instant zero torque)"

    def __init__(self, robot):
        self.robot = robot
        self.engaged_at = None

    def engage(self, data: mujoco.MjData, t: float) -> None:
        self.engaged_at = t

    def override(self, data: mujoco.MjData, tau: np.ndarray) -> np.ndarray:
        return np.zeros_like(tau)
