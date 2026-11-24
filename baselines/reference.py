# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Reference arm -- fault with NO containment (control condition, not a baseline).

The trip signal is ignored and the drives keep doing whatever the (possibly
faulted) stack commands. It isolates how much of the outcome is caused by the
fault itself versus by the containment reaction. It is not a safe reaction:
a hung policy whose drives keep holding torque is itself a hazard, which is
why it is reported separately and never as a candidate mitigation.
"""
from __future__ import annotations

import numpy as np


class NoContainmentReference:
    name = "ref_no_containment"
    label = "Reference: fault, no containment"

    def __init__(self, robot):
        self.robot = robot
        self.engaged_at = None

    def engage(self, data, t: float) -> None:
        pass  # trip ignored by design

    def override(self, data, tau: np.ndarray) -> np.ndarray:
        return tau
