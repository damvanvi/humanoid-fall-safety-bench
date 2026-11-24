# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Containment baselines under test (naive kill switches) plus one reference arm."""
from baselines.naive_brake_lock import NaiveBrakeLock
from baselines.naive_sto import NaiveSTO
from baselines.realistic_brake_lock import RealisticBrakeLock, RealisticBrakeLockAllJoints
from baselines.reference import NoContainmentReference

# v0.1 baseline set (order matters for reproducing the v0.1 seed-42 run set exactly)
BASELINES = {
    NaiveSTO.name: NaiveSTO,
    NaiveBrakeLock.name: NaiveBrakeLock,
}
# v0.2 adds the realistic brake model alongside the v0.1 baselines
BASELINES_V02 = {**BASELINES, RealisticBrakeLock.name: RealisticBrakeLock}
REFERENCE_ARMS = {
    NoContainmentReference.name: NoContainmentReference,
}
# Diagnostic ablations (reported separately, not as baselines)
DIAGNOSTIC_ARMS = {RealisticBrakeLockAllJoints.name: RealisticBrakeLockAllJoints}
ALL_ARMS = {**BASELINES_V02, **REFERENCE_ARMS, **DIAGNOSTIC_ARMS}
