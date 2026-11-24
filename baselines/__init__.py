# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Containment baselines under test (naive kill switches) plus one reference arm."""
from baselines.naive_brake_lock import NaiveBrakeLock
from baselines.naive_sto import NaiveSTO
from baselines.reference import NoContainmentReference

BASELINES = {
    NaiveSTO.name: NaiveSTO,
    NaiveBrakeLock.name: NaiveBrakeLock,
}
REFERENCE_ARMS = {
    NoContainmentReference.name: NoContainmentReference,
}
ALL_ARMS = {**BASELINES, **REFERENCE_ARMS}
