# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Physical impact oracle -- deterministic PASS/FAIL verdict per episode.

Measured every physics step (1 kHz):
  * peak head impact velocity (m/s)   -- body-COM speed of the head on the step
                                         *before* each head-floor contact onset;
                                         maximum over all onsets (bounces count)
  * peak torso impact velocity (m/s)  -- same for the torso (chest + neck geoms)
  * kinetic energy at ground contact  -- whole-robot kinetic energy (J) on the step
                                         before the first non-foot floor contact
  * time-to-fall (s)                  -- first fall event minus fault onset
                                         (also reported relative to containment)

Fall event: any robot geom other than the two feet touches the floor, or the
pelvis drops below FALL_PELVIS_Z (whichever comes first). Kinetic energy is
always taken at the first real non-foot floor contact. Verdict = PASS iff no fall event occurs within
the episode horizon; otherwise FAIL with a failure mode
(head_strike > trunk_strike > limb_or_pelvis_contact > collapse_height).

Thresholds are bench conventions (v0.1), not values taken from any standard.
"""
from __future__ import annotations

import hashlib

import mujoco
import numpy as np

FALL_PELVIS_Z = 0.45      # m (nominal pelvis height ~0.92 m)
TRACE_DT = 0.01           # s, decimation of the stored time series


class ImpactOracle:
    def __init__(self, robot, fault_onset: float, frame_dt: float | None = None):
        self.r = robot
        self.m = robot.model
        self.onset = fault_onset
        self.head_b = robot.body_ids["head"]
        self.torso_b = robot.body_ids["torso"]
        self.pelvis_b = robot.body_ids["pelvis"]
        self._v6 = np.zeros(6)
        self._prev = {"head_v": 0.0, "torso_v": 0.0, "head_vz": 0.0, "torso_vz": 0.0, "ke": 0.0}
        self._prev_contact = {"head": False, "torso": False}
        self.peak_head_impact = 0.0
        self.peak_head_impact_vz = 0.0
        self.peak_torso_impact = 0.0
        self.peak_torso_impact_vz = 0.0
        self.head_contact_t = None
        self.torso_contact_t = None
        self.fall_t = None
        self.fall_cause = None
        self.ground_t = None
        self.ke_at_contact = None
        self.ke_peak_post_fault = 0.0
        self.first_body = None
        self.containment_t = None
        self.sha = hashlib.sha256()
        self.trace = {k: [] for k in ("t", "head_z", "torso_z", "pelvis_z", "ke", "head_speed")}
        self._next_trace = 0.0
        self.frame_dt = frame_dt
        self.frames = []
        self.frame_t = []
        self._next_frame = 0.0

    def _speed(self, body: int):
        mujoco.mj_objectVelocity(self.m, self._data, mujoco.mjtObj.mjOBJ_BODY, body, self._v6, 0)
        lin = self._v6[3:6]
        return float(np.linalg.norm(lin)), float(lin[2])

    def step(self, data: mujoco.MjData, t: float) -> None:
        self._data = data
        r = self.r
        head = torso = other = False
        other_body = None
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == r.floor_geom:
                g = g2
            elif g2 == r.floor_geom:
                g = g1
            else:
                continue
            if g in r.foot_geoms:
                continue
            if g in r.head_geoms:
                head = True
            elif g in r.torso_geoms:
                torso = True
            else:
                other = True
                other_body = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[g]))

        hv, hvz = self._speed(self.head_b)
        tv, tvz = self._speed(self.torso_b)
        ke = float(data.energy[1])
        p = self._prev

        if head and not self._prev_contact["head"]:
            if p["head_v"] > self.peak_head_impact:
                self.peak_head_impact, self.peak_head_impact_vz = p["head_v"], p["head_vz"]
            if self.head_contact_t is None:
                self.head_contact_t = t
        if torso and not self._prev_contact["torso"]:
            if p["torso_v"] > self.peak_torso_impact:
                self.peak_torso_impact, self.peak_torso_impact_vz = p["torso_v"], p["torso_vz"]
            if self.torso_contact_t is None:
                self.torso_contact_t = t

        if self.ground_t is None and (head or torso or other):
            self.ground_t = t
            self.ke_at_contact = p["ke"]          # pre-impact (previous step)
            self.first_body = "head" if head else ("torso" if torso else other_body)
        if self.fall_t is None:
            if self.ground_t is not None:
                self.fall_t, self.fall_cause = t, "ground_contact"
            elif data.xpos[self.pelvis_b][2] < FALL_PELVIS_Z:
                self.fall_t, self.fall_cause = t, "collapse_height"

        if t >= self.onset:
            self.ke_peak_post_fault = max(self.ke_peak_post_fault, ke)

        self._prev_contact["head"], self._prev_contact["torso"] = head, torso
        p.update(head_v=hv, head_vz=hvz, torso_v=tv, torso_vz=tvz, ke=ke)
        self.sha.update(data.qpos.tobytes())

        if t + 1e-9 >= self._next_trace:
            tr = self.trace
            tr["t"].append(t)
            tr["head_z"].append(float(data.xpos[self.head_b][2]))
            tr["torso_z"].append(float(data.xipos[self.torso_b][2]))
            tr["pelvis_z"].append(float(data.xpos[self.pelvis_b][2]))
            tr["ke"].append(ke)
            tr["head_speed"].append(hv)
            self._next_trace += TRACE_DT
        if self.frame_dt is not None and t + 1e-9 >= self._next_frame:
            self.frames.append(data.qpos.copy())
            self.frame_t.append(t)
            self._next_frame += self.frame_dt

    def verdict(self) -> dict:
        fell = self.fall_t is not None
        if not fell:
            mode = None
        elif self.head_contact_t is not None:
            mode = "head_strike"
        elif self.torso_contact_t is not None:
            mode = "trunk_strike"
        elif self.ground_t is not None:
            mode = "limb_or_pelvis_contact"
        else:
            mode = "collapse_height"
        rel = lambda x: None if x is None else round(x - self.onset, 4)  # noqa: E731
        ttf_cont = None
        if fell and self.containment_t is not None:
            ttf_cont = round(self.fall_t - self.containment_t, 4)
        return {
            "verdict": "FAIL" if fell else "PASS",
            "failure_mode": mode,
            "fell": fell,
            "first_ground_body": self.first_body,
            "time_to_fall_s": rel(self.fall_t),
            "time_to_ground_contact_s": rel(self.ground_t),
            "time_to_fall_from_containment_s": ttf_cont,
            "peak_head_impact_velocity_mps": round(self.peak_head_impact, 4),
            "peak_head_impact_vz_mps": round(self.peak_head_impact_vz, 4),
            "peak_torso_impact_velocity_mps": round(self.peak_torso_impact, 4),
            "peak_torso_impact_vz_mps": round(self.peak_torso_impact_vz, 4),
            "ke_at_ground_contact_J": None if self.ke_at_contact is None else round(self.ke_at_contact, 4),
            "ke_peak_post_fault_J": round(self.ke_peak_post_fault, 4),
            "head_contact": self.head_contact_t is not None,
            "torso_contact": self.torso_contact_t is not None,
            "trajectory_sha256": self.sha.hexdigest(),
        }
