# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Model loading and index bookkeeping for the benchmark humanoid."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanoid_model.xml")

FOOT_GEOMS = ("r_foot", "l_foot")
HEAD_GEOMS = ("head",)
TORSO_GEOMS = ("torso", "neck")


def model_sha256(path: str = MODEL_PATH) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass
class Robot:
    """Thin wrapper holding the MuJoCo model plus the index maps every module needs."""

    model: mujoco.MjModel
    joint_names: list = field(default_factory=list)   # actuated joints, actuator order
    qpos_adr: np.ndarray = None                       # qpos address per actuator
    dof_adr: np.ndarray = None                        # qvel address per actuator
    gear: np.ndarray = None                           # rated torque per actuator (N m)
    eq_ids: np.ndarray = None                         # brake-lock equality id per actuator
    floor_geom: int = -1
    foot_geoms: frozenset = frozenset()
    head_geoms: frozenset = frozenset()
    torso_geoms: frozenset = frozenset()
    body_ids: dict = field(default_factory=dict)
    total_mass: float = 0.0

    def idx(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)


def load_robot(path: str = MODEL_PATH) -> Robot:
    model = mujoco.MjModel.from_xml_path(path)
    # Energy bookkeeping (data.energy[1] = kinetic energy) is used by the impact oracle.
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)

    names, qadr, dadr, gear, eqs = [], [], [], [], []
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names.append(jn)
        qadr.append(int(model.jnt_qposadr[j]))
        dadr.append(int(model.jnt_dofadr[j]))
        gear.append(float(model.actuator_gear[a, 0]))
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "lock_" + jn)
        if eq < 0:
            raise RuntimeError(f"missing brake equality lock_{jn} in model")
        eqs.append(eq)

    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    bodies = {n: bid(n) for n in ("pelvis", "torso", "head", "r_foot", "l_foot")}

    return Robot(
        model=model,
        joint_names=names,
        qpos_adr=np.array(qadr),
        dof_adr=np.array(dadr),
        gear=np.array(gear),
        eq_ids=np.array(eqs),
        floor_geom=gid("floor"),
        foot_geoms=frozenset(gid(n) for n in FOOT_GEOMS),
        head_geoms=frozenset(gid(n) for n in HEAD_GEOMS),
        torso_geoms=frozenset(gid(n) for n in TORSO_GEOMS),
        body_ids=bodies,
        total_mass=float(np.sum(model.body_mass)),
    )
