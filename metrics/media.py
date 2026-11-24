# Copyright 2026 BivectorAI. Licensed under the Apache License, Version 2.0.
"""Media export: comparison PNG (always, no OpenGL needed) and comparison GIF (if offscreen GL works).

The GIF is rendered in a *separate process* (``python -m metrics.media gif ...``)
launched by run_bench.py once per candidate MUJOCO_GL backend (egl, osmesa, glfw).
A missing or broken OpenGL stack -- common on WSL -- can abort a process outright
rather than raise, so isolating it guarantees the benchmark itself never crashes;
run_bench.py simply falls back to the PNG.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Palette: first three categorical slots of the dataviz reference palette (validated
# all-pairs for CVD), plus neutral gray for the control.
COLORS = {
    "ref_no_containment": "#2a78d6",
    "naive_sto": "#eb6834",
    "naive_brake_lock": "#1baf7a",
    "nominal_control": "#8a8985",
}
SHORT = {
    "ref_no_containment": "Reference (no containment)",
    "naive_sto": "Naive STO",
    "naive_brake_lock": "Naive brake lock",
    "nominal_control": "Control C0 (no fault)",
}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAULTS = ("F1_policy_hang", "F2_dcbus_sag", "F3_knee_overheat")
ARMS = ("ref_no_containment", "naive_sto", "naive_brake_lock")
TIER_MARK = {"quiet": "o", "push": "s", "stress": "^"}


# --------------------------------------------------------------------------- PNG
def plot_comparison(rep: dict, runs: list, out_path: str, rep_label: str) -> str:
    """rep[(fault, arm)] = {"trace": ..., "onset": t}; runs = list of per-run result dicts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
                         "axes.titlecolor": TEXT, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), gridspec_kw={"height_ratios": [1.0, 1.0]})
    fig.suptitle("humanoid-fall-safety-bench v0.1 - naive containment vs. fault", fontsize=13,
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.935, f"Top: head height after fault onset ({rep_label}). "
             "Bottom: whole-robot kinetic energy at first non-foot ground contact, every falling run.",
             color=TEXT2, fontsize=9)

    for j, fault in enumerate(FAULTS):
        ax = axes[0, j]
        ctrl = rep.get(("C0_no_fault", "nominal_control"))
        series = [("nominal_control", ctrl)] + [(a, rep.get((fault, a))) for a in ARMS]
        for arm, item in series:
            if item is None:
                continue
            t = np.asarray(item["trace"]["t"]) - item["onset"]
            z = np.asarray(item["trace"]["head_z"])
            ls = (0, (3, 2)) if arm == "nominal_control" else "-"
            ax.plot(t, z, color=COLORS[arm], lw=2, ls=ls, label=SHORT[arm])
        ax.axvline(0.0, color=TEXT2, lw=0.8)
        ax.text(0.02, 0.04, "fault onset", transform=ax.get_xaxis_transform(), color=TEXT2, fontsize=7.5)
        ax.set_xlim(-0.5, 3.4)
        ax.set_ylim(0.0, 1.7)
        ax.set_title(fault, loc="left", fontsize=10)
        ax.set_xlabel("time since fault onset (s)")
        if j == 0:
            ax.set_ylabel("head height (m)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="center right")

    for j, fault in enumerate(FAULTS):
        ax = axes[1, j]
        for i, arm in enumerate(ARMS):
            rs = [r for r in runs if r["fault_id"] == fault and r["arm"] == arm]
            for tier, mk in TIER_MARK.items():
                ks = [r["ke_at_ground_contact_J"] for r in rs
                      if r["tier"] == tier and r["fell"] and r["ke_at_ground_contact_J"] is not None]
                if ks:
                    off = {"quiet": -0.22, "push": 0.0, "stress": 0.22}[tier]
                    jit = np.linspace(-0.06, 0.06, len(ks)) if len(ks) > 1 else np.zeros(1)
                    ax.scatter(i + off + jit, ks, s=34, marker=mk, color=COLORS[arm],
                               edgecolor=SURFACE, linewidth=1.0, zorder=3)
            nf = sum(1 for r in rs if r["fell"])
            ax.text(i, 1.0, f"falls: {nf}/{len(rs)} runs", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=8, color=TEXT2)
        ax.set_xticks(range(len(ARMS)), [SHORT[a] for a in ARMS], fontsize=8)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("kinetic energy at ground contact (J)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker=m, ls="", color=TEXT2, label=f"tier: {t}") for t, m in TIER_MARK.items()]
    axes[1, 0].legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- GIF
def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_gif(frames_npz: str, out_path: str, width: int = 256, height: int = 192, fps: int = 20) -> str:
    """Replay stored qpos frames (no re-simulation) into a labelled grid GIF."""
    import mujoco
    from PIL import Image, ImageDraw

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.robot import MODEL_PATH

    blob = np.load(frames_npz, allow_pickle=True)
    rows = list(blob["rows"])            # row labels
    cols = list(blob["cols"])            # column arm ids
    grid = blob["grid"].item()           # {(r, c): {"qpos": [T, nq], "t": [T], "onset": float, "verdict": str}}

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.65]
    cam.distance, cam.azimuth, cam.elevation = 3.6, 145.0, -12.0

    t_rel = np.arange(-0.4, 2.6, 1.0 / fps)
    head_h, left_w = 24, 0
    W, H = width * len(cols), height * len(rows) + head_h
    f_lbl, f_small = _font(13), _font(11)
    out = []
    for tr in t_rel:
        canvas = Image.new("RGB", (W, H), SURFACE)
        dr = ImageDraw.Draw(canvas)
        for c, arm in enumerate(cols):
            dr.text((c * width + 6, 5), SHORT.get(arm, arm), fill=COLORS.get(arm, TEXT), font=f_lbl)
        for r, row in enumerate(rows):
            for c, arm in enumerate(cols):
                item = grid.get((r, c))
                if item is None:
                    continue
                ts = np.asarray(item["t"]) - item["onset"]
                k = int(np.clip(np.searchsorted(ts, tr), 0, len(ts) - 1))
                data.qpos[:] = item["qpos"][k]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                tile = Image.fromarray(renderer.render())
                td = ImageDraw.Draw(tile)
                td.rectangle((0, 0, width, 20), fill=SURFACE)
                td.text((6, 4), row, fill=TEXT, font=f_small)
                td.text((6, height - 18), f"t - t_fault = {tr:+.2f} s", fill=TEXT2, font=f_small)
                if tr >= ts[-1] - 1e-6 or tr > 2.0:
                    v = item["verdict"]
                    td.text((width - 44, 3), v, fill="#c0392b" if v == "FAIL" else "#1d7f3e", font=f_lbl)
                canvas.paste(tile, (c * width + left_w, r * height + head_h))
        out.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
    renderer.close()
    out[0].save(out_path, save_all=True, append_images=out[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out_path


def _probe() -> None:
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 32, 32)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    img = r.render()
    r.close()
    assert img.shape == (32, 32, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="media helper (run by run_bench.py in a subprocess)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    g = sub.add_parser("gif")
    g.add_argument("--frames", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        _probe()
        print("render-probe-ok", os.environ.get("MUJOCO_GL", "<default>"))
        return 0
    _probe()
    render_gif(a.frames, a.out)
    print("gif-ok", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
