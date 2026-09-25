"""A THOUGHT: timeline, frame rendering, overlays and encoding.

Every flash is a spike of the stored simulation (results/spikes/*.npz, the
representative trial of each condition). Brain time t (ms) maps to film time
through a fixed slow-motion factor. Per sub-frame, each neuron's glow is

    a_i(t) = sum over its spikes s <= t of exp(-(t - s) / 8 ms)

(so a flash has decayed by ~95 % after 25 ms of brain time), computed directly
from the spikes of the last 80 ms, which makes every frame independent.

Usage:
  python -m render.film preview  [--start S --end E]    1280x720, 30 fps quick check
  python -m render.film master   [--start S --end E]    3840x2160, 60 fps (chunked, resumable)
  python -m render.film vertical                         1080x1920, 60 fps
  python -m render.film poster                           out/poster.png
  python -m render.film 1080p                            out/a_thought_1080p.mp4 from the master
  python -m render.film still T [--format master]        one frame as PNG
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, PREPARE_LOG, RESULTS, SPIKES, load_json, log, save_json  # noqa: E402

from .cpu import CPURenderer  # noqa: E402
from .fly import fly_rgba  # noqa: E402
from .gl import GLRenderer  # noqa: E402
from .overlay import TextBlock, composite, dot_rgba, text_rgba  # noqa: E402
from .pathway import PATHWAY_JSON, SKELETON_DIR  # noqa: E402
from .scene import LIN, PALETTE, BrainScene, CameraPose, orbit_pose  # noqa: E402

TAU_FLASH_MS = 8.0          # flash decay constant (brain time)
PULSE_FILM_S = 0.03         # visual aid: pulse travel time along an arc (film time)
MN9_WINDOW_MS = 100.0       # live readout window
FILM_NUMBERS = RESULTS / "film_numbers.json"
EXPERIMENTS = ("A_sugar", "B_bitter", "C_sugar_bitter")
EXP_COLOR = {"A_sugar": "sugar", "B_bitter": "bitter", "C_sugar_bitter": "mixed"}


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)  # smootherstep


def ramp(T, a, b):
    """0 before a, 1 after b, smooth in between."""
    return float(ease((T - a) / (b - a))) if b > a else float(T >= a)


def window(T, a, b, fade=0.8):
    """1 inside [a, b] with smooth fades of `fade` seconds at both ends."""
    return min(ramp(T, a, a + fade), 1.0 - ramp(T, b - fade, b))


def wrap(text, weight, size, max_w):
    """Greedy word wrap using the real glyph widths."""
    words, lines, cur = text.split(" "), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if cur and text_rgba(trial, weight, size).shape[1] > max_w:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    return lines + [cur]


# ============================================================================ data
class FilmData:
    """Everything the film shows, loaded from the pipeline outputs."""

    def __init__(self):
        self.scene = BrainScene()
        exp = load_json(RESULTS / "experiments.json")
        self.exp_meta = exp
        prep = load_json(PREPARE_LOG)
        self.n = self.scene.n
        nd = self.scene.neurons
        pos = {r: i for i, r in enumerate(nd.root_id)}
        self.sugar = np.array([pos[r] for r in exp["stimulus"]["sugar_root_ids"]])
        self.bitter = np.array([pos[r] for r in exp["stimulus"]["bitter_root_ids"]])
        self.scene.set_stimulus(self.sugar, self.bitter)
        self.mn9 = np.array(exp["mn9"]["indices"])
        self.mn9_sides = exp["mn9"]["sides"]
        self.trials = {}
        for key in EXPERIMENTS:
            c = exp["conditions"][key]
            d = np.load(SPIKES / f"{key}.npz")
            m = d["trial"] == c["representative_trial"]
            t = d["steps"][m] * float(d["dt"])
            nn = d["neurons"][m]
            o = np.argsort(t, kind="stable")
            self.trials[key] = dict(t=t[o].astype(np.float64), n=nn[o].astype(np.int64), meta=c)
        self.long_idx = {k: np.unique(v["n"]) for k, v in self.trials.items()}
        self._pathway()
        self._numbers(prep, exp)

    # -------------------------------------------------------------- pathway
    def _pathway(self):
        pw = load_json(PATHWAY_JSON)
        prep = load_json(PREPARE_LOG)["render_space"]
        center, sign, scale = np.array(prep["center_nm"]), np.array(prep["axis_sign"]), prep["scale_per_nm"]
        self.pw_members = np.array([r["index"] for r in pw["neurons"]])
        self.pw_level = {r["index"]: r["level"] for r in pw["neurons"]}
        segs, owner = [], []
        for r in pw["neurons"]:
            s = np.load(SKELETON_DIR / f"{r['root_id']}.npz")["segments_nm"].astype(np.float64)
            s = ((s - center) * sign * scale).astype(np.float32)
            segs.append(s)
            owner.append(np.full(len(s), r["index"]))
        self.skel = np.concatenate(segs)
        self.skel_owner = np.concatenate(owner)
        self.skel_color = self.scene.spike_color_film[self.skel_owner]
        # MN9's own skeleton spans the whole region and fires at 60-110 Hz; its point already flashes white
        self.skel_gain = np.where(np.isin(self.skel_owner, self.mn9), 0.025, 0.055).astype(np.float32)
        # arcs: quadratic Bezier between the two neurons' positions, bulging away from the brain centre
        P = self.scene.pos
        arcs, arc_pre = [], []
        for a in pw["arcs"]:
            p0, p1 = P[a["pre"]].astype(np.float64), P[a["post"]].astype(np.float64)
            mid = (p0 + p1) / 2
            out = mid - np.array([0.0, -0.2, 0.0])
            out /= np.linalg.norm(out) + 1e-9
            ctrl = mid + out * 0.25 * np.linalg.norm(p1 - p0) + np.array([0, 0.02, 0])
            u = np.linspace(0, 1, 33)[:, None]
            curve = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * ctrl + u ** 2 * p1
            arcs.append(curve)
            arc_pre.append(a["pre"])
        self.arc_curves = np.array(arcs)                        # (A, 33, 3)
        self.arc_pre = np.array(arc_pre)
        self.arc_segs = np.stack([self.arc_curves[:, :-1], self.arc_curves[:, 1:]], 2).reshape(-1, 2, 3).astype(np.float32)
        self.arc_seg_owner = np.repeat(self.arc_pre, 32)
        self.arc_sources = np.unique(self.arc_pre)
        self.sez_center = P[self.pw_members].mean(0)

    # -------------------------------------------------------------- numbers
    def _numbers(self, prep, exp):
        conds = exp["conditions"]
        nums = dict(
            neurons=prep["neurons"]["simulated_total"],
            connections=prep["connectivity"]["connections"],
            synapses=prep["connectivity"]["synapses"],
            sugar_n=exp["stimulus"]["sugar_n"], bitter_n=exp["stimulus"]["bitter_n"],
            rate_hz=exp["stimulus"]["rate_hz"],
            trials_per_condition=conds["A_sugar"]["n_trials"],
            mn9_sides=self.mn9_sides,
        )
        for key in EXPERIMENTS:
            c = conds[key]
            tr = self.trials[key]
            nums[key] = dict(
                representative_seed=c["representative_seed"],
                mn9_spikes_trial=[int((tr["n"] == m).sum()) for m in self.mn9],
                mn9_rate_mean_30=c["mn9_rate_hz_mean"],
                spikes_trial=int(len(tr["t"])),
                active_neurons_trial=int(len(np.unique(tr["n"]))),
            )
        nums["sources"] = dict(
            neurons="results/prepare_log.json neurons.simulated_total",
            connections="results/prepare_log.json connectivity.connections",
            mn9_spikes_trial="results/spikes/<condition>.npz, representative trial (experiments.json)",
            mn9_rate_mean_30="results/experiments.json conditions.<condition>.mn9_rate_hz_mean",
            spikes_trial="results/spikes/<condition>.npz, representative trial",
        )
        self.numbers = nums
        save_json(FILM_NUMBERS, nums)

    # -------------------------------------------------------------- per-time queries
    def activity(self, key, t_ms):
        tr = self.trials[key]
        lo = np.searchsorted(tr["t"], t_ms - 10 * TAU_FLASH_MS, "left")
        hi = np.searchsorted(tr["t"], t_ms, "right")
        a = np.zeros(self.n, np.float32)
        if hi > lo:
            np.add.at(a, tr["n"][lo:hi], np.exp(-(t_ms - tr["t"][lo:hi]) / TAU_FLASH_MS).astype(np.float32))
        return a

    def counts_until(self, key, t_ms):
        tr = self.trials[key]
        hi = np.searchsorted(tr["t"], t_ms, "right")
        return np.bincount(tr["n"][:hi], minlength=self.n)

    def spikes_between(self, key, t0_ms, t1_ms, neurons=None):
        tr = self.trials[key]
        lo, hi = np.searchsorted(tr["t"], t0_ms, "right"), np.searchsorted(tr["t"], t1_ms, "right")
        t, n = tr["t"][lo:hi], tr["n"][lo:hi]
        if neurons is not None:
            k = np.isin(n, neurons)
            t, n = t[k], n[k]
        return t, n

    def mn9_rate(self, key, t_ms, window_ms=MN9_WINDOW_MS):
        """Spikes of each MN9 in (t - window, t], expressed in Hz. Before t = 0 the model is silent."""
        t, n = self.spikes_between(key, t_ms - window_ms, t_ms, self.mn9)
        return [float((n == m).sum()) * 1000.0 / window_ms for m in self.mn9]


# ============================================================================ timeline
@dataclass
class Format:
    name: str
    size: tuple
    fps: int
    subframes: int
    slow: float                  # film seconds per brain second
    portrait: bool = False
    crf: int = 18
    preset: str = "medium"
    shutter: float = 0.5         # fraction of the frame interval covered by sub-frames
    exposure: float = 1.6
    extra: dict = field(default_factory=dict)


FORMATS = {
    "master": Format("master", (3840, 2160), 60, 4, 25.0, crf=20, preset="medium"),
    "preview": Format("preview", (1280, 720), 30, 2, 25.0, crf=22, preset="veryfast"),
    "vertical": Format("vertical", (1080, 1920), 60, 4, 10.0, portrait=True, crf=20, preset="medium"),
    "vertical_preview": Format("vertical_preview", (540, 960), 30, 2, 10.0, portrait=True, crf=22, preset="veryfast"),
}


@dataclass
class Segment:
    key: str        # experiment key
    t0: float       # film time the brain clock starts (0 ms)
    lead: float     # seconds of labels before t0
    hold: float     # seconds after the trial ends (1000 ms)
    reset: float    # seconds of reset at the start (labels switch, clock resets)


class Timeline:
    def __init__(self, fmt: Format, data: FilmData):
        self.f = fmt
        self.d = data
        s = fmt.slow
        if not fmt.portrait:
            self.darkness = (0.0, 6.0)
            self.reveal = (6.0, 20.0)
            self.glide = (22.0, 30.0)
            self.segments = [Segment("A_sugar", 32.0, 2.0, 8.0, 0.0),
                             Segment("B_bitter", 66.5, 0.0, 1.5, 1.5),
                             Segment("C_sugar_bitter", 94.5, 0.0, 1.5, 1.5)]
            self.long = (121.0, 141.0)
            self.accum = (124.0, 134.0)
            self.credits = (141.0, 156.0)
            self.captions = [
                (1.0, 5.3, "This is the wiring of a real brain."),
                (8.5, 20.5, "{neurons:,} neurons. {connections:,} connections. Mapped one by one."),
                (23.0, 29.8, "Now let's give it a taste of sugar."),
                (41.0, 55.5, "One neuron type, MN9, decides: eat."),
                (73.0, 88.5, "Bitter. The decision neurons stay silent."),
                (101.0, 116.5, "Mixed together, bitter holds sugar back."),
                (129.0, 140.3, "Every point of light was a spike in the simulation."),
            ]
        else:
            self.darkness = (0.0, 3.0)
            self.reveal = (3.0, 8.5)
            self.glide = (9.0, 12.0)
            self.segments = [Segment("A_sugar", 13.0, 1.0, 2.0, 0.0),
                             Segment("B_bitter", 25.5, 0.0, 1.5, 0.5),
                             Segment("C_sugar_bitter", 37.5, 0.0, 1.5, 0.5)]
            self.long = (49.0, 55.0)
            self.accum = (50.0, 54.0)
            self.credits = (55.0, 62.0)
            self.captions = [
                (0.4, 2.9, "This is the wiring of a real brain."),
                (3.4, 8.9, "{neurons:,} neurons.\n{connections:,} connections.\nMapped one by one."),
                (9.2, 12.4, "Now let's give it a taste of sugar."),
                (15.0, 22.9, "One neuron type, MN9,\ndecides: eat."),
                (27.4, 35.4, "Bitter. The decision\nneurons stay silent."),
                (39.4, 47.4, "Mixed together, bitter\nholds sugar back."),
                (50.6, 54.9, "Every point of light was\na spike in the simulation."),
            ]
        for seg in self.segments:
            seg.t1 = seg.t0 + s              # brain clock reaches 1000 ms
            seg.start = seg.t0 - seg.lead - seg.reset
            seg.end = seg.t1 + seg.hold
        self.duration = self.credits[1]
        self._camera_keys()

    # ------------------------------------------------------------ camera
    def _camera_keys(self):
        c = self.d.sez_center
        sez = (float(c[0]), float(c[1]) + 0.075, float(c[2]))
        mid = (0.0, -0.08, 0.0)
        if not self.f.portrait:
            k = [
                (0.0, dict(az=-34, el=12, dist=3.35, tgt=(0, 0.0, 0), fov=30)),
                (21.0, dict(az=16, el=15, dist=3.05, tgt=(0, 0.0, 0), fov=30)),
                (30.0, dict(az=-12, el=4, dist=1.55, tgt=sez, fov=30)),
                (65.0, dict(az=-4, el=5, dist=1.45, tgt=sez, fov=30)),
                (93.0, dict(az=4, el=6, dist=1.47, tgt=sez, fov=30)),
                (121.0, dict(az=10, el=6, dist=1.5, tgt=sez, fov=30)),
                (129.0, dict(az=-16, el=12, dist=2.45, tgt=mid, fov=30)),
                (141.0, dict(az=-8, el=12, dist=2.35, tgt=mid, fov=30)),
                (156.0, dict(az=-2, el=12, dist=3.1, tgt=(0, 0.0, 0), fov=30)),
            ]
        else:
            k = [
                (0.0, dict(az=-20, el=10, dist=4.3, tgt=(0, 0.0, 0), fov=30)),
                (8.5, dict(az=12, el=13, dist=4.1, tgt=(0, 0.0, 0), fov=30)),
                (12.0, dict(az=-10, el=4, dist=1.9, tgt=sez, fov=30)),
                (25.0, dict(az=-4, el=5, dist=1.8, tgt=sez, fov=30)),
                (37.0, dict(az=3, el=6, dist=1.82, tgt=sez, fov=30)),
                (49.0, dict(az=8, el=6, dist=1.85, tgt=sez, fov=30)),
                (52.0, dict(az=-14, el=12, dist=3.4, tgt=mid, fov=30)),
                (62.0, dict(az=-6, el=11, dist=4.1, tgt=(0, 0.0, 0), fov=30)),
            ]
        self.cam_t = np.array([t for t, _ in k])
        self.cam_v = np.array([[v["az"], v["el"], v["dist"], *v["tgt"], v["fov"]] for _, v in k], float)

    def camera(self, T) -> CameraPose:
        t, v = self.cam_t, self.cam_v
        i = int(np.clip(np.searchsorted(t, T, "right") - 1, 0, len(t) - 2))
        s = float(ease((T - t[i]) / (t[i + 1] - t[i])))
        x = v[i] + (v[i + 1] - v[i]) * s
        return orbit_pose(x[0], x[1], x[2], target=tuple(x[3:6]), fovy=x[6])

    # ------------------------------------------------------------ state
    def segment_at(self, T):
        for seg in self.segments:
            if seg.start <= T < seg.end:
                return seg
        return None

    def brain_ms(self, seg, T):
        return (T - seg.t0) * 1000.0 / self.f.slow


# ============================================================================ frames
class FilmRenderer:
    def __init__(self, fmt_name: str, backend: str | None = None):
        self.f = FORMATS[fmt_name]
        self.d = FilmData()
        self.tl = Timeline(self.f, self.d)
        W, H = self.f.size
        backend = backend or os.environ.get("A_THOUGHT_RENDERER", "gl")
        if backend == "gl":
            try:
                self.r = GLRenderer(W, H)
            except Exception as e:  # no OpenGL at all: slow NumPy fallback
                log(f"OpenGL unavailable ({e}); using the CPU fallback renderer")
                self.r = CPURenderer(W, H)
        else:
            self.r = CPURenderer(W, H)
        sc = self.d.scene
        self.r.set_shell(sc.shell_vertices, sc.shell_normals, sc.shell_faces)
        self.base_size = np.full(sc.n, 0.0024, np.float32)
        self.base_size[self.d.mn9] = 0.006
        self.u = H / 2160.0 if not self.f.portrait else W / 1080.0 * 0.9
        self._legend_cache = {}

    # ------------------------------------------------------------------ layers
    def _neuron_layer(self, T, act, vis, gain):
        d, sc = self.d, self.d.scene
        rest = LIN["resting"][None, :] * (0.16 * self.rest_level * vis)[:, None]
        col = rest + sc.spike_color_film * (act * 2.4)[:, None]
        size = self.base_size * self.size_scale * (1.0 + 1.6 * np.sqrt(np.minimum(act, 1.5)))
        col[~sc.drawn] = 0
        ref = self.f.size[1] if not self.f.portrait else self.f.size[0] * 1.2
        self.r.draw_points("neurons", sc.pos, col.astype(np.float32), size.astype(np.float32), self.cam, gain=gain,
                           min_px=max(1.1, 1.3 * ref / 1080))

    def _markers(self, T, act, gain, mn9_on, sugar_mark, bitter_mark):
        d = self.d
        pts, cols, sizes = [], [], []
        if mn9_on > 0:
            for m in d.mn9:
                pts.append(d.scene.pos[m])
                cols.append(LIN["mn9"] * (0.14 + 0.30 * min(act[m], 1.5)) * mn9_on)
                sizes.append(0.028)
        if pts:
            self.r.draw_points("mn9ring", np.array(pts, np.float32), np.array(cols, np.float32),
                               np.array(sizes, np.float32), self.cam, gain=gain, ring=True)
        # steady soft glow marking where the taste input enters (a label, not activity)
        for idx, key, amt in ((d.sugar, "sugar", sugar_mark), (d.bitter, "bitter", bitter_mark)):
            if amt > 0:
                self.r.draw_points(f"mark_{key}", d.scene.pos[idx], np.repeat(LIN[key][None] * 0.55 * amt, len(idx), 0).astype(np.float32),
                                   np.full(len(idx), 0.012, np.float32), self.cam, gain=gain)

    def _pathway_layer(self, key, t_ms, act, gain, amount, pulses=True):
        d = self.d
        if amount <= 0:
            return
        a = np.minimum(act[d.skel_owner], 1.5)[:, None]
        col = (LIN["resting"][None] * 0.006 + d.skel_color * d.skel_gain[:, None] * a) * amount
        self.r.draw_lines("skel", d.skel, col.astype(np.float32).repeat(2, 0), self.cam,
                          width_px=max(1.0, 2.0 * self.u), gain=gain)
        arc_col = np.repeat(LIN["resting"][None] * 0.035 * amount, len(d.arc_segs), 0).astype(np.float32)
        self.r.draw_lines("arcs", d.arc_segs, arc_col.repeat(2, 0), self.cam, width_px=max(1.0, 1.6 * self.u), gain=gain)
        # pulses: each spike of a pathway neuron sends a light pulse along its outgoing arcs
        # (a visual aid: the model itself has a fixed 1.8 ms delay and no travel along wires)
        dt_brain = PULSE_FILM_S * 1000.0 / self.f.slow
        if not pulses:          # only while the brain clock runs: no frozen pulses before or after the trial
            return
        t, n = d.spikes_between(key, t_ms - dt_brain, t_ms, d.arc_sources)
        if len(t):
            pos, colp = [], []
            for ts, pre in zip(t, n):
                u = (t_ms - ts) / dt_brain
                for ai in np.flatnonzero(d.arc_pre == pre):
                    curve = d.arc_curves[ai]
                    x = u * (len(curve) - 1)
                    i0 = int(min(x, len(curve) - 2))
                    p = curve[i0] + (curve[i0 + 1] - curve[i0]) * (x - i0)
                    pos.append(p)
                    colp.append(d.scene.spike_color_film[pre] * 2.4 * amount)
            self.r.draw_points("pulses", np.array(pos, np.float32), np.array(colp, np.float32),
                               np.full(len(pos), 0.009, np.float32), self.cam, gain=gain)

    # ------------------------------------------------------------------ one frame
    def frame(self, fi: int, overlays: bool = True) -> np.ndarray:
        f, tl, d = self.f, self.tl, self.d
        T_frame = fi / f.fps
        K = f.subframes
        W, H = f.size
        self.r.begin()
        # shell once per frame (slow camera; no need for motion blur)
        T_mid = T_frame + 0.5 * f.shutter / f.fps
        self.cam = tl.camera(T_mid).matrices(W, H, shift=(0.0, 0.34) if f.portrait else (0.0, 0.0))
        shell_amt = ramp(T_mid, tl.reveal[0] - 1.0, tl.reveal[0] + 6.0)
        if T_mid > tl.credits[0]:
            shell_amt *= 1.0 - 0.65 * ramp(T_mid, tl.credits[0], tl.credits[0] + 3.0)
        self.r.draw_shell(self.cam, LIN["shell"], opacity=0.017 * shell_amt, power=4.0)
        state = None
        for k in range(K):
            T = T_frame + (k + 0.5) / K * f.shutter / f.fps
            self.cam = tl.camera(T).matrices(W, H, shift=(0.0, 0.34) if f.portrait else (0.0, 0.0))
            state = self._draw_subframe(T, 1.0 / K)
        fade = 1.0 - ramp(T_frame, tl.duration - 1.6, tl.duration - 0.2)
        fade *= ramp(T_frame, 0.0, 0.01)
        img = self.r.finish(exposure=f.exposure, bloom_strength=0.9, bloom_radius=1.0, grain=0.008,
                            seed=fi * 0.618, fade=fade)
        img = np.ascontiguousarray(img)
        if overlays:
            self._overlays(img, T_frame, state)
        return img

    def _draw_subframe(self, T, gain):
        tl, d, sc = self.tl, self.d, self.d.scene
        dist = float(np.linalg.norm(self.cam["eye"] - tl.camera(T).target))
        self.size_scale = float(np.clip(dist / 3.3, 0.3, 1.1))
        # during the long exposure the resting neurons fade almost out, so the points of light are the spikes
        self.rest_level = 1.0 - 0.88 * ramp(T, tl.accum[0] - 1.0, tl.accum[0] + 2.0)
        p = (T - tl.reveal[0]) / (tl.reveal[1] - tl.reveal[0])
        vis = np.clip((p - sc.reveal_at) / 0.08, 0.0, 1.0).astype(np.float32)
        if T > tl.credits[0]:
            vis = vis * (1.0 - 0.55 * ramp(T, tl.credits[0], tl.credits[0] + 3.0))
        seg = tl.segment_at(T)
        act = np.zeros(sc.n, np.float32)
        state = dict(seg=seg, T=T)
        sugar_mark = ramp(T, tl.glide[0] + 1.0, tl.glide[1] - 1.5) * (1 - ramp(T, tl.segments[0].t0 - 0.5, tl.segments[0].t0 + 1.5))
        bitter_mark = 0.0
        mn9_on = ramp(T, tl.segments[0].start, tl.segments[0].t0) * (1 - ramp(T, tl.long[0], tl.long[0] + 2.0))
        pathway_amt = 0.0
        if seg is not None:
            t_ms = tl.brain_ms(seg, T)
            state["t_ms"] = t_ms
            if 0 <= t_ms:
                t_eff = min(t_ms, 1000.0)
                act = d.activity(seg.key, t_eff)
                if t_ms > 1000.0:  # trial over: remaining glow fades out, no new spikes exist
                    act *= np.float32(1.0 - ramp(T, seg.t1, seg.t1 + 1.0))
            fade_in = ramp(T, seg.start, seg.start + max(seg.reset, 0.5)) if seg.reset > 0 else 1.0
            act *= np.float32(fade_in)
            pathway_amt = ramp(T, seg.start, seg.t0) * (1 - ramp(T, seg.end - 0.3, seg.end)) if seg.key != "A_sugar" else \
                ramp(T, seg.t0 - 1.0, seg.t0 + 2.0) * (1 - ramp(T, seg.end - 0.3, seg.end))
            if seg.key == "B_bitter":
                bitter_mark = 0.0
            state["pathway"] = pathway_amt
            self._pathway_layer(seg.key, min(max(t_ms, 0.0), 1000.0), act, gain, pathway_amt,
                                pulses=0.0 <= t_ms <= 1000.0)
        # long exposure: all spikes of the three experiments accumulate, coloured by experiment
        if tl.long[0] <= T:
            acc = np.clip((T - tl.accum[0]) / (tl.accum[1] - tl.accum[0]), 0, 1)
            t_acc = acc * 1000.0
            state["long"] = acc
            lvl = ramp(T, tl.accum[0] - 0.5, tl.accum[0] + 0.5)
            if T > tl.credits[0]:
                lvl *= 1.0 - 0.6 * ramp(T, tl.credits[0], tl.credits[0] + 3.0)
            for key in EXPERIMENTS:
                idx = d.long_idx[key]
                c = d.counts_until(key, t_acc)[idx].astype(np.float32)
                flash = d.activity(key, t_acc)[idx] if acc < 1 else np.zeros(len(idx), np.float32)
                glow = (np.sqrt(c) * 0.30 + flash * 1.6) * lvl
                self.r.draw_points(f"long_{key}", sc.pos[idx], (LIN[EXP_COLOR[key]][None] * glow[:, None]).astype(np.float32),
                                   (0.0042 * self.size_scale * (1 + 0.6 * np.sqrt(np.minimum(flash, 1.5)))).astype(np.float32),
                                   self.cam, gain=gain)
        self._neuron_layer(T, act, vis, gain)
        self._markers(T, act, gain, mn9_on, sugar_mark, bitter_mark)
        return state

    # ------------------------------------------------------------------ overlays
    def _overlays(self, img, T, state):
        f, tl, d, u = self.f, self.tl, self.d, self.u
        W, H = f.size
        nums = d.numbers
        white = (236, 239, 246)
        margin_x = int(0.06 * W) if not f.portrait else int(0.075 * W)
        # captions
        for a, b, text in tl.captions:
            al = window(T, a, b, 0.9)
            if al <= 0:
                continue
            lines = text.format(**nums).split("\n")
            size = int(round((80 if not f.portrait else 54) * u))
            lh = int(size * 1.3)
            y0 = int(H - (0.085 * H if not f.portrait else 0.075 * H) - lh * len(lines))
            for i, line in enumerate(lines):
                composite(img, text_rgba(line, "light", size, white), margin_x, y0 + i * lh, 0.92 * al)
        seg = state.get("seg") if state else None
        if seg is not None:
            self._experiment_overlays(img, T, seg, state)
        if tl.long[0] <= T < tl.credits[0] + 1.0:
            self._long_legend(img, T)
        if T >= tl.credits[0]:
            self._credits(img, T)

    def _experiment_overlays(self, img, T, seg, state):
        f, tl, d, u = self.f, self.tl, self.d, self.u
        W, H = f.size
        P = f.portrait
        nums = d.numbers
        white = (236, 239, 246)
        al = window(T, seg.start, seg.end, 0.6)
        if al <= 0:
            return
        mx = int(0.06 * W) if not P else int(0.075 * W)
        my = int(0.07 * H) if not P else int(0.035 * H)
        rx = W - mx
        title = {"A_sugar": "EXPERIMENT A  ·  SUGAR", "B_bitter": "EXPERIMENT B  ·  BITTER",
                 "C_sugar_bitter": "EXPERIMENT C  ·  SUGAR + BITTER"}[seg.key]
        stim = {"A_sugar": f"{nums['sugar_n']} sugar-sensing taste neurons stimulated at {nums['rate_hz']:.0f} Hz",
                "B_bitter": f"{nums['bitter_n']} bitter-sensing taste neurons stimulated at {nums['rate_hz']:.0f} Hz",
                "C_sugar_bitter": f"{nums['sugar_n']} sugar + {nums['bitter_n']} bitter taste neurons, each at {nums['rate_hz']:.0f} Hz"}[seg.key]
        s1, s2, s3 = int(34 * u), int(40 * u), int(32 * u)
        y = my
        composite(img, text_rgba(title, "medium", s1, white, tracking=0.16), mx, y, 0.85 * al)
        y += int(1.9 * s1)
        for line in wrap(stim, "regular", s2, (W - 2 * mx) if P else int(0.55 * W)):
            composite(img, text_rgba(line, "regular", s2, white), mx, y, 0.75 * al)
            y += int(1.4 * s2)
        composite(img, text_rgba("Simulation on the FlyWire connectome, release 783", "regular", s3, white), mx, y, 0.5 * al)
        y += int(2.1 * s3)
        items = [("excitatory", "excitatory"), ("inhibitory", "inhibitory"), ("other", "other transmitter")]
        if seg.key in ("A_sugar", "C_sugar_bitter"):
            items.append(("sugar", "sugar input"))
        if seg.key in ("B_bitter", "C_sugar_bitter"):
            items.append(("bitter", "bitter input"))
        x, ls = mx, int(30 * u)
        for ck, label in items:
            dot = dot_rgba(max(2, int(8 * u)), hex_rgb(PALETTE[ck]))
            t = text_rgba(label, "regular", ls, white)
            if P and x + dot.shape[1] + t.shape[1] > W - mx:
                x, y = mx, y + int(ls * 1.6)
            composite(img, dot, x, y + int(ls * 0.45) - dot.shape[0] // 2 + 2, 0.9 * al)
            x += dot.shape[1] + int(8 * u)
            composite(img, t, x, y, 0.6 * al)
            x += t.shape[1] + int(26 * u)
        # brain clock: top right (landscape) or the lower third (portrait)
        t_ms = state.get("t_ms", -1)
        clock = "0" if t_ms < 0 else f"{min(t_ms, 1000.0):.0f}"
        big = text_rgba(f"{clock} ms", "light", int(88 * u), white)
        sub = text_rgba(f"brain time  ·  slowed {self.f.slow:.0f}×", "regular", int(30 * u), white)
        cy = my - int(12 * u) if not P else int(0.635 * H)
        composite(img, big, rx - big.shape[1], cy, 0.9 * al)
        composite(img, sub, rx - sub.shape[1], cy + big.shape[0], 0.55 * al)
        # MN9 readout
        if t_ms <= 1000.0:
            r = d.mn9_rate(seg.key, max(min(t_ms, 1000.0), 0.0)) if t_ms >= 0 else [0.0, 0.0]
            label = "MN9 FIRING RATE  (LAST 100 MS)"
            vals = "   ".join(f"{s} {v:3.0f} Hz" for s, v in zip(d.mn9_sides, r))
        else:
            cnt = nums[seg.key]["mn9_spikes_trial"]
            label = "MN9 OVER THE WHOLE 1 S TRIAL"
            vals = "   ".join(f"{s} {c} spikes" for s, c in zip(d.mn9_sides, cnt))
        v_img = text_rgba(vals, "regular", int(54 * u), white)
        l_img = text_rgba(label, "medium", int(27 * u), white, tracking=0.14)
        if not P:
            by = int(H - 0.085 * H)                                   # baseline of the values
        else:
            by = cy + big.shape[0] + sub.shape[0] + int(40 * u) + l_img.shape[0] + int(6 * u) + v_img.shape[0]
        composite(img, v_img, rx - v_img.shape[1], by - v_img.shape[0], 0.9 * al)
        composite(img, l_img, rx - l_img.shape[1], by - v_img.shape[0] - l_img.shape[0] - int(6 * u), 0.6 * al)
        # comparison at the end of experiment C (MN9 spikes of the two film trials)
        if seg.key == "C_sugar_bitter" and T > seg.t1 - 3.0:
            ca = ramp(T, seg.t1 - 3.0, seg.t1 - 2.0) * al
            a_n, c_n = nums["A_sugar"]["mn9_spikes_trial"], nums["C_sugar_bitter"]["mn9_spikes_trial"]
            if not P:
                lines = ["MN9, 1 s trials:  sugar " + " / ".join(str(x) for x in a_n) + " spikes  →  sugar + bitter "
                         + " / ".join(str(x) for x in c_n) + " spikes  (left / right)"]
            else:
                lines = ["MN9 spikes, 1 s trials (left / right):",
                         "sugar " + " / ".join(str(x) for x in a_n) + "  →  sugar + bitter " + " / ".join(str(x) for x in c_n)]
            yy = by + int(14 * u)
            for line in lines:
                ci = text_rgba(line, "regular", int(34 * u), white)
                composite(img, ci, rx - ci.shape[1], yy, 0.8 * ca)
                yy += int(34 * u * 1.45)
        # fly inset: an illustration driven by the simulated MN9 rate
        if t_ms >= -500:
            rr = d.mn9_rate(seg.key, max(min(t_ms, 1000.0), 0.0), window_ms=50.0) if t_ms >= 0 else [0.0, 0.0]
            if t_ms > 1000.0:
                rr = [x * (1.0 - ramp(T, seg.t1, seg.t1 + 1.0)) for x in rr]
            ext = float(np.clip(np.mean(rr) / 60.0, 0.0, 1.0))
            size = int(380 * u) if not P else int(330 * u)
            fly = fly_rgba(size, int(round(ext * 100)))
            if not P:
                fx, fy = rx - size, by - v_img.shape[0] - l_img.shape[0] - int(30 * u) - size
            else:
                fx, fy = mx - int(20 * u), int(0.625 * H)
            composite(img, fly, fx, fy, 0.75 * al)
            cap = text_rgba("illustration", "regular", int(26 * u), white)
            composite(img, cap, fx + size - cap.shape[1], fy + size - int(10 * u), 0.45 * al)

    def _long_legend(self, img, T):
        f, u, tl = self.f, self.u, self.tl
        W, H = f.size
        al = window(T, tl.accum[0], tl.credits[0] + 1.0, 0.8)
        mx = int(0.06 * W) if not f.portrait else int(0.075 * W)
        my = int(0.07 * H) if not f.portrait else int(0.05 * H)
        white = (236, 239, 246)
        composite(img, text_rgba("LONG EXPOSURE  ·  ALL SPIKES OF THE THREE TRIALS", "medium", int(32 * u), white, tracking=0.14), mx, my, 0.8 * al)
        x, y = mx, my + int(66 * u)
        for ck, label in (("sugar", "sugar"), ("bitter", "bitter"), ("mixed", "sugar + bitter")):
            dot = dot_rgba(max(2, int(9 * u)), hex_rgb(PALETTE[ck]))
            composite(img, dot, x, y + int(6 * u), 0.9 * al)
            x += dot.shape[1] + int(8 * u)
            t = text_rgba(label, "regular", int(32 * u), white)
            composite(img, t, x, y, 0.7 * al)
            x += t.shape[1] + int(30 * u)

    def _credits(self, img, T):
        f, u, tl = self.f, self.u, self.tl
        W, H = f.size
        white = (236, 239, 246)
        c0, c1 = tl.credits
        mx = int(0.06 * W) if not f.portrait else int(0.075 * W)
        lines = CREDITS_PORTRAIT if f.portrait else CREDITS
        fin_dur = 5.2 if not f.portrait else 3.0               # seconds for the final line
        al = window(T, c0 + 0.6, c1 - fin_dur - 0.2, 0.8)
        size = int(34 * u)
        y = int(0.16 * H) if not f.portrait else int(0.64 * H)
        for weight, text in lines:
            if text == "":
                y += int(size * 0.8)
                continue
            s = int(size * (1.25 if weight == "medium" else 1.0))
            t = text_rgba(text, weight, s, white, tracking=0.12 if weight == "medium" else 0.0)
            composite(img, t, mx, y, (0.85 if weight == "medium" else 0.7) * al)
            y += int(s * 1.55)
        fin = window(T, c1 - fin_dur, c1 - 0.25, 1.0 if f.portrait else 1.2)
        final = text_rgba("A simulation built on the real wiring of a fruit fly's brain.", "light",
                          int((52 if not f.portrait else 40) * u), white)
        if f.portrait:
            final_lines = ["A simulation built on the real", "wiring of a fruit fly's brain."]
            yy = int(H * 0.70)
            for i, ln in enumerate(final_lines):
                t = text_rgba(ln, "light", int(46 * u), white)
                composite(img, t, (W - t.shape[1]) // 2, yy + i * int(62 * u), 0.95 * fin)
        else:
            composite(img, final, (W - final.shape[1]) // 2, (H - final.shape[0]) // 2, 0.95 * fin)


CREDITS = [
    ("medium", "DATA"),
    ("regular", "FlyWire Consortium · Dorkenwald et al. 2024, Nature: Neuronal wiring diagram of an adult brain"),
    ("regular", "Schlegel et al. 2024, Nature: Whole-brain annotation and multi-connectome cell typing of Drosophila"),
    ("regular", "Eckstein et al. 2024, Cell: neurotransmitter predictions · Berg et al. 2025 and Matsliah et al. 2024: annotations"),
    ("", ""),
    ("medium", "MODEL"),
    ("regular", "Shiu et al. 2024, Nature: A Drosophila computational brain model reveals sensorimotor processing"),
    ("regular", "Model code and v783 connectivity: github.com/philshiu/Drosophila_brain_model (MIT)"),
    ("", ""),
    ("medium", "ALSO USED"),
    ("regular", "flypoke (github.com/vshapenko/flypoke) for cross-checks · navis-flybrains FlyWire brain mesh"),
    ("regular", "FlyWire release-783 skeletons (gs://flywire_v141_m783) · Brian2 · Inter typeface"),
    ("", ""),
    ("regular", "Pulses along arcs and the fly drawing are visual aids. Details: NOTES.md"),
]
CREDITS_PORTRAIT = [
    ("medium", "DATA"),
    ("regular", "FlyWire Consortium · Dorkenwald et al. 2024"),
    ("regular", "Schlegel et al. 2024 · Eckstein et al. 2024"),
    ("", ""),
    ("medium", "MODEL"),
    ("regular", "Shiu et al. 2024, Nature"),
    ("", ""),
    ("regular", "Pulses and fly drawing: visual aids"),
]


# ============================================================================ encoding
def ffmpeg_writer(path: Path, fmt: Format):
    W, H = fmt.size
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fmt.fps), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", fmt.preset, "-crf", str(fmt.crf), "-profile:v", "high",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-g", str(fmt.fps * 2), "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_range(fr: FilmRenderer, f0: int, f1: int, path: Path):
    proc = ffmpeg_writer(path, fr.f)
    t0 = time.time()
    for fi in range(f0, f1):
        img = fr.frame(fi)
        proc.stdin.write(img.tobytes())
        if (fi - f0) % 60 == 0:
            el = time.time() - t0
            log(f"  frame {fi}/{f1} ({fi / fr.f.fps:6.2f} s)  {el / max(fi - f0, 1):.2f} s/frame")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def render_film(fmt_name: str, start: float | None = None, end: float | None = None, chunk_s: float = 8.0,
                out: Path | None = None):
    from sim.validate import require_validated

    require_validated()
    fr = FilmRenderer(fmt_name)
    fps = fr.f.fps
    total = int(round(fr.tl.duration * fps))
    f_start = 0 if start is None else int(round(start * fps))
    f_end = total if end is None else min(total, int(round(end * fps)))
    chunk = int(chunk_s * fps)
    cdir = OUT / "chunks" / fmt_name
    cdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for c0 in range(f_start, f_end, chunk):
        c1 = min(c0 + chunk, f_end)
        p = cdir / f"{c0:06d}_{c1:06d}.mp4"
        parts.append(p)
        if p.exists():
            continue
        log(f"{fmt_name}: frames {c0}-{c1}")
        tmp = p.with_suffix(".part.mp4")
        render_range(fr, c0, c1, tmp)
        tmp.rename(p)
    if out is None:
        out = OUT / {"master": "a_thought.mp4", "vertical": "a_thought_vertical.mp4",
                     "preview": "preview.mp4", "vertical_preview": "preview_vertical.mp4"}[fmt_name]
    lst = cdir / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    log(f"wrote {out}")
    return out


def downscale_1080p(src: Path = OUT / "a_thought.mp4", dst: Path = OUT / "a_thought_1080p.mp4", mbit: float = 4.6):
    """1080p version of the master: Lanczos downscale, two-pass H.264 at ~4.6 Mb/s so the file
    stays under GitHub's 100 MB limit (about 90 MB for 2:36)."""
    common = ["-vf", "scale=1920:1080:flags=lanczos+accurate_rnd:in_color_matrix=bt709:out_color_matrix=bt709",
              "-c:v", "libx264", "-preset", "slow", "-b:v", f"{mbit}M", "-maxrate", f"{2 * mbit}M", "-bufsize", f"{4 * mbit}M",
              "-profile:v", "high", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
              "-color_trc", "bt709", "-color_range", "tv", "-g", "120"]
    log_prefix = str(OUT / "x264_2pass")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "1", "-passlogfile", log_prefix,
                    "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *common, "-pass", "2", "-passlogfile", log_prefix,
                    "-movflags", "+faststart", str(dst)], check=True)
    log(f"wrote {dst}")
    return dst


def still(T: float, fmt_name: str = "preview", path: Path | None = None):
    fr = FilmRenderer(fmt_name)
    img = fr.frame(int(round(T * fr.f.fps)))
    path = path or OUT / f"still_{fmt_name}_{T:07.2f}.png"
    Image.fromarray(img).save(path)
    log(f"wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preview", "master", "vertical", "vertical_preview", "still", "poster", "1080p"])
    ap.add_argument("time", nargs="?", type=float)
    ap.add_argument("--format", default="preview")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.mode == "still":
        still(a.time, a.format, a.out)
    elif a.mode == "poster":
        from .poster import make_poster

        make_poster()
    elif a.mode == "1080p":
        downscale_1080p()
    else:
        render_film(a.mode, a.start, a.end, out=a.out)
