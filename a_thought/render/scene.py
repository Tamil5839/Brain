"""Scene content: camera, colours, neuron layers, the brain shell.

Render space (see prepare.py): +X = fly's left, +Y = dorsal, +Z = anterior;
the brain is centred and its largest half-extent is 1.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GROUPS_JSON, NEURONS_PARQUET, SHELL_NPZ, load_json  # noqa: E402


def srgb_hex(h: str) -> np.ndarray:
    """'#RRGGBB' -> linear-light RGB (additive blending happens in linear light)."""
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


PALETTE = dict(
    background="#030407",
    excitatory="#FFB45C",   # acetylcholine (and every neuron the model treats as excitatory)
    inhibitory="#5CD6FF",   # GABA / glutamate
    other="#F2F2FF",        # dopamine, serotonin, octopamine: soft white
    sugar="#7CFF8A",
    bitter="#B07CFF",
    mn9="#FFFFFF",
    resting="#8FA6D6",      # dim cool grey-blue
    shell="#9DB8FF",
    mixed="#FF8FB8",        # long-exposure colour for the sugar + bitter experiment
)
LIN = {k: srgb_hex(v) for k, v in PALETTE.items()}


# --------------------------------------------------------------------------- camera
def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy_deg: float, aspect: float, near: float = 0.01, far: float = 50.0) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


@dataclass
class CameraPose:
    eye: np.ndarray
    target: np.ndarray
    fovy: float = 30.0
    roll: float = 0.0  # degrees

    def matrices(self, width: int, height: int, shift=(0.0, 0.0)) -> dict:
        """shift: offset of the image centre in NDC (to place the brain off-centre)."""
        up = np.array([0.0, 1.0, 0.0])
        if self.roll:
            fwd = self.target - self.eye
            fwd /= np.linalg.norm(fwd)
            a = np.radians(self.roll)
            up = up * np.cos(a) + np.cross(fwd, up) * np.sin(a) + fwd * np.dot(fwd, up) * (1 - np.cos(a))
        view = look_at(self.eye, self.target, up)
        proj = perspective(self.fovy, width / height)
        if shift != (0.0, 0.0):
            t = np.eye(4)
            t[0, 3], t[1, 3] = shift
            proj = t @ proj
        # pixels per (world unit / clip w): half the viewport height times proj[1,1]
        px = 0.5 * height * proj[1, 1]
        return dict(view=view, proj=proj, mvp=proj @ view, eye=np.asarray(self.eye, float), px_per_unit=px)


def orbit_pose(azimuth_deg: float, elevation_deg: float, distance: float, target=(0.0, 0.0, 0.0),
               fovy: float = 30.0) -> CameraPose:
    """Azimuth 0 = in front of the fly (+Z), positive = towards the fly's left (+X)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    t = np.asarray(target, float)
    eye = t + distance * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    return CameraPose(eye=eye, target=t, fovy=fovy)


def lerp_pose(a: CameraPose, b: CameraPose, s: float) -> CameraPose:
    return CameraPose(eye=a.eye + (b.eye - a.eye) * s, target=a.target + (b.target - a.target) * s,
                      fovy=a.fovy + (b.fovy - a.fovy) * s, roll=a.roll + (b.roll - a.roll) * s)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------- content
class BrainScene:
    """Static per-neuron attributes used by every frame."""

    def __init__(self):
        nd = pd.read_parquet(NEURONS_PARQUET)
        self.neurons = nd
        self.n = len(nd)
        self.pos = nd[["rx", "ry", "rz"]].to_numpy(np.float32)
        self.drawn = ~np.isnan(self.pos).any(1)
        self.pos = np.nan_to_num(self.pos)
        self.groups = load_json(GROUPS_JSON)
        cls = nd.color_class.to_numpy()
        self.spike_color = np.where(
            (cls == "inhibitory")[:, None], LIN["inhibitory"],
            np.where((cls == "other")[:, None], LIN["other"], LIN["excitatory"])).astype(np.float32)
        self.region = nd.region.to_numpy()
        self.mn9 = np.array(self.groups["mn9"]["indices"])
        shell = np.load(SHELL_NPZ)
        self.shell_vertices, self.shell_faces, self.shell_normals = shell["vertices"], shell["faces"], shell["normals"]
        # Fade-in order for the opening: region by region, each as a soft wave
        # travelling outward from the region's own centre.
        order = {"central": 0.0, "optic_right": 1.0, "optic_left": 1.0, "sensory": 2.0, "other": 2.0}
        delay = np.array([order[r] for r in self.region])
        wave = np.zeros(self.n)
        for r in np.unique(self.region):
            m = self.region == r
            c = np.median(self.pos[m], 0)
            d = np.linalg.norm(self.pos[m] - c, axis=1)
            wave[m] = d / (np.percentile(d, 98) + 1e-9)
        rng = np.random.default_rng(7)
        self.reveal_at = (delay + 0.85 * np.clip(wave, 0, 1) + 0.15 * rng.random(self.n)) / 3.0  # in [0, 1]

    def set_stimulus(self, sugar: np.ndarray, bitter: np.ndarray) -> None:
        self.sugar, self.bitter = np.asarray(sugar), np.asarray(bitter)
        self.spike_color_film = self.spike_color.copy()
        self.spike_color_film[self.sugar] = LIN["sugar"]
        self.spike_color_film[self.bitter] = LIN["bitter"]
        self.spike_color_film[self.mn9] = LIN["mn9"]

    def region_center(self, mask) -> np.ndarray:
        return self.pos[mask].mean(0)
