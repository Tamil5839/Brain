"""Line-drawing inset of a fly in side view (an illustration, labelled as such).

The proboscis extension is driven by the simulated MN9 firing rate (see
film.py); the model itself has no muscles or behaviour. Disclosed in NOTES.md.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

SS = 4  # supersampling


def _rot(v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _ellipse_pts(cx, cy, rx, ry, tilt=0.0, n=90):
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    if tilt:
        pts = np.array([_rot(p, tilt) for p in pts])
    return pts + [cx, cy]


def _bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _body_parts():
    """Closed shapes drawn back to front (each erases the lines behind it), then open strokes."""
    abdomen = _ellipse_pts(0.665, 0.45, 0.15, 0.08, tilt=10)
    wing = np.vstack([_bezier(np.array([0.47, 0.33]), np.array([0.58, 0.225]), np.array([0.84, 0.195]), np.array([0.95, 0.27])),
                      _bezier(np.array([0.95, 0.27]), np.array([0.90, 0.335]), np.array([0.66, 0.36]), np.array([0.49, 0.345]))])
    thorax = _ellipse_pts(0.455, 0.405, 0.1, 0.088)
    head = _ellipse_pts(0.29, 0.41, 0.074, 0.08)
    eye = _ellipse_pts(0.283, 0.398, 0.052, 0.062, tilt=-12)
    closed = [abdomen, wing, thorax, head, eye]
    strokes = []
    for i, x in enumerate((0.605, 0.665, 0.725)):                         # abdominal stripes
        y0 = 0.385 + i * 0.012
        strokes.append(_bezier(np.array([x, y0]), np.array([x + 0.014, y0 + 0.045]),
                               np.array([x + 0.012, y0 + 0.095]), np.array([x - 0.002, y0 + 0.13]), 16))
    strokes.append(_bezier(np.array([0.52, 0.305]), np.array([0.65, 0.262]), np.array([0.78, 0.25]), np.array([0.885, 0.265]), 20))
    strokes.append(np.array([[0.228, 0.36], [0.196, 0.335], [0.19, 0.372]]))   # antenna
    strokes.append(np.array([[0.196, 0.335], [0.16, 0.312]]))                  # arista
    for leg in ([[0.405, 0.482], [0.352, 0.585], [0.33, 0.705], [0.292, 0.745]],
                [[0.452, 0.492], [0.49, 0.608], [0.468, 0.735], [0.43, 0.77]],
                [[0.505, 0.482], [0.585, 0.585], [0.605, 0.72], [0.648, 0.762]]):
        strokes.append(np.array(leg))
    return closed, strokes


def _proboscis_paths(e: float):
    """e = 0 retracted (short stub under the head) .. 1 fully extended."""
    base = np.array([0.268, 0.482])
    down = np.array([0.0, 1.0])
    rdir = _rot(down, 25 - 45 * e)              # rostrum swings from back-down to forward-down
    j = base + rdir * (0.03 + 0.07 * e)
    hdir = _rot(down, -115 + 150 * e)           # haustellum unfolds
    tip = j + hdir * (0.025 + 0.085 * e)
    lab_r = 0.012 + 0.016 * e                   # labellum lobes open
    lab = _ellipse_pts(*(tip + hdir * lab_r * 0.6), lab_r * 1.15, lab_r * 0.75, tilt=np.degrees(np.arctan2(hdir[1], hdir[0])))
    return [np.array([base, j, tip]), lab]


@lru_cache(maxsize=128)
def fly_rgba(size_px: int, extension_q: int, color=(232, 236, 246), line=0.0065) -> np.ndarray:
    e = extension_q / 100.0
    S = size_px * SS
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    w = max(1, int(round(line * S)))
    closed, strokes = _body_parts()
    for p in closed:
        d.polygon([tuple(x) for x in (p * S)], fill=0, outline=255, width=w)
    for p in strokes + _proboscis_paths(e):
        d.line([tuple(x) for x in (p * S)], fill=255, width=w, joint="curve")
    img = img.resize((size_px, size_px), Image.LANCZOS)
    a = np.asarray(img, np.float32) / 255.0
    out = np.zeros((size_px, size_px, 4), np.float32)
    out[..., :3] = a[..., None] * (np.array(color, np.float32) / 255.0)
    out[..., 3] = a
    out.flags.writeable = False
    return out


if __name__ == "__main__":
    tiles = [fly_rgba(360, q) for q in (0, 50, 100)]
    arr = np.hstack([t[..., 3] for t in tiles])
    Image.fromarray((arr * 255).astype(np.uint8)).save("out/fly_inset_test.png")
