"""poster.png: the whole brain with every spike of the three film trials as one long exposure.

Each neuron that spiked in a trial is drawn in that trial's colour with a
brightness proportional to the square root of its spike count; nothing else
is drawn except the dim resting neurons and the glass shell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, log  # noqa: E402

from .film import FilmRenderer, hex_rgb  # noqa: E402
from .overlay import composite, dot_rgba, text_rgba  # noqa: E402
from .scene import PALETTE, orbit_pose  # noqa: E402


def make_poster(path: Path = OUT / "poster.png"):
    from sim.validate import require_validated

    require_validated()
    fr = FilmRenderer("master")
    tl = fr.tl
    T = tl.accum[1] + 2.0                        # accumulation complete, before the credits
    pose = orbit_pose(-14, 13, 2.45, target=(0.0, -0.05, 0.0), fovy=30)
    tl.camera = lambda _T: pose
    img = fr.frame(int(round(T * fr.f.fps)), overlays=False)
    W, H = fr.f.size
    u = fr.u
    white = (236, 239, 246)
    nums = fr.d.numbers
    mx, my = int(0.06 * W), int(0.075 * H)
    composite(img, text_rgba("A THOUGHT", "light", int(96 * u), white, tracking=0.3), mx, my, 0.95)
    sub = ("Every coloured point of light is a neuron that spiked in a simulation built on the real wiring "
           "of a fruit fly's brain.")
    composite(img, text_rgba(sub, "light", int(40 * u), white), mx, my + int(140 * u), 0.8)
    x, y = mx, my + int(215 * u)
    for ck, label in (("sugar", f"sugar ({nums['sugar_n']} taste neurons, 150 Hz)"),
                      ("bitter", f"bitter ({nums['bitter_n']} taste neurons, 150 Hz)"),
                      ("mixed", "sugar + bitter")):
        dot = dot_rgba(max(2, int(10 * u)), hex_rgb(PALETTE[ck]))
        composite(img, dot, x, y + int(8 * u), 0.9)
        x += dot.shape[1] + int(10 * u)
        t = text_rgba(label, "regular", int(32 * u), white)
        composite(img, t, x, y, 0.7)
        x += t.shape[1] + int(40 * u)
    foot = [
        f"{nums['neurons']:,} neurons  ·  {nums['connections']:,} connections  ·  three simulated trials of 1 s "
        f"(leaky integrate-and-fire model of Shiu et al. 2024)",
        "Data: FlyWire Consortium, Dorkenwald et al. 2024 (Nature); Schlegel et al. 2024 (Nature); "
        "Eckstein et al. 2024 (Cell)  ·  Model: Shiu et al. 2024 (Nature)",
    ]
    for i, line in enumerate(foot):
        t = text_rgba(line, "regular", int(28 * u), white)
        composite(img, t, mx, H - int(0.075 * H) - (len(foot) - i) * int(46 * u), 0.55)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=True)
    log(f"wrote {path} ({W}x{H})")
    return path


if __name__ == "__main__":
    make_poster()
