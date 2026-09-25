"""Stage-2 check: every neuron as a dim point at its true position, inside the glass shell.

Usage: python -m render.draft_static [width height] -> out/poster_draft.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT  # noqa: E402

from .gl import GLRenderer  # noqa: E402
from .scene import LIN, BrainScene, orbit_pose  # noqa: E402


def main(width=1920, height=1080, out=OUT / "poster_draft.png"):
    t0 = time.time()
    scene = BrainScene()
    r = GLRenderer(width, height)
    r.set_shell(scene.shell_vertices, scene.shell_normals, scene.shell_faces)
    cam = orbit_pose(azimuth_deg=-28, elevation_deg=14, distance=4.3, target=(0.0, -0.02, 0.0), fovy=30).matrices(width, height)
    r.begin()
    r.draw_shell(cam, LIN["shell"], opacity=0.05, power=3.0, fill=0.0)
    col = np.where(scene.drawn[:, None], LIN["resting"][None, :] * 0.05, 0.0).astype(np.float32)
    size = np.full(scene.n, 0.0035, np.float32)
    r.draw_points("neurons", scene.pos, col, size, cam, min_px=1.4)
    img = r.finish(exposure=1.0, bloom_strength=0.5, grain=0.01, seed=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    print(f"{out} {width}x{height} rendered with {r.renderer_name} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a) if a else main()
