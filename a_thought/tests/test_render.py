"""Render-side tests: GL and CPU smoke tests, timeline mapping, traceability of on-screen numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.scene import orbit_pose  # noqa: E402

HAVE_DATA = (ROOT / "data/processed/neurons.parquet").exists() and (ROOT / "results/experiments.json").exists()


def _one_point(renderer):
    cam = orbit_pose(0, 0, 3.0, fovy=30).matrices(*renderer.size)
    renderer.begin()
    renderer.draw_points("p", np.zeros((1, 3), np.float32), np.array([[5.0, 5.0, 5.0]], np.float32),
                         np.array([0.05], np.float32), cam)
    return renderer.finish(grain=0.0, bloom_strength=0.0)


def test_gl_smoke():
    from render.gl import GLRenderer

    try:
        r = GLRenderer(96, 54)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"no OpenGL: {e}")
    img = _one_point(r)
    assert img.shape == (54, 96, 3)
    assert img[27, 48].mean() > 150 and img[2, 2].mean() < 20


def test_cpu_fallback_smoke():
    from render.cpu import CPURenderer

    img = _one_point(CPURenderer(96, 54))
    assert img.shape == (54, 96, 3)
    assert img[26:29, 47:50].mean() > img[2, 2].mean() + 50


def test_overlay_text_renders():
    from render.overlay import text_rgba

    t = text_rgba("139,262 neurons", "light", 40)
    assert t.shape[0] > 20 and t[..., 3].max() > 0.9


@pytest.mark.skipif(not HAVE_DATA, reason="needs prepared data and experiments")
@pytest.mark.parametrize("fmt", ["master", "vertical"])
def test_timeline_brain_clock(fmt):
    from render.film import FORMATS, FilmData, Timeline

    tl = Timeline(FORMATS[fmt], FilmData())
    last_end = 0.0
    for seg in tl.segments:
        assert seg.start >= last_end - 1e-9, "experiments overlap"
        assert tl.brain_ms(seg, seg.t0) == pytest.approx(0.0)
        assert tl.brain_ms(seg, seg.t1) == pytest.approx(1000.0)
        assert seg.t1 - seg.t0 == pytest.approx(FORMATS[fmt].slow)   # 1 s of brain time
        last_end = seg.end
    assert tl.long[0] >= last_end - 1e-9 and tl.credits[1] == tl.duration


@pytest.mark.skipif(not HAVE_DATA, reason="needs prepared data and experiments")
def test_on_screen_numbers_are_traced():
    from render.film import FilmData

    d = FilmData()
    prep = json.loads((ROOT / "results/prepare_log.json").read_text())
    exp = json.loads((ROOT / "results/experiments.json").read_text())
    assert d.numbers["neurons"] == prep["neurons"]["simulated_total"]
    assert d.numbers["connections"] == prep["connectivity"]["connections"]
    assert d.numbers["sugar_n"] == len(exp["stimulus"]["sugar_root_ids"])
    for key in ("A_sugar", "B_bitter", "C_sugar_bitter"):
        spk = np.load(ROOT / f"results/spikes/{key}.npz")
        rep = exp["conditions"][key]["representative_trial"]
        n = spk["neurons"][spk["trial"] == rep]
        assert d.numbers[key]["mn9_spikes_trial"] == [int((n == m).sum()) for m in exp["mn9"]["indices"]]
        assert d.numbers[key]["mn9_spikes_trial"] == exp["conditions"][key]["per_trial"][rep]["mn9_spikes"]
