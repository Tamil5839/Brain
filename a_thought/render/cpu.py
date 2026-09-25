"""Slow CPU fallback with the same interface as render.gl.GLRenderer (for previews
on machines without any OpenGL). Points and line samples are splatted with
bilinear weights into a float image, blurred for size and bloom, then tone
mapped like the GL path. Visual output is close to, not identical with, GL.

Select with FilmRenderer(..., backend="cpu") or A_THOUGHT_RENDERER=cpu.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


class CPURenderer:
    def __init__(self, width: int, height: int, bloom_levels: int = 5):
        self.size = (int(width), int(height))
        self.renderer_name = "numpy (CPU fallback)"
        self.levels = bloom_levels
        self._shell = None
        self.accum = np.zeros((height, width, 3), np.float32)

    def set_shell(self, vertices, normals, faces):
        # sample the surface densely enough to read as a rim: triangle centroids + vertices
        tri = vertices[faces]
        self._shell = (np.vstack([vertices, tri.mean(1)]).astype(np.float32),
                       np.vstack([normals, np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])]).astype(np.float32))
        n = np.linalg.norm(self._shell[1], axis=1, keepdims=True)
        self._shell = (self._shell[0], self._shell[1] / np.maximum(n, 1e-12))

    def begin(self):
        self.accum[:] = 0.0

    def _project(self, pos, cam):
        W, H = self.size
        p = np.c_[pos, np.ones(len(pos), np.float32)] @ cam["mvp"].T
        w = p[:, 3]
        ok = w > 1e-4
        x = (p[:, 0] / np.where(ok, w, 1) * 0.5 + 0.5) * W
        y = (1 - (p[:, 1] / np.where(ok, w, 1) * 0.5 + 0.5)) * H
        return x, y, w, ok

    def _splat(self, x, y, col, ok):
        W, H = self.size
        m = ok & (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1) & (col.sum(1) > 0)
        x, y, col = x[m], y[m], col[m]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(self.accum, (y0 + dy, x0 + dx), col * wgt)

    def draw_shell(self, cam, color, opacity, power=3.0, fill=0.0):
        if self._shell is None or opacity <= 0:
            return
        pos, nrm = self._shell
        v = cam["eye"][None, :] - pos
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rim = (1 - np.abs((nrm * v).sum(1))) ** power + fill
        x, y, _, ok = self._project(pos, cam)
        self._splat(x, y, (np.asarray(color)[None] * (rim * opacity * 6.0)[:, None]).astype(np.float32), ok)

    def draw_points(self, key, positions, colors, sizes, cam, gain=1.0, min_px=1.2, max_px=400.0, ring=False):
        x, y, w, ok = self._project(np.asarray(positions, np.float32), cam)
        px = np.asarray(sizes) * cam["px_per_unit"] / np.maximum(w, 1e-4)
        drawn = np.clip(px, min_px, max_px)
        flux = np.minimum((px / drawn) ** 2, 1.0) * (drawn / 2.0) ** 2 * 0.35  # light of a soft sprite of that size
        self._splat(x, y, (np.asarray(colors) * (gain * flux)[:, None]).astype(np.float32), ok)

    def draw_lines(self, key, segments, colors, cam, width_px=1.5, gain=1.0):
        if len(segments) == 0:
            return
        seg = np.asarray(segments, np.float32)
        col = np.asarray(colors, np.float32).reshape(len(seg), 2, 3).mean(1)
        u = np.linspace(0, 1, 4, dtype=np.float32)[None, :, None]
        pts = (seg[:, :1] * (1 - u) + seg[:, 1:] * u).reshape(-1, 3)
        x, y, _, ok = self._project(pts, cam)
        self._splat(x, y, np.repeat(col, 4, 0) * gain * width_px * 0.5, ok)

    def finish(self, exposure=1.0, bloom_strength=0.6, bloom_radius=1.0, grain=0.012, seed=0.0,
               bg=(3 / 255, 4 / 255, 7 / 255), fade=1.0):
        W, H = self.size
        img = gaussian_filter(self.accum, sigma=(0.7, 0.7, 0))
        bloom = np.zeros_like(img)
        for k in range(1, self.levels + 1):
            s = (2 ** k) * H / 1080.0 * bloom_radius
            bloom += gaussian_filter(img, sigma=(s, s, 0))
        hdr = img + bloom_strength / self.levels * bloom
        c = _aces(hdr * exposure) * fade
        c = c ** (1 / 2.2)
        c = np.asarray(bg, np.float32) + c * (1 - np.asarray(bg, np.float32))
        rng = np.random.default_rng(int(seed * 1000) % (2 ** 32))
        lum = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        c += (rng.random((H, W), np.float32) + rng.random((H, W), np.float32) - 1.0)[..., None] * grain * (0.35 + 0.65 * np.sqrt(lum))[..., None]
        return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)

    def release(self):
        pass
