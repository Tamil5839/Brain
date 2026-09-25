"""2-D overlays: captions, labels, clock, readouts, composited onto the tone-mapped frame.

Typography: Inter (SIL Open Font License; Ubuntu package fonts-inter). Numbers
use Inter's tabular figures so the clock and readouts do not jitter.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
FALLBACK = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
WEIGHTS = {"light": "Inter-Light.otf", "regular": "Inter-Regular.otf", "medium": "Inter-Medium.otf",
           "semibold": "Inter-SemiBold.otf", "extralight": "Inter-ExtraLight.otf"}
RAQM = features.check("raqm")


@lru_cache(maxsize=64)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    p = FONT_DIR / WEIGHTS[weight]
    if not p.exists():
        p = FALLBACK
    return ImageFont.truetype(str(p), size, layout_engine=ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC)


@lru_cache(maxsize=512)
def text_rgba(text: str, weight: str, size: int, color: tuple = (235, 238, 245), tracking: float = 0.0,
              tnum: bool = True) -> np.ndarray:
    """Render one line of text to a float RGBA array (premultiplied, 0..1)."""
    f = font(weight, size)
    feats = ["tnum"] if (tnum and RAQM) else None
    kw = dict(features=feats) if feats else {}
    if tracking:
        # letter-spaced small caps style labels: draw character by character
        widths = [f.getlength(ch, **kw) for ch in text]
        w = int(sum(widths) + tracking * size * max(len(text) - 1, 0)) + 4
    else:
        w = int(f.getlength(text, **kw)) + 4
    asc, desc = f.getmetrics()
    h = asc + desc + 4
    img = Image.new("L", (max(w, 1), h), 0)
    d = ImageDraw.Draw(img)
    if tracking:
        x = 2.0
        for ch, cw in zip(text, widths):
            d.text((x, 2), ch, font=f, fill=255, **kw)
            x += cw + tracking * size
    else:
        d.text((2, 2), text, font=f, fill=255, **kw)
    a = np.asarray(img, np.float32) / 255.0
    rgb = np.array(color, np.float32) / 255.0
    out = np.zeros((h, img.width, 4), np.float32)
    out[..., :3] = a[..., None] * rgb
    out[..., 3] = a
    out.flags.writeable = False
    return out


def composite(frame: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-blend a premultiplied RGBA patch onto an sRGB uint8 frame in place (top-left at x, y)."""
    if alpha <= 0.001:
        return
    H, W = frame.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    region = frame[y0:y1, x0:x1].astype(np.float32) / 255.0
    a = patch[..., 3:4] * alpha
    region = patch[..., :3] * alpha + region * (1.0 - a)
    frame[y0:y1, x0:x1] = np.clip(region * 255.0 + 0.5, 0, 255).astype(np.uint8)


def dot_rgba(radius: int, color: tuple, ring: bool = False) -> np.ndarray:
    s = 4
    R = radius * s
    img = Image.new("L", (2 * R + 4 * s, 2 * R + 4 * s), 0)
    d = ImageDraw.Draw(img)
    c = R + 2 * s
    if ring:
        d.ellipse([c - R, c - R, c + R, c + R], outline=255, width=max(s, int(R * 0.22)))
    else:
        d.ellipse([c - R, c - R, c + R, c + R], fill=255)
    img = img.resize((img.width // s, img.height // s), Image.LANCZOS)
    a = np.asarray(img, np.float32) / 255.0
    out = np.zeros(a.shape + (4,), np.float32)
    out[..., :3] = a[..., None] * (np.array(color, np.float32) / 255.0)
    out[..., 3] = a
    return out


class TextBlock:
    """Several lines, drawn relative to an anchor, sharing one alpha."""

    def __init__(self):
        self.items = []  # (rgba, dx, dy)

    def add(self, rgba, dx, dy):
        self.items.append((rgba, int(dx), int(dy)))
        return self

    def draw(self, frame, x, y, alpha=1.0):
        for rgba, dx, dy in self.items:
            composite(frame, rgba, x + dx, y + dy, alpha)
