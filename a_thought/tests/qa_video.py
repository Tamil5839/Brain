"""QA for rendered films.

  python tests/qa_video.py out/a_thought.mp4          container facts (size, fps, frames, duration)
  python tests/qa_video.py --nan master 40            re-render 40 sampled frames of a format at full
                                                      resolution and check the HDR buffer is finite

The second check targets the one render defect seen during development: a
non-finite pixel from the shell shader (pow of a negative base) that the bloom
chain spread into a dark block. The shader now clamps its input and the bloom
and final passes discard non-finite values; this confirms no frame produces one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_packets,pix_fmt,codec_name,profile,color_space:format=duration,size,bit_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s, f = j["streams"][0], j["format"]
    return dict(file=str(path), width=s["width"], height=s["height"], fps=s["r_frame_rate"], frames=int(s["nb_read_packets"]),
                duration_s=round(float(f["duration"]), 3), size_mb=round(int(f["size"]) / 1e6, 1),
                mbit_s=round(int(f["bit_rate"]) / 1e6, 1), codec=f"{s['codec_name']} {s.get('profile', '')}".strip(),
                pix_fmt=s["pix_fmt"], color_space=s.get("color_space"))


def nan_check(fmt_name="master", n=40, seed=0):
    from render.film import FilmRenderer

    fr = FilmRenderer(fmt_name)
    total = int(round(fr.tl.duration * fr.f.fps))
    rng = np.random.default_rng(seed)
    frames = np.sort(rng.choice(total, size=min(n, total), replace=False))
    bad = []
    for fi in frames:
        fr.frame(int(fi), overlays=False)
        acc = np.frombuffer(fr.r.accum_tex.read(), np.float32).reshape(fr.r.size[1], fr.r.size[0], 4)
        k = int((~np.isfinite(acc[..., :3])).any(2).sum())
        if k:
            bad.append((int(fi), k))
    print(json.dumps(dict(format=fmt_name, frames_checked=len(frames), frames_with_non_finite_pixels=bad)))
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "--nan":
        sys.exit(1 if nan_check(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40) else 0)
    print(json.dumps(probe(sys.argv[1])))
