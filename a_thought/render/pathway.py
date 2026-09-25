"""Pathway highlight (a visual aid; see NOTES.md).

Selection, from the representative sugar trial and the connectome only:
  level 1  neurons with excitatory synapses onto MN9, ranked by
           (spikes in the trial) x (synapses onto MN9); top K1
  level 2  neurons with excitatory synapses onto the level-1 set, ranked by
           (spikes) x (synapses onto level 1); top K2
  input    stimulated sugar GRNs with excitatory synapses onto level 1 or 2,
           ranked by (spikes) x (synapses onto level 1 + 2); top K0
Arcs are drawn for every excitatory connection of >= MIN_SYN synapses that runs
from a lower to a higher level (input -> level 2 -> level 1 -> MN9).

Skeletons of the selected neurons are read from the public FlyWire release-783
segmentation bucket (gs://flywire_v141_m783, layer skeletons_mip_1, neuroglancer
sharded format) and recorded in data/manifest.json.

Usage: python -m render.pathway   -> results/pathway.json, data/processed/skeletons/*.npz
"""
from __future__ import annotations

import gzip
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    CONNECTIVITY_NPZ, MANIFEST, NEURONS_PARQUET, PROCESSED, RESULTS, SPIKES, load_json, log, save_json,
    sha256_bytes, utc_now,
)

PATHWAY_JSON = RESULTS / "pathway.json"
SKELETON_DIR = PROCESSED / "skeletons"
SKELETON_LAYER = "https://storage.googleapis.com/flywire_v141_m783/skeletons_mip_1"
K1, K2, K0, MIN_SYN = 8, 8, 6, 5
RESAMPLE_NM = 1500.0


# ------------------------------------------------------------------ selection
def select(condition: str = "A_sugar", save: bool = True) -> dict:
    exp = load_json(RESULTS / "experiments.json")
    c = exp["conditions"][condition]
    rep = c["representative_trial"]
    d = np.load(SPIKES / f"{condition}.npz")
    m = d["trial"] == rep
    nd = pd.read_parquet(NEURONS_PARQUET)
    n = len(nd)
    spikes = np.bincount(d["neurons"][m], minlength=n).astype(float)
    W = sp.load_npz(CONNECTIVITY_NPZ).tocsr()
    Wexc = W.multiply(W > 0).tocsr()                      # excitatory synapse counts
    mn9 = np.array(exp["mn9"]["indices"])
    pos = {r: i for i, r in enumerate(nd.root_id)}
    grn = np.array([pos[r] for r in exp["stimulus"]["sugar_root_ids"]])
    exclude = set(mn9) | set(grn)

    def rank(targets, k, pool_exclude):
        syn = np.asarray(Wexc[:, targets].sum(1)).ravel()
        score = spikes * syn
        score[list(pool_exclude)] = 0
        order = np.argsort(-score, kind="stable")
        return [int(i) for i in order[:k] if score[i] > 0], score

    l1, s1 = rank(mn9, K1, exclude)
    l2, s2 = rank(np.array(l1), K2, exclude | set(l1))
    syn_g = np.asarray(Wexc[grn][:, l1 + l2].sum(1)).ravel()
    sg = spikes[grn] * syn_g
    g_order = np.argsort(-sg, kind="stable")
    l0 = [int(grn[i]) for i in g_order[:K0] if sg[i] > 0]
    level = {**{i: 0 for i in l0}, **{i: 1 for i in l2}, **{i: 2 for i in l1}, **{int(i): 3 for i in mn9}}
    members = list(level)
    arcs = []
    for a in members:
        row = Wexc[a]
        for b, w in zip(row.indices, row.data):
            b = int(b)
            if b in level and level[b] > level[a] and w >= MIN_SYN:
                arcs.append((a, b, int(w)))
    arcs.sort(key=lambda x: -x[2])
    rows = []
    for i in members:
        rows.append(dict(index=i, root_id=int(nd.root_id[i]), level=level[i], spikes=int(spikes[i]),
                         cell_type=None if pd.isna(nd.cell_type[i]) else str(nd.cell_type[i]),
                         cell_class=None if pd.isna(nd.cell_class[i]) else str(nd.cell_class[i]),
                         side=str(nd.side[i]), color_class=str(nd.color_class[i])))
    out = dict(
        created_utc=utc_now(), condition=condition, trial=int(rep), seed=int(c["representative_seed"]),
        rule=dict(K1=K1, K2=K2, K0=K0, min_syn_arc=MIN_SYN,
                  score="spikes in the trial x excitatory synapses onto the next level"),
        neurons=rows, arcs=[dict(pre=a, post=b, synapses=w) for a, b, w in arcs],
    )
    if save:
        save_json(PATHWAY_JSON, out)
    return out


# ------------------------------------------------------------------ skeletons
def _murmur_low64(seg: int) -> int:
    import mmh3

    return mmh3.hash128(struct.pack("<Q", seg), seed=0, x64arch=False, signed=False) & ((1 << 64) - 1)


def fetch_skeleton(seg: int, info: dict, session) -> tuple[bytes, dict]:
    """Read one skeleton from a neuroglancer_uint64_sharded_v1 precomputed layer."""
    sh = info["sharding"]
    h = _murmur_low64(seg) >> sh["preshift_bits"]
    mb, sb = sh["minishard_bits"], sh["shard_bits"]
    minishard = h & ((1 << mb) - 1)
    shard = (h >> mb) & ((1 << sb) - 1)
    url = f"{SKELETON_LAYER}/{format(shard, 'x').zfill(int(math.ceil(sb / 4)))}.shard"

    def get(a, b):
        r = session.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=60)
        r.raise_for_status()
        return r.content

    start, end = struct.unpack("<QQ", get(minishard * 16, minishard * 16 + 16))
    base = (1 << mb) * 16
    idx = get(base + start, base + end)
    if sh["minishard_index_encoding"] == "gzip":
        idx = gzip.decompress(idx)
    arr = np.frombuffer(idx, "<u8").reshape(3, -1)
    ids = np.cumsum(arr[0])
    k = np.flatnonzero(ids == np.uint64(seg))
    if not len(k):
        raise KeyError(f"{seg} not in {url}")
    offs, sizes = arr[1], arr[2]
    starts = np.zeros(len(offs), np.uint64)
    p = np.uint64(0)
    for i in range(len(offs)):
        starts[i] = p + offs[i]
        p = starts[i] + sizes[i]
    a = base + int(starts[k[0]])
    blob = get(a, a + int(sizes[k[0]]))
    if sh["data_encoding"] == "gzip":
        blob = gzip.decompress(blob)
    return blob, dict(url=url, byte_range=[a, a + int(sizes[k[0]])])


def decode_skeleton(blob: bytes):
    nv, ne = struct.unpack("<II", blob[:8])
    v = np.frombuffer(blob, "<f4", nv * 3, 8).reshape(-1, 3).astype(np.float64)
    e = np.frombuffer(blob, "<u4", ne * 2, 8 + nv * 12).reshape(-1, 2).astype(np.int64)
    return v, e


def resample(v: np.ndarray, e: np.ndarray, step: float = RESAMPLE_NM) -> np.ndarray:
    """Keep branch points, end points and one vertex per `step` nm along each path.

    Returns line segments (M, 2, 3) in nm. Structure is preserved; only density drops.
    """
    n = len(v)
    deg = np.bincount(e.ravel(), minlength=n)
    adj = [[] for _ in range(n)]
    for a, b in e:
        adj[a].append(b)
        adj[b].append(a)
    key = deg != 2
    if not key.any():
        key[0] = True
    seen_edge = set()
    segs = []
    for s in np.flatnonzero(key):
        for nb in adj[s]:
            if (s, nb) in seen_edge:
                continue
            last, prev, cur, acc = s, s, nb, 0.0
            while True:
                seen_edge.add((prev, cur))
                seen_edge.add((cur, prev))
                acc += float(np.linalg.norm(v[cur] - v[prev]))
                if key[cur] or acc >= step:
                    segs.append((last, cur))
                    last, acc = cur, 0.0
                if key[cur]:
                    break
                nxt = [x for x in adj[cur] if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
    idx = np.array(segs, np.int64).reshape(-1, 2)
    return v[idx]


def fetch_all(pathway: dict) -> dict:
    import requests

    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    info = session.get(f"{SKELETON_LAYER}/info", timeout=60).json()
    manifest = load_json(MANIFEST)
    records = manifest.get("skeletons", {}) or {}
    for row in pathway["neurons"]:
        rid = row["root_id"]
        out = SKELETON_DIR / f"{rid}.npz"
        if out.exists() and str(rid) in records:
            continue
        blob, where = fetch_skeleton(rid, info, session)
        v, e = decode_skeleton(blob)
        segs = resample(v, e)
        np.savez_compressed(out, segments_nm=segs.astype(np.float32), n_vertices=len(v), n_edges=len(e))
        records[str(rid)] = dict(
            source=f"{SKELETON_LAYER} (FlyWire public release 783 segmentation, precomputed skeletons)",
            shard_url=where["url"], byte_range=where["byte_range"], retrieved_utc=utc_now(),
            decoded_sha256=sha256_bytes(blob), vertices=int(len(v)), edges=int(len(e)),
            resampled_segments=int(len(segs)), resample_step_nm=RESAMPLE_NM,
        )
        log(f"  skeleton {rid}: {len(v):,d} vertices -> {len(segs):,d} segments")
    manifest["skeletons"] = records
    save_json(MANIFEST, manifest)
    return records


if __name__ == "__main__":
    pw = select()
    lv = {0: "input GRN", 1: "level 2", 2: "level 1", 3: "MN9"}
    for r in pw["neurons"]:
        log(f"  {lv[r['level']]:9s} {r['root_id']} {r['cell_type'] or '':10s} {r['side']:6s} spikes {r['spikes']}")
    log(f"  arcs: {len(pw['arcs'])}")
    fetch_all(pw)
