"""Shared paths and small helpers for the A THOUGHT pipeline."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.json"
RESULTS = ROOT / "results"
SPIKES = RESULTS / "spikes"
OUT = ROOT / "out"

# Processed products written by prepare.py
NEURONS_PARQUET = PROCESSED / "neurons.parquet"
CONNECTIVITY_NPZ = PROCESSED / "connectivity_783.npz"
GROUPS_JSON = PROCESSED / "groups.json"
SHELL_NPZ = PROCESSED / "brain_shell.npz"
PREPARE_LOG = RESULTS / "prepare_log.json"

# The decision neuron. Shiu et al. 2024 use root id 720575940660219265 for MN9
# (figures.ipynb). In FlyWire release 783 that id is unchanged and annotated as
# cell_type CB0701 (ingestion motor neuron); CB0701 has one neuron per side.
SHIU_MN9_ROOT_ID = 720575940660219265
MN9_CELL_TYPE = "CB0701"

# Stimulus lists of Shiu et al. 2024 (figures.ipynb, FlyWire snapshot 630): the labellar
# sugar-sensing, bitter-sensing and water-sensing gustatory receptor neurons the paper
# stimulated ("right hemisphere" in the paper; the v783 annotation lists them as side=left).
# The film's experiments stimulate the ids of these lists that exist unchanged in release 783.
SHIU_SUGAR_R_630 = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
SHIU_BITTER_R_630 = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
SHIU_WATER_R_630 = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053, 720575940622902535,
    720575940616177458, 720575940660292225, 720575940622486922, 720575940613786774, 720575940629852866,
    720575940625861168, 720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
    720575940630553415, 720575940635172191, 720575940634796536,
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default) + "\n")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stdout)
