"""Run the authors' unmodified Brian2 code (data/raw/shiu_model/model.py) for reference trials.

Usage: python -m sim.brian2_reference <network: 630|783> <stim: sugarR630|paper_sugar|paper_bitter|paper_sugar_bitter|sugar|bitter|sugar_bitter> <rate_hz> <n_trials> [n_jobs]
Writes results/brian2/<name>.parquet in the authors' output format (t, trial, flywire_id, exp_name).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/raw/shiu_model"))
sys.path.insert(0, str(ROOT))

import model as shiu  # noqa: E402  (the authors' model.py, downloaded by fetch.py)
from brian2 import Hz  # noqa: E402

from sim.networks import load_groups  # noqa: E402
from common import SHIU_SUGAR_R_630  # noqa: E402

PATHS = {
    "630": (ROOT / "data/raw/shiu_model/2023_03_23_completeness_630_final.csv",
            ROOT / "data/raw/shiu_model/2023_03_23_connectivity_630_final.parquet"),
    "783": (ROOT / "data/raw/shiu_model/Completeness_783.csv",
            ROOT / "data/raw/shiu_model/Connectivity_783.parquet"),
}


def main():
    net, stim, rate, n_trials = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    comp, con = PATHS[net]
    params = dict(shiu.default_params)
    params["n_run"] = n_trials
    params["r_poi"] = rate * Hz
    params["r_poi2"] = rate * Hz
    g = load_groups()
    exc2 = []
    if stim == "sugarR630":
        exc = SHIU_SUGAR_R_630
    elif stim == "paper_sugar":
        exc = g["paper_lists_630"]["sugar_right_in_783"]
    elif stim == "paper_bitter":
        exc = g["paper_lists_630"]["bitter_right_in_783"]
    elif stim == "paper_sugar_bitter":
        exc, exc2 = g["paper_lists_630"]["sugar_right_in_783"], g["paper_lists_630"]["bitter_right_in_783"]
    elif stim == "sugar":
        exc = g["sugar"]["root_ids"]
    elif stim == "bitter":
        exc = g["bitter"]["root_ids"]
    elif stim == "sugar_bitter":
        exc, exc2 = g["sugar"]["root_ids"], g["bitter"]["root_ids"]
    else:
        raise SystemExit(stim)
    name = f"b2_{net}_{stim}_{int(rate)}Hz_{n_trials}tr"
    out = ROOT / "results/brian2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shiu.run_exp(exp_name=name, neu_exc=list(exc), neu_exc2=list(exc2), path_res=out, path_comp=comp,
                 path_con=con, params=params, n_proc=n_jobs, force_overwrite=True)
    print(f"done {name} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
