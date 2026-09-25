"""Run the film's experiments and store every spike of every neuron.

Conditions (all on the authors' v783 network, parameters from their model.py):
  A_sugar         sugar GRNs of Shiu et al. (ids present in release 783), 150 Hz Poisson
  B_bitter        bitter GRNs of Shiu et al., 150 Hz
  C_sugar_bitter  both groups, 150 Hz each
  D_baseline      no stimulus
Each condition: 30 trials of 1 s (the authors' n_run), seeds 0..29.

Outputs:
  results/spikes/<condition>.npz  steps, neurons, trial (every spike), compressed
  results/experiments.json        per-trial summaries, MN9 rates, representative trials

Usage: python -m sim.run [--trials 30] [--only A_sugar ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS, SPIKES, load_json, log, save_json, utc_now  # noqa: E402

from .model import Params, Stimulus, simulate  # noqa: E402
from .networks import load_groups, load_neurons, network_783  # noqa: E402

EXPERIMENTS_JSON = RESULTS / "experiments.json"
RATE_HZ = 150.0  # r_poi default in the authors' model.py
N_TRIALS = 30    # n_run default in the authors' model.py


def stimulus_groups() -> dict:
    """Film stimulus groups: Shiu et al.'s lists, restricted to ids present in v783."""
    g = load_groups()
    nd = load_neurons()
    index = {r: i for i, r in enumerate(nd.root_id.to_numpy())}
    sugar_ids = g["paper_lists_630"]["sugar_right_in_783"]
    bitter_ids = g["paper_lists_630"]["bitter_right_in_783"]
    sugar = np.array([index[r] for r in sugar_ids])
    bitter = np.array([index[r] for r in bitter_ids])
    ann_sugar, ann_bitter = set(g["sugar"]["indices"]), set(g["bitter"]["indices"])
    assert set(sugar) <= ann_sugar, "a paper sugar GRN is not annotated sugar/water"
    assert set(bitter) <= ann_bitter, "a paper bitter GRN is not annotated bitter"
    return dict(sugar=sugar, bitter=bitter, mn9=np.array(g["mn9"]["indices"]),
                sugar_root_ids=sugar_ids, bitter_root_ids=bitter_ids)


def conditions(groups: dict, rate: float = RATE_HZ) -> dict:
    s = Stimulus(groups["sugar"], rate, "sugar")
    b = Stimulus(groups["bitter"], rate, "bitter")
    return {"A_sugar": [s], "B_bitter": [b], "C_sugar_bitter": [s, b], "D_baseline": []}


def representative_trial(mn9_rate: np.ndarray, total_spikes: np.ndarray, seeds: np.ndarray) -> int:
    """Trial whose MN9 rate (mean of the two MN9 neurons) is closest to the condition mean.

    Ties (e.g. every trial at 0 Hz) are broken by total spike count closest to the
    condition mean, then by the lowest seed.
    """
    d1 = np.abs(mn9_rate - mn9_rate.mean())
    d2 = np.abs(total_spikes - total_spikes.mean())
    order = np.lexsort((seeds, d2, np.round(d1, 9)))
    return int(order[0])


def run_condition(net, name, stimuli, groups, n_trials, params) -> dict:
    mn9 = groups["mn9"]
    steps, neurons, trial_ix = [], [], []
    per_trial = []
    t0 = time.time()
    for k in range(n_trials):
        tr = simulate(net, stimuli, params, seed=k)
        steps.append(tr.steps)
        neurons.append(tr.neurons)
        trial_ix.append(np.full(len(tr.steps), k, np.int16))
        counts = np.bincount(tr.neurons, minlength=net.n)
        stim_idx = np.concatenate([s.indices for s in stimuli]) if stimuli else np.empty(0, int)
        per_trial.append(dict(
            seed=k,
            total_spikes=int(len(tr.steps)),
            active_neurons=int((counts > 0).sum()),
            active_non_stimulated=int((counts > 0).sum() - (counts[stim_idx] > 0).sum()),
            mn9_spikes=[int(counts[m]) for m in mn9],
            stimulated_mean_rate_hz=float(counts[stim_idx].mean() / (params.t_run / 1000)) if len(stim_idx) else 0.0,
        ))
    SPIKES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPIKES / f"{name}.npz", steps=np.concatenate(steps), neurons=np.concatenate(neurons),
                        trial=np.concatenate(trial_ix), dt=params.dt, n_steps=int(round(params.t_run / params.dt)))
    t_s = params.t_run / 1000.0
    mn9_rates = np.array([t["mn9_spikes"] for t in per_trial]) / t_s  # trials x 2
    total = np.array([t["total_spikes"] for t in per_trial])
    seeds = np.array([t["seed"] for t in per_trial])
    rep = representative_trial(mn9_rates.mean(1), total, seeds)
    out = dict(
        stimuli=[dict(label=s.label, rate_hz=s.rate, n=len(s.indices)) for s in stimuli],
        n_trials=n_trials,
        seeds=seeds.tolist(),
        mn9_rate_hz_mean=mn9_rates.mean(0).tolist(),
        mn9_rate_hz_sd=mn9_rates.std(0).tolist(),
        mn9_pair_mean_hz=float(mn9_rates.mean()),
        total_spikes_mean=float(total.mean()),
        active_neurons_mean=float(np.mean([t["active_neurons"] for t in per_trial])),
        representative_trial=rep,
        representative_seed=int(seeds[rep]),
        representative_mn9_rate_hz=mn9_rates[rep].tolist(),
        per_trial=per_trial,
        wall_time_s=round(time.time() - t0, 1),
    )
    log(f"  {name:15s} MN9 L/R {np.round(out['mn9_rate_hz_mean'], 1)} Hz (sd {np.round(out['mn9_rate_hz_sd'], 1)}), "
        f"{out['total_spikes_mean']:.0f} spikes/trial, representative trial seed {out['representative_seed']} "
        f"({out['wall_time_s']:.0f} s)")
    return out


def main(n_trials: int = N_TRIALS, only: list[str] | None = None) -> dict:
    params = Params()
    net = network_783()
    groups = stimulus_groups()
    conds = conditions(groups)
    prev = load_json(EXPERIMENTS_JSON, default={}) or {}
    results = prev.get("conditions", {})
    for name, stimuli in conds.items():
        if only and name not in only:
            continue
        results[name] = run_condition(net, name, stimuli, groups, n_trials, params)
    nd = load_neurons()
    exp = dict(
        created_utc=utc_now(),
        network=net.info,
        params=params.as_dict(),
        neurons_simulated=int(net.n),
        connections=int(net.W.nnz),
        stimulus=dict(
            description="Shiu et al. 2024 GRN lists (figures.ipynb, snapshot 630), ids unchanged in release 783",
            sugar_root_ids=groups["sugar_root_ids"], bitter_root_ids=groups["bitter_root_ids"],
            sugar_n=len(groups["sugar"]), bitter_n=len(groups["bitter"]), rate_hz=RATE_HZ,
            sugar_annotation=sorted(set(nd.cell_sub_class.iloc[groups["sugar"]])),
            bitter_annotation=sorted(set(nd.cell_sub_class.iloc[groups["bitter"]])),
            sides_annotation=sorted(set(nd.side.iloc[np.r_[groups["sugar"], groups["bitter"]]])),
        ),
        mn9=dict(indices=groups["mn9"].tolist(), root_ids=nd.root_id.iloc[groups["mn9"]].tolist(),
                 sides=nd.side.iloc[groups["mn9"]].tolist()),
        conditions=results,
    )
    save_json(EXPERIMENTS_JSON, exp)
    return exp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(a.trials, a.only)
