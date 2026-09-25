"""Consistency of the stored results: statistics, representative trials and the pathway
can all be re-derived from the raw spike files and the connectome."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXP = ROOT / "results/experiments.json"
pytestmark = pytest.mark.skipif(not EXP.exists() or not (ROOT / "data/processed/neurons.parquet").exists(),
                                reason="needs prepared data and experiments")


def load():
    return json.loads(EXP.read_text())


@pytest.mark.parametrize("key", ["A_sugar", "B_bitter", "C_sugar_bitter", "D_baseline"])
def test_stats_match_spike_files(key):
    exp = load()
    c = exp["conditions"][key]
    d = np.load(ROOT / f"results/spikes/{key}.npz")
    mn9 = exp["mn9"]["indices"]
    per = np.array([[int(((d["trial"] == k) & (d["neurons"] == m)).sum()) for m in mn9] for k in range(c["n_trials"])])
    assert per.tolist() == [t["mn9_spikes"] for t in c["per_trial"]]
    assert np.allclose(per.mean(0), c["mn9_rate_hz_mean"])        # 1 s trials: count == rate in Hz
    assert [int((d["trial"] == k).sum()) for k in range(c["n_trials"])] == [t["total_spikes"] for t in c["per_trial"]]


@pytest.mark.parametrize("key", ["A_sugar", "B_bitter", "C_sugar_bitter", "D_baseline"])
def test_representative_trial_rule(key):
    from sim.run import representative_trial

    c = load()["conditions"][key]
    mn9 = np.array([t["mn9_spikes"] for t in c["per_trial"]], float)
    tot = np.array([t["total_spikes"] for t in c["per_trial"]], float)
    rep = representative_trial(mn9.mean(1), tot, np.array(c["seeds"]))
    assert rep == c["representative_trial"]
    assert abs(mn9.mean(1)[rep] - mn9.mean(1).mean()) == pytest.approx(np.abs(mn9.mean(1) - mn9.mean(1).mean()).min())


def test_pathway_reproducible():
    from render.pathway import PATHWAY_JSON, select

    stored = json.loads(PATHWAY_JSON.read_text())
    fresh = select(stored["condition"], save=False)
    assert fresh["neurons"] == stored["neurons"] and fresh["arcs"] == stored["arcs"]


def test_stimulus_groups_are_annotated_grns():
    import pandas as pd

    exp = load()
    nd = pd.read_parquet(ROOT / "data/processed/neurons.parquet").set_index("root_id")
    assert set(nd.loc[exp["stimulus"]["sugar_root_ids"], "cell_sub_class"]) == {"sugar/water"}
    assert set(nd.loc[exp["stimulus"]["bitter_root_ids"], "cell_sub_class"]) == {"bitter"}
    assert set(nd.loc[exp["mn9"]["root_ids"], "cell_type"]) == {"CB0701"}
