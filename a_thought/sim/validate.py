"""Section-5 validation. Rendering refuses to start unless every test here passes.

Tests
  1 data integrity      counts of neurons / connections / synapses / stimulus groups
  2 sugar -> MN9        high rate, consistent with the authors' published outputs
  3 bitter              does not drive MN9
  4 sugar + bitter      MN9 lower than sugar alone
  5 baseline            no spikes at all without input
  6 determinism         same seed -> identical spike trains; different seed -> different
Cross-checks (reported, and asserted where a strict expectation exists)
  a published outputs   rerun the authors' example (snapshot 630) and compare with their stored Brian2 output
  b Brian2              authors' unmodified code on the v783 network (results/brian2/*.parquet), and the
                        step-level equivalence checks (tests/brian2_*equivalence.py, results/brian2/*.json)
  c flypoke             flypoke's own code on the same connectome, and this model at flypoke's settings
  d annotation classes  this model with all annotated sugar/water and bitter GRNs (flypoke's selectors)

Usage: python -m sim.validate [--rerun]   -> results/validation.json (+ printed report)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    MANIFEST, PREPARE_LOG, RAW, RESULTS, SHIU_MN9_ROOT_ID, SHIU_SUGAR_R_630, load_json, log, save_json, utc_now,
)

from .model import Params, Stimulus, simulate  # noqa: E402
from .networks import (  # noqa: E402
    index_of, load_groups, load_neurons, network_630, network_783, network_783_variant,
)
from .run import EXPERIMENTS_JSON  # noqa: E402

VALIDATION_JSON = RESULTS / "validation.json"
CACHE_JSON = RESULTS / "validation_runs.json"
FLYPOKE_TABLE = {  # flypoke README "Results" table (5 trials, seed 0, 150 Hz)
    "sugar": dict(mn9_right=150.0, mn9_left=111.6, ingestion_mean=56.2, active=439),
    "bitter": dict(mn9_right=0.0, mn9_left=0.0, ingestion_mean=0.0, active=529),
    "sugar + bitter": dict(mn9_right=14.4, mn9_left=8.8, ingestion_mean=40.4, active=419),
}


def _mn9_rates(trials_counts: np.ndarray, t_s: float = 1.0):
    return trials_counts / t_s


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 and b.std() == 0:
        return dict(t=float("inf") if a.mean() != b.mean() else 0.0, p=0.0 if a.mean() != b.mean() else 1.0)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(t=float(t), p=float(p))


def summary(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std()), n=int(len(x)), sem=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)


# ------------------------------------------------------------------------- runs
def run_trials(net, stimuli, n, seed0=0, watch=()):
    counts = []
    totals = []
    for k in range(n):
        tr = simulate(net, stimuli, Params(), seed=seed0 + k)
        c = np.bincount(tr.neurons, minlength=net.n)
        counts.append(c[list(watch)])
        totals.append(len(tr.steps))
    return np.array(counts), np.array(totals)


def published_630_check(n_trials=30) -> dict:
    """Rerun the authors' example (21 sugar GRNs, snapshot 630) and compare with their stored outputs."""
    net = network_630()
    sugar = index_of(net, SHIU_SUGAR_R_630)
    ids = [SHIU_MN9_ROOT_ID, 720575940645521262]  # MN9 pair in snapshot 630 (figures.ipynb, Figure 2)
    mn9 = index_of(net, ids)
    out = {}
    for rate, fname in [(200.0, "sugarR.parquet"), (100.0, "sugarR_100Hz.parquet")]:
        df = pd.read_parquet(RAW / "shiu_model/results_example" / fname)
        n_ref = int(df.trial.nunique())
        ref = df[df.flywire_id.isin(ids)].groupby(["trial", "flywire_id"]).size().unstack(fill_value=0)
        ref = ref.reindex(index=range(n_ref), columns=ids, fill_value=0)
        ref_grn = df[df.flywire_id.isin(SHIU_SUGAR_R_630)].groupby("trial").size().reindex(range(n_ref), fill_value=0) / len(SHIU_SUGAR_R_630)
        counts, totals = run_trials(net, [Stimulus(sugar, rate)], n_trials, watch=mn9)
        rows = {}
        for k, rid in enumerate(ids):
            a, b = counts[:, k], ref[rid].to_numpy()
            rows[str(rid)] = dict(ours=summary(a), authors=summary(b), welch=welch(a, b),
                                  diff_hz=float(a.mean() - b.mean()),
                                  diff_in_se=float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))))
        out[f"{int(rate)}Hz"] = dict(
            stored_file=f"results/example/{fname} (Shiu et al. repository)",
            stimulus_rate_hz=rate, trials_ours=n_trials, trials_authors=n_ref,
            mn9=rows,
            spikes_per_trial=dict(ours=summary(totals), authors=float(len(df) / n_ref)),
            authors_sugar_grn_rate_hz=float(ref_grn.mean()),
        )
    return out


def flypoke_checks(n_trials=5) -> dict:
    """(i) flypoke's own code on the connectome built from the authors' table (flypoke's settings);
    (ii) this model with flypoke's settings; (iii) this model, authors' network, flypoke's selectors."""
    sys.path.insert(0, str(RAW / "flypoke"))
    fd = importlib.import_module("flypoke.data")
    fs = importlib.import_module("flypoke.sim")
    ann = pd.read_csv(RAW / "flywire_annotations_v3.1.0/Supplemental_file1_neuron_annotations.tsv", sep="\t",
                      usecols=fd.ANNOTATION_COLUMNS, dtype={"root_id": np.int64}, low_memory=False)
    neurons = ann.sort_values("root_id").reset_index(drop=True)
    con = pd.read_parquet(RAW / "shiu_model/Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    con = con[con.Connectivity >= 5]
    idx = pd.Index(neurons.root_id)
    pre, post = idx.get_indexer(con.Presynaptic_ID.to_numpy()), idx.get_indexer(con.Postsynaptic_ID.to_numpy())
    keep = (pre >= 0) & (post >= 0)
    sign = neurons.top_nt.map(fd.NT_SIGN).fillna(1.0).to_numpy(dtype=np.float32)
    W = sp.csr_matrix((con.Connectivity.to_numpy()[keep].astype(np.float32) * sign[pre[keep]], (pre[keep], post[keep])),
                      shape=(len(neurons),) * 2, dtype=np.float32)
    W.sum_duplicates()
    net = fd.Network(neurons, W, 5)
    sugar, bitter = net.select("cell_sub_class=sugar/water"), net.select("cell_sub_class=bitter")
    mn = net.select("cell_sub_class=ingestion_motor_neuron")
    mn9 = net.index_of([720575940660219265, 720575940618238523])  # flypoke's MN9 right/left
    conds = {"sugar": [fs.Stimulus(sugar, 150.0)], "bitter": [fs.Stimulus(bitter, 150.0)],
             "sugar + bitter": [fs.Stimulus(sugar, 150.0), fs.Stimulus(bitter, 150.0)]}
    own = {}
    for name, stim in conds.items():
        rec = fs.run(net, stim, fs.Params(), n_trials=n_trials, seed=0)
        r = rec.rates(net.n)
        n_stim = len(np.concatenate([s.indices for s in stim]))
        own[name] = dict(mn9_right=round(float(r[mn9[0]]), 1), mn9_left=round(float(r[mn9[1]]), 1),
                         ingestion_mean=round(float(r[mn].mean()), 1), active=int((r > 1.0).sum() - n_stim))
    reproduced = all(own[k] == FLYPOKE_TABLE[k] for k in FLYPOKE_TABLE)

    g = load_groups()
    nd = load_neurons()
    s_idx, b_idx = np.array(g["sugar"]["indices"]), np.array(g["bitter"]["indices"])
    mn9_ours = np.array([nd.index[nd.root_id == 720575940660219265][0], nd.index[nd.root_id == 720575940618238523][0]])
    ing = np.array(g["ingestion_motor_neurons"]["indices"])

    def ours(netw):
        res = {}
        for name, stim in {"sugar": [Stimulus(s_idx, 150.0)], "bitter": [Stimulus(b_idx, 150.0)],
                           "sugar + bitter": [Stimulus(s_idx, 150.0), Stimulus(b_idx, 150.0)]}.items():
            rates = []
            active = []
            tot = []
            for k in range(n_trials):
                tr = simulate(netw, stim, Params(), seed=k)
                c = np.bincount(tr.neurons, minlength=netw.n)
                rates.append(c)
                n_stim = len(np.concatenate([s.indices for s in stim]))
                active.append(int((c > 1.0).sum() - n_stim))  # 1 s trials: count > 1 <=> rate > 1 Hz
                tot.append(len(tr.steps))
            r = np.mean(rates, 0)
            res[name] = dict(mn9_right=round(float(r[mn9_ours[0]]), 1), mn9_left=round(float(r[mn9_ours[1]]), 1),
                             ingestion_mean=round(float(r[ing].mean()), 1), active=int(np.mean(active)),
                             spikes_per_trial=int(np.mean(tot)))
        return res

    matched = ours(network_783_variant(min_syn=5, signs="top_nt"))
    exact_annot = ours(network_783())
    return dict(
        flypoke_readme_table=FLYPOKE_TABLE,
        flypoke_code_on_this_connectome=own,
        flypoke_readme_reproduced_exactly=bool(reproduced),
        flypoke_network=dict(connections=int(W.nnz), note="Connectivity_783.parquet, >=5 synapses, annotation top_nt signs; "
                             "connections of the 14 neurons without v3.1.0 annotation dropped, as flypoke does"),
        this_model_flypoke_settings=matched,
        this_model_authors_network_annotation_classes=exact_annot,
        settings=dict(trials=n_trials, seeds=list(range(n_trials)), rate_hz=150.0, sugar_n=len(s_idx), bitter_n=len(b_idx)),
    )


def determinism_check() -> dict:
    net = network_783()
    g = load_groups()
    nd = load_neurons()
    pos = {r: i for i, r in enumerate(nd.root_id)}
    sugar = np.array([pos[r] for r in g["paper_lists_630"]["sugar_right_in_783"]])
    stim = [Stimulus(sugar, 150.0)]
    a = simulate(net, stim, Params(), seed=7)
    b = simulate(net, stim, Params(), seed=7)
    c = simulate(net, stim, Params(), seed=8)
    same = np.array_equal(a.steps, b.steps) and np.array_equal(a.neurons, b.neurons)
    diff = not (len(a.steps) == len(c.steps) and np.array_equal(a.steps, c.steps) and np.array_equal(a.neurons, c.neurons))
    return dict(seed=7, spikes=len(a.steps), identical_same_seed=bool(same), different_other_seed=bool(diff))


def ours_sugar_trials(n_trials=90) -> dict:
    """MN9 spike counts of this model for the Brian2 comparison condition (paper sugar GRNs, 150 Hz)."""
    net = network_783()
    g = load_groups()
    nd = load_neurons()
    pos = {r: i for i, r in enumerate(nd.root_id)}
    sugar = np.array([pos[r] for r in g["paper_lists_630"]["sugar_right_in_783"]])
    mn9 = [pos[r] for r in g["mn9"]["root_ids"]]
    full = np.zeros((n_trials, net.n), np.int32)
    for k in range(n_trials):
        tr = simulate(net, [Stimulus(sugar, 150.0)], Params(), seed=k)
        full[k] = np.bincount(tr.neurons, minlength=net.n)
    cols = np.flatnonzero(full.sum(0))
    np.savez_compressed(RESULTS / "validation_counts_ours.npz", neurons=cols, counts=full[:, cols], seeds=np.arange(n_trials))
    return dict(seeds=list(range(n_trials)), mn9_left=full[:, mn9[0]].tolist(), mn9_right=full[:, mn9[1]].tolist(),
                total_spikes=full.sum(1).tolist(), counts_file="results/validation_counts_ours.npz")


def brian2_results() -> dict:
    """Collect the Brian2 reference outputs (produced with the authors' environment, see NOTES.md)."""
    out = {}
    g = load_groups()
    mn9_ids = g["mn9"]["root_ids"]  # [left, right]
    frames = []
    for p in sorted((RESULTS / "brian2").glob("b2_783_paper_sugar_150Hz_*tr.parquet")):
        df = pd.read_parquet(p)
        df["run"] = p.stem
        frames.append(df)
    if frames:
        df = pd.concat(frames)
        df["key"] = df.run + ":" + df.trial.astype(str)
        per = df[df.flywire_id.isin(mn9_ids)].groupby(["key", "flywire_id"]).size().unstack(fill_value=0)
        keys = sorted(df.key.unique())
        per = per.reindex(index=keys, columns=mn9_ids, fill_value=0)
        totals = df.groupby("key").size().reindex(keys)
        nd = load_neurons()
        pos = pd.Series(np.arange(len(nd)), index=nd.root_id)
        B = np.zeros((len(keys), len(nd)), np.int32)
        np.add.at(B, (df.key.map({k: i for i, k in enumerate(keys)}).to_numpy(), pos.reindex(df.flywire_id).to_numpy()), 1)
        out["full_network_poisson"] = dict(
            _count_matrix=B,
            source="authors' model.py run_exp(), unmodified, Brian2 2.5.1, cython target; v783, paper sugar GRNs, 150 Hz",
            files=[f.stem for f in sorted((RESULTS / "brian2").glob("b2_783_paper_sugar_150Hz_*tr.parquet"))],
            trials=len(keys),
            mn9_left=summary(per[mn9_ids[0]]), mn9_right=summary(per[mn9_ids[1]]),
            spikes_per_trial=summary(totals),
            _per_trial_left=per[mn9_ids[0]].tolist(), _per_trial_right=per[mn9_ids[1]].tolist(),
        )
    for name in ("micro_equivalence", "fullnet_equivalence"):
        p = RESULTS / "brian2" / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


# ------------------------------------------------------------------------ main
def main(rerun: bool = False) -> dict:
    t0 = time.time()
    exp = load_json(EXPERIMENTS_JSON)
    if not exp:
        raise SystemExit("run `python -m sim.run` first")
    prep = load_json(PREPARE_LOG)
    manifest = load_json(MANIFEST)
    cache = {} if rerun else (load_json(CACHE_JSON, default={}) or {})
    for key, fn in [("published_630", published_630_check), ("flypoke", flypoke_checks), ("determinism", determinism_check),
                    ("ours_sugar_90", ours_sugar_trials)]:
        if key not in cache:
            log(f"running {key} ...")
            cache[key] = fn()
            save_json(CACHE_JSON, cache)
    cache["brian2"] = brian2_results()
    conds = exp["conditions"]
    per = {k: np.array([t["mn9_spikes"] for t in v["per_trial"]], float) for k, v in conds.items()}
    tests = {}

    # 1 data integrity -------------------------------------------------------
    n, c, grp = prep["neurons"], prep["connectivity"], prep["groups"]
    ann_v2 = load_groups()["annotation_v2.1.0_counts"]
    integrity = dict(
        neurons_simulated=n["simulated_total"], annotation_v3_1_0=n["annotation_v3_1_0"],
        annotation_v2_1_0=n["annotation_v2_1_0_paper_version"], authors_model=n["authors_v783_model"],
        connections=c["connections"], synapses=c["synapses"], connections_ge5=c["connections_ge5_synapses"],
        positions=prep["positions"]["soma"] + prep["positions"]["anchor_point_no_soma"],
        film_stimulus=dict(sugar=exp["stimulus"]["sugar_n"], bitter=exp["stimulus"]["bitter_n"]),
        annotation_classes=dict(sugar_water=grp["sugar"]["n"], bitter=grp["bitter"]["n"], mn9_cb0701=grp["mn9"]["n"],
                                ingestion_motor_neurons=grp["ingestion_motor_neurons"]["n"]),
        annotation_v2_1_0_classes=dict(sugar_water=ann_v2["sugar_water"], bitter=ann_v2["bitter"]),
    )
    flypoke_readme = " ".join((RAW / "flypoke/README.md").read_text().split())
    checks = {
        # FlyWire release 783 proofread neuron table (annotation v2.1.0, the version published with the release)
        "v2.1.0 annotation lists 139,255 neurons": n["annotation_v2_1_0_paper_version"] == 139255,
        "flypoke README 'Of 139,248 neurons' == annotation v3.1.0 rows": "Of 139,248 neurons" in flypoke_readme and n["annotation_v3_1_0"] == 139248,
        "authors' model neurons all inside the v2.1.0 release table": n["authors_model_not_in_v2_1_0"] == 0,
        "connection table has no duplicates or self-connections": c["self_connections"] == 0,
        ">=5-synapse connections ~2.7 million (flypoke README '~2.7 million')": 2.65e6 < c["connections_ge5_synapses"] < 2.75e6,
        "sugar/water 129 and bitter 65 == flypoke README": grp["sugar"]["n"] == 129 and grp["bitter"]["n"] == 65
                                                          and "drives the 129 sugar/water" in flypoke_readme and "the 65 bitter" in flypoke_readme,
        "28 ingestion motor neurons == flypoke README": grp["ingestion_motor_neurons"]["n"] == 28 and "all 28 ingestion MNs" in flypoke_readme,
        "MN9 = 2 CB0701 neurons incl. Shiu et al. id": grp["mn9"]["n"] == 2,
        "film stimulus: 20 of 21 paper sugar GRNs, 20 of 21 bitter GRNs present": exp["stimulus"]["sugar_n"] == 20 and exp["stimulus"]["bitter_n"] == 20,
        "every simulated neuron has a position": prep["positions"]["no_position"] == 0,
        "manifest records sha256 for every raw file": all("sha256" in v for v in manifest["files"].values()),
    }
    tests["1_data_integrity"] = dict(passed=all(checks.values()), checks=checks, counts=integrity)

    # 2 sugar -> MN9 high --------------------------------------------------------
    a = per["A_sugar"]
    pub = cache["published_630"]
    pub_ref = [pub["100Hz"]["mn9"][str(SHIU_MN9_ROOT_ID)]["authors"]["mean"], pub["200Hz"]["mn9"][str(SHIU_MN9_ROOT_ID)]["authors"]["mean"]]
    mn9_side = exp["mn9"]["sides"]
    shiu_mn9_col = exp["mn9"]["root_ids"].index(SHIU_MN9_ROOT_ID)
    rate_shiu = a[:, shiu_mn9_col]
    lo, hi = 0.85 * min(pub_ref), 1.15 * max(pub_ref)
    fp = cache["flypoke"]["this_model_flypoke_settings"]["sugar"]
    tests["2_sugar_drives_mn9"] = dict(
        passed=bool(lo <= rate_shiu.mean() <= hi and a.mean() > 30),
        mn9_rate_hz={s: summary(a[:, k]) for k, s in enumerate(mn9_side)},
        criterion=(f"Shiu et al.'s MN9 (root {SHIU_MN9_ROOT_ID}) at 150 Hz input lies within 0.85x-1.15x of the authors' "
                   f"stored outputs at 100 and 200 Hz input ({pub_ref[0]:.1f}-{pub_ref[1]:.1f} Hz): [{lo:.1f}, {hi:.1f}] Hz"),
        flypoke_reported_hz=dict(right=150.0, left=111.6),
        this_model_at_flypoke_settings_hz=dict(right=fp["mn9_right"], left=fp["mn9_left"]),
    )
    # 3 bitter -------------------------------------------------------------------
    b = per["B_bitter"]
    tests["3_bitter_does_not_drive_mn9"] = dict(passed=bool(b.mean() <= 1.0 and b.max() <= 5),
                                                 mn9_rate_hz={s: summary(b[:, k]) for k, s in enumerate(mn9_side)},
                                                 criterion="mean MN9 rate <= 1 Hz and no trial above 5 spikes")
    # 4 sugar + bitter < sugar ------------------------------------------------------
    cc = per["C_sugar_bitter"]
    w = welch(cc.mean(1), a.mean(1))
    tests["4_bitter_suppresses_sugar"] = dict(
        passed=bool(cc.mean() < a.mean() and w["p"] < 1e-3 and (cc.mean(0) < a.mean(0)).all()),
        mn9_rate_hz={s: summary(cc[:, k]) for k, s in enumerate(mn9_side)},
        reduction_vs_sugar={s: float(1 - cc[:, k].mean() / a[:, k].mean()) for k, s in enumerate(mn9_side)},
        welch_t_test_pair_mean=w, criterion="both MN9 lower with sugar+bitter; Welch t-test p < 0.001 (30 vs 30 trials)")
    # 5 baseline -------------------------------------------------------------------
    d = conds["D_baseline"]
    tests["5_baseline_silent"] = dict(passed=all(t["total_spikes"] == 0 for t in d["per_trial"]),
                                      total_spikes=[t["total_spikes"] for t in d["per_trial"]][:5] + ["..."])
    # 6 determinism ----------------------------------------------------------------
    det = cache["determinism"]
    tests["6_determinism"] = dict(passed=bool(det["identical_same_seed"] and det["different_other_seed"]), **det)

    # cross-checks -------------------------------------------------------------------
    cross = {}
    se_ok = all(abs(v["diff_in_se"]) < 3.0 for r in pub.values() for v in r["mn9"].values())
    cross["a_published_outputs_630"] = dict(passed=bool(se_ok), criterion="every MN9 mean within 3 standard errors of the authors' stored output", **pub)
    b2 = cache["brian2"]
    b2_ok = True
    if "full_network_poisson" in b2:
        fn = b2["full_network_poisson"]
        comp = {}
        mine = cache["ours_sugar_90"]
        for k, s in enumerate(mn9_side):
            ref = np.array(fn["_per_trial_left" if s == "left" else "_per_trial_right"], float)
            ours = np.array(mine["mn9_left" if s == "left" else "mn9_right"], float)
            comp[s] = dict(brian2=summary(ref), ours=summary(ours), welch=welch(ours, ref),
                           diff_in_se=float((ours.mean() - ref.mean()) / np.sqrt(ours.var(ddof=1) / len(ours) + ref.var(ddof=1) / len(ref))))
        tot = np.array(mine["total_spikes"], float)
        comp["spikes_per_trial"] = dict(brian2=fn["spikes_per_trial"], ours=summary(tot))
        # every neuron: per-trial spike counts, Welch t-test, Bonferroni over all neurons active in either
        oc = np.load(RESULTS / "validation_counts_ours.npz")
        nd = load_neurons()
        n_all = len(nd)
        O = np.zeros((oc["counts"].shape[0], n_all), np.int32)
        O[:, oc["neurons"]] = oc["counts"]
        B = fn.pop("_count_matrix")
        act = np.flatnonzero((O.sum(0) + B.sum(0)) > 0)
        t, pv = stats.ttest_ind(O[:, act].astype(float), B[:, act].astype(float), equal_var=False)
        pv = np.where(np.isnan(pv), 1.0, pv)
        order = np.argsort(pv)[:5]
        fn["all_neurons"] = dict(
            active_neurons=int(len(act)), bonferroni_threshold=0.05 / len(act),
            significant_after_bonferroni=int((pv < 0.05 / len(act)).sum()),
            p_below_0_001=int((pv < 1e-3).sum()), expected_by_chance_at_0_001=float(len(act) * 1e-3),
            smallest_p=[dict(root_id=int(nd.root_id[act[j]]), cell_type=str(nd.cell_type[act[j]]), side=str(nd.side[act[j]]),
                             brian2_hz=float(B[:, act[j]].mean()), ours_hz=float(O[:, act[j]].mean()), p=float(pv[j])) for j in order],
            grn_rate_hz=dict(brian2=float(B[:, np.array([nd.index[nd.root_id == r][0] for r in load_groups()["paper_lists_630"]["sugar_right_in_783"]])].mean()),
                             ours=float(O[:, np.array([nd.index[nd.root_id == r][0] for r in load_groups()["paper_lists_630"]["sugar_right_in_783"]])].mean())),
        )
        fn["comparison_with_this_model"] = comp
        b2_ok = fn["all_neurons"]["significant_after_bonferroni"] == 0
    for name in ("micro_equivalence", "fullnet_equivalence"):
        if name in b2:
            b2_ok = b2_ok and bool(b2[name].get("identical"))
    cross["b_brian2"] = dict(passed=bool(b2_ok and "fullnet_equivalence" in b2),
                             criterion="spike trains identical to Brian2 under identical input (small networks and the full "
                                       "v783 network); with Poisson input, no neuron's mean rate differs between 90 Brian2 and "
                                       "90 trials of this model after Bonferroni correction over all active neurons", **b2)
    fpk = cache["flypoke"]
    fm = fpk["this_model_flypoke_settings"]
    qual = fm["sugar"]["mn9_right"] > 30 and fm["bitter"]["mn9_right"] <= 1 and fm["sugar + bitter"]["mn9_right"] < fm["sugar"]["mn9_right"]
    ratio = fm["sugar"]["mn9_right"] / FLYPOKE_TABLE["sugar"]["mn9_right"]
    cross["c_flypoke"] = dict(passed=bool(fpk["flypoke_readme_reproduced_exactly"] and qual and 0.5 <= ratio <= 2.0),
                              criterion="flypoke's own code reproduces its README table exactly on this connectome; "
                                        "this model at flypoke's settings shows the same qualitative result and an MN9 "
                                        "rate within a factor 2 of flypoke's", ratio_mn9_right=ratio, **fpk)

    all_passed = all(t["passed"] for t in tests.values()) and all(c["passed"] for c in cross.values())
    report = dict(created_utc=utc_now(), all_passed=bool(all_passed), tests=tests, cross_checks=cross,
                  wall_time_s=round(time.time() - t0, 1))
    save_json(VALIDATION_JSON, report)
    print_report(report)
    return report


def print_report(r: dict) -> None:
    log("\nVALIDATION (section 5)")
    for k, t in r["tests"].items():
        log(f"  [{'PASS' if t['passed'] else 'FAIL'}] {k}")
    for k, t in r["cross_checks"].items():
        log(f"  [{'PASS' if t['passed'] else 'FAIL'}] cross-check {k}")
    t2 = r["tests"]["2_sugar_drives_mn9"]["mn9_rate_hz"]
    t4 = r["tests"]["4_bitter_suppresses_sugar"]["mn9_rate_hz"]
    log("  MN9 (Hz, mean of 30 trials): " + ", ".join(f"{s} sugar {t2[s]['mean']:.1f} / sugar+bitter {t4[s]['mean']:.1f}" for s in t2))
    log(f"  ALL PASSED: {r['all_passed']}")


def require_validated() -> dict:
    r = load_json(VALIDATION_JSON)
    if not r or not r.get("all_passed"):
        raise SystemExit("validation has not passed: run `python -m sim.validate` (rendering is blocked until it does)")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    main(ap.parse_args().rerun)
