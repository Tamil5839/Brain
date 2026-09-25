"""Full-network, fixed-input comparison with the authors' Brian2 code.

Uses the authors' create_model() (unchanged) to build the v783 network. The
Poisson input is replaced, in both implementations, by the same pre-drawn input
events, delivered through a SpikeGeneratorGroup -> 'v += w_syn * f_poi' synapse
(same schedule slot as PoissonInput). Reports spike-train agreement.
Usage: python tests/brian2_fullnet_equivalence.py [t_run_ms] [seed]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/raw/shiu_model"))


def main(t_run_ms=1000.0, seed=0):
    import brian2 as b2
    import model as shiu

    from sim.model import Params, Stimulus, poisson_schedule, simulate
    from sim.networks import index_of, network_shiu_files

    comp = ROOT / "data/raw/shiu_model/Completeness_783.csv"
    con = ROOT / "data/raw/shiu_model/Connectivity_783.parquet"
    g = json.loads((ROOT / "data/processed/groups.json").read_text())
    net = network_shiu_files(comp, con)                      # authors' index order
    sugar = index_of(net, g["paper_lists_630"]["sugar_right_in_783"])
    mn9 = index_of(net, g["mn9"]["root_ids"])
    p = Params()
    stim = [Stimulus(sugar, 150.0)]
    ev = poisson_schedule(stim, int(round(t_run_ms / p.dt)), p.dt, np.random.default_rng(seed))
    t0 = time.time()
    ours = simulate(net, stim, p, events=ev, t_run=t_run_ms)
    t_ours = time.time() - t0

    params = dict(shiu.default_params)
    t0 = time.time()
    neu, syn, mon = shiu.create_model(str(comp), str(con), params)
    for i in sugar:
        neu[int(i)].rfc = 0 * b2.ms
    gen = b2.SpikeGeneratorGroup(net.n, ev[1], ev[0] * p.dt * b2.ms)
    kick = b2.Synapses(gen, neu, on_pre="v += w_syn * f_poi", namespace=params)
    kick.connect(j="i")
    b2.Network(neu, syn, mon, gen, kick).run(t_run_ms * b2.ms)
    t_b2 = time.time() - t0
    b_steps = np.round(np.asarray(mon.t / b2.ms) / p.dt).astype(np.int64)
    b_neur = np.asarray(mon.i).astype(np.int64)

    def keyset(s, n):
        return set(zip(s.tolist(), n.tolist()))
    A, B = keyset(ours.steps.astype(np.int64), ours.neurons.astype(np.int64)), keyset(b_steps, b_neur)
    only_a, only_b = sorted(A - B), sorted(B - A)
    first = min(only_a[:1] + only_b[:1]) if (only_a or only_b) else None
    res = dict(
        t_run_ms=t_run_ms, seed=seed, input_events=len(ev[0]),
        spikes_ours=len(A), spikes_brian2=len(B), common=len(A & B),
        only_ours=len(only_a), only_brian2=len(only_b), first_difference=first,
        identical=(not only_a and not only_b),
        mn9_ours=[int((ours.neurons == m).sum()) for m in mn9],
        mn9_brian2=[int((b_neur == m).sum()) for m in mn9],
        wall_s=dict(ours=round(t_ours, 1), brian2=round(t_b2, 1)),
    )
    print(json.dumps(res), flush=True)
    out = ROOT / "results/brian2/fullnet_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res["description"] = ("authors' create_model() on Connectivity_783.parquet in Brian2 vs sim.model, paper sugar GRNs, "
                          "identical fixed input events (SpikeGeneratorGroup -> v += w_syn*f_poi)")
    out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
