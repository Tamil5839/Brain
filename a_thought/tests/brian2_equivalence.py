"""Step-level equivalence of sim.model.simulate with the authors' Brian2 model.

Builds a small random network with Brian2 using the equations, threshold, reset,
refractoriness, integration method and parameters of the authors' model.py, and
the same network in sim.model. Poisson input is replaced in *both* by the same
fixed list of input events (Brian2: SpikeGeneratorGroup -> v += w_syn*f_poi in
the 'synapses' slot, exactly where PoissonInput acts). Returns max |dv| and
whether the spike trains are identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/raw/shiu_model"))


def run(n=80, n_targets=6, t_run_ms=400.0, rate=150.0, seed=3, codegen="numpy"):
    import brian2 as b2
    import model as shiu  # authors' model.py

    from sim.model import Network, Params, Stimulus, poisson_schedule, simulate

    b2.prefs.codegen.target = codegen
    b2.defaultclock.dt = 0.1 * b2.ms
    params = dict(shiu.default_params)
    rng = np.random.default_rng(seed)
    # random signed connectivity with a heavy-tailed synapse-count distribution
    m = int(n * 12)
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    pairs = np.unique(np.stack([pre, post], 1), axis=0)
    pre, post = pairs[:, 0], pairs[:, 1]
    count = np.minimum(rng.geometric(0.12, len(pre)), 80)
    sign = np.where(rng.random(n) < 0.3, -1, 1)
    signed = count * sign[pre]
    targets = np.arange(n_targets)

    p = Params()
    net = Network.from_arrays(pre, post, signed, n)
    stim = [Stimulus(targets, rate)]
    ev = poisson_schedule(stim, int(round(t_run_ms / p.dt)), p.dt, np.random.default_rng(seed + 1))
    probe = np.arange(n)
    ours = simulate(net, stim, p, events=ev, t_run=t_run_ms, probe=probe)

    # ---- Brian2, mirroring create_model() + poi()
    neu = b2.NeuronGroup(N=n, model=params["eqs"], method="linear", threshold=params["eq_th"],
                         reset=params["eq_rst"], refractory="rfc", name="default_neurons", namespace=params)
    neu.v = params["v_0"]
    neu.g = 0
    neu.rfc = params["t_rfc"]
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"], name="default_synapses")
    syn.connect(i=pre, j=post)
    syn.w = signed * params["w_syn"]
    for i in targets:
        neu[int(i)].rfc = 0 * b2.ms
    gen = b2.SpikeGeneratorGroup(n, ev[1], ev[0] * p.dt * b2.ms)
    kick = b2.Synapses(gen, neu, on_pre="v += w_syn * f_poi", namespace=params)  # delay 0 -> same step
    kick.connect(j="i")
    mon = b2.SpikeMonitor(neu)
    vm = b2.StateMonitor(neu, "v", record=True, when="end")
    b2.Network(neu, syn, gen, kick, mon, vm).run(t_run_ms * b2.ms)

    b_steps = np.round(np.asarray(mon.t / b2.ms) / p.dt).astype(int)
    b_neur = np.asarray(mon.i)
    o = np.lexsort((ours.neurons, ours.steps))
    b = np.lexsort((b_neur, b_steps))
    same = len(o) == len(b) and np.array_equal(ours.steps[o], b_steps[b]) and np.array_equal(ours.neurons[o], b_neur[b])
    dv = np.abs(ours.v_trace - np.asarray(vm.v / b2.mV).T).max()
    return dict(spikes_ours=len(o), spikes_brian2=len(b), identical_spikes=bool(same), max_abs_dv_mV=float(dv),
                n_input_events=len(ev[0]))


if __name__ == "__main__":
    import json
    runs = [run(seed=s) for s in range(3)]
    for r in runs:
        print(r, flush=True)
    res = dict(description="random 80-neuron networks, authors' equations/parameters in Brian2 vs sim.model, "
                           "identical fixed input events, 400 ms each", runs=runs,
               identical=all(r["identical_spikes"] and r["max_abs_dv_mV"] < 1e-9 for r in runs))
    out = ROOT / "results/brian2/micro_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
