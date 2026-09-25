"""Unit tests for sim.model (no FlyWire data needed except where noted)."""
from __future__ import annotations

import re
import sys
from math import exp
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.model import Network, Params, Stimulus, poisson_schedule, simulate  # noqa: E402

AUTHORS_MODEL = ROOT / "data/raw/shiu_model/model.py"


def small_net(n=300, seed=1, density=12):
    rng = np.random.default_rng(seed)
    m = n * density
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = pre != post
    pairs = np.unique(np.stack([pre[keep], post[keep]], 1), axis=0)
    count = np.minimum(rng.geometric(0.12, len(pairs)), 80)
    sign = np.where(rng.random(n) < 0.3, -1, 1)
    return Network.from_arrays(pairs[:, 0], pairs[:, 1], count * sign[pairs[:, 0]], n)


@pytest.mark.skipif(not AUTHORS_MODEL.exists(), reason="run fetch.py first")
def test_parameters_are_the_authors():
    """Every constant in Params equals default_params in the authors' model.py (units converted)."""
    src = AUTHORS_MODEL.read_text()
    unit = {"ms": 1.0, "mV": 1.0, "Hz": 1.0}
    found = {}
    for key, num, u in re.findall(r"'(\w+)'\s*:\s*([-\d.]+)\s*\*\s*(ms|mV|Hz)", src):
        found[key] = float(num) * unit[u]
    found["n_run"] = float(re.search(r"'n_run'\s*:\s*(\d+)", src).group(1))
    found["f_poi"] = float(re.search(r"'f_poi'\s*:\s*(\d+)", src).group(1))
    p = Params().as_dict()
    assert set(found) == set(p) - {"dt"}, set(found) ^ (set(p) - {"dt"})
    for k, v in found.items():
        assert p[k] == pytest.approx(v), k
    assert "defaultclock" not in src  # Brian2 default dt (0.1 ms) applies
    assert "dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)" in src
    assert "dg/dt = -g / tau               : volt (unless refractory)" in src
    assert "'eq_th'     : 'v > v_th'" in src
    assert "method='linear'" in src


def test_exact_integration_single_synapse():
    """One input event -> the postsynaptic trace equals the analytic solution of the linear ODEs."""
    p = Params()
    net = Network.from_arrays([0], [1], [10], 2)
    ev = (np.array([10], np.int32), np.array([0], np.int32))
    tr = simulate(net, [Stimulus(np.array([0]), 150.0)], p, events=ev, t_run=6.0, probe=np.arange(2))
    assert tr.steps.tolist() == [11] and tr.neurons.tolist() == [0]  # kick at step 10 -> spike at step 11
    g0 = 10 * p.w_syn
    arrival = 11 + round(p.t_dly / p.dt)                              # 18 steps later
    for k in range(arrival, 60):
        t = (k - arrival) * p.dt
        u = g0 * p.tau / (p.tau - p.t_mbr) * (exp(-t / p.tau) - exp(-t / p.t_mbr))
        assert tr.v_trace[k, 1] == pytest.approx(p.v_0 + u, abs=1e-9)
    assert (tr.v_trace[:arrival, 1] == p.v_0).all()


def test_refractory_blocks_and_freezes():
    """A neuron is frozen at v_rst for 21 steps after a spike (t_rfc = 2.2 ms), and synaptic
    input arriving in that window is discarded (Brian2 masks synaptic updates of
    '(unless refractory)' variables), while the same input arriving later is not."""
    p = Params()
    net = Network.from_arrays([0, 2], [1, 1], [10, 400], 3)          # two Poisson targets drive neuron 1
    stim = [Stimulus(np.array([0, 2]), 150.0)]

    def run(extra_kick_step=None):
        steps, neurons = [0, 0], [0, 2]
        if extra_kick_step is not None:
            steps += [extra_kick_step] * 2
            neurons += [0, 2]
        ev = (np.array(steps, np.int32), np.array(neurons, np.int32))
        return simulate(net, stim, p, events=ev, t_run=12.0, probe=np.arange(3))

    base = run()
    s = int(base.steps[base.neurons == 1][0])
    assert s > 19                                                     # first input arrives at step 1 + 18
    assert (base.v_trace[s:s + 22, 1] == p.v_rst).all()               # reset, then frozen for 21 steps
    assert base.v_trace[s + 22, 1] == p.v_rst                          # nothing arrived: stays at rest
    # a second volley arriving 5 steps after the spike (refractory) has no effect at all
    during = run(extra_kick_step=s + 5 - 19)
    assert np.array_equal(during.v_trace[:, 1], base.v_trace[:, 1])
    # the same volley arriving 25 steps after the spike (not refractory) does
    after = run(extra_kick_step=s + 25 - 19)
    assert not np.array_equal(after.v_trace[:, 1], base.v_trace[:, 1])


def test_baseline_is_silent():
    tr = simulate(small_net(), [], Params(), seed=0, t_run=200.0)
    assert len(tr.steps) == 0


def test_determinism():
    net = small_net()
    stim = [Stimulus(np.arange(20), 150.0)]
    a = simulate(net, stim, Params(), seed=3, t_run=300.0)
    b = simulate(net, stim, Params(), seed=3, t_run=300.0)
    c = simulate(net, stim, Params(), seed=4, t_run=300.0)
    assert np.array_equal(a.steps, b.steps) and np.array_equal(a.neurons, b.neurons)
    assert not (len(a.steps) == len(c.steps) and np.array_equal(a.steps, c.steps))


def test_poisson_targets_fire_at_rate():
    """Each Poisson event forces one spike one step later; an event in the step of a spike is lost."""
    net = Network.from_arrays([0], [1], [1], 200)
    stim = [Stimulus(np.arange(200), 150.0)]
    tr = simulate(net, stim, Params(), seed=0, t_run=1000.0)
    rate = len(tr.steps) / 200
    p = 150 * 1e-4
    assert rate == pytest.approx(150 * (1 - p), rel=0.03)


def test_schedule_matches_random_draws():
    """poisson_schedule and on-the-fly draws have the same statistics."""
    p = Params()
    stim = [Stimulus(np.arange(50), 150.0)]
    ev = poisson_schedule(stim, 10000, p.dt, np.random.default_rng(1))
    assert len(ev[0]) / 50 == pytest.approx(150.0, rel=0.03)


@pytest.mark.skipif(not (ROOT / "results/prepare_log.json").exists(), reason="run prepare.py first")
def test_data_integrity_counts():
    import json

    r = json.loads((ROOT / "results/prepare_log.json").read_text())
    assert r["neurons"]["annotation_v2_1_0_paper_version"] == 139255
    assert r["neurons"]["simulated_total"] == 139262
    assert r["connectivity"]["connections"] == 15091983
    assert r["connectivity"]["synapses"] == 54492922
    assert r["positions"]["no_position"] == 0
    assert r["groups"]["sugar"]["n"] == 129 and r["groups"]["bitter"]["n"] == 65
    assert r["groups"]["mn9"]["n"] == 2


@pytest.mark.skipif(not (ROOT / "results/validation.json").exists(), reason="run sim.validate first")
def test_validation_passed():
    import json

    v = json.loads((ROOT / "results/validation.json").read_text())
    failed = [k for k, t in {**v["tests"], **v["cross_checks"]}.items() if not t["passed"]]
    assert v["all_passed"], failed
