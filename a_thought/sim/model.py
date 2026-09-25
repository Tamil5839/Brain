"""Whole-brain leaky integrate-and-fire model of Shiu et al. 2024, in NumPy/SciPy.

This reproduces the Brian2 model in the authors' public code
(github.com/philshiu/Drosophila_brain_model, model.py, commit 91bdd1e) step by
step. Every constant below is copied from ``default_params`` in that file; the
time step is Brian2's default clock (0.1 ms), which model.py does not change.

Brian2 model (verbatim from model.py):

    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    threshold  v > v_th            reset  v = v_rst; g = 0 mV
    refractory rfc  (t_rfc = 2.2 ms; 0 ms for Poisson-stimulated neurons)
    Synapses(on_pre='g += w', delay=t_dly),  w = (sign x synapse count) * w_syn
    PoissonInput(target_var='v', N=1, rate=r_poi, weight=w_syn * f_poi)
    NeuronGroup(method='linear')  -> exact integration of the linear system

What one simulation step does, in Brian2's default schedule order
(start, groups, thresholds, synapses, resets, end):

  groups      not_refractory = timestep(t - lastspike) >= timestep(rfc)
              exact update of (v, g) over dt for non-refractory neurons;
              refractory neurons are frozen ('unless refractory' multiplies
              both right-hand sides by not_refractory)
  thresholds  spike if v > v_th and not refractory; lastspike = t
  synapses    spikes emitted round(t_dly/dt) = 18 steps ago:  g_post += w,
              but only for postsynaptic neurons that are not refractory:
              Brian2 masks synaptic updates of '(unless refractory)' variables
              (generated code: add.at(g, post[not_refractory], w[not_refractory])),
              so input arriving during the refractory period is discarded
              Poisson input: v += w_syn * f_poi with probability rate * dt
  resets      spiking neurons: v = v_rst, g = 0

The state is stored as u = v - v_0 (mV). With v_rst = v_0 the reset value is
u = 0. Exact integration over one step (T = t_mbr):

  u' = u e^{-dt/T} + g * tau/(tau - T) * (e^{-dt/tau} - e^{-dt/T})
  g' = g e^{-dt/tau}

Synaptic input uses a scipy.sparse CSR matrix (rows = presynaptic neurons);
the membrane update is vectorised NumPy over all neurons.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from math import exp

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Params:
    """Constants of Shiu et al. model.py ``default_params`` (units: ms, mV, Hz)."""

    t_run: float = 1000.0   # 't_run'  : 1000 * ms   duration of trial
    n_run: int = 30         # 'n_run'  : 30          number of runs
    v_0: float = -52.0      # 'v_0'    : -52 * mV    resting potential
    v_rst: float = -52.0    # 'v_rst'  : -52 * mV    reset potential after spike
    v_th: float = -45.0     # 'v_th'   : -45 * mV    threshold for spiking
    t_mbr: float = 20.0     # 't_mbr'  :  20 * ms    membrane time scale
    tau: float = 5.0        # 'tau'    :   5 * ms    synaptic time constant
    t_rfc: float = 2.2      # 't_rfc'  : 2.2 * ms    refractory period
    t_dly: float = 1.8      # 't_dly'  : 1.8 * ms    synaptic delay
    w_syn: float = 0.275    # 'w_syn'  : .275 * mV   weight per synapse
    r_poi: float = 150.0    # 'r_poi'  : 150 * Hz    rate of the Poisson inputs
    r_poi2: float = 0.0     # 'r_poi2' :   0 * Hz    rate of a 2nd class of Poisson inputs
    f_poi: float = 250.0    # 'f_poi'  : 250         scaling factor for Poisson synapse
    dt: float = 0.1         # Brian2 defaultclock.dt (not set in model.py)

    def as_dict(self) -> dict:
        return asdict(self)


def brian2_timestep(t: float, dt: float) -> int:
    """brian2.core.functions.timestep: int((t + 1e-3*dt) / dt)."""
    return int((t + 1e-3 * dt) / dt)


@dataclass
class Network:
    """Signed connectivity. W[pre, post] = sign(pre) * synapse count."""

    W: sp.csr_matrix
    root_ids: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @classmethod
    def from_arrays(cls, pre, post, signed_count, n, root_ids=None, **info) -> "Network":
        W = sp.csr_matrix((np.asarray(signed_count, np.float64), (pre, post)), shape=(n, n))
        W.sum_duplicates()
        W.sort_indices()
        return cls(W, root_ids, dict(info))


@dataclass
class Stimulus:
    """Neurons driven by independent Poisson input (Brian2 PoissonInput, N=1)."""

    indices: np.ndarray
    rate: float  # Hz
    label: str = ""


@dataclass
class Trial:
    steps: np.ndarray    # int32 time-step index of each spike (time = step * dt)
    neurons: np.ndarray  # int32 neuron index of each spike
    dt: float
    n_steps: int
    seed: int | None
    v_trace: np.ndarray | None = None  # (steps, probed neurons) membrane potential in mV, if requested

    @property
    def times_ms(self) -> np.ndarray:
        return self.steps * self.dt

    def counts(self, n: int) -> np.ndarray:
        return np.bincount(self.neurons, minlength=n)


def poisson_schedule(stimuli: list[Stimulus], n_steps: int, dt: float, rng: np.random.Generator):
    """Pre-draw the Poisson input events (step, position-in-target-list).

    Used only for implementation-equivalence tests (the same events can be fed
    to Brian2). ``simulate`` draws on the fly with the same distribution.
    """
    targets = np.concatenate([s.indices for s in stimuli])
    p = np.concatenate([np.full(len(s.indices), s.rate * dt * 1e-3) for s in stimuli])
    hits = rng.random((n_steps, len(targets))) < p
    steps, pos = np.nonzero(hits)
    return steps.astype(np.int32), targets[pos].astype(np.int32)


def simulate(
    net: Network,
    stimuli: list[Stimulus],
    params: Params = Params(),
    seed: int | None = 0,
    t_run: float | None = None,
    events: tuple[np.ndarray, np.ndarray] | None = None,
    probe: np.ndarray | None = None,
) -> Trial:
    """Run one trial. Returns every spike of every neuron.

    ``events`` optionally replaces the random Poisson draws by a fixed list of
    (step, neuron) input events, e.g. from ``poisson_schedule``.
    ``probe`` optionally records v (mV) of these neurons at the end of every
    step (after resets), like a Brian2 StateMonitor(when='end').
    """
    p = params
    dt = p.dt
    N = net.n
    W = net.W
    indptr, indices = W.indptr, W.indices
    weights = W.data * p.w_syn  # mV per spike, sign * count * w_syn

    t_run = p.t_run if t_run is None else t_run
    n_steps = brian2_timestep(t_run, dt)
    delay = int(np.round(p.t_dly / dt))          # SpikeQueue: np.round(delay / dt) = 18
    ref_steps = brian2_timestep(p.t_rfc, dt)     # timestep(rfc, dt) = 22
    A = exp(-dt / p.t_mbr)
    C = exp(-dt / p.tau)
    B = p.tau / (p.tau - p.t_mbr) * (C - A)
    u_th = p.v_th - p.v_0
    u_rst = p.v_rst - p.v_0
    kick = p.w_syn * p.f_poi                     # PoissonInput weight = w_syn * f_poi

    targets = np.concatenate([s.indices for s in stimuli]).astype(np.int64) if stimuli else np.empty(0, np.int64)
    if len(np.unique(targets)) != len(targets):
        raise ValueError("a neuron appears in two stimulus groups")
    probs = np.concatenate([np.full(len(s.indices), s.rate * dt * 1e-3) for s in stimuli]) if stimuli else np.empty(0)
    is_target = np.zeros(N, bool)
    is_target[targets] = True                    # neu[i].rfc = 0 ms for Poisson targets

    if events is not None:
        ev_steps, ev_neurons = events
        order = np.argsort(ev_steps, kind="stable")
        ev_steps, ev_neurons = ev_steps[order], ev_neurons[order]
        ev_bounds = np.searchsorted(ev_steps, np.arange(n_steps + 1))
        rng = None
    else:
        rng = np.random.default_rng(seed)

    u = np.zeros(N)                              # v - v_0, mV
    g = np.zeros(N)                              # mV
    tmp = np.empty(N)
    lastspike = np.full(N, -(10 ** 9), np.int64)  # step of the last spike
    rfc = np.where(is_target, 0, ref_steps)      # refractory period in steps
    in_flight: deque[np.ndarray] = deque(np.empty(0, np.int64) for _ in range(delay))
    refractory: deque[np.ndarray] = deque(np.empty(0, np.int64) for _ in range(ref_steps - 1))
    out_steps, out_neurons = [], []
    trace = np.empty((n_steps, len(probe))) if probe is not None else None

    for n in range(n_steps):
        # ---- groups: exact integration, refractory neurons frozen
        frozen = np.concatenate(refractory) if ref_steps > 1 else np.empty(0, np.int64)
        if len(frozen):
            u_keep, g_keep = u[frozen], g[frozen]
        np.multiply(g, B, out=tmp)
        u *= A
        u += tmp
        g *= C
        if len(frozen):
            u[frozen] = u_keep
            g[frozen] = g_keep

        # ---- thresholds (a frozen neuron sits at u_rst < u_th and cannot fire)
        spk = np.flatnonzero(u > u_th)
        lastspike[spk] = n

        # ---- synapses: delayed synaptic events, then Poisson input
        arriving = in_flight.popleft() if delay > 0 else spk
        if len(arriving):
            starts, ends = indptr[arriving], indptr[arriving + 1]
            lens = ends - starts
            total = int(lens.sum())
            if total:
                pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
                post, w = indices[pos], weights[pos]
                open_ = (n - lastspike[post]) >= rfc[post]  # not_refractory of the target
                post, w = post[open_], w[open_]
                if len(post) < 20000:
                    np.add.at(g, post, w)
                else:
                    g += np.bincount(post, weights=w, minlength=N)
        if events is not None:
            kicked = ev_neurons[ev_bounds[n]:ev_bounds[n + 1]]
        elif len(targets):
            kicked = targets[rng.random(len(targets)) < probs]
        else:
            kicked = targets
        if len(kicked):
            u[kicked] += kick

        # ---- resets
        if len(spk):
            u[spk] = u_rst
            g[spk] = 0.0
            out_steps.append(np.full(len(spk), n, np.int32))
            out_neurons.append(spk.astype(np.int32))
        if delay > 0:
            in_flight.append(spk)
        if ref_steps > 1:
            refractory.popleft()
            refractory.append(spk[~is_target[spk]])
        if trace is not None:
            trace[n] = u[probe] + p.v_0

    if out_steps:
        steps, neurons = np.concatenate(out_steps), np.concatenate(out_neurons)
    else:
        steps, neurons = np.empty(0, np.int32), np.empty(0, np.int32)
    return Trial(steps=steps, neurons=neurons, dt=dt, n_steps=n_steps, seed=seed, v_trace=trace)


def rates_hz(trial: Trial, n: int) -> np.ndarray:
    return trial.counts(n) / (trial.n_steps * trial.dt * 1e-3)
