"""Build the networks used by the experiments and the cross-checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONNECTIVITY_NPZ, GROUPS_JSON, NEURONS_PARQUET, RAW  # noqa: E402

from .model import Network  # noqa: E402

# Sign convention used by flypoke (and stated there as following Shiu et al.):
# GABA and glutamate inhibitory, all other predicted transmitters excitatory.
FLYPOKE_NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
                   "gaba": -1.0, "glutamate": -1.0}


def load_neurons() -> pd.DataFrame:
    return pd.read_parquet(NEURONS_PARQUET)


def load_groups() -> dict:
    return json.loads(GROUPS_JSON.read_text())


def network_783() -> Network:
    """Main network: the authors' v783 connection table, unthresholded, their signs."""
    W = sp.load_npz(CONNECTIVITY_NPZ).tocsr().astype(np.float64)
    W.sort_indices()
    ids = load_neurons().root_id.to_numpy()
    return Network(W, ids, dict(name="shiu_783", source="Connectivity_783.parquet (Shiu et al.), no threshold"))


def network_783_variant(min_syn: int = 1, signs: str = "authors") -> Network:
    """Same connections with a synapse threshold and/or signs from annotation top_nt.

    signs='top_nt' uses the per-neuron predicted transmitter from the FlyWire
    annotation (flypoke's convention; unknown -> excitatory).
    """
    W = sp.load_npz(CONNECTIVITY_NPZ).tocsr().astype(np.float64)
    W.data[np.abs(W.data) < min_syn] = 0.0
    if signs == "top_nt":
        nt = load_neurons().top_nt.map(FLYPOKE_NT_SIGN).fillna(1.0).to_numpy()
        W = sp.csr_matrix(sp.diags(nt) @ abs(W))
    elif signs != "authors":
        raise ValueError(signs)
    W.eliminate_zeros()
    W.sort_indices()
    ids = load_neurons().root_id.to_numpy()
    return Network(W, ids, dict(name=f"783_min{min_syn}_{signs}", min_syn=min_syn, signs=signs))


def network_shiu_files(completeness_csv: Path, connectivity_parquet: Path) -> Network:
    """Network exactly as the authors' create_model() builds it (their index order)."""
    comp = pd.read_csv(completeness_csv, index_col=0)
    con = pd.read_parquet(connectivity_parquet,
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    net = Network.from_arrays(con.Presynaptic_Index.to_numpy(), con.Postsynaptic_Index.to_numpy(),
                              con["Excitatory x Connectivity"].to_numpy(), len(comp),
                              root_ids=comp.index.to_numpy(), source=str(connectivity_parquet.name))
    return net


def network_630() -> Network:
    return network_shiu_files(RAW / "shiu_model/2023_03_23_completeness_630_final.csv",
                              RAW / "shiu_model/2023_03_23_connectivity_630_final.parquet")


def index_of(net: Network, root_ids) -> np.ndarray:
    pos = pd.Index(net.root_ids).get_indexer(np.asarray(root_ids, np.int64))
    if (pos < 0).any():
        raise KeyError(f"root ids not in network: {np.asarray(root_ids)[pos < 0][:5]}")
    return pos
