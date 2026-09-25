"""Download every raw input file and record provenance in data/manifest.json.

Each entry records: URL, retrieval date (UTC), version / commit, file size,
SHA-256, and the licence / citation terms that could be verified. Sources that
cannot be reached from the build environment are recorded as attempts with the
error, so the manifest also documents what was *not* used and why.

Usage:  python fetch.py            (skips files already present with matching hash)
        python fetch.py --force    (re-download everything)
"""
from __future__ import annotations

import argparse
import io
import zipfile

import requests

from common import MANIFEST, RAW, load_json, log, save_json, sha256_bytes, sha256_file, utc_now

ANN_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"  # flywire_annotations tag v3.1.0
ANN_PAPER_COMMIT = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"  # tag v2.1.0 (Schlegel et al. 2024 version)
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"  # philshiu/Drosophila_brain_model
FLYPOKE_COMMIT = "24814fe83224ca2d25c116c1107440b877b07762"  # vshapenko/flypoke
FLYBRAINS_WHEEL = (
    "https://files.pythonhosted.org/packages/12/aa/10522349cba5ec65b8827cd7f4f189cb3cebd912d58c19db7f74f1e64b0f/"
    "flybrains-0.6.3-py3-none-any.whl"
)
FLYBRAINS_WHEEL_SHA256 = "a74755ea5db431864458356384a234ce2f30d7856e099a2bcf9e55e117196d95"  # from PyPI JSON API

GH = "https://raw.githubusercontent.com"

FLYWIRE_TERMS = (
    "FlyWire release 783 data. Citation required: Dorkenwald et al. 2024 (Nature 634:124-138) and "
    "Schlegel et al. 2024 (Nature 634:139-152); annotation versions >= 3.0.0 also ask for Berg et al. 2025 "
    "(bioRxiv 10.1101/2025.10.09.680999) and Matsliah et al. 2024 (Nature). The flywire_annotations "
    "repository has no LICENSE file; its README 'How to cite' section is the stated requirement. "
    "flypoke's README states 'FlyWire data are released under CC BY 4.0'; the primary licence pages "
    "(zenodo.org record 10676866, flywire.ai, codex.flywire.ai) returned HTTP 403 from this build "
    "environment's egress proxy, so the exact licence wording is UNVERIFIED here and must be confirmed "
    "before public release."
)

SOURCES = [
    dict(
        key="flywire_annotations",
        url=f"{GH}/flyconnectome/flywire_annotations/{ANN_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        dest="flywire_annotations_v3.1.0/Supplemental_file1_neuron_annotations.tsv",
        version=f"flyconnectome/flywire_annotations tag v3.1.0, commit {ANN_COMMIT}; FlyWire materialization 783",
        role="neuron annotations: cell classes/types, side, predicted top_nt, soma and anchor positions (4x4x40 nm voxels)",
        license=FLYWIRE_TERMS,
    ),
    dict(
        key="flywire_annotations_paper_version",
        url=f"{GH}/flyconnectome/flywire_annotations/{ANN_PAPER_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        dest="flywire_annotations_v2.1.0/Supplemental_file1_neuron_annotations.tsv",
        version=f"flyconnectome/flywire_annotations tag v2.1.0, commit {ANN_PAPER_COMMIT} (version reported in Schlegel et al. 2024)",
        role="reference only: stimulus-group counts compared against v3.1.0 in NOTES.md",
        license=FLYWIRE_TERMS,
    ),
    dict(
        key="shiu_connectivity_783",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/Connectivity_783.parquet",
        dest="shiu_model/Connectivity_783.parquet",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}; built by the authors from FlyWire public release 783",
        role="connection table used by the simulation: presynaptic/postsynaptic root ids, synapse count, sign (Excitatory)",
        license="Repository: MIT License (c) 2023 Philip Shiu and Nico Spiller. Underlying data: " + FLYWIRE_TERMS,
    ),
    dict(
        key="shiu_completeness_783",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/Completeness_783.csv",
        dest="shiu_model/Completeness_783.csv",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}",
        role="neuron list of the authors' v783 model (index order of Connectivity_783.parquet)",
        license="MIT (repository); FlyWire terms for the data",
    ),
    dict(
        key="shiu_model_py",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/model.py",
        dest="shiu_model/model.py",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}",
        role="reference Brian2 implementation; all model parameters are taken from default_params in this file",
        license="MIT License (c) 2023 Philip Shiu and Nico Spiller",
    ),
    dict(
        key="shiu_connectivity_630",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/2023_03_23_connectivity_630_final.parquet",
        dest="shiu_model/2023_03_23_connectivity_630_final.parquet",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}; FlyWire snapshot 630 (the version used in the paper)",
        role="validation only: reproduce the authors' published example outputs",
        license="MIT (repository); FlyWire terms for the data",
    ),
    dict(
        key="shiu_completeness_630",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/2023_03_23_completeness_630_final.csv",
        dest="shiu_model/2023_03_23_completeness_630_final.csv",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}",
        role="validation only",
        license="MIT (repository); FlyWire terms for the data",
    ),
    dict(
        key="shiu_example_sugarR",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/results/example/sugarR.parquet",
        dest="shiu_model/results_example/sugarR.parquet",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}",
        role="published reference output (authors' Brian2 run, v630, 21 right sugar GRNs, 30 trials)",
        license="MIT (repository)",
    ),
    dict(
        key="shiu_example_sugarR_100Hz",
        url=f"{GH}/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/results/example/sugarR_100Hz.parquet",
        dest="shiu_model/results_example/sugarR_100Hz.parquet",
        version=f"philshiu/Drosophila_brain_model commit {SHIU_COMMIT}",
        role="published reference output (authors' Brian2 run, v630, 21 right sugar GRNs at 100 Hz, 30 trials)",
        license="MIT (repository)",
    ),
    dict(
        key="flypoke_sim",
        url=f"{GH}/vshapenko/flypoke/{FLYPOKE_COMMIT}/src/flypoke/sim.py",
        dest="flypoke/flypoke/sim.py",
        version=f"vshapenko/flypoke commit {FLYPOKE_COMMIT}",
        role="independent NumPy reimplementation used as a cross-check",
        license="MIT License (c) 2026 vshapenko",
    ),
    dict(
        key="flypoke_data",
        url=f"{GH}/vshapenko/flypoke/{FLYPOKE_COMMIT}/src/flypoke/data.py",
        dest="flypoke/flypoke/data.py",
        version=f"vshapenko/flypoke commit {FLYPOKE_COMMIT}",
        role="flypoke network construction (min 5 synapses, top_nt signs)",
        license="MIT License (c) 2026 vshapenko",
    ),
    dict(
        key="flypoke_init",
        url=f"{GH}/vshapenko/flypoke/{FLYPOKE_COMMIT}/src/flypoke/__init__.py",
        dest="flypoke/flypoke/__init__.py",
        version=f"vshapenko/flypoke commit {FLYPOKE_COMMIT}",
        role="package marker",
        license="MIT License (c) 2026 vshapenko",
    ),
    dict(
        key="flypoke_readme",
        url=f"{GH}/vshapenko/flypoke/{FLYPOKE_COMMIT}/README.md",
        dest="flypoke/README.md",
        version=f"vshapenko/flypoke commit {FLYPOKE_COMMIT}",
        role="flypoke's published feeding results table (compared in validation)",
        license="MIT License (c) 2026 vshapenko",
    ),
    dict(
        key="flybrains_wheel",
        url=FLYBRAINS_WHEEL,
        dest="flybrains/flybrains-0.6.3-py3-none-any.whl",
        expected_sha256=FLYBRAINS_WHEEL_SHA256,
        extract={
            "flybrains/meshes/FLYWIRE_whole_brain.ply": "flybrains/FLYWIRE_whole_brain.ply",
            "flybrains/meshes/FLYWIRE.ply": "flybrains/FLYWIRE.ply",
            "flybrains-0.6.3.dist-info/licenses/LICENSE": "flybrains/LICENSE",
        },
        version="navis-flybrains 0.6.3 (PyPI wheel)",
        role="brain surface mesh in FlyWire space (nm): FLYWIRE_whole_brain.ply = outer brain shell incl. optic lobes",
        license=(
            "flybrains is GNU GPL v3. The FLYWIRE neuropil mesh is the FAFB14 mesh transformed into FlyWire space "
            "(flybrains docstring); cite Zheng et al. 2018 (Cell), Dorkenwald et al. 2022 (Nat Methods) and "
            "navis-flybrains (github.com/navis-org/navis-flybrains)."
        ),
    ),
]

# Official sources we tried first. They are blocked by the environment's
# network policy; the attempt is logged so the manifest is complete.
UNREACHABLE_PROBES = [
    dict(
        key="zenodo_10676866_connections",
        url="https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
        role="official FlyWire 783 connection table (pre, post, neuropil, syn_count, NT averages)",
        substitute="shiu_connectivity_783 (authors' aggregation of the same release; equivalence evidence in NOTES.md)",
    ),
    dict(
        key="zenodo_10877326_skeletons",
        url="https://zenodo.org/records/10877326",
        role="official FlyWire 783 skeleton download",
        substitute="public segmentation bucket gs://flywire_v141_m783, layer skeletons_mip_1 (fetched per neuron by render/pathway.py)",
    ),
    dict(
        key="flywire_codex",
        url="https://codex.flywire.ai/api/download",
        role="FlyWire Codex downloads and documentation",
        substitute="none needed",
    ),
]


def _download(url: str) -> bytes:
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content


def fetch(force: bool = False) -> dict:
    manifest = load_json(MANIFEST, default={}) or {}
    files = manifest.get("files", {})
    for src in SOURCES:
        dest = RAW / src["dest"]
        prev = files.get(src["key"])
        if dest.exists() and prev and not force and sha256_file(dest) == prev["sha256"]:
            log(f"  ok   {src['key']:34s} {prev['size_bytes']:>12,d} B  (cached)")
            continue
        log(f"  get  {src['key']:34s} {src['url']}")
        data = _download(src["url"])
        digest = sha256_bytes(data)
        if "expected_sha256" in src and digest != src["expected_sha256"]:
            raise RuntimeError(f"{src['key']}: sha256 {digest} != expected {src['expected_sha256']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entry = {k: v for k, v in src.items() if k not in ("key", "extract", "expected_sha256")}
        entry.update(path=f"data/raw/{src['dest']}", size_bytes=len(data), sha256=digest, retrieved_utc=utc_now())
        if "extract" in src:
            entry["extracted"] = {}
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for member, out in src["extract"].items():
                    blob = zf.read(member)
                    p = RAW / out
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(blob)
                    entry["extracted"][member] = dict(
                        path=f"data/raw/{out}", size_bytes=len(blob), sha256=sha256_bytes(blob)
                    )
        files[src["key"]] = entry
        log(f"       {len(data):,d} B  sha256 {digest[:16]}...")

    probes = []
    for p in UNREACHABLE_PROBES:
        try:
            r = requests.head(p["url"], timeout=30, allow_redirects=True)
            status = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # proxy refuses CONNECT -> ProxyError
            status = f"unreachable: {type(e).__name__}: {str(e)[:160]}"
        probes.append(dict(p, checked_utc=utc_now(), status=status))
        log(f"  probe {p['key']:33s} {status[:90]}")

    manifest = dict(
        description="Raw inputs for A THOUGHT. Paths are relative to a_thought/. Re-create with `python fetch.py`.",
        flywire_release="783 (FlyWire public release; materialization 783)",
        files=files,
        unreachable_official_sources=probes,
        skeletons=manifest.get("skeletons", {}),
    )
    save_json(MANIFEST, manifest)
    log(f"manifest written: {MANIFEST}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    fetch(force=ap.parse_args().force)
