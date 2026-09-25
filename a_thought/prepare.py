"""Clean and assemble the inputs for simulation and rendering.

Outputs (data/processed/):
  neurons.parquet       one row per simulated neuron (sorted by root_id; row = model index)
  connectivity_783.npz  CSR matrix pre -> post of signed synapse counts (authors' v783 table)
  groups.json           stimulus groups (sugar/water GRNs, bitter GRNs), MN9, ingestion MNs
  brain_shell.npz       brain surface mesh in render space (vertices, faces, normals)
and results/prepare_log.json with every count printed below.

Neuron set: the union of (a) all neurons in the FlyWire v783 annotation table and
(b) all neurons in the authors' v783 model (Completeness_783.csv). Neurons without
any connection simply never receive input; nothing is dropped from the simulation.

Positions: soma coordinates where annotated, otherwise the annotation's anchor
point (pos_x/y/z, "typically on the backbone of the neuron"). Both are in
4 x 4 x 40 nm voxels and are converted to nanometres, then to a render space
that is centred on the brain mesh and scaled so its largest half-extent is 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import trimesh

from common import (
    CONNECTIVITY_NPZ, GROUPS_JSON, MN9_CELL_TYPE, NEURONS_PARQUET, PREPARE_LOG, PROCESSED, RAW,
    SHELL_NPZ, SHIU_BITTER_R_630, SHIU_MN9_ROOT_ID, SHIU_SUGAR_R_630, log, save_json,
)

VOXEL_NM = np.array([4.0, 4.0, 40.0])
ANN_FILE = RAW / "flywire_annotations_v3.1.0/Supplemental_file1_neuron_annotations.tsv"
ANN_PAPER_FILE = RAW / "flywire_annotations_v2.1.0/Supplemental_file1_neuron_annotations.tsv"
COMP_FILE = RAW / "shiu_model/Completeness_783.csv"
CON_FILE = RAW / "shiu_model/Connectivity_783.parquet"
SHELL_FILE = RAW / "flybrains/FLYWIRE_whole_brain.ply"

ANN_COLS = [
    "root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "flow", "super_class",
    "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "top_nt", "top_nt_conf", "side",
]

OTHER_NT = {"dopamine", "serotonin", "octopamine"}


def load_annotations(path=ANN_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=ANN_COLS, dtype={"root_id": np.int64}, low_memory=False)


def anatomical_axes(ann: pd.DataFrame, xyz_nm: np.ndarray) -> dict:
    """Work out which way each FlyWire axis points, from anatomy in the data itself.

    x: soma/anchor x of neurons annotated side=right vs side=left.
    y: gustatory afferents (enter the ventral SEZ) vs pars intercerebralis + ocellar
       neurons (dorsal).
    z: olfactory receptor neurons (antennal lobes, anterior) vs Kenyon cells
       (mushroom body calyx / KC somata, posterior).
    """
    def mean_of(mask):
        return np.nanmean(xyz_nm[mask.to_numpy()], axis=0)

    right = mean_of(ann.side == "right")
    left = mean_of(ann.side == "left")
    ventral = mean_of(ann.cell_class == "gustatory")
    dorsal = mean_of(ann.cell_class.isin(["pars_intercerebralis", "ocellar"]))
    anterior = mean_of(ann.cell_class == "olfactory")
    posterior = mean_of(ann.cell_class == "Kenyon_Cell")
    return dict(
        x_increases_toward_fly_right=bool(right[0] > left[0]),
        y_increases_ventrally=bool(ventral[1] > dorsal[1]),
        z_increases_posteriorly=bool(posterior[2] > anterior[2]),
        evidence_nm=dict(
            mean_x_right=right[0], mean_x_left=left[0],
            mean_y_gustatory=ventral[1], mean_y_PI_ocellar=dorsal[1],
            mean_z_olfactory=anterior[2], mean_z_kenyon=posterior[2],
        ),
    )


def render_transform(axes: dict, shell_vertices_nm: np.ndarray) -> dict:
    """nm -> render space. Render frame: +X = fly's left, +Y = dorsal, +Z = anterior.

    (+X left, +Y up, +Z forward is right-handed, so a camera in front of the fly
    looking along -Z sees the fly's left on the right of the screen, as in a
    frontal view.) Centred on the brain-mesh bounding box, scaled so the largest
    half-extent is 1.
    """
    sign = np.array([
        -1.0 if axes["x_increases_toward_fly_right"] else 1.0,
        -1.0 if axes["y_increases_ventrally"] else 1.0,
        -1.0 if axes["z_increases_posteriorly"] else 1.0,
    ])
    lo, hi = shell_vertices_nm.min(0), shell_vertices_nm.max(0)
    center = (lo + hi) / 2
    scale = 1.0 / ((hi - lo).max() / 2)
    return dict(center_nm=center, axis_sign=sign, scale_per_nm=scale)


def to_render(xyz_nm: np.ndarray, tf: dict) -> np.ndarray:
    return ((xyz_nm - tf["center_nm"]) * tf["axis_sign"] * tf["scale_per_nm"]).astype(np.float32)


def main() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ---------------------------------------------------------------- neurons
    ann = load_annotations()
    paper_ann = load_annotations(ANN_PAPER_FILE)
    comp = pd.read_csv(COMP_FILE, index_col=0)
    con = pd.read_parquet(CON_FILE)
    assert ann.root_id.is_unique and paper_ann.root_id.is_unique and comp.index.is_unique
    ann_ids, paper_ids, model_ids = set(ann.root_id), set(paper_ann.root_id), set(comp.index)
    ids = np.array(sorted(ann_ids | paper_ids | model_ids), dtype=np.int64)
    report["neurons"] = dict(
        annotation_v3_1_0=len(ann_ids),
        annotation_v2_1_0_paper_version=len(paper_ids),
        authors_v783_model=len(model_ids),
        authors_model_not_in_v3_1_0=len(model_ids - ann_ids),
        authors_model_not_in_v2_1_0=len(model_ids - paper_ids),
        v3_1_0_only=len(ann_ids - paper_ids),
        v2_1_0_only=len(paper_ids - ann_ids),
        without_connections=len((ann_ids | paper_ids) - model_ids),
        simulated_total=len(ids),
    )

    # Latest annotation (v3.1.0) wherever available; the paper-version table
    # (v2.1.0) supplies the rows for neurons that v3.1.0 no longer lists.
    fallback = paper_ann[~paper_ann.root_id.isin(ann_ids)]
    both = pd.concat([ann.assign(annotation_version="v3.1.0"), fallback.assign(annotation_version="v2.1.0")])
    df = pd.DataFrame({"root_id": ids}).merge(both, on="root_id", how="left")
    df["annotation_version"] = df.annotation_version.fillna("none")
    df["in_annotation"] = df.annotation_version != "none"
    df["in_authors_model"] = df.root_id.isin(model_ids)

    soma = df[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_NM
    anchor = df[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM
    has_soma = ~np.isnan(soma).any(1)
    has_anchor = ~np.isnan(anchor).any(1)
    xyz = np.where(has_soma[:, None], soma, anchor)
    df["pos_source"] = np.where(has_soma, "soma", np.where(has_anchor, "anchor", "none"))
    df[["x_nm", "y_nm", "z_nm"]] = xyz
    report["positions"] = dict(
        soma=int(has_soma.sum()),
        anchor_point_no_soma=int((~has_soma & has_anchor).sum()),
        no_position=int((~has_soma & ~has_anchor).sum()),
        no_position_root_ids=df.root_id[~has_soma & ~has_anchor].tolist(),
        note="neurons without a position are simulated but not rendered",
    )

    # ------------------------------------------------------ orientation / space
    shell = trimesh.load_mesh(SHELL_FILE)
    axes = anatomical_axes(df, xyz)
    tf = render_transform(axes, np.asarray(shell.vertices))
    render = np.full((len(df), 3), np.nan, np.float32)
    ok = df.pos_source.to_numpy() != "none"
    render[ok] = to_render(xyz[ok], tf)
    df[["rx", "ry", "rz"]] = render
    inside = np.nan_to_num(np.abs(render), nan=0).max(1) <= 1.0
    report["render_space"] = dict(
        axes=axes, center_nm=tf["center_nm"], axis_sign=tf["axis_sign"], scale_per_nm=tf["scale_per_nm"],
        frame="+X = fly's left, +Y = dorsal, +Z = anterior (right-handed)",
        positions_inside_unit_box=int(inside[ok].sum()), positioned=int(ok.sum()),
    )

    # ------------------------------------------------------------ connectivity
    index = pd.Index(ids)
    pre = index.get_indexer(con.Presynaptic_ID.to_numpy())
    post = index.get_indexer(con.Postsynaptic_ID.to_numpy())
    assert (pre >= 0).all() and (post >= 0).all()
    signed = con["Excitatory x Connectivity"].to_numpy().astype(np.int32)
    assert np.array_equal(signed, con.Connectivity.to_numpy() * con.Excitatory.to_numpy())
    W = sp.csr_matrix((signed, (pre, post)), shape=(len(ids), len(ids)), dtype=np.int32)
    W.sort_indices()
    assert W.nnz == len(con), "duplicate pre/post pairs in connection table"
    sp.save_npz(CONNECTIVITY_NPZ, W, compressed=True)
    n_syn = int(con.Connectivity.sum())
    report["connectivity"] = dict(
        source="Connectivity_783.parquet (Shiu et al. repository)",
        connections=int(W.nnz),
        synapses=n_syn,
        min_synapses_per_connection=int(con.Connectivity.min()),
        max_synapses_per_connection=int(con.Connectivity.max()),
        excitatory_connections=int((con.Excitatory > 0).sum()),
        inhibitory_connections=int((con.Excitatory < 0).sum()),
        connections_ge5_synapses=int((con.Connectivity >= 5).sum()),
        synapses_in_connections_ge5=int(con.Connectivity[con.Connectivity >= 5].sum()),
        self_connections=int((pre == post).sum()),
        connections_touching_unannotated=int((~df.in_annotation.to_numpy()[pre] | ~df.in_annotation.to_numpy()[post]).sum()),
    )

    # ------------------------------------------ transmitter sign and colour class
    per_pre = con.groupby("Presynaptic_ID").Excitatory.agg(["min", "max"])
    assert (per_pre["min"] == per_pre["max"]).all(), "sign is not constant per presynaptic neuron"
    model_sign = df.root_id.map(per_pre["min"]).fillna(0).astype(np.int8).to_numpy()
    df["model_sign"] = model_sign  # +1 excitatory, -1 inhibitory, 0 = no outgoing connections
    nt = df.top_nt.fillna("unknown").to_numpy()
    # Colour follows what the neuron does in the simulation (the sign used by the
    # model); "other" marks neurons predicted to use dopamine/serotonin/octopamine,
    # which the model treats as excitatory. Neurons without outgoing connections
    # fall back to their predicted transmitter.
    color = np.where(model_sign < 0, "inhibitory", "excitatory").astype(object)
    color[(model_sign > 0) & np.isin(nt, list(OTHER_NT))] = "other"
    no_out = model_sign == 0
    color[no_out & np.isin(nt, ["gaba", "glutamate"])] = "inhibitory"
    color[no_out & np.isin(nt, list(OTHER_NT) + ["unknown"])] = "other"
    df["color_class"] = color
    ct = pd.crosstab(df.top_nt.fillna("unknown"), df.model_sign)
    report["transmitters"] = dict(
        model_sign_counts={str(k): int(v) for k, v in df.model_sign.value_counts().items()},
        color_class_counts={str(k): int(v) for k, v in df.color_class.value_counts().items()},
        annotation_top_nt_vs_model_sign={f"{r}|{c}": int(ct.loc[r, c]) for r in ct.index for c in ct.columns},
        sign_disagrees_with_top_nt=int(
            ((df.model_sign > 0) & df.top_nt.isin(["gaba", "glutamate"])).sum()
            + ((df.model_sign < 0) & ~df.top_nt.isin(["gaba", "glutamate"]) & df.top_nt.notna()).sum()
        ),
    )

    # ------------------------------------------------------------------ groups
    def sel(mask):
        return np.flatnonzero(mask.to_numpy())

    sugar = sel(df.cell_sub_class == "sugar/water")
    bitter = sel(df.cell_sub_class == "bitter")
    mn9 = sel(df.cell_type == MN9_CELL_TYPE)
    ingestion = sel(df.cell_sub_class == "ingestion_motor_neuron")
    assert SHIU_MN9_ROOT_ID in set(ids[mn9]), "Shiu et al. MN9 root id is not CB0701 in this release"
    mn9_sides = df.side.to_numpy()[mn9]
    mn9 = mn9[np.argsort(mn9_sides)]  # left, right

    def group(idx, selector):
        sub = df.iloc[idx]
        return dict(
            selector=selector,
            n=len(idx),
            by_side={str(k): int(v) for k, v in sub.side.value_counts().items()},
            indices=idx.tolist(),
            root_ids=sub.root_id.tolist(),
            with_outgoing_connections=int((sub.model_sign != 0).sum()),
            in_authors_model=int(sub.in_authors_model.sum()),
        )

    groups = dict(
        sugar=group(sugar, "cell_sub_class == 'sugar/water' (annotation v3.1.0)"),
        bitter=group(bitter, "cell_sub_class == 'bitter' (annotation v3.1.0)"),
        mn9=group(mn9, f"cell_type == '{MN9_CELL_TYPE}'; contains Shiu et al. MN9 root id {SHIU_MN9_ROOT_ID}"),
        ingestion_motor_neurons=group(ingestion, "cell_sub_class == 'ingestion_motor_neuron'"),
    )
    groups["mn9"]["sides"] = df.side.to_numpy()[mn9].tolist()
    groups["mn9"]["shiu_mn9_root_id"] = SHIU_MN9_ROOT_ID
    groups["paper_lists_630"] = dict(
        note="Shiu et al. figures.ipynb lists (snapshot 630, right hemisphere); ids unchanged in 783 are listed",
        sugar_right_in_783=[int(i) for i in SHIU_SUGAR_R_630 if i in set(ids)],
        bitter_right_in_783=[int(i) for i in SHIU_BITTER_R_630 if i in set(ids)],
        n_sugar_630=len(SHIU_SUGAR_R_630), n_bitter_630=len(SHIU_BITTER_R_630),
    )
    groups["annotation_v2.1.0_counts"] = dict(
        sugar_water=int((paper_ann.cell_sub_class == "sugar/water").sum()),
        bitter=int((paper_ann.cell_sub_class == "bitter").sum()),
        cb0701=int((paper_ann.cell_type == MN9_CELL_TYPE).sum()),
        neurons=int(len(paper_ann)),
    )
    save_json(GROUPS_JSON, groups)
    report["groups"] = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("indices", "root_ids")} for k, v in groups.items()
        if isinstance(v, dict)
    }

    # ------------------------------------------------------------ render regions
    sc = df.super_class.fillna("unannotated").to_numpy()
    side = df.side.fillna("na").to_numpy()
    region = np.full(len(df), "other", dtype=object)
    region[np.isin(sc, ["central", "endocrine", "motor", "descending", "ascending"])] = "central"
    optic = np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])
    region[optic & (side == "left")] = "optic_left"
    region[optic & (side == "right")] = "optic_right"
    region[optic & ~np.isin(side, ["left", "right"])] = "central"
    region[np.isin(sc, ["sensory", "sensory_ascending"])] = "sensory"
    df["region"] = region
    report["regions"] = {str(k): int(v) for k, v in pd.Series(region).value_counts().items()}

    df.to_parquet(NEURONS_PARQUET, index=False)

    # --------------------------------------------------------------- the shell
    verts = to_render(np.asarray(shell.vertices, float), tf)
    faces = np.asarray(shell.faces, np.int32)
    if tf["axis_sign"].prod() < 0:  # a reflection flips triangle winding
        faces = faces[:, ::-1].copy()
    m = trimesh.Trimesh(verts, faces, process=False)
    normals = np.asarray(m.vertex_normals, np.float32)
    np.savez_compressed(SHELL_NPZ, vertices=verts, faces=faces, normals=normals)
    report["shell"] = dict(
        source="flybrains 0.6.3 meshes/FLYWIRE_whole_brain.ply", vertices=len(verts), faces=len(faces),
        bounds_render=[verts.min(0).tolist(), verts.max(0).tolist()],
        neurons_inside_mesh_bbox=int(
            ((render[ok] >= verts.min(0) - 1e-6) & (render[ok] <= verts.max(0) + 1e-6)).all(1).sum()
        ),
    )
    save_json(PREPARE_LOG, report)
    return report


def print_report(r: dict) -> None:
    n, p, c, g = r["neurons"], r["positions"], r["connectivity"], r["groups"]
    log("FlyWire release 783 -- data summary")
    log(f"  neurons simulated          {n['simulated_total']:>12,d}")
    log(f"    annotation v3.1.0        {n['annotation_v3_1_0']:>12,d}")
    log(f"    annotation v2.1.0        {n['annotation_v2_1_0_paper_version']:>12,d}  (paper version)")
    log(f"    in authors' v783 model   {n['authors_v783_model']:>12,d}")
    log(f"    model ids not in v3.1.0  {n['authors_model_not_in_v3_1_0']:>12,d}  (annotated from v2.1.0)")
    log(f"    without any connection   {n['without_connections']:>12,d}  (never receive input)")
    log(f"  connections (>=1 synapse)  {c['connections']:>12,d}")
    log(f"  synapses                   {c['synapses']:>12,d}")
    log(f"  connections >= 5 synapses  {c['connections_ge5_synapses']:>12,d}")
    log(f"  positions: soma {p['soma']:,d} | anchor point {p['anchor_point_no_soma']:,d} | none {p['no_position']:,d}")
    log(f"  sugar/water GRNs {g['sugar']['n']} {g['sugar']['by_side']} | bitter GRNs {g['bitter']['n']} {g['bitter']['by_side']}")
    log(f"  MN9 (CB0701) {g['mn9']['n']} {g['mn9']['by_side']} | ingestion motor neurons {g['ingestion_motor_neurons']['n']}")
    ax = r["render_space"]["axes"]
    log(f"  axes: x->fly right {ax['x_increases_toward_fly_right']}, y->ventral {ax['y_increases_ventrally']}, "
        f"z->posterior {ax['z_increases_posteriorly']}")


if __name__ == "__main__":
    print_report(main())
