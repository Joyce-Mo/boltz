#!/usr/bin/env python3
"""UMAP analysis of per-residue Boltz single representations on CATH-20.

Reads per-domain `<domain_id>_s.pt` files (saved by extract_cath_reps.py), pulls
the matching PDB to get sequences + CA coordinates, then produces the same
seven UMAP plots as protpardelle-1c/representation_extraction/umap_representations.py:

  1. amino acid identity (20 AAs)
  2. amino acid group (nonpolar/aromatic/polar/positive/negative/special)
  3. side-chain charge at pH 7
  4. Kyte-Doolittle hydrophobicity
  5. relative SASA (Tien et al. max ASA proxy)
  6. CA-CA distance to chain center
  7. mean pairwise CA-CA distance

UMAP parameters: n_neighbors=100, min_dist=0.3, metric='euclidean'.

Usage:
    python masking_code/plot_cath_umaps.py \
        --rep_dir /wynton/scratch/jqmo/rotation_datasets/boltz1 \
        --pdb_dir /wynton/home/rotation/jqmo/rotation3/datasets/cath20-filtered-foldseek \
        --out_dir figures/boltz1_umap \
        --model_tag boltz1 \
        --max_residues 200000 \
        --max_structures 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Amino acid properties (matches protpardelle-1c/umap_representations.py)
# ---------------------------------------------------------------------------

AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0, "X": 0,
}

AA_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, "X": 0.0,
}

AA_MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174, "X": 200,
}

AA_GROUP = {
    "A": "nonpolar", "V": "nonpolar", "I": "nonpolar", "L": "nonpolar",
    "M": "nonpolar", "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "P": "nonpolar", "G": "special",
    "S": "polar", "T": "polar", "C": "special", "N": "polar", "Q": "polar",
    "D": "negative", "E": "negative",
    "K": "positive", "R": "positive", "H": "positive",
    "X": "unknown",
}


# ---------------------------------------------------------------------------
# PDB parsing (CA coords + 1-letter sequence, in residue order)
# ---------------------------------------------------------------------------
def parse_pdb_ca(pdb_path: Path):
    """Return (seq_1letter: str, ca_coords: np.ndarray[L,3]).

    Walks the first model only, picks the first chain encountered, and
    returns standard residues in residue order.
    """
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    for model in structure:
        for chain in model:
            seq_chars = []
            cas = []
            for res in chain:
                if res.get_id()[0] != " ":
                    continue  # hetero / water
                if "CA" not in res:
                    continue
                seq_chars.append(seq1(res.get_resname()))
                cas.append(res["CA"].coord)
            if seq_chars:
                return "".join(seq_chars), np.asarray(cas, dtype=np.float64)
        break
    return "", np.zeros((0, 3), dtype=np.float64)


def compute_residue_properties(seq: str, ca_coords: np.ndarray) -> pd.DataFrame:
    aa_names = [c if c in AA_CHARGE else "X" for c in seq]
    charges = [AA_CHARGE[a] for a in aa_names]
    hydro = [AA_HYDROPHOBICITY[a] for a in aa_names]
    max_asa = [AA_MAX_ASA[a] for a in aa_names]
    groups = [AA_GROUP[a] for a in aa_names]

    L = len(aa_names)
    if ca_coords.shape[0] != L:
        # Defensive: trim to the shorter length so we don't crash
        L = min(L, ca_coords.shape[0])
        aa_names = aa_names[:L]
        charges = charges[:L]
        hydro = hydro[:L]
        max_asa = max_asa[:L]
        groups = groups[:L]
        ca_coords = ca_coords[:L]

    if L == 0:
        return pd.DataFrame()

    center = ca_coords.mean(axis=0)
    dist_to_center = np.linalg.norm(ca_coords - center, axis=1)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    pw = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(pw, 0.0)
    mean_pw = pw.sum(axis=1) / max(L - 1, 1)

    return pd.DataFrame({
        "aa": aa_names,
        "aa_group": groups,
        "charge": charges,
        "hydrophobicity": hydro,
        "max_asa": max_asa,
        "dist_to_center": dist_to_center,
        "mean_pairwise_dist": mean_pw,
        "residue_index": np.arange(L),
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_single_reps(
    rep_dir: Path,
    pdb_dir: Path,
    max_residues: int = 0,
    max_structures: int = 0,
    seed: int = 42,
):
    """Load (s) representations + per-residue properties for all CATH domains.

    Returns:
        embeddings : np.ndarray [N_residues, D]
        properties : pd.DataFrame [N_residues, ...]
    """
    s_files = sorted(rep_dir.glob("*_s.pt"))
    if not s_files:
        raise FileNotFoundError(f"No *_s.pt files found in {rep_dir}")
    print(f"Found {len(s_files)} single-rep files in {rep_dir}")

    if max_structures > 0 and len(s_files) > max_structures:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(s_files), size=max_structures, replace=False))
        s_files = [s_files[i] for i in idx]
        print(f"Randomly sampled {max_structures} structures (seed={seed})")

    all_embeds = []
    all_props = []
    total = 0
    skipped = 0
    for f in s_files:
        domain_id = f.stem[:-2]  # strip trailing "_s"
        pdb_path = pdb_dir / f"{domain_id}.pdb"
        if not pdb_path.exists():
            skipped += 1
            continue
        try:
            seq, ca = parse_pdb_ca(pdb_path)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN parse {domain_id}: {e}")
            skipped += 1
            continue
        if not seq:
            skipped += 1
            continue

        s = torch.load(f, map_location="cpu", weights_only=True)
        s = s.squeeze(0) if s.dim() == 3 else s  # -> [L, D]
        if s.dim() != 2:
            skipped += 1
            continue
        L = min(s.shape[0], len(seq), ca.shape[0])
        if L == 0:
            skipped += 1
            continue
        emb = s[:L].float().numpy()

        props = compute_residue_properties(seq[:L], ca[:L])
        if props.empty:
            skipped += 1
            continue
        props["domain_id"] = domain_id

        all_embeds.append(emb)
        all_props.append(props)
        total += L
        if max_residues > 0 and total >= max_residues:
            print(f"  Reached max_residues={max_residues} after {len(all_embeds)} structures")
            break

    if not all_embeds:
        raise RuntimeError("No embeddings loaded.")

    embeddings = np.concatenate(all_embeds, axis=0)
    properties = pd.concat(all_props, ignore_index=True)
    print(f"Loaded {embeddings.shape[0]} residues ({embeddings.shape[1]}D) "
          f"from {len(all_embeds)} domains (skipped {skipped})")
    return embeddings, properties


def subsample_residues(embeddings, properties, max_points, seed=42):
    n_total = embeddings.shape[0]
    if max_points <= 0 or n_total <= max_points:
        return embeddings, properties
    rng = np.random.RandomState(seed)
    domain_ids = properties["domain_id"].values
    unique = properties["domain_id"].unique()
    selected = []
    indices_per_dom = []
    for d in unique:
        idx = np.where(domain_ids == d)[0]
        indices_per_dom.append(idx)
        selected.append(rng.choice(idx, size=1))
    remaining = max_points - len(unique)
    if remaining > 0:
        sizes = np.array([len(i) for i in indices_per_dom], dtype=np.float64)
        extra = (sizes / sizes.sum() * remaining).astype(int)
        for i, (idx, n_extra) in enumerate(zip(indices_per_dom, extra)):
            avail = np.setdiff1d(idx, selected[i])
            n_pick = min(n_extra, len(avail))
            if n_pick > 0:
                selected.append(rng.choice(avail, size=n_pick, replace=False))
    chosen = np.sort(np.concatenate(selected))
    print(f"Subsampled {len(chosen)} / {n_total} residues "
          f"({len(chosen) / n_total * 100:.1f}%) covering "
          f"{len(np.unique(domain_ids[chosen]))} domains")
    return embeddings[chosen], properties.iloc[chosen].reset_index(drop=True)


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 100,
    min_dist: float = 0.3,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    import umap

    n_neighbors = min(n_neighbors, embeddings.shape[0] - 1)
    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric={metric}, n={embeddings.shape[0]}) ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_jobs=-1,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Plotting (matches protpardelle-1c formatting / palette)
# ---------------------------------------------------------------------------
def plot_umaps(umap_emb, properties, output_dir: Path, model_tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_tag}_single"

    properties = properties.copy()
    properties["umap1"] = umap_emb[:, 0]
    properties["umap2"] = umap_emb[:, 1]
    properties.to_csv(output_dir / f"{prefix}_umap_data.csv", index=False)

    point_size = 2
    alpha = 0.5
    figsize = (10, 8)

    # 1. AA identity
    fig, ax = plt.subplots(figsize=figsize)
    aa_list = sorted(properties["aa"].unique())
    cmap = plt.cm.get_cmap("tab20", len(aa_list))
    aa_to_color = {aa: cmap(i) for i, aa in enumerate(aa_list)}
    for aa in aa_list:
        m = properties["aa"] == aa
        ax.scatter(umap_emb[m, 0], umap_emb[m, 1],
                   c=[aa_to_color[aa]], s=point_size, alpha=alpha, label=aa)
    ax.legend(fontsize=7, markerscale=4, loc="best", ncol=2, title="Amino Acid")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by amino acid")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_aa.png", dpi=200)
    plt.close(fig)

    # 2. AA group
    group_colors = {
        "nonpolar": "#1f77b4",
        "aromatic": "#ff7f0e",
        "polar": "#2ca02c",
        "positive": "#d62728",
        "negative": "#9467bd",
        "special": "#8c564b",
        "unknown": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=figsize)
    for group in sorted(group_colors.keys()):
        m = properties["aa_group"] == group
        if not m.any():
            continue
        ax.scatter(umap_emb[m, 0], umap_emb[m, 1],
                   c=group_colors[group], s=point_size, alpha=alpha, label=group)
    ax.legend(fontsize=8, markerscale=4, loc="best", title="AA Group")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by AA group")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_aa_group.png", dpi=200)
    plt.close(fig)

    # 3. charge
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=properties["charge"], s=point_size, alpha=alpha,
                    cmap="coolwarm", vmin=-1.1, vmax=1.1)
    plt.colorbar(sc, ax=ax, label="Charge (pH 7)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by charge")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_charge.png", dpi=200)
    plt.close(fig)

    # 4. hydrophobicity
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=properties["hydrophobicity"], s=point_size, alpha=alpha,
                    cmap="RdYlBu_r")
    plt.colorbar(sc, ax=ax, label="Hydrophobicity (Kyte-Doolittle)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by hydrophobicity")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_hydrophobicity.png", dpi=200)
    plt.close(fig)

    # 5. SASA proxy
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=properties["max_asa"], s=point_size, alpha=alpha,
                    cmap="YlOrRd")
    plt.colorbar(sc, ax=ax, label="Max ASA (Å², Tien et al.)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by SASA (max ASA)")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_sasa.png", dpi=200)
    plt.close(fig)

    # 6. dist to chain center
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=properties["dist_to_center"], s=point_size, alpha=alpha,
                    cmap="viridis")
    plt.colorbar(sc, ax=ax, label="CA distance to chain center (Å)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by distance to center")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_dist_to_center.png", dpi=200)
    plt.close(fig)

    # 7. mean pairwise CA-CA dist
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=properties["mean_pairwise_dist"], s=point_size, alpha=alpha,
                    cmap="plasma")
    plt.colorbar(sc, ax=ax, label="Mean pairwise CA-CA distance (Å)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Boltz {model_tag} single rep — colored by mean pairwise distance")
    fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_by_mean_pw_dist.png", dpi=200)
    plt.close(fig)

    print(f"Saved 7 UMAP plots to {output_dir}/")


def main():
    p = argparse.ArgumentParser(description="UMAP per-residue Boltz reps on CATH-20")
    p.add_argument("--rep_dir", type=Path, required=True,
                   help="Dir with <domain_id>_s.pt files (output of extract_cath_reps.py)")
    p.add_argument("--pdb_dir", type=Path, required=True,
                   help="Dir with the matching CATH PDB files")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_tag", type=str, default="boltz",
                   help="Tag inserted into plot titles / filenames (e.g. boltz1, boltz2)")
    p.add_argument("--max_residues", type=int, default=200_000,
                   help="Stop loading after this many residues (0 = no limit)")
    p.add_argument("--max_structures", type=int, default=0,
                   help="Randomly sample this many domains before loading (0 = all)")
    p.add_argument("--max_points", type=int, default=200_000,
                   help="Subsample residues passed into UMAP for tractability")
    p.add_argument("--n_neighbors", type=int, default=100)
    p.add_argument("--min_dist", type=float, default=0.3)
    p.add_argument("--metric", type=str, default="euclidean")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    embeddings, properties = load_single_reps(
        rep_dir=args.rep_dir,
        pdb_dir=args.pdb_dir,
        max_residues=args.max_residues,
        max_structures=args.max_structures,
        seed=args.seed,
    )
    embeddings, properties = subsample_residues(
        embeddings, properties, max_points=args.max_points, seed=args.seed,
    )
    umap_emb = run_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )
    plot_umaps(umap_emb, properties, args.out_dir, args.model_tag)


if __name__ == "__main__":
    main()
