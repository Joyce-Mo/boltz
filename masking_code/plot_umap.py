"""Visualizations of Boltz-1 single and pair representations.

Loads saved .pt representation files from test_reps.py and produces:
  1. Per-residue UMAP of single (s) representations, colored by protein.
  2. Per-residue UMAP of single (s) representations, colored by residue index.
  3. Row-averaged UMAP of pair (z) representations, colored by protein.
  4. Contact map heatmaps from pair representations (cosine similarity).
  5. 3D backbone reconstruction via the model's diffusion decoder.
  6. Approximate 3D shape from pair-rep distance matrix + MDS.

Usage:
    # UMAPs + contact maps (no model needed):
    python plot_umap.py --rep_dir representations --out_dir figures

    # Also run diffusion decoder to generate 3D structures:
    python plot_umap.py --rep_dir representations --out_dir figures \
        --wts_path /Users/joycemo/Documents/GitHub/boltz/boltz1_conf.ckpt --device mps

    # Also run MDS 3D reconstruction from pair reps:
    python plot_umap.py --rep_dir representations --out_dir figures --mds

    Make sure the environment has umap and matplotlib
    pip install umap-learn 
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import umap


# Protein metadata for consistent coloring / labeling
PROTEIN_INFO = {
    "1smg": "Switch parent (GTPase)",
    "9cif": "Switch parent",
    "9cid": "Switch state 1",
    "9cig": "Switch state 2",
    "2lv8": "De novo Rossmann",
}


def load_representations(rep_dir: Path):
    """Load all *_s.pt and *_z.pt files, return dicts keyed by protein name."""
    singles, pairs = {}, {}
    for f in sorted(rep_dir.glob("*_s.pt")):
        # filename like: 1smg_0recycles_nomsa_s.pt
        name = f.stem.split("_")[0]
        singles[name] = torch.load(f, map_location="cpu", weights_only=True)
    for f in sorted(rep_dir.glob("*_z.pt")):
        name = f.stem.split("_")[0]
        pairs[name] = torch.load(f, map_location="cpu", weights_only=True)
    return singles, pairs


# ---------------------------------------------------------------------------
# 1. UMAP of single reps colored by protein
# ---------------------------------------------------------------------------
def plot_single_umap_by_protein(singles: dict, out_dir: Path):
    """UMAP of per-residue single representations, colored by protein."""
    all_vecs, labels = [], []
    for name, s in singles.items():
        s = s.squeeze()
        if s.dim() != 2:
            print(f"  Skipping {name}: unexpected s shape {list(s.shape)}")
            continue
        all_vecs.append(s.numpy())
        labels.extend([name] * s.shape[0])

    X = np.concatenate(all_vecs, axis=0)
    labels = np.array(labels)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_names = list(singles.keys())
    cmap = plt.cm.get_cmap("tab10", len(unique_names))

    for i, name in enumerate(unique_names):
        mask = labels == name
        label = f"{name} ({PROTEIN_INFO.get(name, '')})"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=12, alpha=0.7, label=label)

    ax.set_title("UMAP of single representations (per residue)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_single_by_protein.png", dpi=200)
    plt.close(fig)
    print(f"  Saved umap_single_by_protein.png")


# ---------------------------------------------------------------------------
# 2. UMAP of single reps colored by residue position
# ---------------------------------------------------------------------------
def plot_single_umap_by_position(singles: dict, out_dir: Path):
    """UMAP of per-residue single representations, colored by normalized residue index."""
    all_vecs, positions, labels = [], [], []
    for name, s in singles.items():
        s = s.squeeze()
        if s.dim() != 2:
            continue
        n_res = s.shape[0]
        all_vecs.append(s.numpy())
        positions.extend(np.linspace(0, 1, n_res).tolist())
        labels.extend([name] * n_res)

    X = np.concatenate(all_vecs, axis=0)
    positions = np.array(positions)
    labels = np.array(labels)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=positions, cmap="viridis", s=12, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Normalized residue position (N\u2192C)")

    for name in singles:
        mask = labels == name
        if mask.any():
            cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
            ax.annotate(name, (cx, cy), fontsize=9, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_title("UMAP of single representations (colored by residue position)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(out_dir / "umap_single_by_position.png", dpi=200)
    plt.close(fig)
    print(f"  Saved umap_single_by_position.png")


# ---------------------------------------------------------------------------
# 3. UMAP of pair reps (row-averaged) colored by protein
# ---------------------------------------------------------------------------
def plot_pair_umap(pairs: dict, out_dir: Path, max_residues: int = 50):
    """UMAP of pair representations, row-averaged to per-residue vectors."""
    all_vecs, labels = [], []
    for name, z in pairs.items():
        z = z.squeeze()
        if z.dim() != 3:
            print(f"  Skipping {name}: unexpected z shape {list(z.shape)}")
            continue
        n_res = z.shape[0]
        z_per_res = z.mean(dim=1)
        if n_res > max_residues:
            idx = np.linspace(0, n_res - 1, max_residues, dtype=int)
            z_per_res = z_per_res[idx]
            n_res = max_residues
        all_vecs.append(z_per_res.numpy())
        labels.extend([name] * n_res)

    X = np.concatenate(all_vecs, axis=0)
    labels = np.array(labels)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_names = list(pairs.keys())
    cmap = plt.cm.get_cmap("tab10", len(unique_names))

    for i, name in enumerate(unique_names):
        mask = labels == name
        label = f"{name} ({PROTEIN_INFO.get(name, '')})"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=12, alpha=0.7, label=label)

    ax.set_title("UMAP of pair representations (row-averaged, per residue)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(out_dir / "umap_pair_by_protein.png", dpi=200)
    plt.close(fig)
    print(f"  Saved umap_pair_by_protein.png")


# ---------------------------------------------------------------------------
# 4. Contact map heatmaps from pair representations
# ---------------------------------------------------------------------------
def plot_contact_maps(pairs: dict, out_dir: Path):
    """Plot cosine-similarity contact maps derived from pair representations.

    For each protein, computes cosine similarity between z[i,j,:] vectors
    to produce a residue-residue similarity matrix.  High similarity at
    off-diagonal positions indicates predicted contacts.
    """
    n_proteins = len(pairs)
    if n_proteins == 0:
        return

    cols = min(n_proteins, 3)
    rows = (n_proteins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    for idx, (name, z) in enumerate(pairs.items()):
        ax = axes[idx // cols][idx % cols]
        z = z.squeeze()  # [N, N, D]
        if z.dim() != 3:
            ax.set_title(f"{name} (bad shape)")
            continue

        # Cosine similarity between row-vectors: sim[i,j] = cos(z[i,:,:].mean, z[j,:,:].mean)
        # But more informative: use the z[i,j,:] norm as a proxy for interaction strength
        z_norm = z.norm(dim=-1).numpy()  # [N, N]

        im = ax.imshow(z_norm, cmap="hot", origin="upper", aspect="equal")
        ax.set_title(f"{name}\n{PROTEIN_INFO.get(name, '')}", fontsize=10)
        ax.set_xlabel("Residue j")
        ax.set_ylabel("Residue i")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="||z[i,j]||")

    # Hide unused axes
    for idx in range(n_proteins, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Pair representation contact maps (L2 norm of z[i,j,:])", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "contact_maps.png", dpi=200)
    plt.close(fig)
    print(f"  Saved contact_maps.png")

    # Also save individual high-res contact maps
    for name, z in pairs.items():
        z = z.squeeze()
        if z.dim() != 3:
            continue
        z_norm = z.norm(dim=-1).numpy()

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(z_norm, cmap="hot", origin="upper", aspect="equal")
        ax2.set_title(f"{name} - {PROTEIN_INFO.get(name, '')}")
        ax2.set_xlabel("Residue j")
        ax2.set_ylabel("Residue i")
        plt.colorbar(im, ax=ax2, label="||z[i,j]||")
        plt.tight_layout()
        fig2.savefig(out_dir / f"contact_map_{name}.png", dpi=200)
        plt.close(fig2)

    print(f"  Saved individual contact_map_*.png files")


# ---------------------------------------------------------------------------
# 5. 3D structure from diffusion decoder
# ---------------------------------------------------------------------------
def generate_structures_from_decoder(
    singles: dict,
    pairs: dict,
    rep_dir: Path,
    out_dir: Path,
    wts_path: Path,
    device: str,
):
    """Use the Boltz-1 diffusion decoder to generate 3D coordinates from cached
    representations, then plot Ca traces.

    This re-loads the model checkpoint, injects the saved s/z representations,
    and runs the full diffusion sampling loop.
    """
    import diffusion_stepper as ds
    from dataclasses import asdict

    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    # We need the test_reps PROTEINS dict for sequences
    from test_reps import PROTEINS, make_yaml_input, extract_sequence_from_pdb

    device_obj = torch.device(device) if device != "cpu" else None
    predict_args = ds.PredictArgs(recycling_steps=0)
    stepper = None

    all_coords = {}

    for name in singles:
        if name not in pairs:
            print(f"  Skipping {name}: no pair rep")
            continue
        if name not in PROTEINS:
            print(f"  Skipping {name}: not in PROTEINS dict")
            continue

        info = PROTEINS[name]
        # We need sequences — try to get from PDB or use what's stored
        if not any(v for v in info["chains"].values()):
            print(f"  Skipping {name}: no sequence")
            continue

        print(f"  Generating structure for {name}...")

        work_dir = struct_dir / f"_work_{name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        work_out = work_dir / "out"
        work_out.mkdir(parents=True, exist_ok=True)

        input_dir = make_yaml_input(name, info["chains"], work_dir)

        if stepper is None:
            stepper = ds.DiffusionStepper(
                checkpoint_path=wts_path,
                data_path=input_dir,
                out_dir=work_out,
                use_msa_server=False,
                predict_args=predict_args,
                device=device_obj,
            )
        else:
            stepper.setup(data_path=input_dir, out_dir=work_out, use_msa_server=False)

        # Load saved representations into the stepper cache
        s = singles[name].to(stepper.device)
        z = pairs[name].to(stepper.device)

        # We need feats from the data module
        batch = stepper.prepare_feats_from_datamodule_batch()
        # Run compute_representations to populate s_inputs, feats, etc.
        stepper.compute_representations(batch, recycling_steps=0)
        # Override s and z with saved representations
        stepper.cached_representations["s"] = s
        stepper.cached_representations["z"] = z

        # Run diffusion
        stepper.initialize_diffusion(num_samples=1, sampling_steps=200)
        coords = stepper.cached_diffusion_init["atom_coords"]

        for step_i in range(200):
            coords = stepper.step(
                atom_coords=coords,
                align_to_input=False,
                return_denoised=False,
                inject_noise=True,
            )

        # Extract unpadded coords
        pad_mask = stepper.cached_representations["feats"]["atom_pad_mask"].squeeze().bool()
        final_coords = coords[:, pad_mask, :].detach().cpu().numpy()
        all_coords[name] = final_coords[0]  # first sample

        # Save as .pt
        torch.save(torch.tensor(final_coords), struct_dir / f"{name}_coords.pt")

    if not all_coords:
        print("  No structures generated.")
        return

    # Plot Ca traces in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.get_cmap("tab10", len(all_coords))

    for i, (name, coords) in enumerate(all_coords.items()):
        # coords shape: [n_atoms, 3]
        # Subsample every 4th atom as rough Ca proxy (atom ordering: N, Ca, C, O, ...)
        ca_coords = coords[1::4]  # Ca is typically the 2nd atom per residue
        ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
                color=cmap(i), linewidth=1.5,
                label=f"{name} ({PROTEIN_INFO.get(name, '')})")

    ax.set_title("Diffusion decoder 3D backbone traces")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "decoder_3d_traces.png", dpi=200)
    plt.close(fig)
    print(f"  Saved decoder_3d_traces.png")


# ---------------------------------------------------------------------------
# 6. MDS 3D reconstruction from pair representations
# ---------------------------------------------------------------------------
def plot_mds_from_pairs(pairs: dict, out_dir: Path):
    """Reconstruct approximate 3D shapes from pair representations using MDS.

    Computes a pseudo-distance matrix from z as:
        dist[i,j] = 1 / (1 + cosine_similarity(z[i,j,:], z[j,i,:]))
    then uses classical MDS to embed in 3D.
    """
    from sklearn.manifold import MDS

    n_proteins = len(pairs)
    if n_proteins == 0:
        return

    all_mds_coords = {}

    for name, z in pairs.items():
        z = z.squeeze()  # [N, N, D]
        if z.dim() != 3:
            continue
        n_res = z.shape[0]

        # Cosine similarity between z[i,j,:] and z[j,i,:] (symmetrized)
        z_np = z.numpy()
        # Compute pairwise norms
        z_norm = np.linalg.norm(z_np, axis=-1, keepdims=True)
        z_norm = np.clip(z_norm, 1e-8, None)
        z_normalized = z_np / z_norm

        # Similarity: dot product of z[i,j,:] and z[j,i,:] for each (i,j)
        # This gives a scalar per residue pair
        sim = np.einsum("ijd,jid->ij", z_normalized, z_normalized)
        sim = (sim + sim.T) / 2  # symmetrize
        sim = np.clip(sim, -1, 1)

        # Convert to distance: high similarity = close
        dist = 1.0 - sim

        mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        coords_3d = mds.fit_transform(dist)
        all_mds_coords[name] = coords_3d

    if not all_mds_coords:
        return

    # Plot all proteins in a single 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap_colors = plt.cm.get_cmap("tab10", len(all_mds_coords))

    for i, (name, coords) in enumerate(all_mds_coords.items()):
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color=cmap_colors(i), linewidth=1.5, alpha=0.8,
                label=f"{name} ({PROTEIN_INFO.get(name, '')})")
        # Mark N and C termini
        ax.scatter(*coords[0], color=cmap_colors(i), marker="o", s=50, zorder=5)
        ax.scatter(*coords[-1], color=cmap_colors(i), marker="^", s=50, zorder=5)

    ax.set_title("MDS 3D reconstruction from pair representations")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("MDS 3")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "mds_3d_all.png", dpi=200)
    plt.close(fig)
    print(f"  Saved mds_3d_all.png")

    # Individual per-protein plots colored by residue index
    for name, coords in all_mds_coords.items():
        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        n_res = coords.shape[0]
        colors = np.arange(n_res)

        # Plot backbone as line
        ax2.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                 color="gray", linewidth=0.8, alpha=0.5)
        # Scatter colored by residue index
        sc = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                         c=colors, cmap="viridis", s=20, alpha=0.9)
        plt.colorbar(sc, ax=ax2, label="Residue index", shrink=0.6)

        ax2.set_title(f"{name} - {PROTEIN_INFO.get(name, '')}\nMDS from pair reps ({n_res} residues)")
        ax2.set_xlabel("MDS 1")
        ax2.set_ylabel("MDS 2")
        ax2.set_zlabel("MDS 3")
        plt.tight_layout()
        fig2.savefig(out_dir / f"mds_3d_{name}.png", dpi=200)
        plt.close(fig2)

    print(f"  Saved individual mds_3d_*.png files")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize Boltz-1 representations")
    parser.add_argument("--rep_dir", type=str, default="representations",
                        help="Directory containing *_s.pt and *_z.pt files")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--wts_path", type=str, default=None,
                        help="Path to boltz1_conf.ckpt (needed for diffusion decoder)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device for diffusion decoder")
    parser.add_argument("--mds", action="store_true",
                        help="Run MDS 3D reconstruction from pair representations")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="PDB directory (needed for diffusion decoder to get sequences)")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading representations from {rep_dir}")
    singles, pairs = load_representations(rep_dir)
    print(f"  Found {len(singles)} single reps, {len(pairs)} pair reps")

    if not singles:
        print("No representations found. Run test_reps.py first.")
        return

    # UMAPs
    print("Plotting single UMAP (by protein)...")
    plot_single_umap_by_protein(singles, out_dir)

    print("Plotting single UMAP (by position)...")
    plot_single_umap_by_position(singles, out_dir)

    if pairs:
        print("Plotting pair UMAP...")
        plot_pair_umap(pairs, out_dir)

    # Contact maps
    if pairs:
        print("Plotting contact maps...")
        plot_contact_maps(pairs, out_dir)

    # Diffusion decoder (requires model checkpoint)
    if args.wts_path and pairs:
        print("Generating 3D structures via diffusion decoder...")
        # Resolve sequences from PDB if pdb_dir given
        if args.pdb_dir:
            from test_reps import PROTEINS, extract_sequence_from_pdb
            for name, info in PROTEINS.items():
                pdb_path = Path(args.pdb_dir) / info["pdb_source"]
                if pdb_path.exists():
                    info["chains"] = extract_sequence_from_pdb(pdb_path)
        generate_structures_from_decoder(
            singles, pairs, rep_dir, out_dir,
            wts_path=Path(args.wts_path),
            device=args.device,
        )

    # MDS 3D reconstruction
    if args.mds and pairs:
        print("Running MDS 3D reconstruction from pair reps...")
        plot_mds_from_pairs(pairs, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
