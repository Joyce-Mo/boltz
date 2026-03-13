"""Extract single (s) and pair (z) representations from Boltz-1 for a set of proteins.

Computes trunk representations with 0 recycles (no MSA needed for small, well-folded
proteins) and saves them as PyTorch tensors with systematic names.

Target proteins:
  - 1SMG, 9CIF: Amy's switch single-state parents
  - 9CID, 9CIG: Two states of the switch
  - 2LV8: De novo Rossmann fold

Usage:
 python test_reps.py --wts_path /Users/joycemo/Documents/GitHub/boltz/boltz1_conf.ckpt --device mps

 """

import os
import sys
import argparse
import shutil
import tempfile
from pathlib import Path

import torch

# Assumes running from masking_code/ directory
import diffusion_stepper as ds


# ---- Protein sequences (extracted from PDB files) ---- #
# These are the canonical sequences so we don't need to parse PDB at runtime.
# If you want to re-extract from PDB, use BioPython.
PROTEINS = {
    "1smg": {
        "description": "Switch single-state parent (GTPase)",
        "pdb_source": "1SMG.pdb",
        "chains": {
            "A": None,  # will be filled from PDB or set manually
        },
    },
    "9cif": {
        "description": "Switch single-state parent",
        "pdb_source": "9CIF.pdb",
        "chains": {
            "A": None,
        },
    },
    "9cid": {
        "description": "Switch state 1",
        "pdb_source": "9CID.pdb",
        "chains": {
            "A": None,
        },
    },
    "9cig": {
        "description": "Switch state 2",
        "pdb_source": "9CIG.pdb",
        "chains": {
            "A": None,
        },
    },
    "2lv8": {
        "description": "De novo Rossmann fold",
        "pdb_source": "2LV8.pdb",
        "chains": {
            "A": None,
        },
    },
}


def extract_sequence_from_pdb(pdb_path: Path) -> dict[str, str]:
    """Extract chain sequences from a PDB file using BioPython."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    sequences = {}
    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.get_id()[0] == " "]
            if residues:
                seq = "".join(seq1(r.get_resname()) for r in residues)
                sequences[chain.id] = seq
        break  # first model only
    return sequences


def make_yaml_input(name: str, chain_sequences: dict[str, str], out_dir: Path) -> Path:
    """Create a Boltz YAML input file for a single protein."""
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = input_dir / f"{name}.yaml"
    lines = ["version: 1", "sequences:"]
    for chain_id, seq in chain_sequences.items():
        lines.append(f"  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        lines.append(f"      msa: empty")

    yaml_path.write_text("\n".join(lines) + "\n")
    return input_dir


def run_extraction(
    pdb_dir: Path,
    wts_path: Path,
    save_dir: Path,
    device: str = "cpu",
    recycling_steps: int = 0,
    use_msa_server: bool = False,
):
    """Extract and save representations for all target proteins.

    Parameters
    ----------
    pdb_dir : Path
        Directory containing source PDB files.
    wts_path : Path
        Path to boltz1_conf.ckpt.
    save_dir : Path
        Directory to save representation tensors.
    device : str
        'cpu', 'cuda', or 'mps'.
    recycling_steps : int
        Number of recycling steps (0 for efficiency).
    use_msa_server : bool
        Whether to query MSA server. False for small well-folded proteins.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sequences from PDB files
    for name, info in PROTEINS.items():
        pdb_path = pdb_dir / info["pdb_source"]
        if not pdb_path.exists():
            print(f"WARNING: {pdb_path} not found, skipping {name}")
            continue

        sequences = extract_sequence_from_pdb(pdb_path)
        info["chains"] = sequences

    # Use a single model load, re-setup data for each protein
    stepper = None

    for name, info in PROTEINS.items():
        if not any(v for v in info["chains"].values()):
            print(f"Skipping {name}: no sequence available")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {name}: {info['description']}")
        print(f"  Chains: {list(info['chains'].keys())}")
        for cid, seq in info["chains"].items():
            print(f"  Chain {cid}: {len(seq)} residues")
        print(f"  Recycling steps: {recycling_steps}")
        print(f"  MSA: {'server' if use_msa_server else 'none'}")
        print(f"{'='*60}")

        # Create temporary working directory for this protein
        work_dir = save_dir / f"_work_{name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir = work_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write YAML input
        input_dir = make_yaml_input(name, info["chains"], work_dir)

        predict_args = ds.PredictArgs(recycling_steps=recycling_steps)

        if stepper is None:
            # First protein: full initialization
            stepper = ds.DiffusionStepper(
                checkpoint_path=wts_path,
                data_path=input_dir,
                out_dir=out_dir,
                use_msa_server=use_msa_server,
                predict_args=predict_args,
                device=torch.device(device) if device != "cpu" else None,
            )
        else:
            # Reuse model, just re-setup data
            stepper.setup(data_path=input_dir, out_dir=out_dir, use_msa_server=use_msa_server)

        # Compute representations
        batch = stepper.prepare_feats_from_datamodule_batch()
        stepper.compute_representations(batch, recycling_steps=recycling_steps)

        # Save with systematic name
        tag = f"{name}_{recycling_steps}recycles"
        if use_msa_server:
            tag += "_msa"
        else:
            tag += "_nomsa"

        stepper.save_representations(save_dir, tag)

        s = stepper.cached_representations["s"]
        z = stepper.cached_representations["z"]
        print(f"  Saved: {tag}_s.pt  shape={list(s.shape)}")
        print(f"  Saved: {tag}_z.pt  shape={list(z.shape)}")

        # Clean up working directory
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nAll representations saved to: {save_dir}")
    print("Files:")
    for f in sorted(save_dir.glob("*.pt")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Extract Boltz-1 trunk representations")
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default="/Users/joycemo/Documents/GitHub/protpardelle-1c/examples/march_rep_test",
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--wts_path",
        type=str,
        default="/Users/joycemo/Documents/GitHub/boltz/boltz1_conf.ckpt",
        help="Path to Boltz-1 checkpoint",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="representations",
        help="Output directory for saved tensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=0,
        help="Number of recycling steps (0 for efficiency)",
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Query ColabFold MSA server (default: off for small proteins)",
    )
    args = parser.parse_args()

    run_extraction(
        pdb_dir=Path(args.pdb_dir),
        wts_path=Path(args.wts_path).expanduser(),
        save_dir=Path(args.save_dir),
        device=args.device,
        recycling_steps=args.recycling_steps,
        use_msa_server=args.use_msa_server,
    )


if __name__ == "__main__":
    main()
