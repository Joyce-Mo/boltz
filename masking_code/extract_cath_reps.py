# extract_cath_reps.py
"""Extract single (s) and pair (z) representations from Boltz for CATH domains.

Reads PDB files from a directory, extracts sequences, builds YAML inputs,
and runs representation extraction through Boltz1 or Boltz2.

Usage (from repo root):
  python masking_code/extract_cath_reps.py \
      --model_version boltz1 \
      --checkpoint boltz1_conf.ckpt \
      --pdb_dir /wynton/home/rotation/jqmo/rotation3/datasets/cath20-filtered-foldseek \
      --save_dir /wynton/home/rotation/jqmo/rotation3/datasets/cath20_reps/boltz1 \
      --device cuda
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

# Ensure masking_code/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_reps import extract_sequence_from_pdb, make_yaml_input


def load_cath_pdbs(pdb_dir: Path) -> dict[str, dict[str, str]]:
    """Load CATH domain sequences from a directory of PDB files.

    Scans for *.pdb files, extracts chain sequences from each.
    Returns dict like {"1abcA00": {"A": "MKTL..."}, ...}
    """
    pdb_dir = Path(pdb_dir)
    sequences = {}
    pdb_files = sorted(pdb_dir.glob("*.pdb"))

    if not pdb_files:
        print(f"ERROR: No .pdb files found in {pdb_dir}")
        sys.exit(1)

    for pdb_path in pdb_files:
        domain_id = pdb_path.stem
        try:
            chains = extract_sequence_from_pdb(pdb_path)
            if chains:
                sequences[domain_id] = chains
            else:
                print(f"  WARNING: No chains extracted from {pdb_path.name}, skipping")
        except Exception as e:
            print(f"  WARNING: Failed to parse {pdb_path.name}: {e}, skipping")

    return sequences


def get_stepper_module(model_version: str):
    """Import the correct diffusion_stepper module for boltz1 or boltz2."""
    if model_version == "boltz1":
        import diffusion_stepper as ds
    elif model_version == "boltz2":
        import diffusion_stepper_boltz2 as ds
    else:
        raise ValueError(f"Unknown model version: {model_version}. Use 'boltz1' or 'boltz2'.")
    return ds


def run_extraction(
    model_version: str,
    checkpoint: Path,
    cath_sequences: dict[str, dict[str, str]],
    save_dir: Path,
    device: str = "cuda",
    recycling_steps: int = 0,
):
    ds = get_stepper_module(model_version)

    save_dir.mkdir(parents=True, exist_ok=True)
    stepper = None
    predict_args = ds.PredictArgs(recycling_steps=recycling_steps)

    total = len(cath_sequences)
    for i, (domain_id, chains) in enumerate(cath_sequences.items(), 1):
        # Skip if already extracted
        if (save_dir / f"{domain_id}_s.pt").exists() and (save_dir / f"{domain_id}_z.pt").exists():
            print(f"[{i}/{total}] {domain_id} already exists, skipping")
            continue

        print(f"\n[{i}/{total}] Processing {domain_id} ({model_version})")
        for cid, seq in chains.items():
            print(f"  Chain {cid}: {len(seq)} residues")

        work_dir = save_dir / f"_work_{domain_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir = work_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        input_dir = make_yaml_input(domain_id, chains, work_dir)

        try:
            if stepper is None:
                stepper = ds.DiffusionStepper(
                    checkpoint_path=checkpoint,
                    data_path=input_dir,
                    out_dir=out_dir,
                    use_msa_server=False,
                    predict_args=predict_args,
                    device=torch.device(device),
                )
            else:
                stepper.setup(data_path=input_dir, out_dir=out_dir, use_msa_server=False)

            batch = stepper.prepare_feats_from_datamodule_batch()
            stepper.compute_representations(batch, recycling_steps=recycling_steps)
            stepper.save_representations(save_dir, domain_id)

            s = stepper.cached_representations["s"]
            z = stepper.cached_representations["z"]
            print(f"  Saved: {domain_id}_s.pt shape={list(s.shape)}")
            print(f"  Saved: {domain_id}_z.pt shape={list(z.shape)}")
        except Exception as e:
            print(f"  ERROR processing {domain_id}: {e}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nDone. Processed {total} domains -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract Boltz representations for CATH domains")
    parser.add_argument("--model_version", type=str, required=True, choices=["boltz1", "boltz2"],
                        help="Which model to use: boltz1 or boltz2")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing CATH PDB files")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for saved representation tensors")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"],
                        help="Device for inference (default: cuda)")
    parser.add_argument("--recycling_steps", type=int, default=0,
                        help="Number of recycling steps (default: 0)")
    args = parser.parse_args()

    cath_sequences = load_cath_pdbs(Path(args.pdb_dir))
    print(f"Loaded {len(cath_sequences)} CATH domains from {args.pdb_dir}")

    run_extraction(
        model_version=args.model_version,
        checkpoint=Path(args.checkpoint),
        cath_sequences=cath_sequences,
        save_dir=Path(args.save_dir),
        device=args.device,
        recycling_steps=args.recycling_steps,
    )


if __name__ == "__main__":
    main()
