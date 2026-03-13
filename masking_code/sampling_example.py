"""Example: compare stochastic samplers on a single protein.

Demonstrates the four available samplers:
  1. default    - EDM deterministic (Karras et al.)
  2. langevin   - Annealed Langevin dynamics
  3. sde        - Euler-Maruyama SDE
  4. predictor_corrector - Predictor-Corrector (Song et al.)

Usage:
    python sampling_example.py
"""

import torch
import os
import pickle
from pathlib import Path

import diffusion_stepper as ds
from export import write_mmcif
from boltz.data.types import Structure

# ---- Configuration ---- #
wts_path = Path("~/.boltz/boltz1_conf.ckpt").expanduser()
ccd_path = Path("~/.boltz/ccd.pkl").expanduser()
data_path = Path("in")  # should contain 1 yaml or fasta file
out_dir = Path("out")

sampling_steps = 200
step_scale = 1.0
num_images = 1

# ---- Choose sampler ---- #
# Options: 'default', 'langevin', 'sde', 'predictor_corrector'
SAMPLER = "langevin"

# Sampler-specific kwargs
SAMPLER_KWARGS = {
    "default": {
        "inject_noise": True,
    },
    "langevin": {
        "n_langevin_steps": 3,       # Langevin corrections per noise level
        "langevin_step_size": 0.005,  # base step size (annealed by sigma)
    },
    "sde": {
        "s_churn": 40.0,  # stochasticity (0=deterministic, higher=more noise)
    },
    "predictor_corrector": {
        "n_corrector_steps": 2,        # Langevin corrections after each Euler step
        "corrector_step_size": 0.01,   # Langevin step size
    },
}

# ---- Setup ---- #
predict_args = ds.PredictArgs(recycling_steps=0)

stepper = ds.DiffusionStepper(
    checkpoint_path=wts_path,
    data_path=data_path,
    out_dir=out_dir,
    use_msa_server=False,
    predict_args=predict_args,
)

stepper.initialize_diffusion(num_samples=num_images, sampling_steps=sampling_steps)
stepper.model.structure_module.step_scale = step_scale

# Get structure for mmCIF export
input_file = None
with os.scandir(data_path) as d:
    for entry in d:
        if entry.name.endswith(".yaml") or entry.name.endswith(".fasta"):
            input_file = entry.name

npz_name = input_file.replace(".yaml", "").replace(".fasta", "")
structure = Structure.load(out_dir / f"processed/structures/{npz_name}.npz")

# ---- Run sampling ---- #
prefix = f"{npz_name}_{SAMPLER}"
print(f"Running {SAMPLER} sampler with {sampling_steps} steps...")

coords = stepper.run_sampling(
    sampler=SAMPLER,
    save_structs_every_n=50,
    save_current=True,
    save_perceived=True,
    prefix=prefix,
    out_dir=out_dir,
    structure=structure,
    **SAMPLER_KWARGS[SAMPLER],
)

# Save final structure
pad_mask = stepper.cached_representations["feats"]["atom_pad_mask"].squeeze().bool()
write_mmcif(coords[:, pad_mask, ...], structure, Path(out_dir / f"{prefix}_final.cif"))
print(f"Done. Final structure saved to {out_dir}/{prefix}_final.cif")
