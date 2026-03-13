# masking_code

`masking_code` is an experimental toolkit for representation-level control of **Boltz-1** inference.

It is designed for workflows where you want to:
- run Boltz preprocessing and trunk representation computation,
- intervene on internal representations (especially pair representation `z`),
- step diffusion manually (instead of one-shot `boltz predict`),
- inspect and export intermediate structures across denoising steps.

This directory is best understood as a research playground built on top of Boltz internals, not as a stable product API.

## Purpose

Boltz's standard inference path (`boltz predict ...`) is optimized for complete predictions from input to final structures. `masking_code` exists for experiments where you need explicit control over the latent conditioning during structure generation.

Typical use cases:
- zero out selected token-token communication channels in pair representation `z`,
- perform partial diffusion from an initial structure and selectively noise atoms,
- save trajectories and denoised targets at each step for analysis,
- prototype guided generation loops that combine model denoising with external guidance.

## How It Integrates With Boltz

`masking_code` reuses major pieces of the main Boltz stack:
- model loading: `Boltz1.load_from_checkpoint(...)`
- input preprocessing: `check_inputs(...)` and `process_inputs(...)`
- inference data module: `BoltzInferenceDataModule`
- structure diffusion network: `model.structure_module.preconditioned_network_forward(...)`
- alignment utility: `weighted_rigid_align(...)`
- mmCIF writer backend: `boltz.data.write.mmcif.to_mmcif(...)`

At a high level, integration looks like:

1. Preprocess YAML/FASTA input into `out/processed/...` (same style as core Boltz).
2. Build one inference batch via `BoltzInferenceDataModule`.
3. Compute trunk representations (`s`, `z`, `s_inputs`, positional features).
4. Apply user-supplied representation edits (`pair_rep_func`, intended `single_rep_func`).
5. Initialize diffusion state (from noise or from a provided structure).
6. Run denoising one step at a time with explicit control over noise injection and schedule usage.
7. Export intermediate coordinates as mmCIF.

## Directory Contents

- `masking_example.py`
  - Example script that configures paths/params, defines a pair-representation mask, runs iterative diffusion, and writes stepwise outputs.
- `diffusion_stepper.py`
  - Core class (`DiffusionStepper`) for trunk caching + controlled diffusion stepping.
  - Also includes `DensityGuidedDiffusionStepper` for gradient-guided updates.
- `export.py`
  - Lightweight mmCIF export helper for batch coordinates using Boltz `Structure` metadata.
- `mmcif.py`
  - Local parser fork for reading less-ideal CIF/PDB formatting into Boltz structure types.
- `rcsb.py`
  - Bulk data processing script for local CIF collections (separate from the online masking workflow).
- `in/9C91.yaml`
  - Minimal sample YAML input.

## End-to-End Workflow (`masking_example.py`)

`masking_example.py` is the reference run loop:

1. Configure paths and hyperparameters:
   - checkpoint (`wts_path`), CCD (`ccd_path`), input dir (`data_path`), output dir (`out_dir`)
   - diffusion controls (`sampling_steps`, `step_scale`, `inject_noise`, etc.)
2. Define representation intervention function(s):
   - `pair_rep_mask_interdomain(...)`
   - `pair_rep_explicit_mask(...)`
3. Construct `DiffusionStepper(...)` with `pair_rep_func=...`.
4. Call `initialize_diffusion(...)` (or partial diffusion alternative).
5. Iterate `stepper.step(...)` for each denoising iteration.
6. Save selected states as mmCIF:
   - current iterative coordinates (`..._current.cif`)
   - model denoised target estimate (`..._perceived.cif`)

### Key intervention point

The important concept is that edits happen after trunk computation and before structure denoising:

- pair rep tensor shape is roughly `[B, N, N, C_pair]`
- masking selected blocks in `z` changes what residue/token pairs can communicate downstream
- diffusion then runs with this modified conditioning for all subsequent steps

## `DiffusionStepper` API and Behavior

### Constructor

`DiffusionStepper(...)`:
- loads a Boltz-1 checkpoint if no model object is provided,
- runs `setup(...)` to preprocess inputs and prepare a dataloader,
- stores optional hooks:
  - `pair_rep_func(z) -> z`
  - `single_rep_func(s) -> s` (intended; see compatibility notes)

### Representation stage

`compute_representations(feats, recycling_steps=None)`:
- computes input embeddings,
- executes recycling + pairformer stack,
- applies optional representation hook(s),
- caches tensors needed for repeated diffusion steps.

Cached fields include:
- `s`, `z`
- `s_inputs`
- `relative_position_encoding`
- `feats` (contains masks and structural metadata)

### Diffusion initialization

Two initialization modes are implemented:

- `initialize_diffusion(...)`
  - starts from Gaussian coordinates at initial sigma.
- `initialize_partial_diffusion(structure, noising_steps, selector, ...)`
  - starts from provided coordinates,
  - adds noise globally or only on selected atoms,
  - supports controlled perturbation around a known structure.

### Single-step denoising

`step(atom_coords, ...)` performs:
- schedule lookup (`sigma_tm`, `sigma_t`, `gamma`),
- optional stochastic noise injection,
- structure module forward pass,
- rigid alignment,
- Euler-like update to next coordinates,
- optional alignment back to initial structure,
- trajectory cache update (`diffusion_trajectory[step_k]`).

It can return:
- next coordinates,
- optionally denoised estimate from current noise level (`return_denoised=True`).

## Output Artifacts

From a typical run with `out_dir = out`:

- Boltz preprocessing outputs:
  - `out/processed/manifest.json`
  - `out/processed/structures/*.npz`
  - `out/processed/msa/*`
- masking trajectory outputs:
  - `out/<prefix>_<step>_current.cif`
  - `out/<prefix>_<step>_perceived.cif`

`export.write_mmcif(...)` supports batched coordinates and writes one file per sample if batch size > 1.

## `rcsb.py` Scope

`rcsb.py` is not required for normal `masking_example.py` runs.

It is a separate preprocessing utility for local mmCIF collections:
- scans CIF files,
- parses structures through local `mmcif.py`,
- applies static filters,
- writes NPZ structures + JSON records + manifest.

Use this when building custom local datasets or debugging parse/filter behavior at scale.

## Environment and Runtime Expectations

The scripts assume:
- a working Boltz Python environment,
- a compatible Boltz-1 checkpoint,
- CCD dictionary available locally,
- optional internet access if MSA server usage is enabled.

`masking_example.py` path defaults are user-specific and should be updated for your machine.

## Compatibility Notes (Important)

This folder is experimental and may drift from the current Boltz API.

Known mismatches in this repository state:
- `DiffusionStepper.setup()` calls `check_inputs(input_path, out_dir, False)`, while current `check_inputs` accepts only one argument.
- `process_inputs(...)` now expects `mol_dir` in core Boltz, but the call in `diffusion_stepper.py` does not provide it.
- In `compute_representations(...)`, the `single_rep_func` branch currently calls `self.pair_rep_func(s)` instead of `self.single_rep_func(s)`.

Practical implication:
- treat `masking_code` as a reference implementation of ideas,
- expect light refactoring to run cleanly against the latest `src/boltz` APIs.

## Extending `masking_code`

Recommended extension pattern:

1. Keep preprocessing and datamodule behavior aligned with `src/boltz/main.py`.
2. Add intervention functions that are pure tensor transforms.
3. Log masks, schedules, and output metrics per step for reproducibility.
4. Keep trajectory exports deterministic when debugging (`inject_noise=False`, fixed seeds).
5. Separate stable interfaces (runner + config) from experimental logic (mask/guidance functions).

## Conceptual Difference vs FX-Only Instrumentation

Compared with a pure PyTorch FX approach:
- `masking_code` is model-aware and directly integrated with Boltz diffusion semantics.
- FX is model-agnostic graph tracing/transformation; useful for extraction but not a complete controlled diffusion workflow by itself.

If your goal is representation intervention + downstream structural effect analysis, `masking_code` is the more direct workflow.

## Quick Start Checklist

1. Install Boltz and dependencies in a fresh environment.
2. Update paths in `masking_example.py` (`wts_path`, `ccd_path`, `data_path`, `out_dir`).
3. Put one input YAML/FASTA in `masking_code/in/`.
4. Define your `pair_rep_func` (or patch single-rep hook behavior).
5. Run `python masking_code/masking_example.py`.
6. Inspect `out/*.cif` trajectory files.

## Current Limitations

- Assumes batch size 1 in several places.
- `masking_example.py` is script-style (no CLI or config schema).
- No formal tests under `tests/` for this folder.
- API drift risk as Boltz evolves.

---

If you plan to rely on this for repeated studies, the next step is to harden it into a versioned module with:
- explicit config files,
- compatibility adapters for `src/boltz/main.py` changes,
- unit/integration tests around representation hooks and trajectory outputs.
