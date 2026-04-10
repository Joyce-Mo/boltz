#!/bin/bash
#SBATCH -A ucb368
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -J extract_boltz_reps
#SBATCH -o logs/extract_boltz_reps_%j.out
#SBATCH -e logs/extract_boltz_reps_%j.err
#SBATCH -p gpu
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

date
hostname
nvidia-smi

## Load modules — match the toolchain used by the pixi `boltz` env (cu126).
module purge
module load gpu/0.15.4
module load gcc/10.2.0
module load cuda/12.2.1

## ---------------- paths ----------------
REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/boltz"
PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/datasets/cath20-filtered-foldseek"
OUTPUT_BASE="/expanse/lustre/scratch/jmo/temp_project/boltz2_embeddings"


## Boltz expects ckpt + ccd.pkl + mols/ to live in the same `cache` directory.
CACHE_DIR="${REPO_ROOT}"
BOLTZ2_CKPT="${CACHE_DIR}/boltz2_conf.ckpt"

mkdir -p "${OUTPUT_BASE}/boltz2" logs

cd "$REPO_ROOT"

## ---------------- pixi env / CUDA wiring ----------------
## Point torch.utils.cpp_extension at the pixi-provided CUDA toolkit so any
## JIT-compiled CUDA ops use the same CUDA the boltz env's torch was built
## against (cu126).
PIXI_BOLTZ_PREFIX="${REPO_ROOT}/.pixi/envs/boltz"
export CUDA_HOME="$PIXI_BOLTZ_PREFIX"
export PATH="${PIXI_BOLTZ_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PIXI_BOLTZ_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.torch_extensions/${SLURM_JOB_ID:-local}"
export PYTORCH_KERNEL_CACHE_PATH="${REPO_ROOT}/.torch_kernel_cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$PYTORCH_KERNEL_CACHE_PATH"

pixi run -e boltz python -c "import boltz, torch; print('boltz', boltz.__file__); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

## ---------------- ensure ccd.pkl + mols/ are present ----------------
## download_boltz2() can time out on this cluster, so fetch mols.tar directly
## from HuggingFace if it's missing.
MOLS_DIR="${CACHE_DIR}/mols"
if [ ! -d "$MOLS_DIR" ] || [ -z "$(ls -A "$MOLS_DIR" 2>/dev/null || true)" ]; then
    echo "mols/ not found at $MOLS_DIR — downloading mols.tar from HuggingFace"
    curl -L --retry 5 --retry-delay 10 \
        -o "${CACHE_DIR}/mols.tar" \
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
    tar -xf "${CACHE_DIR}/mols.tar" -C "$CACHE_DIR"
    rm -f "${CACHE_DIR}/mols.tar"
    echo "mols/ ready at $MOLS_DIR"
fi

## ---------------- run boltz2 ----------------
echo "==================== boltz2 reps ===================="
pixi run -e boltz python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
    --model_version boltz2 \
    --checkpoint "${BOLTZ2_CKPT}" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}/boltz2" \
    --device cuda

## ---------------- UMAPs ----------------
FIG_BASE="${REPO_ROOT}/masking_code/figures"
mkdir -p "${FIG_BASE}/boltz2_umap"

echo "==================== boltz2 UMAPs ===================="
pixi run -e boltz python "${REPO_ROOT}/masking_code/plot_cath_umaps.py" \
    --rep_dir  "${OUTPUT_BASE}/boltz2" \
    --pdb_dir  "${PDB_DIR}" \
    --out_dir  "${FIG_BASE}/boltz2_umap" \
    --model_tag boltz2 \
    --max_residues 200000 \
    --max_points   200000

## ---------------- end-of-job ----------------
date
echo "Job $SLURM_JOB_ID complete."
if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State || true
fi
