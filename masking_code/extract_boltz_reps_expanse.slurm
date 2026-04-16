#!/bin/bash
#SBATCH -A ucb368
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J extract_boltz_reps
#SBATCH -o logs/extract_boltz_reps_04152026_%j.out
#SBATCH -e logs/extract_boltz_reps_04152026_%j.err
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

date
hostname
nvidia-smi

## Load modules
module purge
module load cpu/0.15.4
module load gcc/10.2.0
module load gpu/0.15.4

## paths
REPO_ROOT="/expanse/lustre/scratch/jmo/temp_project/boltz"
PDB_DIR="/expanse/lustre/scratch/jmo/temp_project/cath20-filtered-foldseek"
OUTPUT_BASE="/expanse/lustre/scratch/jmo/temp_project/boltz2_embeddings"

## Boltz resolves ccd.pkl and mols/ relative to the checkpoint's parent dir,
## so keep boltz2_conf.ckpt, ccd.pkl, and mols/ in REPO_ROOT.
BOLTZ2_CKPT="${REPO_ROOT}/boltz2_conf.ckpt"

mkdir -p "${OUTPUT_BASE}/boltz2" logs

cd "$REPO_ROOT"

## ---------------- conda env ----------------
## On Expanse, `module load anaconda3` does not initialize conda for
## non-interactive shells, so `conda activate` fails. The eval hook
## runs the equivalent of `conda init bash` inline for this session.
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate boltz

python -c "import boltz, torch; print('boltz', boltz.__file__); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

## ---------------- ensure ccd.pkl + mols/ are present ----------------
## Boltz2 needs mols/ extracted from mols.tar in the same dir as the ckpt.
## This call only downloads what is missing.
# python - <<PY
# from pathlib import Path
# from boltz.main import download_boltz2
# cache = Path("${REPO_ROOT}")
# cache.mkdir(parents=True, exist_ok=True)
# download_boltz2(cache)
# print("cache ready:", sorted(p.name for p in cache.iterdir()))
# PY

## ---------------- run boltz2 ----------------
echo "==================== boltz2 reps ===================="
python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
    --model_version boltz2 \
    --checkpoint "${BOLTZ2_CKPT}" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}/boltz2" \
    --device cuda

## ---------------- UMAPs ----------------
FIG_BASE="${REPO_ROOT}/masking_code/figures"
mkdir -p "${FIG_BASE}/boltz2_umap"

echo "==================== boltz2 UMAPs ===================="
python "${REPO_ROOT}/masking_code/plot_cath_umaps.py" \
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
