#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=4G
#$ -l scratch=2G
#$ -l h_rt=18:00:00
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu
#$ -q gpu.q
#$ -l gpu_mem=16G

date
hostname

# ---------- modules ----------
if command -v module >/dev/null 2>&1; then
    module load gcc/12.4.0
    module load cuda
fi

# ---------- environment ----------
if [ -z "$ENV_DIR" ]; then
    conda activate boltz
else
    source "$ENV_DIR"/boltz/bin/activate
fi

# ---------- paths (edit these) ----------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Directory of CATH-20 PDB files
PDB_DIR="/wynton/home/rotation/jqmo/rotation3/datasets/cath20-filtered-foldseek"

# Where to save output representations (separate subdirs per model)
OUTPUT_BASE="/wynton/home/rotation/jqmo/rotation3/datasets/cath20_reps"

# Checkpoints (relative to repo root, or use absolute paths)
BOLTZ1_CKPT="${REPO_ROOT}/boltz1_conf.ckpt"
BOLTZ2_CKPT="${REPO_ROOT}/boltz2_conf.ckpt"

# ---------- run boltz1 ----------
echo "========== Extracting Boltz-1 representations =========="
python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
    --model_version boltz1 \
    --checkpoint "${BOLTZ1_CKPT}" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}/boltz1" \
    --device cuda

# ---------- run boltz2 ----------
echo "========== Extracting Boltz-2 representations =========="
python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
    --model_version boltz2 \
    --checkpoint "${BOLTZ2_CKPT}" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}/boltz2" \
    --device cuda

# ---------- end-of-job ----------
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
