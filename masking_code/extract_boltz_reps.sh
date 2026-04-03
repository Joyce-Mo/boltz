#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o logs/extract_boltz1_reps_$JOB_ID_$date.out
#$ -e logs/extract_boltz1_reps_$JOB_ID_$date.err
#$ -l mem_free=48G
#$ -l scratch=2G
#$ -l h_rt=96:00:00
#$ -r y
#$ -m bea
#$ -M jqmo@berkeley.edu

date
hostname

source activate boltz

# paths 
REPO_ROOT="/wynton/home/rotation/jqmo/rotation3/boltz"

# Directory of CATH-20 PDB files
PDB_DIR="/wynton/home/rotation/jqmo/rotation3/datasets/cath20-filtered-foldseek"

# Where to save output representations (separate subdirs per model)
OUTPUT_BASE="/wynton/scratch/jqmo/rotation_datasets/cath20_reps/boltz1_reps"

# Checkpoints 
BOLTZ1_CKPT="${REPO_ROOT}/boltz1_conf.ckpt"
BOLTZ2_CKPT="${REPO_ROOT}/boltz2_conf.ckpt"

# ---------- run boltz1 ----------
# echo "boltz1 reps" d
# python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
#     --model_version boltz1 \
#     --checkpoint "${BOLTZ1_CKPT}" \
#     --pdb_dir "${PDB_DIR}" \
#     --save_dir "${OUTPUT_BASE}/boltz1" \

# ---------- run boltz2 ----------
echo "boltz2 reps"
python "${REPO_ROOT}/masking_code/extract_cath_reps.py" \
    --model_version boltz2 \
    --checkpoint "${BOLTZ2_CKPT}" \
    --pdb_dir "${PDB_DIR}" \
    --save_dir "${OUTPUT_BASE}/boltz2" \

# ---------- end-of-job ----------
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
