#!/bin/bash
#SBATCH --job-name=train_ensemble
#SBATCH --output=logs_hyper/%x_%A_%a.out
#SBATCH --error=logs_hyper/%x_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --array=0-9%4

source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215

mkdir -p logs_hyper

export PYTHONUNBUFFERED=1
export FIT_WORKERS=1

SCRATCH="${SLURM_TMPDIR:-/tmp}"

nvidia-smi || true

cp /home/kchand/input_data/data_split_Simple_structures/train_data_temp.h5 "$SCRATCH/"
cp /home/kchand/input_data/data_split_Simple_structures/val_data_temp.h5   "$SCRATCH/"

export TRAIN_H5="$SCRATCH/train_data_temp.h5"
export VAL_H5="$SCRATCH/val_data_temp.h5"

cd /home/kchand/image_registration
srun python -m src.training.train_ensemble_simple_structures
