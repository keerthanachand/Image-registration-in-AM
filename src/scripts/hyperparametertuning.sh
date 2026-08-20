#!/bin/bash
#SBATCH --job-name=vxm-optuna
#SBATCH --output=logs_hyper/%x_%A_%a.out
#SBATCH --error=logs_hyper/%x_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=6-00:00:00
#SBATCH --array=0-3

mkdir -p logs_hyper

# env
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215

export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"



# (optional) sanity
nvidia-smi || true

# run
cd /home/kchand/image_registration
srun python -m src.training.hyperparameter_tuning_TPMS


