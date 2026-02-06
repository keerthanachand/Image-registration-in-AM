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



# env
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215

export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"



# (optional) sanity
nvidia-smi || true




srun python /home/kchand/image_registration/src/train_ensemble.py 


