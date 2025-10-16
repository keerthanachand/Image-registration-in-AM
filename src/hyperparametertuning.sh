#!/bin/bash
#SBATCH -J vxm-optuna
#SBATCH -p batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH -o logs_hyper/%x_%A_%a.out
#SBATCH -e logs_hyper/%x_%A_%a.err
#SBATCH --array=0-3    # 4 parallel Optuna workers

# --- Environment setup ---


source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215      # your conda environment



# --- Run tuning script ---
python /home/kchand/image_registration/src/hyperparameter_tuning.py