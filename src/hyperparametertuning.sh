#!/bin/bash
#SBATCH -J vxm-optuna
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-3    # 4 parallel Optuna workers

# --- Environment setup ---

source ~/miniconda3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215      # your conda environment



# --- Run tuning script ---
python finetune_two_stage_optuna.py \
    --storage "$STORAGE" \
    --study "vxm_two_stage" \
    --trials $TRIALS_PER_WORKER \
    --seed $SEED
