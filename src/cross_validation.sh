#!/bin/bash
#SBATCH --job-name=voxel_cv
#SBATCH --output=logs/cross_val_%j.out
#SBATCH --error=logs/cross_val_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00  # 3 days, adjust as needed


# Load environment (adjust path if needed)
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215


# Run the test
python /home/kchand/image_registration/src/cross_validation.py
