#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

# Load environment (adjust path if needed)
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf2

# Run the test
python /home/kchand/image_registration/src/gpu_test.py
