#!/bin/bash
#SBATCH --job-name=voxel_cv
#SBATCH --output=logs/sample15_steps50epoch200_%j.out
#SBATCH --error=logs/sample15_steps50epoch200_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00  # 3 days, adjust as needed

### Mail to user when job start, terminate or abort
### Options: BEGIN|END|FAIL|REQUEUE|ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=keerthana.chand@bam.de


# Load environment (adjust path if needed)
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215


# Run the test
python /home/kchand/image_registration/src/cross_validation.py
