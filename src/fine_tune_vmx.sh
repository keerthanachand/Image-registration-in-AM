#!/bin/bash
#SBATCH --job-name=fine_tune_
#SBATCH --output=logs/fine_tune_%j.out
#SBATCH --error=logs/fine_tune_%j.err
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
python /home/kchand/image_registration/src/fine_tune_vxm.py
