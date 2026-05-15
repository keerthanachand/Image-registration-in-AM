#!/bin/bash
#SBATCH --job-name=eval_with_uncert
#SBATCH --output=logs/evalmodelTPMSonsimpledata_%j.out
#SBATCH --error=logs/evalmodelTPMSonsimpledata_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00  # 3 days, adjust as needed

### Mail to user when job start, terminate or abort
### Options: BEGIN|END|FAIL|REQUEUE|ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=keerthana.chand@bam.de


# Load environment (adjust path if needed)
source /home/kchand/miniforge3/etc/profile.d/conda.sh
conda activate voxelmorph_tf215


# Run the test
python /home/kchand/image_registration/src/ensemble_uncertainity.py