#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --partition small
#SBATCH --time 10:00:00
#SBATCH --job-name download_ct_rate_data
#SBATCH --output logs/download/download_log-%J.txt
#SBATCH -A project_465001111

SCRATCH_DIR=/scratch/project_465001111/ct_clip
PROJECT_DIR=/projappl/project_465001111/ct_clip
CONTAINER=/projappl/project_465001111/ct_clip/ct_clip.sif

srun singularity exec -B $SCRATCH_DIR:/mnt/ct_clip_data \
                      -B $PROJECT_DIR:/mnt/ct_clip \
                      $CONTAINER python /mnt/ct_clip/scripts/dataset/download_dataset.py \
                      --split train --start_at 0 --end_at 50
