#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4  # Match with the number of workers you want to use
#SBATCH --mem 32G
#SBATCH --partition small
#SBATCH --time 03:00:00
#SBATCH --job-name preprocess_ct_rate_data
#SBATCH --output logs/preprocess/preprocess_log-%J.txt
#SBATCH -A project_465001111

SCRATCH_DIR=/scratch/project_465001111/ct_clip
PROJECT_DIR=/projappl/project_465001111/ct_clip
CONTAINER=/projappl/project_465001111/ct_clip/ct_clip.sif

srun singularity exec -B $SCRATCH_DIR:/mnt/ct_clip_data \
                      -B $PROJECT_DIR:/mnt/ct_clip \
                      $CONTAINER python /mnt/ct_clip/scripts/dataset/preprocess_dataset.py \
                      --split valid --workers 4
