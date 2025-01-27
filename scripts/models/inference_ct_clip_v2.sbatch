#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 60G
#SBATCH --partition small-g
#SBATCH --time 24:00:00
#SBATCH --job-name ct_clip_v2_inference
#SBATCH --output logs/ct_clip_v2/inference_log-%J.txt
#SBATCH -A project_465001111

HOME_DIR=/users/injarabi
SCRATCH_DIR=/scratch/project_465001111/ct_clip
PROJECT_DIR=/projappl/project_465001111/ct_clip
CONTAINER=/projappl/project_465001111/ct_clip/ct_clip.sif

srun singularity exec -B $HOME_DIR:/mnt/ct_clip_home \
                      -B $SCRATCH_DIR:/mnt/ct_clip_data \
                      -B $PROJECT_DIR:/mnt/ct_clip \
                      $CONTAINER python /mnt/ct_clip/CT-CLIP-UT/scripts/run_zero_shot.py
