#!/bin/bash
#SBATCH -p GPU                  # Use the GPU partition
#SBATCH --gres=gpu:v100-32:8     # Request 8 V100 GPUs
#SBATCH -t 48:00:00              # Set a 48-hour time limit
#SBATCH -A cis250053p            # Charge job to your group account
#SBATCH --job-name=train_emg     # Set a job name
#SBATCH --output=logs/%x-%j.out  # Save output logs in logs/ folder
#SBATCH --error=logs/%x-%j.err   # Save error logs separately

# Load necessary modules
module load cuda/12.6.1          # Ensure correct CUDA version
module load python/3.8.6         # Use the correct Python version

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emg2qwerty

# Debugging info
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA version:"
nvcc --version

# Resume training from the best checkpoint
python -m emg2qwerty.train \
  +user=generic \
  +trainer.accelerator=gpu +trainer.devices=8 \
  +trainer.resume_from_checkpoint="/ocean/projects/cis250053p/clee18/emg2qwerty/logs/2025-03-15/12-16-38/job0_trainer.devices=8,user=generic/checkpoints/epoch=144-step=129920.ckpt"
