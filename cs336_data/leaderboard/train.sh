#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=a4-batch
#SBATCH --qos=a4-batch-qos
#SBATCH --time=12:00:00
#SBATCH --gpus=2
#SBATCH --output=train_model_%j.out
#SBATCH --error=train_model_%j.err

uv run torchrun --standalone --nproc_per_node=2 \
    scripts/train.py \
    --config-name=experiment/v01-thresholded-005 \
    +training.save_checkpoints=True