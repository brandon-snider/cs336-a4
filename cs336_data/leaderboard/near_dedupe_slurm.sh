#!/bin/bash
#SBATCH --job-name=near_dedupe
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --output=near_dedupe_%j.out
#SBATCH --error=near_dedupe_%j.err

uv run -m cs336_data.leaderboard.near_dedupe

