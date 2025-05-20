#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=a4-cpu
#SBATCH --qos=a4-cpu-qos
#SBATCH --cpus-per-task=1
#SBATCH -c 1
#SBATCH --time=00:10:00
#SBATCH --output=filter_%j.out
#SBATCH --error=filter_%j.err

uv run -m cs336_data.leaderboard.filter --max-files 1001

