#!/bin/bash
#SBATCH --job-name=download_positive_urls
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH -c 1
#SBATCH --time=12:00:00
#SBATCH --output=download_positive_urls_%j.out
#SBATCH --error=download_positive_urls_%j.err

./cs336_data/quality_classifier/02-download_positive_urls.sh

