#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -J "01_process_data"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -o logs/01_process_data_%j.out
#SBATCH -e logs/01_process_data_%j.err

set -euo pipefail

module load python/3.13.5/6anz4qy

export PYTHONUNBUFFERED=1
source .venv/bin/activate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 01_process_data"


uv run src/01_process_data.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished 01_process_data"
