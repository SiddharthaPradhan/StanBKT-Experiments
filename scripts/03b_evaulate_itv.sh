#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH -J "03_evaulate_itv"
#SBATCH -p short
#SBATCH -t 04:00:00
#SBATCH -o logs/03_evaulate_itv_%j.out
#SBATCH -e logs/03_evaulate_itv_%j.err

set -euo pipefail

module load python/3.13.5/6anz4qy

export PYTHONUNBUFFERED=1
source .venv/bin/activate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 03_evaulate_itv"

uv run src/03_evaulate_itv.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished 03_evaulate_itv"