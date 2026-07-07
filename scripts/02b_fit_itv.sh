#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=70
#SBATCH --mem=64G
#SBATCH -J "02b_fit_itv"
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -o logs/02b_fit_itv_%j.out
#SBATCH -e logs/02b_fit_itv_%j.err

set -euo pipefail

module load python/3.13.5/6anz4qy

export PYTHONUNBUFFERED=1
source .venv/bin/activate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 02b_fit_itv"

uv run src/02_fit_itv.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished 02b_fit_itv"