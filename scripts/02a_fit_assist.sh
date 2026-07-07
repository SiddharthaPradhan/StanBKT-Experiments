#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH -J "02_fit_assist"
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -o logs/02_fit_assist_%j.out
#SBATCH -e logs/02_fit_assist_%j.err

set -euo pipefail

module load python/3.13.5/6anz4qy

export PYTHONUNBUFFERED=1
source .venv/bin/activate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 02_fit_assist"

uv run src/02_fit_assist.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished 02_fit_assist"