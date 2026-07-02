#!/bin/bash
#SBATCH --job-name=FishYOLO11
#SBATCH --partition=h200
#SBATCH --gres=gpu:1g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

cd ~/FishProject

source ~/env/bin/activate

python scripts/train_yolo11.py
