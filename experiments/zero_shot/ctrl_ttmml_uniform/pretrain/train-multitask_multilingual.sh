#!/bin/bash
#SBATCH --job-name=multitask
#SBATCH --ntasks=1
#SBATCH -p gpu --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

JSON_FILE=ttml_pretrain.json

cd ../../../../volta
python train_concap_ttml.py \
  --config_ttml_paths ttml/config/$JSON_FILE





