#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -J test-gpu-access
#SBATCH -o logs/test_gpu_%j.out
#SBATCH -e logs/test_gpu_%j.err
#SBATCH -p short
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1

module load python/3.13.3/23ldx7y
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
