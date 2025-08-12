#!/bin/bash
#SBATCH --job-name=geno-pheno
#SBATCH --account=gftabor
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

module load miniconda3/25.1.1/24g7bpu
source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/etc/profile.d/conda.sh
conda activate geno-pheno-env

cd /home/vmeel/2025-geno-pheno-attention
export PYTHONPATH=$PWD
python src/analysis/train.py --train.phenotypes 30C
