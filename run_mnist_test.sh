#!/bin/bash 
#SBATCH --job-name=gpu_check
#SBATCH --time=0-0:20
#SBATCH --partition=gpu
#SBATCH --exclusive

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

python mnist_test.py
