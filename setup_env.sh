#!/bin/bash
#SBATCH --job-name=envcreate
#SBATCH --output=envcreate.log
#SBATCH --mem=32G
#SBATCH --gpus=1

conda env create -f env.yml
conda activate dpm-pc-gen
