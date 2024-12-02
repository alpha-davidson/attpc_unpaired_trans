#!/bin/bash
### job parameters:
#SBATCH --job-name "testPointDPM"
#SBATCH --output testing.log
#SBATCH --mem 32G
#SBATCH --gpus 1

# email alerts
# SBATCH --mail-type ALL
# SBATCH --mail-user etcramer@davidson.edu

cd /home/DAVIDSON/etcramer/project/attpc_dpm/diffusion-point-cloud/

# conda activate dpm-pc-gen


#test ae
python test_ae.py

# Test a generator
python test_gen.py