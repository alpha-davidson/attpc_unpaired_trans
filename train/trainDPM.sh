#!/bin/bash
### job parameters:
#SBATCH --job-name "trainPointDPM"
#SBATCH --mem 64G
#SBATCH --gpus 1

# source /opt/conda/bin/activate dpm-pc-gen

cd /home/DAVIDSON/allandolfi/attpc_unpaired_trans
# train generator
 python -m train.train_gen
