#!/bin/bash
### job parameters:
#SBATCH --job-name "trainPointDPM"
#SBATCH --mem 64G
#SBATCH --gpus 1

# source /opt/conda/bin/activate dpm-pc-gen

cd /home/DAVIDSON/<yournamehere>/attpc_unpaired_trans
# train generator
python -m train.train_gen # Add arguments like --max_iters, --lr, --dataset_path, etc.


