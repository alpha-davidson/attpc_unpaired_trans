#!/bin/bash
### job parameters:
#SBATCH --job-name "trainPointDPM"
#SBATCH --mem 64G
#SBATCH --gpus 1

# source /opt/conda/bin/activate dpm-pc-gen

cd /home/DAVIDSON/allandolfi/attpc_unpaired_trans
# train generator
python -m train.train_gen --dataset_path data/Fission/fission_data/Fission_exp.npy --max_iters 1510000 --val_freq 100 --tag FissionExp --lr 1e-3 --end_lr 1e-7

