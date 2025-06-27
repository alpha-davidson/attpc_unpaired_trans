#!/bin/bash
### job parameters:
#SBATCH --job-name "trainPointDPM"
#SBATCH --mem 64G
#SBATCH --gpus 1

source /opt/conda/bin/activate dpm-pc-gen

# train generator
python -m train.train_gen --dataset_path data/Fission/fission_data/Fission_sim.npy --max_iters 1000000 --val_freq 100 --tag FissionSim --lr 1e-3 --end_lr 1e-6

