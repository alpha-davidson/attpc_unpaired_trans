#!/bin/bash
### job parameters:
#SBATCH --job-name "trainPointDPM"
#SBATCH --mem 64G
#SBATCH --gpus 1

# source /opt/conda/bin/activate dpm-pc-gen

cd ../
# train generator
python -m train.train_gen --dataset_path ## add the dataset path, as well as change hyperparameters

