#!/bin/bash
#SBATCH --job-name "Jupyter Instance"
#SBATCH --output jupyter.log
#SBATCH --mem 32g
#SBATCH --gpus 1

source /opt/conda/bin/activate dpm-pc-gen
jupyter lab --port=2201 --no-browser
