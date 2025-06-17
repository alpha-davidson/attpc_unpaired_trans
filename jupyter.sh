#!/bin/bash
#SBATCH --job-name "Jupyter Instance"
#SBATCH --output jupyter.log
#SBATCH --mem 32g

source /opt/conda/bin/activate dpm-pc-gen
jupyter lab --port=9876 --ip="0.0.0.0" --no-browser
