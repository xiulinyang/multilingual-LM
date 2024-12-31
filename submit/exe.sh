#!/usr/bin/env bash

source /<username>/<projdir>/submit/rename_gpus.sh

# Activate basic conda environment. 
# Change filepath in this command if your conda environment is not installed under nethome.
source /nethome/<username>/miniconda3/etc/profile.d/conda.sh

conda activate <envname>

echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# run misc. stuff for sanity check
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list


#run any commands here
WANDB_API_KEY=<wandbkey> CUDA_VISIBLE_DEVICES=0 bash /scratch/<username>/<projdir>/submit/commands.sh
