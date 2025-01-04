#!/usr/bin/env bash


source /xiulyang/multilingual-LM/submit/rename_gpus.sh

# Activate basic conda environment. 
# Change filepath in this command if your conda environment is not installed under nethome.
source /scratch/xiulyang/anaconda3/etc/profile.d/conda.sh

conda activate mission

echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# run misc. stuff for sanity check
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list

cd /scratch/xiulyang/multilingual-LM/

LANG=$1
PERT=$2
SEED=$3
PERT2=$4
SEED2=$5

#run any commands here
WANDB_API_KEY=3576739fd3c81c328720ae979b8a8b4106ef409c bash run.sh $1 $2 $3
WANDB_API_KEY=3576739fd3c81c328720ae979b8a8b4106ef409c bash run.sh $1 $4 $5
