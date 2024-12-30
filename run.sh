#!/bin/bash

# Accept a language argument
LANGUAGE=$1
GPU=$2
PERTURB=$3
RANDOM=$4
# Default value if no argument is provided
if [ -z "$LANGUAGE" ]; then
	  echo "No language specified. Using 'EN' as default."
	    LANGUAGE="EN"
fi

cd data
cd multilingual/multilingual_data_perturbed/

rm -rf ${PERTURB}_${LANGUAGE,,}/

cd ..
cd ..

python perturb.py ${PERTURB} $LANGUAGE train
python perturb.py ${PERTURB} $LANGUAGE dev
python perturb.py ${PERTURB} $LANGUAGE test

cd ..
cd training

bash prepare_training.sh ${PERTURB} $LANGUAGE $RANDOM randinit

cd ..
cd mistral

conda init bash
source ~/.bashrc
conda deactivate
conda activate mistral

CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config conf/train_${PERTURB}_${LANGUAGE,,}_${LANGUAGE}_randinit_seed${RANDOM}.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 120 --training_arguments.max_steps 1200
    
cd ..
cd perplexities
conda activate mission
CUDA_VISIBLE_DEVICES=$GPU python perplexities.py ${PERTURB} ${PERTURB} $LANGUAGE $RANDOM randinit pretrained
