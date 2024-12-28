#!/bin/bash

# Accept a language argument
LANGUAGE=$1
VOCAB=$2
GPU=$3

# Default value if no argument is provided
if [ -z "$LANGUAGE" ]; then
	  echo "No language specified. Using 'EN' as default."
	    LANGUAGE="EN"
fi

# Navigate to the tokenizers directory and perform operations
cd tokenizers/

rm -rf $LANGUAGE
cd ..
python train_tokenizer.py $LANGUAGE train $VOCAB

cd data
cd multilingual/multilingual_data_perturbed/

rm -rf shuffle_control_${LANGUAGE,,}/

cd ..
cd ..

python perturb.py shuffle_control_${LANGUAGE,,} $LANGUAGE train
python perturb.py shuffle_control_${LANGUAGE,,} $LANGUAGE dev
python perturb.py shuffle_control_${LANGUAGE,,} $LANGUAGE test

cd ..
cd training

bash prepare_training.sh shuffle_control_${LANGUAGE,,} $LANGUAGE 41 randinit

cd ..
cd mistral

conda init bash
source ~/.bashrc
conda deactivate
conda activate mistral

CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config conf/train_shuffle_control_${LANGUAGE,,}_${LANGUAGE}_randinit_seed41.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 50 --training_arguments.max_steps 500
    
cd ..
cd perplexities
conda activate mission
CUDA_VISIBLE_DEVICES=5 python perplexities.py shuffle_control_${LANGUAGE,,} shuffle_control_${LANGUAGE,,} $LANGUAGE 41 randinit $VOCAB
