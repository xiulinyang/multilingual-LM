#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local3 shuffle_local3 RO 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local3 shuffle_local3 RO 81 randinit pretrained
