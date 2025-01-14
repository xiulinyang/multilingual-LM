#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python perplexities_exp.py shuffle_remove_fw shuffle_remove_fw EN 41 randinit pretrained
CUDA_VISIBLE_DEVICES=1 python perplexities_exp.py shuffle_remove_fw shuffle_remove_fw EN 53 randinit pretrained
CUDA_VISIBLE_DEVICES=1 python perplexities_exp.py shuffle_remove_fw shuffle_remove_fw EN 81 randinit pretrained
