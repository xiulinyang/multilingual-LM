#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py perturb_num_adj perturb_num_adj EN 53 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py perturb_num_adj perturb_num_adj EN 81 randinit pretrained
