#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py perturb_adj_num perturb_adj_num EN 81 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py perturb_num_adj perturb_num_adj EN 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py perturb_adj_num perturb_adj_num EN 53 randinit pretrained
