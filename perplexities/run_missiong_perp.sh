#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local3 shuffle_local3 PL 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local5 shuffle_local5 PL 53 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local10 shuffle_local10 PL 81 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_even_odd shuffle_even_odd PL 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_even_odd shuffle_even_odd PL 53 randinit pretrained
