#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_even_odd shuffle_even_odd RU 53 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_nondeterministic shuffle_nondeterministic RU 81 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_deterministic21 shuffle_deterministic21 RU 41 randinit pretrained

