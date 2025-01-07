#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local3 shuffle_local3 ZH 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local5 shuffle_local5 ZH 41 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_deterministic21 shuffle_deterministic21 ZH 41 randinit pretrained
