#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local3 shuffle_local3 RU 81 randinit pretrained
CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local5 shuffle_local5 RU 81 randinit pretrained

