#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_even_odd shuffle_even_odd ZH 41 randinit pretrained
