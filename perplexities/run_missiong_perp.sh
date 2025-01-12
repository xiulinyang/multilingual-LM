#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python perplexities_exp.py shuffle_remove_fw shuffle_remove_fw ZH 41 randinit pretrained
