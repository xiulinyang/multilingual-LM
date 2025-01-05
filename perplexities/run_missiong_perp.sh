#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python perplexities_exp.py shuffle_local10 shuffle_local10 RO 53 randinit pretrained
