#!/bin/bash
cd /root/parameter-golf
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export WARMDOWN_ITERS=4000
export TARGET_MB=15.9
export SEED=314
torchrun --standalone --nproc_per_node=8 answer_q1.py >> /root/answer_q1.log 2>&1
