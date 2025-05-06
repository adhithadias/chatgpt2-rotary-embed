#!/bin/bash

BATCH_SIZES=(4 8)
BLOCK_SIZES=(128 256 512 1024 2048 4096)

for B in "${BATCH_SIZES[@]}"; do
  for S in "${BLOCK_SIZES[@]}"; do
    echo "Running benchmark with batch size $B and block size $S"
    CUDA_VISIBLE_DEVICES=1 python test_tiny_llama_rotary_embedding.py --batch_size $B --block_size $S
  done
done
echo "All benchmarks completed."