#!/bin/bash

# Download GSM8K dataset preparation script
set -e

echo "Downloading and preparing GSM8K dataset..."

# Create data directory
mkdir -p ~/data/gsm8k

# Download train dataset
echo "Preparing GSM8K train dataset..."
python examples/prepare_gsm8k.py \
    --local_dir ~/data/gsm8k \
    --split train \
    --max_length 2048

# Download test dataset  
echo "Preparing GSM8K test dataset..."
python examples/prepare_gsm8k.py \
    --local_dir ~/data/gsm8k \
    --split test \
    --max_length 2048

echo "GSM8K dataset preparation complete!"
echo "Train dataset: ~/data/gsm8k/train.parquet"
echo "Test dataset: ~/data/gsm8k/test.parquet"
