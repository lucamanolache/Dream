import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


def format_gsm8k_row(row):
    """Format GSM8K row to match the expected format with prompt and response keys."""
    # GSM8K has 'question' and 'answer' fields
    # We'll use 'question' as prompt and 'answer' as response
    row["prompt"] = row["question"]
    row["response"] = row["answer"]
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split", default="train", choices=["train", "test"])

    args = parser.parse_args()

    # Load GSM8K dataset
    dataset = datasets.load_dataset("gsm8k", "main", split=args.split)
    
    # Format the dataset to have prompt and response keys
    dataset = dataset.map(format_gsm8k_row)
    
    # Remove the original question and answer columns to avoid confusion
    dataset = dataset.remove_columns(["question", "answer"])

    # Filter out too long samples if max_length is specified
    if args.max_length is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        def filter_by_length(row):
            prompt = row["prompt"]
            response = row["response"]
            
            # Create a simple format for length calculation
            full_text = f"Question: {prompt}\nAnswer: {response}"
            tokens = tokenizer.encode(full_text)
            return len(tokens) <= args.max_length

        dataset = dataset.filter(filter_by_length)

    # Create local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Save the dataset
    if args.split == "train":
        output_file = os.path.join(local_dir, "train.parquet")
    else:
        output_file = os.path.join(local_dir, "test.parquet")
    
    dataset.to_parquet(output_file)
    print(f"Saved {len(dataset)} samples to {output_file}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
