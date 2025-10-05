set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: examples/run_sft_tulu3.sh <save_path> [other_configs...]"
    exit 1
fi

save_path=$1

# Shift the arguments so $@ refers to the rest
shift 1

python -m src.trainer.standard_sft_trainer \
    diffusion.time_reweighting=cart \
    data.train_files=$HOME/data/tulu3/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.max_length=1024 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=2e-6 \
    data.micro_batch_size_per_gpu=1 \
    data.perbatch_cutoff_type=random_with_input_pad \
    model.partial_pretrain=Dream-org/Dream-v0-Base-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=diff-verl \
    trainer.experiment_name=single_gpu_exp \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3
