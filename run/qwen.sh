#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="qwen35"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32GB

BASE_MODELS=(
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    Qwen/Qwen3.5-9B
)

PREFIX_INITS=(
    pretrained
    random
)

DATASET=multinerd

TRAIN_SAMPLES=20000
VAL_SAMPLES=1024
EPOCHS=3

LOG_LEVEL=debug
SEED=0

for BASE in "${BASE_MODELS[@]}"; do
    uv run --no-sync cli few-shot \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --model "$BASE" \
        --seed $SEED

    uv run --no-sync cli fine-tune \
        --train-samples $TRAIN_SAMPLES \
        --val-samples $VAL_SAMPLES \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --model "$BASE" \
        --seed $SEED

    for PREFIX_INIT in "${PREFIX_INITS[@]}"; do
        uv run --no-sync cli prompt-tune \
            --train-samples $TRAIN_SAMPLES \
            --prefix-init "$PREFIX_INIT" \
            --val-samples $VAL_SAMPLES \
            --log-level $LOG_LEVEL \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --model "$BASE" \
            --seed $SEED
    done
done
