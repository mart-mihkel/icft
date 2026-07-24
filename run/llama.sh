#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="llama32"
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32GB

BASE_MODELS=(
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
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
