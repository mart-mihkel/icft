#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="deberta"
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16GB

BASE_MODELS=(
    distilbert/distilbert-base-cased
    jhu-clsp/mmBERT-small
    jhu-clsp/mmBERT-base
    EuroBERT/EuroBERT-210m
    EuroBERT/EuroBERT-610m
    EuroBERT/EuroBERT-2.1B
    microsoft/deberta-v3-xsmall
    microsoft/deberta-v3-small
    microsoft/deberta-v3-base
    microsoft/deberta-v3-large
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
SEED=42

for BASE in "${BASE_MODELS[@]}"; do
    uv run --no-sync cli fine-tune \
        --train-samples $TRAIN_SAMPLES \
        --val-samples $VAL_SAMPLES \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --model "$BASE" \
        --seed $SEED \
        --head-only

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
