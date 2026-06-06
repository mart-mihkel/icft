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

N_TRAIN_SAMPLES=20000
N_DEV_SAMPLES=1024
DATASET=multinerd
LOG_LEVEL=debug
EPOCHS=3
SEED=0

for BASE in ${BASE_MODELS[@]}; do
    uv run --no-sync cli fine-tune \
        --n-train-samples $N_TRAIN_SAMPLES \
        --n-dev-samples $N_DEV_SAMPLES \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --model $BASE \
        --seed $SEED \
        --head-only

    uv run --no-sync cli fine-tune \
        --n-train-samples $N_TRAIN_SAMPLES \
        --n-dev-samples $N_DEV_SAMPLES \
        --log-level $LOG_LEVEL \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --model $BASE \
        --seed $SEED

    for PREFIX_INIT in ${PREFIX_INITS[@]}; do
        uv run --no-sync cli prompt-tune \
            --n-train-samples $N_TRAIN_SAMPLES \
            --n-dev-samples $N_DEV_SAMPLES \
            --prefix-init $PREFIX_INIT \
            --log-level $LOG_LEVEL \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --model $BASE \
            --seed $SEED
    done
done
