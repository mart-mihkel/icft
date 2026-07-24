#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --job-name="test"
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

# BASE=hf-internal-testing/tiny-random-bert
# BASE=distilbert/distilbert-base-cased
# BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base
# BASE=EuroBERT/EuroBERT-210m
# BASE=EuroBERT/EuroBERT-610m
# BASE=EuroBERT/EuroBERT-2.1B
# BASE=microsoft/deberta-v3-xsmall
# BASE=microsoft/deberta-v3-small
# BASE=microsoft/deberta-v3-base
# BASE=microsoft/deberta-v3-large

BASE=hf-internal-testing/tiny-random-gpt2
# BASE=openai-community/gpt2
# BASE=openai-community/gpt2-medium
# BASE=openai-community/gpt2-large
# BASE=openai-community/gpt2-xl
# BASE=openai-community/gpt2-xxl
# BASE=EleutherAI/pythia-70m
# BASE=EleutherAI/pythia-160m
# BASE=EleutherAI/pythia-410m
# BASE=EleutherAI/pythia-1b
# BASE=EleutherAI/pythia-1.4b
# BASE=EleutherAI/pythia-2.8b
# BASE=EleutherAI/pythia-6.9b
# BASE=google/gemma-3-270m-it
# BASE=google/gemma-3-1b-it
# BASE=google/gemma-3-4b-it
# BASE=google/gemma-4-E2B-it
# BASE=google/gemma-4-E4B-it
# BASE=Qwen/Qwen2.5-0.5B
# BASE=Qwen/Qwen2.5-1.5B
# BASE=Qwen/Qwen3-0.6B
# BASE=Qwen/Qwen3-1.7B
# BASE=Qwen/Qwen3-4B
# BASE=Qwen/Qwen3-8B
# BASE=Qwen/Qwen3.5-0.8B
# BASE=Qwen/Qwen3.5-2B
# BASE=Qwen/Qwen3.5-4B
# BASE=Qwen/Qwen3.5-9B
# BASE=meta-llama/Llama-3.2-1B
# BASE=meta-llama/Llama-3.2-3B
# BASE=meta-llama/Llama-3.1-8B
# BASE=meta-llama/Llama-3.2-1B-Instruct
# BASE=meta-llama/Llama-3.2-3B-Instruct
# BASE=meta-llama/Llama-3.1-8B-Instruct

# BASE=hf-internal-testing/tiny-random-t5
# BASE=google-t5/t5-small
# BASE=google-t5/t5-base
# BASE=google-t5/t5-large
# BASE=google-t5/t5-xl
# BASE=google-t5/t5-xxl
# BASE=google/flan-t5-small
# BASE=google/flan-t5-base
# BASE=google/flan-t5-large
# BASE=google/flan-t5-xl
# BASE=google/flan-t5-xxl
# BASE=google/t5gemma-2-270m-270m
# BASE=google/t5gemma-2-1b-1b
# BASE=google/t5gemma-2-4b-4b

PREFIX_INIT=pretrained
PREFIX_LR=1e-3
N_SHOT=3

DATASET=multinerd

TRAIN_SAMPLES=1024
VAL_SAMPLES=128
BATCH_SIZE=8
EPOCHS=5

LOG_LEVEL=debug
SEED=0

if [[ $1 = few-shot ]]; then
    uv run --no-sync cli few-shot \
        --run-name "test/$BASE/$N_SHOT-shot" \
        --batch-size "$BATCH_SIZE" \
        --log-level "$LOG_LEVEL" \
        --dataset "$DATASET" \
        --experiment test \
        --n-shot "$N_SHOT" \
        --model "$BASE" \
        --seed "$SEED"

    exit 0
fi

if [[ $1 = cls-head ]]; then
    uv run --no-sync cli fine-tune \
        --run-name "test/$BASE/cls-head/$TASK" \
        --n-train-samples "$TRAIN_SAMPLES" \
        --n-validation-samples "$VAL_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --log-level "$LOG_LEVEL" \
        --dataset "$DATASET" \
        --experiment test \
        --epochs "$EPOCHS" \
        --n-shot "$N_SHOT" \
        --model "$BASE" \
        --seed "$SEED" \
        --head-only \
        --do-eval

    exit 0
fi

if [[ $1 = fine-tune ]]; then
    uv run --no-sync cli fine-tune \
        --run-name "test/$BASE/fine-tune/$TASK" \
        --n-train-samples "$TRAIN_SAMPLES" \
        --n-validation-samples "$VAL_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --log-level "$LOG_LEVEL" \
        --dataset "$DATASET" \
        --experiment test \
        --epochs "$EPOCHS" \
        --n-shot "$N_SHOT" \
        --no-head-only \
        --model "$BASE" \
        --seed "$SEED" \
        --do-eval

    exit 0
fi

if [[ $1 = prompt-tune ]]; then
    uv run --no-sync cli prompt-tune \
        --run-name "test/$BASE/$PREFIX_INIT-prefix/$TASK" \
        --n-train-samples "$TRAIN_SAMPLES" \
        --n-validation-samples "$VAL_SAMPLES" \
        --learning-rate "$PREFIX_LR" \
        --prefix-init "$PREFIX_INIT" \
        --batch-size "$BATCH_SIZE" \
        --log-level "$LOG_LEVEL" \
        --dataset "$DATASET" \
        --experiment test \
        --epochs "$EPOCHS" \
        --n-shot "$N_SHOT" \
        --model "$BASE" \
        --seed "$SEED" \
        --do-eval

    exit 0
fi

echo "usage: $0 [few-shot|cls-head|fine-tune|prompt-tune]"
exit 1
