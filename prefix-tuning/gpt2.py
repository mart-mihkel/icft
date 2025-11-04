import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    from datasets import load_dataset
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    from transformers import (
        GPT2TokenizerFast,
        GPT2LMHeadModel,
        TrainingArguments,
        Trainer,
    )

    return (
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        PrefixTuningConfig,
        TaskType,
        Trainer,
        TrainingArguments,
        get_peft_model,
        load_dataset,
    )


@app.cell
def _(GPT2TokenizerFast, load_dataset):
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = (
        load_dataset("wikitext", "wikitext-2-raw-v1")
        .map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            ),
            remove_columns=["text"],
            batched=True,
        )
        .map(
            lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]},
            batched=True,
        )
    )
    return dataset, tokenizer


@app.cell
def _(
    GPT2LMHeadModel,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    tokenizer,
):
    _cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=64,
    )

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, _cfg)
    return (model,)


@app.cell
def _(Trainer, TrainingArguments, dataset, model):
    _args = TrainingArguments(
        output_dir="./trainer-out/gpt2-prefix-tuning",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=_args,
        eval_dataset=dataset["test"].select(range(2500)),
        train_dataset=dataset["train"],
    )

    trainer.train()
    return


if __name__ == "__main__":
    app.run()
