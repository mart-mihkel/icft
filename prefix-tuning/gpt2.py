import marimo

__generated_with = "0.15.3"
app = marimo.App()

with app.setup:
    from datasets import load_dataset
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    from transformers import (
        GPT2TokenizerFast,
        GPT2LMHeadModel,
        TrainingArguments,
        Trainer,
    )


@app.cell
def _():
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenized_dataset = dataset.map(
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

    lm_dataset = tokenized_dataset.map(
        lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]},
        batched=True,
    )
    return lm_dataset, tokenizer


@app.cell
def _(tokenizer):
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model.resize_token_embeddings(len(tokenizer))

    prefix_cfg = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=64,
    )

    peft_model = get_peft_model(model, prefix_cfg)
    peft_model.print_trainable_parameters()
    return (peft_model,)


@app.cell
def _(lm_dataset, peft_model):
    training_args = TrainingArguments(
        output_dir="./trainer-out/gpt2-prefix-tuning",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=128,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        eval_dataset=lm_dataset["test"].select(range(2500)),
        train_dataset=lm_dataset["train"],
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer_out = trainer.train()
    return


@app.cell
def _(peft_model, tokenizer):
    _inputs = tokenizer("Sid from the hit movie", return_tensors="pt").to("cuda")
    _outputs = peft_model.to("cuda").generate(**_inputs, max_new_tokens=100)
    tokenizer.decode(*_outputs)
    return


if __name__ == "__main__":
    app.run()
