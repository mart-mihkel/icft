from typing import cast

from transformers import (
    BertForMultipleChoice,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
)

from .util.datasets import prep_swag


def main(
    pretrained_model: str,
    num_virtual_tokens: int,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
):
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    tokenizer = cast(BertTokenizerFast, tokenizer)
    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    ds = prep_swag(tokenizer)

    model = BertForMultipleChoice.from_pretrained(pretrained_model)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    _inference(model, tokenizer)


def _inference(model: BertForMultipleChoice, tokenizer: BertTokenizerFast):
    import torch

    prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
    candidate1 = "The law does not apply to croissants and brioche."
    candidate2 = "The law applies to baguettes."

    inputs = tokenizer(
        [[prompt, candidate1], [prompt, candidate2]],
        return_tensors="pt",
        padding=True,
    )

    labels = torch.tensor(0).unsqueeze(0)

    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits

    predicted_class = logits.argmax().item()
    print(predicted_class)
