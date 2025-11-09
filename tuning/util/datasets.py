from typing import cast

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerFast


def prep_swag(tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
    endings = ["ending0", "ending1", "ending2", "ending3"]

    def _preprocess(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Make four copies of `sent1` field and combine each of them with `sent2`
        to recreate how a sentence starts.

        Combine `sent2` with each of the four possible sentence endings.

        Flatten for tokenization.

        Unflatten afterward so each example has a corresponding `input_ids`,
        `attention_mask`, and `labels` field.
        """
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][header_idx]}" for end in endings]
            for header_idx, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True
        )

        return {
            k: [v[i : i + 4] for i in range(0, len(v), 4)]
            for k, v in tokenized_examples.items()
        }

    swag = load_dataset("swag", "regular")
    swag = cast(DatasetDict, swag)

    return swag.map(_preprocess, batched=True)
