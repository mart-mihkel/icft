import logging
from collections import defaultdict
from typing import Annotated, TypedDict, cast

import evaluate
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from torch._C import FloatTensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import QuestionAnsweringModelOutput

logger = logging.getLogger(__name__)


class SQuADMetrics(TypedDict):
    exact_match: float
    f1: float


class SQuADBatchAnswer(TypedDict):
    answer_start: tuple[int]
    text: tuple[str]


class SQuADBatch(TypedDict):
    id: list[int]
    question: list[str]
    context: list[str]
    answers: list[SQuADBatchAnswer]


class SQuAD:
    _max_len_tokenizer = 384
    _stride_tokenizer = 128

    _n_best_eval = 20
    _max_answer_len_eval = 30

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        logger.info("init SQuAD")

        data = load_dataset("squad")
        assert isinstance(data, DatasetDict)

        self.data = data
        self.tokenizer = tokenizer
        self.tokenized = self._tokenize()
        self.metric = evaluate.loading.load("squad")

    def _tokenize(self) -> DatasetDict:
        logger.info("tokenize SQuAD")

        train = self.data["train"].map(
            self._preprocess_train_batch,
            batched=True,
            remove_columns=self.data["train"].column_names,
        )

        val = self.data["validation"].map(
            self._preprocess_val_batch,
            batched=True,
            remove_columns=self.data["validation"].column_names,
        )

        return DatasetDict({"train": train, "validation": val})

    def _preprocess_train_batch(self, examples: SQuADBatch):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self._stride_tokenizer,
            max_length=self._max_len_tokenizer,
        )

        answers = examples["answers"]
        offset_mapping: list[list[tuple[int, int]]] = inputs.pop("offset_mapping")
        sample_map: list[int] = inputs.pop("overflow_to_sample_mapping")

        start_positions: list[int] = []
        end_positions: list[int] = []
        for i, (offset, sample_idx) in enumerate(zip(offset_mapping, sample_map)):
            answer = answers[sample_idx]
            start_chr = answer["answer_start"][0]
            end_chr = answer["answer_start"][0] + len(answer["text"][0])
            start_pos, end_pos = self._find_label_span(
                offset=offset,
                seq_ids=inputs.sequence_ids(i),
                answer_span=(start_chr, end_chr),
            )

            start_positions.append(start_pos)
            end_positions.append(end_pos)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    def _preprocess_val_batch(self, examples: SQuADBatch):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self._stride_tokenizer,
            max_length=self._max_len_tokenizer,
        )

        sample_map: list[int] = inputs.pop("overflow_to_sample_mapping")
        example_ids: list[int] = []
        for i in range(len(inputs.data["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            inputs.data["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(inputs.data["offset_mapping"][i])
            ]

        inputs["example_id"] = example_ids

        return inputs

    @staticmethod
    def _find_label_span(
        offset: list[tuple[int, int]],
        seq_ids: list[int | None],
        answer_span: tuple[int, int],
    ) -> tuple[int, int]:
        start_chr, end_chr = answer_span

        idx = 0
        while seq_ids[idx] != 1:
            idx += 1

        ctx_start = idx

        while seq_ids[idx] == 1:
            idx += 1

        ctx_end = idx - 1

        if offset[ctx_start][0] > end_chr or offset[ctx_end][1] < start_chr:
            return 0, 0

        idx = ctx_start
        while idx <= ctx_end and offset[idx][0] <= start_chr:
            idx += 1

        start_pos = idx - 1

        idx = ctx_end
        while idx >= ctx_start and offset[idx][1] >= end_chr:
            idx -= 1

        end_pos = idx + 1

        return start_pos, end_pos

    def compute_metrics(
        self,
        out: QuestionAnsweringModelOutput,
        examples: Dataset,
        tokenized_examples: Dataset,
    ) -> SQuADMetrics:
        predicted_answers = self._postprocess_predictions(
            out=out,
            examples=examples,
            tokenized_examples=tokenized_examples,
        )

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
        ]

        metrics = self.metric.compute(
            predictions=predicted_answers,
            references=theoretical_answers,
        )

        return cast(SQuADMetrics, metrics)

    def _postprocess_predictions(
        self,
        out: QuestionAnsweringModelOutput,
        examples: Dataset,
        tokenized_examples: Dataset,
    ) -> list[dict[str, str | int]]:
        start_logits = out.start_logits
        end_logits = out.end_logits

        assert start_logits is not None
        assert end_logits is not None

        example_to_features: dict[int, list[int]] = defaultdict(list)
        for idx, feature in enumerate(tokenized_examples):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            answers = self._extract_answers(
                start_logits=start_logits,
                end_logits=end_logits,
                context=example["context"],
                example_features=example_to_features[example_id],
                offset_mapping=tokenized_examples["offset_mapping"],
            )

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        return predicted_answers

    def _extract_answers(
        self,
        start_logits: Annotated[FloatTensor, "batch seq"],
        end_logits: Annotated[FloatTensor, "batch seq"],
        context: list[str],
        example_features: list[int],
        offset_mapping: list[list[tuple[int, int]]],
    ) -> list[dict[str, str | int]]:
        answers = []
        for f_idx in example_features:
            start_logit = start_logits[f_idx]
            end_logit = end_logits[f_idx]
            offsets = offset_mapping[f_idx]

            start_indexes = np.argsort(start_logit)[
                -1 : -self._n_best_eval - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logit)[
                -1 : -self._n_best_eval - 1 : -1
            ].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue

                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > self._max_answer_len_eval
                    ):
                        continue

                    text = context[offsets[start_index][0] : offsets[end_index][1]]
                    score = start_logit[start_index] + end_logit[end_index]
                    answers.append({"text": text, "logit_score": score})

        return answers
