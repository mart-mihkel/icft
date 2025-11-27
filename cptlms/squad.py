import logging
from typing import cast

from datasets.load import load_dataset
from datasets.dataset_dict import DatasetDict
from evaluate.module import EvaluationModule
from transformers import BatchEncoding, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class SQuAD:
    data: DatasetDict
    metric: EvaluationModule

    def __init__(self) -> None:
        logger.info("init SQuAD")

        self.data = cast(DatasetDict, load_dataset("squad"))

    def tokenize(self, tokenizer: PreTrainedTokenizerFast):
        logger.info("tokenize SQuAD")

        def _preprocess(examples: dict) -> BatchEncoding:
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=384,
                padding="max_length",
                truncation="only_second",
                return_offsets_mapping=True,
            )

            answers = examples["answers"]
            offset_mapping: list[list[tuple[int, int]]] = inputs.pop("offset_mapping")

            start_positions: list[int] = []
            end_positions: list[int] = []
            for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
                start_chr = answer["answer_start"][0]
                end_chr = start_chr + len(answer["text"][0])
                start_pos, end_pos = self._get_squad_offset_span(
                    offset=offset,
                    seq_ids=inputs.sequence_ids(i),
                    start_chr=start_chr,
                    end_chr=end_chr,
                )

                start_positions.append(start_pos)
                end_positions.append(end_pos)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

            return inputs

        self.data = self.data.map(
            _preprocess,
            batched=True,
            remove_columns=self.data["train"].column_names,
        )

    @staticmethod
    def _get_squad_offset_span(
        offset: list[tuple[int, int]],
        seq_ids: list[int | None],
        start_chr: int,
        end_chr: int,
    ) -> tuple[int, int]:
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
