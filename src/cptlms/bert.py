import logging
from typing import Annotated, cast

import torch
from torch import Tensor, FloatTensor
from torch.nn import Embedding, Module
from transformers import BertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from cptlms.prompt_encoder import PromptEncoder

logger = logging.getLogger("cptlms")


class PTuningBert(Module):
    def __init__(
        self,
        bert: BertForQuestionAnswering,
        num_virtual_tokens: int = 32,
        train_new_layers: bool = True,
    ) -> None:
        super().__init__()

        self.num_virtual_tokens = num_virtual_tokens

        self.bert = bert
        self._freeze_params(self.bert, train_new_layers)

        bert_embedding = bert.get_input_embeddings()
        assert isinstance(bert_embedding, Embedding)

        self.bert_embedding = bert_embedding

        self.prompt_encoder = PromptEncoder(
            num_virtual_tokens=num_virtual_tokens,
            hidden_size=self.bert_embedding.embedding_dim,
        )

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch seq"],
        attention_mask: Annotated[Tensor, "batch seq"],
        start_positions: Annotated[Tensor, "batch"] | None = None,
        end_positions: Annotated[Tensor, "batch"] | None = None,
    ) -> QuestionAnsweringModelOutput:
        batch_size = input_ids.size(0)

        virtual_embeds: Annotated[Tensor, "batch virtual hidden"] = (
            self.prompt_encoder().unsqueeze(0).expand(batch_size, -1, -1)
        )

        bert_embeds: Annotated[Tensor, "batch seq hidden"] = self.bert_embedding(
            input_ids
        )

        virtual_attention: Annotated[Tensor, "batch virtual"] = torch.ones(
            batch_size,
            self.num_virtual_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        out = self.bert(
            inputs_embeds=torch.cat([virtual_embeds, bert_embeds], dim=1),
            attention_mask=torch.cat([virtual_attention, attention_mask], dim=1),
            start_positions=start_positions,
            end_positions=end_positions,
        )

        assert isinstance(out, QuestionAnsweringModelOutput)

        start_logits = out.start_logits
        end_logits = out.end_logits

        assert start_logits is not None
        assert end_logits is not None

        out.start_logits = cast(FloatTensor, start_logits[:, self.num_virtual_tokens :])
        out.end_logits = cast(FloatTensor, end_logits[:, self.num_virtual_tokens :])

        return out

    @staticmethod
    def _freeze_params(
        bert: BertForQuestionAnswering,
        train_new_layers: bool = True,
    ):
        logger.info("freeze bert parameters")
        for name, param in bert.named_parameters():
            if train_new_layers and "qa_outputs" in name:
                logger.info("skip %s", name)
                continue

            param.requires_grad = False
