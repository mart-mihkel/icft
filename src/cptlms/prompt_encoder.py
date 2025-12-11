import logging
from typing import Annotated

import torch
from torch._tensor import Tensor
from torch.nn import LSTM, Embedding, Linear, Module, ReLU, Sequential

logger = logging.getLogger("cptlms")


class PromptEncoder(Module):
    def __init__(
        self,
        num_virtual_tokens: int = 32,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.seq_idx = torch.arange(num_virtual_tokens)

        self.embedding = Embedding(
            num_virtual_tokens,
            hidden_size,
        )

        self.lstm_head = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.mlp_head = Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
        )

    def forward(self) -> Annotated[Tensor, "batch seq"]:
        device = next(self.parameters()).device
        seq_idx = self.seq_idx.to(device)

        input_embeds = self.embedding.forward(seq_idx).unsqueeze(0)
        lstm_embeds = self.lstm_head.forward(input_embeds)[0]
        output_embeds = self.mlp_head.forward(lstm_embeds).squeeze()
        return output_embeds
