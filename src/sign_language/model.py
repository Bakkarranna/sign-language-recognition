"""BiGRU sequence classifier over normalised landmark sequences."""

from __future__ import annotations

import torch
from torch import nn

from . import config


class SignGRU(nn.Module):
    """Frame landmarks -> class logits.

    Each frame is 225 numbers (75 landmarks x 3). They are projected to the
    GRU width, run through a bidirectional GRU, and classified from the final
    states of both directions. The whole sequence model is ~1.2M parameters,
    which trains in minutes on CPU for 100 classes.
    """

    def __init__(self, n_classes: int = config.N_CLASSES,
                 hidden: int = config.HIDDEN, layers: int = config.LAYERS,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.FEAT_DIM, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, 75, 3) -> (batch, frames, FEAT_DIM)
        b, t = x.shape[0], x.shape[1]
        feats = x.reshape(b, t, config.FEAT_DIM)
        h, _ = self.gru(torch.relu(self.input_proj(feats)))
        # last frame of the forward direction + last frame of the backward one
        last = torch.cat([h[:, -1, :self.gru.hidden_size],
                          h[:, 0, self.gru.hidden_size:]], dim=1)
        return self.head(self.dropout(last))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
