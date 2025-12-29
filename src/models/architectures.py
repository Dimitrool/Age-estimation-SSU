from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def assemble_head_pipeline(
    in_channels: int,
    out_channels: int,
    hidden_channels: List[int],
    activation: str = "relu",
    dropout_p: List[float] = [],
):
    pipeline = []

    if len(hidden_channels) == 0:
        pipeline.append(nn.Linear(in_channels, out_channels))
        return nn.Sequential(*pipeline)
    

    if activation:
        if hasattr(F, activation):
            activation_func = getattr(F, activation)
        else:
            raise ValueError(f"Activation type '{activation}' is not supported.")
    else:
        activation_func = nn.Identity()

    prev_out = in_channels
    for i in range(len(hidden_channels)):
        pipeline.append(nn.Linear(prev_out, hidden_channels[i]))
        pipeline.append(nn.BatchNorm1d(hidden_channels[i]))
        pipeline.append(activation_func)

        if dropout_p:
            pipeline.append(nn.Dropout(dropout_p[i]))

    pipeline.append(nn.Linear(prev_out, out_channels))
    return nn.Sequential(*pipeline)



class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        activation: str = "relu",
        dropout_p: List[float] = [],
    ):
        super().__init__()
        self.head = assemble_head_pipeline(
            in_channels,
            out_channels,
            hidden_channels,
            activation,
            dropout_p
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


