import torch
import torch.nn as nn

from imitation_learning.utils.utils import build_mlp


class CostNetwork(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape,
        hidden_units=(64, 64),
        hidden_activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
