from typing import Union
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.env_info import ActionType


class NN(nn.Module):
    def __init__(self, inputs: int, outputs: int = 1, action_type: ActionType = ActionType.Discrete,
                 max_nn_parameters: Union[str, int] = 'standard'):
        super().__init__()

        self.action_type = action_type
        self.model = self._build_model(inputs, outputs, max_nn_parameters)
        self.hidden = len(list(self.model.modules())) // 2
        self._disable_grad()

    @staticmethod
    def _calculate_hidden_layers_count(inputs: int, outputs: int, max_nn_parameters: int, hidden_neurons: int):
        hidden_parameters = max(max_nn_parameters - (inputs + 1) * hidden_neurons - (hidden_neurons + 1) * outputs, 0)
        if hidden_parameters == 0:
            hidden = 0
        else:
            hidden = hidden_parameters // ((hidden_neurons + 1) * hidden_neurons) + 1
        return hidden

    @staticmethod
    def _build_model(inputs: int, outputs: int, max_nn_parameters: Union[str, int]) -> nn.Sequential:
        hidden_neurons = min(2 * (inputs + outputs), 8)
        if max_nn_parameters == 'standard':
            hidden_layers_count = 1
        elif max_nn_parameters == 'minimal':
            hidden_layers_count = 0
        elif isinstance(max_nn_parameters, int):
            hidden_layers_count = NN._calculate_hidden_layers_count(inputs, outputs, max_nn_parameters, hidden_neurons)
        else:
            raise f'Invalid {max_nn_parameters=}'

        layers = []
        _inputs = inputs
        for i in range(hidden_layers_count):
            _outputs = hidden_neurons
            layers.append(nn.Linear(_inputs, _outputs, bias=True))
            layers.append(nn.ReLU())
            _inputs = hidden_neurons
        _outputs = outputs
        layers.append(nn.Linear(_inputs, _outputs, bias=True))
        return nn.Sequential(*layers)

    def _disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def map_to_action(self, state: Union[np.ndarray, torch.Tensor]) -> Union[int, list]:
        if isinstance(state, np.ndarray):
            state = np.reshape(state, [1, -1])
            state = torch.from_numpy(state).float()

        x = state
        x = self.model(x)

        if self.action_type == ActionType.Discrete:
            x = F.softmax(x, dim=1)
            x = x.squeeze().tolist()
            max_val = max(x)
            max_idx = x.index(max_val)
            return max_idx

        # continuous
        x = x.squeeze().tolist()
        return x

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_weights(self, w: np.ndarray):
        assert len(w) == self.parameters_count(), 'Weights count mismatch.'

        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                idx += self._set_weights(m, w[idx:])

    @staticmethod
    def _set_weights(module: nn.Linear, w: np.ndarray) -> int:
        start_idx = 0
        weights_shape = module.weight.data.shape
        bias_shape = module.bias.data.shape
        weights_count = math.prod(weights_shape)
        bias_count = math.prod(bias_shape)
        end_idx = start_idx + weights_count + bias_count
        assert end_idx <= len(w), 'Weights count mismatch.'

        weights = np.reshape(w[start_idx:end_idx - bias_count], weights_shape)
        bias = np.reshape(w[end_idx - bias_count: end_idx], bias_shape)
        module.weight = nn.Parameter(torch.from_numpy(weights).float())
        module.bias = nn.Parameter(torch.from_numpy(bias).float())
        return end_idx
