from typing import Union
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.env_info import ActionType


class NN(nn.Module):
    HIDDEN_NEURONS = 8

    def __init__(self, inputs: int, outputs: int = 1, action_type: ActionType = ActionType.Discrete,
                 max_nn_parameters: Union[str, int] = 'standard', bias: bool = False):
        super().__init__()

        self.action_type = action_type
        self.hidden = self._get_hidden_layers_count(inputs, outputs, max_nn_parameters, bias)
        self.model = self._build_model(inputs, outputs, self.hidden, bias)
        self.bias = bias
        self._disable_grad()

    @staticmethod
    def _calculate_hidden_layers_count(inputs: int, outputs: int, max_nn_parameters: int, bias: bool) -> int:
        # hidden_neurons = min(2 * (inputs + outputs), 8)
        hidden_neurons = NN.HIDDEN_NEURONS
        _bias = 1 if bias else 0
        hidden_parameters = max(max_nn_parameters - (inputs + _bias) * hidden_neurons - (hidden_neurons + _bias) * outputs, 0)
        if hidden_parameters == 0:
            hidden = 0
        else:
            hidden = hidden_parameters // ((hidden_neurons + _bias) * hidden_neurons) + 1
        return hidden

    @staticmethod
    def _get_hidden_layers_count(inputs: int, outputs: int, max_nn_parameters: Union[str, int], bias: bool) -> int:
        if max_nn_parameters == 'standard':
            hidden_layers_count = 1
        elif max_nn_parameters == 'minimal':
            hidden_layers_count = 0
        elif isinstance(max_nn_parameters, int):
            hidden_layers_count = NN._calculate_hidden_layers_count(inputs, outputs, max_nn_parameters, bias)
        else:
            raise ValueError('Invalid {max_nn_parameters=}')
        return hidden_layers_count

    @staticmethod
    def _build_model(inputs: int, outputs: int, hidden: int, bias: bool) -> nn.Sequential:
        layers = []
        _inputs = inputs
        for i in range(hidden):
            _outputs = NN.HIDDEN_NEURONS
            layers.append(nn.Linear(_inputs, _outputs, bias=bias))
            layers.append(nn.ReLU())
            _inputs = NN.HIDDEN_NEURONS
        _outputs = outputs
        layers.append(nn.Linear(_inputs, _outputs, bias=bias))
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
                idx += self._set_weights(m, w[idx:], self.bias)

    @staticmethod
    def _set_weights(module: nn.Linear, w: np.ndarray, bias: bool) -> int:
        start_idx = 0
        weights_shape = module.weight.data.shape
        weights_count = math.prod(weights_shape)
        bias_shape = (0,)
        bias_count = 0
        if bias:
            bias_shape = module.bias.data.shape
            bias_count = math.prod(bias_shape)
        end_idx = start_idx + weights_count + bias_count
        assert end_idx <= len(w), 'Weights count mismatch.'

        weights = np.reshape(w[start_idx:end_idx - bias_count], weights_shape)
        module.weight = nn.Parameter(torch.from_numpy(weights).float())
        if bias:
            bias = np.reshape(w[end_idx - bias_count: end_idx], bias_shape)
            module.bias = nn.Parameter(torch.from_numpy(bias).float())
        return end_idx
