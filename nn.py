from typing import Union
from math import prod, sqrt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cmaes_agent import ActionType


class NN(nn.Module):
    def __init__(self, inputs: int, outputs: int = 1, action_type: ActionType = ActionType.Discrete):
        self.action_type = action_type

        super().__init__()
        hidden_neurons = round(sqrt(inputs * outputs)) + 1
        self.fc1 = nn.Linear(inputs, hidden_neurons, bias=False)
        self.fc2 = nn.Linear(hidden_neurons, outputs, bias=False)
        self.submodules = [self.fc1, self.fc2]

        for param in self.parameters():
            param.requires_grad = False

    def map_to_action(self, state: Union[np.ndarray, torch.Tensor]) -> Union[int, list]:
        if isinstance(state, np.ndarray):
            state = np.reshape(state, [1, -1])
            state = torch.from_numpy(state).float()

        x = state
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

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
        for m in self.submodules:
            if isinstance(m, nn.Linear):
                idx = self._set_weights(m, w[idx:])

    @staticmethod
    def _set_weights(module: nn.Linear, w: np.ndarray, start_idx: int = 0) -> int:
        shape = module.weight.data.shape
        weights_count = prod(shape)
        weights = np.reshape(w[start_idx:start_idx + weights_count], shape)
        module.weight = nn.Parameter(torch.from_numpy(weights).float())
        return start_idx + weights_count
