# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Actor Network that generates distributions."""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from .lstm_encoding_network import LSTMEncodingNetwork


class ActorNetwork(nn.Module):
    """Actor network that outputs action distributions.
    
    Supports both continuous actions (Normal distribution) and 
    discrete actions (Categorical distribution).
    """

    def __init__(
            self,
            input_size: int,
            action_size: int,
            continuous: bool = True,
            conv_layer_params: Optional[List[Tuple[int, int, int]]] = None,
            input_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            lstm_size: Optional[Tuple[int, ...]] = None,
            output_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            activation_fn: nn.Module = nn.ReLU,
            init_action_stddev: float = 0.35,
            dtype: torch.dtype = torch.float32,
    ):
        """Creates an instance of `ActorNetwork`.

        Args:
            input_size: The size of the input features.
            action_size: The number of action dimensions (continuous) or 
                number of discrete actions.
            continuous: If True, outputs Normal distribution for continuous actions.
                If False, outputs Categorical distribution for discrete actions.
            conv_layer_params: Optional list of convolution layer parameters.
            input_fc_layer_params: Optional tuple of FC layer sizes before LSTM.
            lstm_size: Tuple of LSTM hidden sizes.
            output_fc_layer_params: Optional tuple of FC layer sizes after LSTM.
            activation_fn: Activation function class.
            init_action_stddev: Initial standard deviation for continuous actions.
            dtype: The dtype to use for layers.
        """
        super(ActorNetwork, self).__init__()

        if lstm_size is None:
            raise ValueError('Need to provide lstm_size.')

        self._continuous = continuous
        self._action_size = action_size
        self._dtype = dtype

        # Feature extractor using LSTMEncodingNetwork
        self._encoder = LSTMEncodingNetwork(
            input_size=input_size,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn,
            dtype=dtype,
        )

        encoder_output_size = self._encoder.output_size

        if continuous:
            # For continuous actions: output mean and log_std
            self._mean_layer = nn.Linear(encoder_output_size, action_size)
            # Initialize mean layer with small weights
            nn.init.uniform_(self._mean_layer.weight, -0.1, 0.1)
            nn.init.zeros_(self._mean_layer.bias)

            # Log std as learnable parameter (initialized based on init_action_stddev)
            init_log_std = np.log(np.exp(init_action_stddev) - 1)
            self._log_std = nn.Parameter(
                torch.full((action_size,), init_log_std, dtype=dtype)
            )
        else:
            # For discrete actions: output logits
            self._logits_layer = nn.Linear(encoder_output_size, action_size)
            # Initialize logits layer with small weights
            nn.init.uniform_(self._logits_layer.weight, -0.1, 0.1)
            nn.init.zeros_(self._logits_layer.bias)

    def get_initial_state(
            self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial hidden state for the LSTM."""
        return self._encoder.get_initial_state(batch_size, device)

    def forward(
            self,
            observations: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Union[Normal, Categorical], Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the network.

        Args:
            observations: Input tensor of shape (batch, features) or
                (batch, time, features).
            state: Optional tuple of (h, c) hidden states for the LSTM.

        Returns:
            A tuple of (action_distribution, next_state):
                - action_distribution: Normal or Categorical distribution.
                - next_state: Tuple of (h, c) for the next LSTM state.
        """
        # Get features from encoder
        features, next_state = self._encoder(observations, state)

        if self._continuous:
            # Compute mean and std for Normal distribution
            mean = self._mean_layer(features)
            # Use softplus to ensure positive std
            std = torch.nn.functional.softplus(self._log_std)
            std = std.expand_as(mean)
            distribution = Normal(mean, std)
        else:
            # Compute logits for Categorical distribution
            logits = self._logits_layer(features)
            distribution = Categorical(logits=logits)

        return distribution, next_state
