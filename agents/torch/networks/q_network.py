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

"""PyTorch Q-Network for DQN."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .lstm_encoding_network import LSTMEncodingNetwork


class QNetwork(nn.Module):
    """Q-Network that outputs Q-values for all actions."""

    def __init__(
            self,
            input_size: int,
            num_actions: int,
            conv_layer_params: Optional[List[Tuple[int, int, int]]] = None,
            input_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            lstm_size: Optional[Tuple[int, ...]] = None,
            output_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            activation_fn: nn.Module = nn.ReLU,
            q_layer_activation_fn: Optional[nn.Module] = None,
            dtype: torch.dtype = torch.float32,
    ):
        """Creates an instance of `QNetwork`.

        Args:
            input_size: The size of the input features.
            num_actions: The number of discrete actions.
            conv_layer_params: Optional list of convolution layer parameters.
            input_fc_layer_params: Optional tuple of FC layer sizes before LSTM.
            lstm_size: Tuple of LSTM hidden sizes.
            output_fc_layer_params: Optional tuple of FC layer sizes after LSTM.
            activation_fn: Activation function class.
            q_layer_activation_fn: Optional activation for the Q-value layer.
            dtype: The dtype to use for layers.
        """
        super(QNetwork, self).__init__()

        if lstm_size is None:
            raise ValueError('Need to provide lstm_size.')

        self._dtype = dtype
        self._num_actions = num_actions

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

        # Q-value layer
        self._q_value_layer = nn.Linear(encoder_output_size, num_actions)
        # Initialize with small uniform weights and negative bias similar to TF version
        nn.init.uniform_(self._q_value_layer.weight, -0.03, 0.03)
        nn.init.constant_(self._q_value_layer.bias, -0.2)

        self._q_layer_activation = q_layer_activation_fn

    def get_initial_state(
            self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial hidden state for the LSTM."""
        return self._encoder.get_initial_state(batch_size, device)

    def forward(
            self,
            observations: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the network.

        Args:
            observations: Input tensor of shape (batch, features) or
                (batch, time, features).
            state: Optional tuple of (h, c) hidden states for the LSTM.

        Returns:
            A tuple of (q_values, next_state):
                - q_values: Q-values for all actions, shape (batch, num_actions)
                    or (batch, time, num_actions).
                - next_state: Tuple of (h, c) for the next LSTM state.
        """
        # Get features from encoder
        features, next_state = self._encoder(observations, state)

        # Compute Q-values
        q_values = self._q_value_layer(features)

        if self._q_layer_activation is not None:
            q_values = self._q_layer_activation(q_values)

        return q_values, next_state
