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

"""PyTorch LSTM Encoding Network.

Implements a network that will generate the following layers:

  [optional]: Conv1D layers  # conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM layer
  [optional]: Dense  # output_fc_layer_params
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class LSTMEncodingNetwork(nn.Module):
    """Recurrent network with optional Conv1D, FC, and LSTM layers."""

    def __init__(
            self,
            input_size: int,
            conv_layer_params: Optional[List[Tuple[int, int, int]]] = None,
            input_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            lstm_size: Optional[Tuple[int, ...]] = None,
            output_fc_layer_params: Optional[Tuple[int, ...]] = (75, 40),
            activation_fn: nn.Module = nn.ReLU,
            dtype: torch.dtype = torch.float32,
    ):
        """Creates an instance of `LSTMEncodingNetwork`.

        Args:
            input_size: The size of the input features (number of input channels
                for conv layers or input dimension for FC layers).
            conv_layer_params: Optional list of convolution layer parameters, where
                each item is a length-three tuple indicating (filters, kernel_size,
                stride).
            input_fc_layer_params: Optional tuple of fully connected layer sizes.
                These feed into the recurrent layer.
            lstm_size: A tuple of ints specifying the LSTM hidden sizes to use.
                For stacked LSTMs, provide multiple sizes.
            output_fc_layer_params: Optional tuple of fully connected layer sizes.
                These are applied on top of the recurrent layer.
            activation_fn: Activation function class, e.g., nn.ReLU.
            dtype: The dtype to use for the layers.

        Raises:
            ValueError: If `lstm_size` is not provided.
        """
        super(LSTMEncodingNetwork, self).__init__()

        if lstm_size is None:
            raise ValueError('Need to provide lstm_size.')

        self._dtype = dtype
        self._lstm_size = lstm_size
        self._num_lstm_layers = len(lstm_size)

        # Build preprocessing Conv1D layers
        self._conv_layers = nn.ModuleList()
        current_channels = input_size
        if conv_layer_params:
            for filters, kernel_size, stride in conv_layer_params:
                self._conv_layers.append(
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                )
                self._conv_layers.append(activation_fn())
                current_channels = filters

        # Track the flattened size after conv layers
        # This will be computed dynamically in forward pass if conv layers exist
        self._has_conv = len(self._conv_layers) > 0
        self._post_conv_size = current_channels  # Will be updated if conv layers exist

        # Build input FC layers
        self._input_fc_layers = nn.ModuleList()
        if input_fc_layer_params:
            for num_units in input_fc_layer_params:
                self._input_fc_layers.append(nn.LazyLinear(num_units))
                self._input_fc_layers.append(activation_fn())
                current_channels = num_units

        # Build LSTM layer(s)
        # PyTorch LSTM supports stacked layers natively
        if len(lstm_size) == 1:
            self._lstm = nn.LSTM(
                input_size=current_channels,
                hidden_size=lstm_size[0],
                num_layers=1,
                batch_first=True,
            )
            self._lstm_output_size = lstm_size[0]
        else:
            # For variable hidden sizes per layer, we need separate LSTM layers
            self._lstm = nn.ModuleList()
            lstm_input_size = current_channels
            for size in lstm_size:
                self._lstm.append(
                    nn.LSTM(
                        input_size=lstm_input_size,
                        hidden_size=size,
                        num_layers=1,
                        batch_first=True,
                    )
                )
                lstm_input_size = size
            self._lstm_output_size = lstm_size[-1]

        # Build output FC layers
        self._output_fc_layers = nn.ModuleList()
        current_size = self._lstm_output_size
        if output_fc_layer_params:
            for num_units in output_fc_layer_params:
                self._output_fc_layers.append(nn.Linear(current_size, num_units))
                self._output_fc_layers.append(activation_fn())
                current_size = num_units

        self._output_size = current_size

    def get_initial_state(
            self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial hidden state for the LSTM.

        Args:
            batch_size: The batch size for the initial state.
            device: The device to create the tensors on.

        Returns:
            A tuple of (h_0, c_0) tensors for LSTM initial state.
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(self._lstm, nn.LSTM):
            h_0 = torch.zeros(
                1, batch_size, self._lstm_size[0],
                dtype=self._dtype, device=device
            )
            c_0 = torch.zeros(
                1, batch_size, self._lstm_size[0],
                dtype=self._dtype, device=device
            )
        else:
            # Stacked LSTMs with different sizes
            h_0 = [
                torch.zeros(1, batch_size, size, dtype=self._dtype, device=device)
                for size in self._lstm_size
            ]
            c_0 = [
                torch.zeros(1, batch_size, size, dtype=self._dtype, device=device)
                for size in self._lstm_size
            ]
        return (h_0, c_0)

    @property
    def output_size(self) -> int:
        """Returns the output size of the network."""
        return self._output_size

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
                If None, zeros will be used.

        Returns:
            A tuple of (features, next_state):
                - features: Output tensor of shape (batch, output_size) or
                    (batch, time, output_size).
                - next_state: Tuple of (h, c) for the next LSTM state.
        """
        # Check if we have a time dimension
        has_time_dim = observations.dim() == 3

        if not has_time_dim:
            # Add time dimension: (batch, features) -> (batch, 1, features)
            observations = observations.unsqueeze(1)

        batch_size, seq_len, _ = observations.shape

        # Initialize state if not provided
        if state is None:
            state = self.get_initial_state(batch_size, observations.device)

        # Apply Conv1D layers if present
        if self._has_conv:
            # Conv1D expects (batch, channels, length)
            # We have (batch, time, features), treat time as length
            x = observations.transpose(1, 2)  # (batch, features, time)
            for layer in self._conv_layers:
                x = layer(x)
            # Flatten: (batch, channels, new_length) -> (batch, time_out, channels*new_length)
            # For simplicity, we flatten the channel and remaining spatial dims
            x = x.transpose(1, 2)  # (batch, new_length, channels)
            # If we need to preserve time dimension structure, we keep the sequence
        else:
            x = observations

        # Apply input FC layers
        for layer in self._input_fc_layers:
            x = layer(x)

        # Apply LSTM
        h, c = state
        if isinstance(self._lstm, nn.LSTM):
            x, (h_next, c_next) = self._lstm(x, (h, c))
            next_state = (h_next, c_next)
        else:
            # Multiple LSTM layers with different sizes
            h_next_list = []
            c_next_list = []
            for i, lstm_layer in enumerate(self._lstm):
                x, (h_n, c_n) = lstm_layer(x, (h[i], c[i]))
                h_next_list.append(h_n)
                c_next_list.append(c_n)
            next_state = (h_next_list, c_next_list)

        # Apply output FC layers
        for layer in self._output_fc_layers:
            x = layer(x)

        # Remove time dimension if input didn't have one
        if not has_time_dim:
            x = x.squeeze(1)

        return x, next_state
