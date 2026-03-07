"""Neural network definitions for RT-Gesture."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ReinhardCompression(nn.Module):
    """Dynamic range compression using the Reinhard operator."""

    def __init__(self, range: float, midpoint: float) -> None:
        super().__init__()
        self.range = range
        self.midpoint = midpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.range * inputs / (self.midpoint + torch.abs(inputs))


class DiscreteGesturesArchitecture(nn.Module):
    """Conv + stacked LSTM architecture for discrete gesture recognition."""

    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 512,
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
    ) -> None:
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.left_context = kernel_width - 1
        self.kernel_width = kernel_width
        self.stride = stride

        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        self.conv_layer = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        x, _ = self.lstm(x)
        x = self.post_lstm_layer_norm(x)
        x = self.projection(x)
        return x.permute(0, 2, 1)

    def _init_lstm_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(
            self.lstm_num_layers,
            batch_size,
            self.lstm_hidden_size,
            device=device,
            dtype=dtype,
        )
        c0 = torch.zeros(
            self.lstm_num_layers,
            batch_size,
            self.lstm_hidden_size,
            device=device,
            dtype=dtype,
        )
        return h0, c0

    def forward_streaming(
        self,
        new_samples: torch.Tensor,
        conv_history: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Run streaming inference for incremental samples.

        Parameters
        ----------
        new_samples:
            Tensor with shape (B, C, N).
        conv_history:
            Tail buffer from previous step with shape (B, C, H),
            where 0 <= H <= left_context.
        lstm_state:
            Previous LSTM hidden state tuple (h, c), or None for init.

        Returns
        -------
        logits:
            Tensor with shape (B, 9, T_out).
        new_history:
            Updated tail buffer for next call.
        new_state:
            Updated LSTM hidden state.
        """
        if new_samples.ndim != 3:
            raise ValueError(f"new_samples must be 3D, got shape {tuple(new_samples.shape)}")
        if conv_history.ndim != 3:
            raise ValueError(f"conv_history must be 3D, got shape {tuple(conv_history.shape)}")
        if new_samples.shape[:2] != conv_history.shape[:2]:
            raise ValueError("new_samples and conv_history must match batch/channel dimensions")

        batch_size, channels, _ = new_samples.shape
        history_len = conv_history.shape[2]
        input_len = history_len + new_samples.shape[2]
        full_input = torch.cat([conv_history, new_samples], dim=2)

        if lstm_state is None:
            lstm_state = self._init_lstm_state(
                batch_size=batch_size,
                device=new_samples.device,
                dtype=new_samples.dtype,
            )

        if input_len < self.kernel_width:
            zero_logits = torch.zeros(
                batch_size,
                self.projection.out_features,
                0,
                device=new_samples.device,
                dtype=new_samples.dtype,
            )
            return zero_logits, full_input, lstm_state

        x = self.compression(full_input)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        x, new_state = self.lstm(x, lstm_state)
        x = self.post_lstm_layer_norm(x)
        x = self.projection(x)
        logits = x.permute(0, 2, 1)

        produced_frames = logits.shape[2]
        consumed_samples = produced_frames * self.stride
        new_history = full_input[:, :, consumed_samples:]
        return logits, new_history, new_state


class WristArchitecture(nn.Module):
    """Placeholder for future implementation."""

    def __init__(self, *_: Sequence[object], **__: object) -> None:
        raise NotImplementedError("WristArchitecture is not part of C01/C02 scope.")
