"""Lightweight CNN-ConvLSTM used by the Xiao BEV method."""

from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.hidden_channels, *spatial_size)
        return (
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.conv(torch.cat([input_tensor, hidden], dim=1))
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell_next = forget_gate * cell + input_gate * candidate
        hidden_next = output_gate * torch.tanh(cell_next)
        return hidden_next, cell_next


class ConvLSTM(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(channels, channels, kernel_size=kernel_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Process ``(B, T, C, H, W)`` and return all hidden states."""

        batch, steps, _, height, width = sequence.shape
        state = self.cell.init_hidden(
            batch,
            (height, width),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        outputs = []
        for step in range(steps):
            state = self.cell(sequence[:, step], state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1)


class CNNConvLSTM(nn.Module):
    """Xiao-style BEV CNN-ConvLSTM with configurable input channels."""

    def __init__(
        self,
        input_channels: int = 2,
        *,
        output_init: str = "zero",
    ) -> None:
        super().__init__()
        if output_init not in {"zero", "small", "default"}:
            raise ValueError("output_init must be 'zero', 'small', or 'default'")
        self.input_channels = input_channels
        self.output_init = output_init
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convlstm = ConvLSTM(channels=64, kernel_size=3)
        dose_output = nn.ConvTranspose2d(16, 1, kernel_size=1)
        if output_init == "zero":
            nn.init.zeros_(dose_output.weight)
            nn.init.zeros_(dose_output.bias)
        elif output_init == "small":
            nn.init.normal_(dose_output.weight, mean=0.0, std=1.0e-2)
            nn.init.zeros_(dose_output.bias)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            dose_output,
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} inputs, received {len(inputs)}"
            )
        reference_shape = inputs[0].shape
        if len(reference_shape) != 4:
            raise ValueError(f"expected (B,T,H,W), got {reference_shape}")
        if any(tensor.shape != reference_shape for tensor in inputs[1:]):
            raise ValueError("all model inputs must have identical shapes")

        batch, steps, height, width = reference_shape
        stacked = torch.stack(inputs, dim=2)
        encoded = self.encoder(
            stacked.reshape(batch * steps, self.input_channels, height, width)
        )
        _, channels, encoded_h, encoded_w = encoded.shape
        encoded = encoded.reshape(batch, steps, channels, encoded_h, encoded_w)
        recurrent = self.convlstm(encoded)
        decoded = self.decoder(
            recurrent.reshape(batch * steps, channels, encoded_h, encoded_w)
        )
        return decoded.reshape(batch, steps, height, width)
