from __future__ import annotations

import torch

from rt_gesture.networks import DiscreteGesturesArchitecture


def _run_streaming(
    model: DiscreteGesturesArchitecture,
    full_input: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    history = torch.zeros(
        full_input.shape[0],
        full_input.shape[1],
        0,
        dtype=full_input.dtype,
    )
    state = None
    outputs: list[torch.Tensor] = []
    total_samples = full_input.shape[2]
    for start in range(0, total_samples, chunk_size):
        chunk = full_input[:, :, start : start + chunk_size]
        logits, history, state = model.forward_streaming(chunk, history, state)
        outputs.append(logits)
    return torch.cat(outputs, dim=2) if outputs else torch.zeros_like(full_input[:, :9, :0])


def test_forward_streaming_matches_forward_for_multiple_chunk_sizes() -> None:
    torch.manual_seed(0)
    model = DiscreteGesturesArchitecture(
        conv_output_channels=64,
        lstm_hidden_size=32,
        lstm_num_layers=2,
    )
    model.eval()
    full_input = torch.randn(1, 16, 260)
    with torch.no_grad():
        full_logits = model(full_input)
        for chunk_size in (1, 3, 7, 10, 17, 40, 73):
            stream_logits = _run_streaming(model, full_input, chunk_size=chunk_size)
            assert stream_logits.shape == full_logits.shape
            assert torch.allclose(full_logits, stream_logits, atol=1e-6)


def test_forward_streaming_handles_insufficient_context() -> None:
    model = DiscreteGesturesArchitecture(
        conv_output_channels=32,
        lstm_hidden_size=16,
        lstm_num_layers=2,
    )
    model.eval()
    history = torch.zeros(1, 16, 0)
    state = None
    new_samples = torch.randn(1, 16, 10)
    logits, new_history, new_state = model.forward_streaming(new_samples, history, state)
    assert logits.shape == (1, 9, 0)
    assert new_history.shape[2] == 10
    assert new_state[0].shape == (2, 1, 16)
    assert new_state[1].shape == (2, 1, 16)
