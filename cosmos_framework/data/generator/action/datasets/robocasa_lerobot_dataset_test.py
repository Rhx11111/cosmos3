# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    STATE_STREAM_DIM,
    decode_state,
    encode_native16_state,
    normalize_state,
    official_state_to_native16,
)
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation


def _random_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _random_official_state(n: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.cat(
        [
            torch.randn(n, 3, generator=gen, dtype=torch.float64),  # base_pos
            _random_quat(n, gen),  # base_quat
            torch.randn(n, 3, generator=gen, dtype=torch.float64),  # eef_pos
            _random_quat(n, gen),  # hand_quat
            torch.rand(n, 2, generator=gen, dtype=torch.float64) * 0.04,  # finger qpos
        ],
        dim=-1,
    )


def _assert_native16_close(a: torch.Tensor, b: torch.Tensor) -> None:
    non_quat = [i for i in range(16) if i not in (3, 4, 5, 6, 10, 11, 12, 13)]
    torch.testing.assert_close(a[:, non_quat], b[:, non_quat], atol=1e-6, rtol=0)
    for q in (slice(3, 7), slice(10, 14)):  # q and -q are the same rotation
        torch.testing.assert_close((a[:, q] * b[:, q]).sum(-1).abs(), torch.ones(len(a), dtype=a.dtype), atol=1e-6, rtol=0)


def test_official_to_native16_applies_hand_to_grip() -> None:
    state = _random_official_state(1)
    state[:, 10:14] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)  # identity hand
    native = official_state_to_native16(state)
    grip = convert_rotation(native[:, 3:7], input_format="quat_xyzw", output_format="matrix")[0]
    expected = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(grip, expected, atol=1e-9, rtol=0)
    torch.testing.assert_close(native[:, 0:3], state[:, 7:10])  # grip pos = eef pos in base
    torch.testing.assert_close(native[:, 7:14], state[:, 0:7])  # base pose unchanged
    torch.testing.assert_close(native[:, 14:16], state[:, 14:16])


def test_state_stream_round_trip() -> None:
    native = official_state_to_native16(_random_official_state(64))
    encoded = encode_native16_state(native)
    assert encoded.shape == (64, STATE_STREAM_DIM)
    stats = {
        "mean": torch.linspace(-1.0, 1.0, STATE_STREAM_DIM, dtype=torch.float64),
        "std": torch.linspace(0.5, 2.0, STATE_STREAM_DIM, dtype=torch.float64),
    }
    _assert_native16_close(decode_state(normalize_state(encoded, stats), stats), native)


def test_decode_state_projects_noisy_rot6d() -> None:
    native = official_state_to_native16(_random_official_state(16, seed=1))
    stats = {"mean": torch.zeros(STATE_STREAM_DIM, dtype=torch.float64), "std": torch.ones(STATE_STREAM_DIM, dtype=torch.float64)}
    noisy = encode_native16_state(native) + 1e-3 * torch.randn(16, STATE_STREAM_DIM, dtype=torch.float64)
    decoded = decode_state(noisy, stats)
    for q in (slice(3, 7), slice(10, 14)):
        torch.testing.assert_close(decoded[:, q].norm(dim=-1), torch.ones(16, dtype=torch.float64), atol=1e-6, rtol=0)
