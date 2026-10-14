# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.world_models.contract import (
    Cosmos3WorldModelContract,
    compose_canvas,
    letterbox_box,
    split_canvas,
)


def test_default_contract_describes_libero():
    contract = Cosmos3WorldModelContract()

    assert contract.canvas_size == (256, 512)
    assert contract.video_frames == 9
    assert contract.history_actions == 4
    assert contract.raw_row_dim == 8
    assert contract.resolution_tier == "256"


def test_rows_follow_the_vision_clock():
    contract = Cosmos3WorldModelContract()

    frames = contract.frame_indexes(4, 5)
    assert frames[:12] == list(range(-3, 9))  # action k produces frame k - 3
    assert frames[12:] == list(range(-4, 9))  # state j observes frame j - 4
    assert contract.condition_rows(4, 5) == list(range(17))
    domains = contract.domain_ids(4, 5)
    assert domains.tolist() == [get_domain_id("libero")] * 12 + [get_domain_id("libero_state")] * 13


def test_short_history_shrinks_the_payload():
    contract = Cosmos3WorldModelContract()

    assert contract.frame_indexes(0, 1) == list(range(1, 9)) + list(range(0, 9))
    assert contract.condition_rows(0, 1) == list(range(9))
    with pytest.raises(ValueError, match="history_actions"):
        contract.frame_indexes(5, 5)
    with pytest.raises(ValueError, match="history_states"):
        contract.frame_indexes(4, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"view_size": (200, 256)}, "multiples of 32"),
        ({"prediction_frames": 6}, "multiple of 4"),
        ({"condition_frames": 0}, "condition_frames"),
        ({"state_domain": "libero"}, "different domain ids"),
    ],
)
def test_invalid_layouts_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Cosmos3WorldModelContract(**kwargs)


def test_letterbox_box_centers_the_content():
    assert letterbox_box((256, 256), (256, 256)) == (0, 0, 256, 256)
    assert letterbox_box((128, 256), (256, 256)) == (64, 0, 128, 256)
    assert letterbox_box((256, 128), (256, 256)) == (0, 64, 256, 128)


def test_canvas_round_trip_restores_each_view():
    contract = Cosmos3WorldModelContract()
    views = torch.zeros((2, 2, 3, 128, 96, 3), dtype=torch.uint8)
    views[:, 0] = 200  # external camera is bright
    views[:, 1, :, :, :48] = 90  # wrist camera has a left half

    canvas = compose_canvas(views, contract)
    assert canvas.shape == (2, 3, 256, 512) + (3,)
    assert canvas.dtype == torch.uint8
    # The external tile occupies the left half of the canvas, letterboxed vertically.
    assert torch.all(canvas[0, 0, :, :256][canvas[0, 0, :, :256] != 0] == 200)
    assert torch.all(canvas[0, 0, 8:-8, 128 - 48 : 128 + 48] == 200)

    restored = split_canvas(canvas, contract, (128, 96))
    assert restored.shape == views.shape
    assert torch.all(restored[:, 0] == 200)
    assert torch.all(restored[:, 1, :, :, :40] == 90)
    assert torch.all(restored[:, 1, :, :, 56:] == 0)


def test_canvas_shapes_are_validated():
    contract = Cosmos3WorldModelContract()
    with pytest.raises(ValueError, match="views must be uint8"):
        compose_canvas(torch.zeros((1, 1, 2, 64, 64, 3), dtype=torch.uint8), contract)
    with pytest.raises(ValueError, match="canvas must be uint8"):
        split_canvas(torch.zeros((1, 2, 256, 256, 3), dtype=torch.uint8), contract, (64, 64))
