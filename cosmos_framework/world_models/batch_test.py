# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import torch

from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.world_models.batch import Cosmos3Sample, build_batch, predicted_states
from cosmos_framework.world_models.contract import Cosmos3WorldModelContract

CONTRACT = Cosmos3WorldModelContract()
MAX_ACTION_DIM = 64


def _sample(history_actions=4, history_states=5, task="put the bowl on the plate"):
    frames = CONTRACT.video_frames
    canvas = torch.randint(0, 255, (frames, *CONTRACT.canvas_size, 3), dtype=torch.uint8)
    actions = torch.arange(history_actions + 8, dtype=torch.float32)[:, None].expand(-1, 7).clone() / 10
    states = torch.arange(history_states + 8, dtype=torch.float32)[:, None].expand(-1, 8).clone() + 100
    return Cosmos3Sample(canvas=canvas, actions=actions, states=states, task=task)


def _tokenize(prompts):
    return [[7, 8, 9] for _ in prompts]


def test_train_batch_lays_out_actions_then_states():
    batch = build_batch([_sample()], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=_tokenize)

    assert batch["video"][0].shape == (3, 9, 256, 512) and batch["video"][0].dtype == torch.uint8
    payload = batch["action"][0][0]
    assert payload.shape == (25, MAX_ACTION_DIM)
    assert torch.equal(payload[:12, :7], torch.arange(12, dtype=torch.float32)[:, None].expand(-1, 7) / 10)
    assert torch.all(payload[:12, 7:] == 0)  # actions pad the state channel and the model-space channels
    assert torch.equal(payload[12:, :8], torch.arange(13, dtype=torch.float32)[:, None].expand(-1, 8) + 100)
    assert torch.all(payload[12:, 8:] == 0)
    assert batch["domain_id"][0][0].tolist() == [get_domain_id("libero")] * 12 + [get_domain_id("libero_state")] * 13
    assert batch["raw_action_dim"][0].item() == 8
    assert batch["action_processing_record"][0].raw_action_dim == 8
    assert batch["image_size"].tolist() == [[256.0, 512.0, 256.0, 512.0]]
    assert batch["conditioning_fps"][0].item() == 20.0
    assert batch["text_token_ids"][0][0].tolist() == [7, 8, 9]
    plan = batch["sequence_plan"][0]
    assert plan.has_text and plan.has_vision and plan.has_action
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == list(range(17))
    assert plan.action_frame_indexes == list(range(-3, 9)) + list(range(-4, 9))


def test_prompt_describes_the_task_canvas_and_clip():
    batch = build_batch([_sample()], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=None)

    prompt = json.loads(batch["ai_caption"][0])
    assert "text_token_ids" not in batch
    assert prompt["actions"][0]["description"] == "put the bowl on the plate."
    assert prompt["actions"][0]["time"] == "0.00-0.45s"
    assert prompt["duration"] == "0.45s" and prompt["fps"] == 20.0
    assert prompt["resolution"] == {"H": 256, "W": 512}
    assert "wrist camera" in prompt["cinematography"]["framing"]


def test_short_history_samples_pack_fewer_rows():
    batch = build_batch(
        [_sample(history_actions=1, history_states=2)], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=None
    )

    plan = batch["sequence_plan"][0]
    assert batch["action"][0][0].shape == (19, MAX_ACTION_DIM)
    assert plan.condition_frame_indexes_action == list(range(11))
    assert plan.action_frame_indexes == list(range(0, 9)) + list(range(-1, 9))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("canvas", torch.zeros((9, 256, 256, 3), dtype=torch.uint8), "canvas must be uint8"),
        ("actions", torch.zeros((12, 6)), r"actions must be \[rows,7\]"),
        ("states", torch.zeros((13, 9)), r"states must be \[rows,8\]"),
        ("actions", torch.zeros((13, 7)), "history_actions"),
    ],
)
def test_sample_shapes_are_validated(field, value, match):
    sample = _sample()
    sample = Cosmos3Sample(**{**sample.__dict__, field: value})
    with pytest.raises(ValueError, match=match):
        build_batch([sample], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=None)


def test_predicted_states_are_the_generated_rows():
    batch = build_batch([_sample()], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=None)
    rows = torch.arange(25, dtype=torch.float32)[:, None].expand(-1, 8)

    states = predicted_states(rows, batch["sequence_plan"][0], CONTRACT)

    assert torch.equal(states, rows[17:])
    with pytest.raises(ValueError, match="generated rows"):
        predicted_states(rows[:-1], batch["sequence_plan"][0], CONTRACT)


def test_packing_puts_actions_and_states_on_one_vision_clock():
    """Rows that describe the same frame share a temporal position; the state stream restarts."""
    batch = build_batch([_sample()], CONTRACT, max_action_dim=MAX_ACTION_DIM, tokenize=_tokenize)
    plan = batch["sequence_plan"][0]
    payload = batch["action"][0][0]
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros((1, 48, 3, 8, 16))],
        fps_vision=torch.tensor([20.0]),
        x0_tokens_action=[payload],
        fps_action=torch.tensor([20.0]),
        action_domain_id=[batch["domain_id"][0][0]],
        raw_action_dim=[batch["raw_action_dim"][0]],
        action_valid_mask=None,
    )

    packed = pack_input_sequence(
        sequence_plans=[plan],
        input_text_indexes=[batch["text_token_ids"][0][0].tolist()],
        gen_data_clean=gen_data_clean,
        input_timesteps=torch.tensor([[0.5]]),
        special_tokens={"eos_token_id": 11, "start_of_generation": 20, "end_of_generation": 21},
        latent_patch_size=2,
        enable_fps_modulation=True,
        base_fps=24.0,
        temporal_compression_factor=4,
        action_dim=MAX_ACTION_DIM,
    )

    condition_mask = packed.action.condition_mask[0].squeeze(1)
    assert condition_mask.tolist() == [1.0] * 17 + [0.0] * 8
    assert packed.action.noisy_frame_indexes[0].tolist() == list(range(17, 25))
    assert packed.action.domain_id[0].tolist() == batch["domain_id"][0][0].tolist()
    temporal = packed.position_ids[0, packed.action.sequence_indexes]
    frames = plan.action_frame_indexes
    by_frame = {}
    for frame, position in zip(frames, temporal.tolist()):
        by_frame.setdefault(frame, set()).add(position)
    assert all(len(positions) == 1 for positions in by_frame.values()), "a frame must map to one temporal position"
    ordered = [next(iter(by_frame[frame])) for frame in sorted(by_frame)]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered)
    # The clean conditioning frame and its state share the vision start of the sample.
    vision_start = packed.position_ids[0, packed.vision.sequence_indexes].min().item()
    assert by_frame[0] == {vision_start}
