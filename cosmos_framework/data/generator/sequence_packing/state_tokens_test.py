# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-frame state tokens appended to a forward-dynamics action stream."""

import pytest
import torch

from cosmos_framework.data.generator.action.utils.transforms import append_state_tokens, build_sequence_plan_from_mode
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder

NUM_ACTIONS = 20
ACTION_DIM = 15
STATE_DIM = 20


def _plan_and_tokens():
    plan = build_sequence_plan_from_mode(
        mode="forward_dynamics", video_length=NUM_ACTIONS + 1, action_length=NUM_ACTIONS
    )
    action = torch.randn(NUM_ACTIONS, ACTION_DIM)
    state = torch.randn(NUM_ACTIONS + 1, STATE_DIM)
    return plan, action, state, append_state_tokens(action, state, plan)


def test_merged_layout_and_plan():
    plan, action, state, merged = _plan_and_tokens()
    assert merged.shape == (2 * NUM_ACTIONS + 1, ACTION_DIM + STATE_DIM)
    assert torch.equal(merged[:NUM_ACTIONS, :ACTION_DIM], action)
    assert torch.equal(merged[NUM_ACTIONS:, ACTION_DIM:], state)
    assert not merged[:NUM_ACTIONS, ACTION_DIM:].any() and not merged[NUM_ACTIONS:, :ACTION_DIM].any()
    # Every action and the current state s_0 are clean; s_1..s_T are predicted.
    assert plan.condition_frame_indexes_action == list(range(NUM_ACTIONS + 1))
    assert plan.action_state_tokens == NUM_ACTIONS + 1


def test_rejects_non_forward_dynamics():
    plan = build_sequence_plan_from_mode(mode="wam", video_length=NUM_ACTIONS + 1, action_length=NUM_ACTIONS)
    with pytest.raises(ValueError):
        append_state_tokens(torch.zeros(NUM_ACTIONS, ACTION_DIM), torch.zeros(NUM_ACTIONS + 1, STATE_DIM), plan)


@pytest.mark.parametrize("fps", [None, 20.0])
def test_state_positions_follow_vision_frames(fps):
    plan, _, _, merged = _plan_and_tokens()
    kwargs = dict(
        input_timestep=0.5,
        action_temporal_offset=7,
        enable_fps_modulation=fps is not None,
        base_fps=24.0,
        action_fps=fps,
        base_temporal_compression_factor=4,
    )
    builder = PackedSequenceBuilder()
    builder.pack_action_tokens(
        input_action_tokens=merged,
        condition_frame_indexes_action=plan.condition_frame_indexes_action,
        action_start_frame_offset=plan.action_start_frame_offset,
        action_state_tokens=plan.action_state_tokens,
        **kwargs,
    )
    positions = torch.cat(builder.position_ids, dim=1)  # [3,T_a+T_s]

    # Reference streams packed alone: actions from frame 1, states from frame 0.
    ref_action, ref_state = PackedSequenceBuilder(), PackedSequenceBuilder()
    ref_action.pack_action_tokens(merged[:NUM_ACTIONS], list(range(NUM_ACTIONS)), action_start_frame_offset=1, **kwargs)
    ref_state.pack_action_tokens(merged[NUM_ACTIONS:], [], action_start_frame_offset=0, **kwargs)
    assert torch.equal(positions[:, :NUM_ACTIONS], torch.cat(ref_action.position_ids, dim=1))
    assert torch.equal(positions[:, NUM_ACTIONS:], torch.cat(ref_state.position_ids, dim=1))
    # Action a_t (transition t-1 -> t) shares its temporal position with state s_t.
    assert torch.equal(positions[0, :NUM_ACTIONS], positions[0, NUM_ACTIONS + 1 :])

    action = builder.action
    assert action.condition_mask[0][: NUM_ACTIONS + 1].eq(1).all()
    assert action.condition_mask[0][NUM_ACTIONS + 1 :].eq(0).all()
    assert len(action.mse_loss_indexes) == NUM_ACTIONS
