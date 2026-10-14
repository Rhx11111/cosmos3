# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build Cosmos3 forward-dynamics data batches from host-side windows."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessingRecord, pad_action_to_max_dim
from cosmos_framework.data.generator.action.utils.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.world_models.contract import Cosmos3WorldModelContract

CANVAS_VIEWPOINT = "concat_view"


@dataclass(frozen=True)
class Cosmos3Sample:
    """One training window or prediction request in canvas space.

    Attributes:
        canvas: uint8 RGB frames ``[T, H, W, 3]``: the conditioning frame followed by
            the target frames for training, or by placeholder frames (the conditioning
            frame repeated) for prediction.
        actions: float ``[history_actions + P, action_dim]`` native actions, oldest
            first; row ``history_actions`` is applied at the conditioning frame.
        states: float ``[history_states + P, state_dim]`` native states, oldest first;
            row ``history_states - 1`` belongs to the conditioning frame and the P
            future rows are zeros for prediction.
        task: Task instruction.
    """

    canvas: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    task: str


def build_batch(
    samples: Sequence[Cosmos3Sample],
    contract: Cosmos3WorldModelContract,
    *,
    max_action_dim: int,
    tokenize: Callable[[list[str]], list[list[int]]] | None,
) -> dict:
    """Return the ``OmniMoTModel`` data batch describing ``samples``.

    Args:
        samples: Windows sharing the contract's canvas and prediction horizon; their
            history lengths may differ.
        contract: Layout of the payload rows and the canvas.
        max_action_dim: Model-space action width every payload row is padded to.
        tokenize: Renders prompts to text token ids for training. Prediction passes
            ``None``: the sampler tokenizes the captions itself.

    Returns:
        Batch-list data dict for ``training_step`` or ``generate_samples_from_batch``.
    """
    if not samples:
        raise ValueError("At least one sample is required.")
    if max_action_dim < contract.raw_row_dim:
        raise ValueError(f"max_action_dim {max_action_dim} is narrower than the payload rows ({contract.raw_row_dim}).")
    canvas_height, canvas_width = contract.canvas_size
    frames = contract.video_frames
    formatter = ActionPromptJsonFormatter(float_seconds=True)
    fps = torch.tensor(contract.fps, dtype=torch.float32)
    image_size = torch.tensor([canvas_height, canvas_width, canvas_height, canvas_width], dtype=torch.float32)
    record = ActionProcessingRecord(raw_action_dim=contract.raw_row_dim, action_normalizer=None)

    videos, prompts, payloads, domain_ids, plans = [], [], [], [], []
    for sample in samples:
        canvas = torch.as_tensor(sample.canvas).detach().cpu()
        if tuple(canvas.shape) != (frames, canvas_height, canvas_width, 3) or canvas.dtype != torch.uint8:
            raise ValueError(
                f"canvas must be uint8 [{frames},{canvas_height},{canvas_width},3]; got {tuple(canvas.shape)} {canvas.dtype}."
            )
        actions = torch.as_tensor(sample.actions, dtype=torch.float32).detach().cpu()
        states = torch.as_tensor(sample.states, dtype=torch.float32).detach().cpu()
        if actions.ndim != 2 or actions.shape[1] != contract.action_dim:
            raise ValueError(f"actions must be [rows,{contract.action_dim}]; got {tuple(actions.shape)}.")
        if states.ndim != 2 or states.shape[1] != contract.state_dim:
            raise ValueError(f"states must be [rows,{contract.state_dim}]; got {tuple(states.shape)}.")
        history_actions = actions.shape[0] - contract.prediction_frames
        history_states = states.shape[0] - contract.prediction_frames
        frame_indexes = contract.frame_indexes(history_actions, history_states)

        rows = torch.zeros((len(frame_indexes), contract.raw_row_dim), dtype=torch.float32)
        rows[: actions.shape[0], : contract.action_dim] = actions
        rows[actions.shape[0] :, : contract.state_dim] = states
        video = canvas.permute(3, 0, 1, 2).contiguous()  # [3,T,H,W]
        prompt = formatter(
            {
                "ai_caption": sample.task,
                "viewpoint": CANVAS_VIEWPOINT,
                "additional_view_description": contract.view_description,
                "video": video,
                "conditioning_fps": fps,
                "image_size": image_size,
            }
        )["ai_caption"]
        videos.append(video)
        prompts.append(json.dumps(prompt))
        payloads.append(pad_action_to_max_dim(rows, max_action_dim))
        domain_ids.append(contract.domain_ids(history_actions, history_states))
        plans.append(
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
                condition_frame_indexes_action=contract.condition_rows(history_actions, history_states),
                action_frame_indexes=frame_indexes,
            )
        )

    batch_size = len(samples)
    batch = {
        "video": videos,
        "ai_caption": prompts,
        "action": [[payload] for payload in payloads],
        "domain_id": [[ids] for ids in domain_ids],
        "raw_action_dim": [torch.tensor(contract.raw_row_dim, dtype=torch.long) for _ in range(batch_size)],
        "action_processing_record": [record] * batch_size,
        "conditioning_fps": [fps.clone() for _ in range(batch_size)],
        "image_size": image_size.expand(batch_size, -1).clone(),
        "sequence_plan": plans,
    }
    if tokenize is not None:
        batch["text_token_ids"] = [[torch.tensor(ids, dtype=torch.long)] for ids in tokenize(prompts)]
    return batch


def predicted_states(rows: torch.Tensor, plan: SequencePlan, contract: Cosmos3WorldModelContract) -> torch.Tensor:
    """Return the generated future states ``[P, state_dim]`` from a sample's decoded payload."""
    future = rows[len(plan.condition_frame_indexes_action) :]
    if future.shape[0] != contract.prediction_frames:
        raise ValueError(f"Expected {contract.prediction_frames} generated rows; got {future.shape[0]}.")
    return future[:, : contract.state_dim]
