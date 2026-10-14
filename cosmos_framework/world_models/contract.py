# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static contract between an MBRL host and Cosmos3 joint video/state forward dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.utils.generator.image_resize import get_vision_data_resolution

VAE_SPATIAL_COMPRESSION = 16
LATENT_PATCH_SIZE = 2
VAE_TEMPORAL_COMPRESSION = 4


@dataclass(frozen=True)
class Cosmos3WorldModelContract:
    """Temporal, visual, action, and state layout shared by training and prediction.

    The model conditions on the last observed frame, the ``condition_frames - 1``
    actions between the observed frames, and the ``condition_frames`` observed
    states, then generates ``prediction_frames`` frames and states. Actions and
    states travel in one Cosmos action payload: the action rows come first and
    stay clean, the state rows follow, and only the future state rows are noised
    and supervised. Each row carries its own domain id and vision frame index.

    Attributes:
        views: Camera names, left to right on the canvas.
        view_size: Height and width of every letterboxed camera tile.
        fps: Frame rate of observations and actions.
        action_domain: Registered Cosmos action domain of the native controller input.
        action_dim: Native action width.
        state_domain: Registered Cosmos action domain that carries the state rows.
        state_dim: Native state width.
        condition_frames: Observed frames; the last one conditions the video.
        prediction_frames: Generated frames, actions, and states.
        view_description: Prompt sentence describing the tile layout.
    """

    views: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    view_size: tuple[int, int] = (256, 256)
    fps: float = 20.0
    action_domain: str = "libero"
    action_dim: int = 7
    state_domain: str = "libero_state"
    state_dim: int = 8
    condition_frames: int = 5
    prediction_frames: int = 8
    view_description: str = "The left tile is the fixed external camera and the right tile is the wrist camera."

    def __post_init__(self) -> None:
        tile = VAE_SPATIAL_COMPRESSION * LATENT_PATCH_SIZE
        if not self.views:
            raise ValueError("At least one camera view is required.")
        if len(self.view_size) != 2 or any(side <= 0 or side % tile for side in self.view_size):
            raise ValueError(f"view_size must be positive multiples of {tile}; got {self.view_size}.")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive; got {self.fps}.")
        if self.condition_frames < 1:
            raise ValueError(f"condition_frames must be at least 1; got {self.condition_frames}.")
        if self.prediction_frames < 1 or self.prediction_frames % VAE_TEMPORAL_COMPRESSION:
            raise ValueError(
                f"prediction_frames must be a positive multiple of {VAE_TEMPORAL_COMPRESSION} so the "
                f"conditioning frame plus the predictions form a 4k+1 video; got {self.prediction_frames}."
            )
        if self.action_dim < 1 or self.state_dim < 1:
            raise ValueError("action_dim and state_dim must be positive.")
        if get_domain_id(self.action_domain) == get_domain_id(self.state_domain):
            raise ValueError("action_domain and state_domain must map to different domain ids.")

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Height and width of the composed canvas: tiles side by side."""
        return (self.view_size[0], self.view_size[1] * len(self.views))

    @property
    def video_frames(self) -> int:
        """Frames per video: the conditioning frame plus the predictions."""
        return self.prediction_frames + 1

    @property
    def history_actions(self) -> int:
        """Actions between the observed frames."""
        return self.condition_frames - 1

    @property
    def raw_row_dim(self) -> int:
        """Native width of a payload row: the wider of action and state."""
        return max(self.action_dim, self.state_dim)

    @property
    def resolution_tier(self) -> str:
        """Cosmos resolution tier the canvas falls into; selects the training noise shift."""
        return get_vision_data_resolution(self.canvas_size)

    def frame_indexes(self, history_actions: int, history_states: int) -> list[int]:
        """Vision frame index of every payload row.

        Action row ``k`` produces frame ``k + 1 - history_actions``; state row ``j``
        observes frame ``j + 1 - history_states``. Frame 0 is the conditioning frame.
        """
        self._check_history(history_actions, history_states)
        return list(range(1 - history_actions, self.prediction_frames + 1)) + list(
            range(1 - history_states, self.prediction_frames + 1)
        )

    def domain_ids(self, history_actions: int, history_states: int) -> torch.Tensor:
        """Per-row domain ids: the action domain, then the state domain."""
        self._check_history(history_actions, history_states)
        return torch.cat(
            (
                torch.full((history_actions + self.prediction_frames,), get_domain_id(self.action_domain)),
                torch.full((history_states + self.prediction_frames,), get_domain_id(self.state_domain)),
            )
        ).to(torch.long)

    def condition_rows(self, history_actions: int, history_states: int) -> list[int]:
        """Clean rows: every action plus the observed states."""
        self._check_history(history_actions, history_states)
        return list(range(history_actions + self.prediction_frames + history_states))

    def _check_history(self, history_actions: int, history_states: int) -> None:
        if not 0 <= history_actions <= self.history_actions:
            raise ValueError(f"history_actions must be in [0, {self.history_actions}]; got {history_actions}.")
        if not 1 <= history_states <= self.condition_frames:
            raise ValueError(f"history_states must be in [1, {self.condition_frames}]; got {history_states}.")


def letterbox_box(source_size: tuple[int, int], tile_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return ``(top, left, height, width)`` of a source image letterboxed into a tile."""
    source_height, source_width = source_size
    tile_height, tile_width = tile_size
    scale = min(tile_height / source_height, tile_width / source_width)
    height = max(1, round(source_height * scale))
    width = max(1, round(source_width * scale))
    return (tile_height - height) // 2, (tile_width - width) // 2, height, width


def compose_canvas(views: torch.Tensor, contract: Cosmos3WorldModelContract) -> torch.Tensor:
    """Letterbox every camera frame into its tile and lay the tiles out side by side.

    Args:
        views: uint8 RGB frames ``[B, V, T, h, w, 3]`` in the contract's view order.
        contract: Layout to compose for.

    Returns:
        uint8 canvas frames ``[B, T, H, W, 3]`` on the CPU.
    """
    views = torch.as_tensor(views).detach().cpu()
    if views.ndim != 6 or views.shape[1] != len(contract.views) or views.shape[-1] != 3 or views.dtype != torch.uint8:
        raise ValueError(
            f"views must be uint8 RGB [B,{len(contract.views)},T,h,w,3]; got {tuple(views.shape)} {views.dtype}."
        )
    tile_width = contract.view_size[1]
    top, left, height, width = letterbox_box((int(views.shape[3]), int(views.shape[4])), contract.view_size)
    canvas = np.zeros((views.shape[0], views.shape[2], *contract.canvas_size, 3), dtype=np.uint8)
    for batch_index, view_index, time_index in np.ndindex(views.shape[0], views.shape[1], views.shape[2]):
        frame = Image.fromarray(views[batch_index, view_index, time_index].numpy())
        tile = np.asarray(frame.resize((width, height), Image.Resampling.BICUBIC))
        column = view_index * tile_width + left
        canvas[batch_index, time_index, top : top + height, column : column + width] = tile
    return torch.from_numpy(canvas)


def split_canvas(
    canvas: torch.Tensor, contract: Cosmos3WorldModelContract, source_size: tuple[int, int]
) -> torch.Tensor:
    """Cut the tiles out of canvas frames and restore each view's source resolution.

    Args:
        canvas: uint8 RGB canvas frames ``[B, T, H, W, 3]``.
        contract: Layout the canvas was composed with.
        source_size: Height and width the views were letterboxed from.

    Returns:
        uint8 RGB frames ``[B, V, T, h, w, 3]`` on the CPU.
    """
    canvas = torch.as_tensor(canvas).detach().cpu()
    if canvas.ndim != 5 or tuple(canvas.shape[2:4]) != contract.canvas_size or canvas.dtype != torch.uint8:
        raise ValueError(
            f"canvas must be uint8 RGB [B,T,{contract.canvas_size[0]},{contract.canvas_size[1]},3]; "
            f"got {tuple(canvas.shape)} {canvas.dtype}."
        )
    source_height, source_width = source_size
    top, left, height, width = letterbox_box(source_size, contract.view_size)
    tile_width = contract.view_size[1]
    views = np.empty(
        (canvas.shape[0], len(contract.views), canvas.shape[1], source_height, source_width, 3), dtype=np.uint8
    )
    for batch_index, view_index, time_index in np.ndindex(views.shape[:3]):
        column = view_index * tile_width + left
        content = canvas[batch_index, time_index, top : top + height, column : column + width].numpy()
        views[batch_index, view_index, time_index] = np.asarray(
            Image.fromarray(content).resize((source_width, source_height), Image.Resampling.BILINEAR)
        )
    return torch.from_numpy(views)
