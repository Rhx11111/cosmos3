# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 forward dynamics as a host-managed world model.

The host framework owns distribution, optimization, checkpoints and weight
exchange of the MoT network; this module owns the model's loss, its joint
video/state prediction, and the frozen runtime assets around the network.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import hydra
import omegaconf
import torch

import cosmos_framework
from cosmos_framework.inference.common.config import CONFIG_DIR, deserialize_config_dict, structure_config
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils import misc
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.world_models.batch import Cosmos3Sample, build_batch, predicted_states
from cosmos_framework.world_models.checkpoint import load_diffusers_net_weights
from cosmos_framework.world_models.contract import Cosmos3WorldModelContract, compose_canvas, split_canvas

EDGE_MODEL_CONFIG = CONFIG_DIR / "model" / "Cosmos3-Edge.yaml"
#: Parameter-name markers trained by the forward-dynamics post-training recipe
#: (``action_fd_droid_posttrain``): the generation expert and the VAE/action projections.
TRAINABLE_PARAMETER_MARKERS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
_PACKAGE_ROOT = Path(cosmos_framework.__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cosmos3WorldModelLoss:
    """A differentiable scalar loss and detached per-modality flow-matching losses."""

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class Cosmos3Prediction:
    """Generated canvas frames ``[B, P, H, W, 3]`` (uint8) and states ``[B, P, state_dim]`` (float32, CPU)."""

    frames: torch.Tensor
    states: torch.Tensor


class Cosmos3MBRLModel(OmniMoTModel):
    """``OmniMoTModel`` kept unsharded so a host framework can wrap and optimize ``net``."""

    def set_up_parallelism(self) -> None:
        # Degree-1 inference-mode dims: no FSDP2 wrap, no device mesh, no process group.
        self.parallel_dims = ParallelDims(world_size=1, dp_shard=1, dp_replicate=1, enable_inference_mode=True)


def edge_model_config(
    checkpoint_dir: Path | str,
    vae_path: Path | str,
    contract: Cosmos3WorldModelContract,
    *,
    config_file: Path = EDGE_MODEL_CONFIG,
) -> dict:
    """Return the Cosmos3-Edge model config bound to local assets and stripped of training infrastructure."""
    # instantiate_model() runs from the package root, which would otherwise
    # resolve a relative asset path against the package instead of the caller.
    checkpoint_dir, vae_path = Path(checkpoint_dir).resolve(), Path(vae_path).resolve()
    model = deserialize_config_dict(Path(config_file))["model"]
    config = model["config"]
    model["_target_"] = f"{Cosmos3MBRLModel.__module__}.{Cosmos3MBRLModel.__qualname__}"
    config["ema"]["enabled"] = False
    config["compile"]["enabled"] = False
    config["activation_checkpointing"]["mode"] = "none"
    config["parallelism"].update(
        data_parallel_shard_degree=1,
        data_parallel_replicate_degree=1,
        context_parallel_shard_degree=1,
        cfg_parallel_shard_degree=1,
        enable_inference_mode=True,
    )
    tokenizer = config["tokenizer"]
    tokenizer.update(vae_path=str(vae_path), bucket_name="", object_store_credential_path_pretrained="")
    tokenizer["encode_exact_durations"] = sorted(
        set(tokenizer.get("encode_exact_durations") or ()) | {contract.video_frames}
    )
    config["vlm_config"]["tokenizer"] = {
        "_target_": "cosmos_framework.data.generator.processors.build_processor_lazy",
        "tokenizer_type": str(checkpoint_dir),
    }
    config["vlm_config"]["pretrained_weights"]["enabled"] = False
    return model


def instantiate_model(model_config: dict) -> Cosmos3MBRLModel:
    """Instantiate the configured model on the process device."""
    misc.set_torch_compile_options(use_duck_shape=False)
    model_dict = structure_config(model_config, omegaconf.DictConfig)
    # Package-relative config paths (the Nemotron backbone JSON) resolve from the package root.
    with contextlib.chdir(_PACKAGE_ROOT):
        return hydra.utils.instantiate(model_dict)


def freeze_for_forward_dynamics(net: torch.nn.Module) -> None:
    """Leave only the forward-dynamics recipe's parameters trainable."""
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(any(marker in name for marker in TRAINABLE_PARAMETER_MARKERS))


def build_cosmos3_world_model(
    checkpoint_dir: Path | str,
    vae_path: Path | str,
    contract: Cosmos3WorldModelContract,
    *,
    config_file: Path = EDGE_MODEL_CONFIG,
) -> Cosmos3WorldModel:
    """Build the Edge network from a diffusers checkpoint and a local Wan2.2 VAE."""
    model = instantiate_model(edge_model_config(checkpoint_dir, vae_path, contract, config_file=config_file))
    load_diffusers_net_weights(model.net, Path(checkpoint_dir))
    freeze_for_forward_dynamics(model.net)
    return Cosmos3WorldModel(model, contract)


class Cosmos3WorldModel(torch.nn.Module):
    """Expose Cosmos3 training loss and joint video/state prediction without owning distribution.

    Only ``net`` is registered, so the host's FSDP wrapper, optimizer, checkpoints
    and weight sync see exactly the MoT network. The VAE tokenizer, text processor,
    schedulers and samplers stay on the unregistered runtime model.

    Args:
        model: Unsharded model whose ``net`` this module registers.
        contract: Layout the batches follow.
    """

    def __init__(self, model: OmniMoTModel, contract: Cosmos3WorldModelContract) -> None:
        super().__init__()
        self.net = model.net
        self.contract = contract
        object.__setattr__(self, "_runtime", model)

    @property
    def runtime(self) -> OmniMoTModel:
        """Return the model that owns loss, sampling, and the frozen assets."""
        return self._runtime

    @property
    def device(self) -> torch.device:
        """Return the device the runtime computes on."""
        return torch.device(self._runtime.tensor_kwargs["device"])

    def build_train_batch(self, samples: Sequence[Cosmos3Sample]) -> dict:
        """Build a training batch with pre-tokenized prompts."""
        return build_batch(
            samples, self.contract, max_action_dim=self._runtime.config.max_action_dim, tokenize=self._tokenize
        )

    def build_predict_batch(self, samples: Sequence[Cosmos3Sample]) -> dict:
        """Build a prediction batch; the sampler tokenizes the prompts."""
        return build_batch(samples, self.contract, max_action_dim=self._runtime.config.max_action_dim, tokenize=None)

    def _tokenize(self, prompts: list[str]) -> list[list[int]]:
        return self._runtime._tokenize_captions(
            prompts,
            use_system_prompt=self._runtime.config.vlm_config.use_system_prompt,
            system_prompt=None,
            is_video=False,
        )

    def forward(self, batch: dict, iteration: int | None = None) -> Cosmos3WorldModelLoss:
        """Compute one batch's flow-matching loss without backward, reduction, or optimizer work.

        Args:
            batch: Output of :meth:`build_train_batch`; consumed in place, so build one per call.
            iteration: Seeds the noise schedule deterministically when given; ``None``
                draws from the global RNG.
        """
        output, loss = self._runtime.training_step(batch, iteration)
        return Cosmos3WorldModelLoss(
            loss,
            {
                "video_loss": output["flow_matching_loss_vision"].detach(),
                "state_loss": output["flow_matching_loss_action"].detach(),
            },
        )

    @torch.no_grad()
    def predict(self, batch: dict, *, num_inference_steps: int, seeds: Sequence[int]) -> Cosmos3Prediction:
        """Generate the future frames and states of every sample in ``batch``.

        Args:
            batch: Output of :meth:`build_predict_batch`; consumed in place.
            num_inference_steps: Denoising steps of the UniPC sampler.
            seeds: One noise seed per sample.
        """
        plans = list(batch["sequence_plan"])
        if len(seeds) != len(plans):
            raise ValueError(f"Expected {len(plans)} seeds, one per sample; got {len(seeds)}.")
        was_training = self.net.training
        self.net.eval()
        try:
            result = self._runtime.generate_samples_from_batch(
                batch,
                seed=[int(seed) for seed in seeds],
                num_steps=num_inference_steps,
                guidance=1.0,
                shift=float(self._training_shift()),
            )
        finally:
            self.net.train(was_training)
        frames = torch.stack([self._decode(latent) for latent in result["vision"]])  # [B,T,H,W,3]
        states = torch.stack(
            [predicted_states(rows.float().cpu(), plan, self.contract) for rows, plan in zip(result["action"], plans)]
        )
        return Cosmos3Prediction(frames[:, 1:], states)

    def _training_shift(self) -> int:
        """Return the noise-schedule shift of the canvas' resolution tier."""
        shift = self._runtime.config.rectified_flow_training_config.shift
        if isinstance(shift, int):
            return shift
        return int(dict(shift)[self.contract.resolution_tier])

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        video = self._runtime.decode(latent)[0]  # [3,T,H,W] in [-1,1]
        return ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu()

    def compose_canvas(self, views: torch.Tensor) -> torch.Tensor:
        """Letterbox camera frames ``[B, V, T, h, w, 3]`` into canvas frames ``[B, T, H, W, 3]``."""
        return compose_canvas(views, self.contract)

    def split_canvas(self, canvas: torch.Tensor, source_size: tuple[int, int]) -> torch.Tensor:
        """Restore camera frames ``[B, V, T, h, w, 3]`` from canvas frames ``[B, T, H, W, 3]``."""
        return split_canvas(canvas, self.contract, source_size)

    def reload_frozen_assets(self, device: str | torch.device) -> None:
        """Move the frozen VAE back to ``device``."""
        self._move_vae(torch.device(device))

    def offload_frozen_assets(self) -> None:
        """Move the frozen VAE to the CPU; the host moves ``net`` itself."""
        self._move_vae(torch.device("cpu"))

    def _move_vae(self, device: torch.device) -> None:
        vae = self._runtime.tokenizer_vision_gen.model
        vae.model.to(device)
        vae.scale = tuple(tensor.to(device) for tensor in vae.scale)
        vae.device = device
