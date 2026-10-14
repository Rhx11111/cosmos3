# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Acceptance check for the Cosmos3 MBRL world model with real weights.

Builds the Edge network from a diffusers checkpoint and a local Wan2.2 VAE, then
walks the host-facing contract on synthetic LIBERO-shaped windows. On the CPU the
check stops before the network forward (the attention backends need a GPU) but
still exercises the real tokenizer, VAE encoding, prompt rendering, sequence
packing and noising. On CUDA it also runs one training forward/backward and one
prediction.

Example:
    COSMOS_DEVICE=cpu CUDA_VISIBLE_DEVICES= python tools/mbrl_check.py \\
        --checkpoint ~/run/models/Cosmos3-Edge --vae assets/wan_world_model/Wan2.2_VAE.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Diffusers-layout Cosmos3-Edge directory")
    parser.add_argument("--vae", required=True, type=Path, help="Wan2.2_VAE.pth")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--view-size", type=int, default=128, help="Synthetic camera frame side length")
    parser.add_argument("--inference-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None, help="Write the summary JSON here as well")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device_name = os.environ.get("COSMOS_DEVICE", "cuda").lower()
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("COSMOS_DEVICE=cuda but CUDA is unavailable; set COSMOS_DEVICE=cpu for the CPU check.")

    from cosmos_framework.world_models import Cosmos3Sample, Cosmos3WorldModelContract, build_cosmos3_world_model
    from cosmos_framework.world_models.mbrl import TRAINABLE_PARAMETER_MARKERS

    summary: dict = {"device": device_name, "checkpoint": str(args.checkpoint), "vae": str(args.vae)}
    contract = Cosmos3WorldModelContract()
    summary["contract"] = {
        "canvas_size": contract.canvas_size,
        "resolution_tier": contract.resolution_tier,
        "video_frames": contract.video_frames,
        "payload_rows": len(contract.frame_indexes(contract.history_actions, contract.condition_frames)),
    }

    started = time.monotonic()
    world_model = build_cosmos3_world_model(args.checkpoint, args.vae, contract)
    summary["build_seconds"] = round(time.monotonic() - started, 1)
    net = world_model.net
    parameters = dict(net.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    summary["parameters"] = {
        "total": sum(p.numel() for p in parameters.values()),
        "trainable": sum(parameters[name].numel() for name in trainable),
        "dtype": str(next(iter(parameters.values())).dtype),
        "trainable_markers_present": sorted(
            marker for marker in TRAINABLE_PARAMETER_MARKERS if any(marker in name for name in trainable)
        ),
    }
    assert trainable, "forward-dynamics parameters must stay trainable"
    assert all(any(marker in name for marker in TRAINABLE_PARAMETER_MARKERS) for name in trainable)
    assert not any(name.startswith("language_model.visual.") for name in trainable)
    assert all(torch.isfinite(p).all() for p in parameters.values()), "loaded weights contain non-finite values"

    generator = torch.Generator().manual_seed(0)
    frames = contract.video_frames
    views = torch.randint(
        0,
        256,
        (args.batch_size, len(contract.views), frames, args.view_size, args.view_size, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    canvas = world_model.compose_canvas(views)
    assert tuple(canvas.shape) == (args.batch_size, frames, *contract.canvas_size, 3)
    history_actions, history_states = contract.history_actions, contract.condition_frames
    samples = [
        Cosmos3Sample(
            canvas=canvas[index],
            actions=torch.randn(history_actions + contract.prediction_frames, contract.action_dim, generator=generator),
            states=torch.randn(history_states + contract.prediction_frames, contract.state_dim, generator=generator),
            task=f"put the bowl {index} on the plate",
        )
        for index in range(args.batch_size)
    ]

    runtime = world_model.runtime
    train_batch = world_model.build_train_batch(samples)
    summary["prompt"] = json.loads(train_batch["ai_caption"][0])
    summary["text_tokens"] = int(train_batch["text_token_ids"][0][0].numel())

    started = time.monotonic()
    text_indexes, plans, gen_data_clean, memory_info, resolutions, _ = runtime._get_training_inputs(train_batch, None)
    latent = gen_data_clean.x0_tokens_vision[0]
    summary["vae"] = {
        "latent_shape": list(latent.shape),
        "encode_seconds": round(time.monotonic() - started, 1),
        "resolutions": resolutions,
    }
    assert list(latent.shape[2:]) == [
        (frames - 1) // 4 + 1,
        contract.canvas_size[0] // 16,
        contract.canvas_size[1] // 16,
    ], latent.shape
    timesteps, sigmas = runtime._get_train_noise_level_vision(
        batch_size=gen_data_clean.batch_size,
        is_image_batch=False,
        resolutions=resolutions,
        num_vision_latent_frames=[x.shape[2] for x in gen_data_clean.x0_tokens_vision],
        num_tokens=[x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision],
        iteration=None,
    )
    packed = runtime._pack_input_sequence(
        plans,
        text_indexes,
        gen_data_clean,
        timesteps.cpu(),
        skip_text_tokens=memory_info["skip_text"],
        initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
    )
    rows = len(contract.frame_indexes(history_actions, history_states))
    condition_mask = packed.action.condition_mask[0].squeeze(1).tolist()
    assert condition_mask == [1.0] * (rows - contract.prediction_frames) + [0.0] * contract.prediction_frames
    noised = runtime._add_noise_to_input(gen_data_clean, packed, sigmas)
    state_sigma = noised.sigmas_action[0].squeeze(1)
    assert torch.all(state_sigma[: rows - contract.prediction_frames] == 0)
    assert torch.all(state_sigma[rows - contract.prediction_frames :] > 0)
    summary["packing"] = {
        "sequence_length": int(packed.sequence_length),
        "action_tokens": int(packed.action.sequence_indexes.numel()),
        "vision_tokens": int(packed.vision.sequence_indexes.numel()),
        "text_tokens": int(packed.text_indexes.numel()),
    }

    if device_name == "cuda":
        started = time.monotonic()
        result = world_model(world_model.build_train_batch(samples))
        result.loss.backward()
        summary["train"] = {
            "loss": float(result.loss),
            "video_loss": float(result.metrics["video_loss"]),
            "state_loss": float(result.metrics["state_loss"]),
            "seconds": round(time.monotonic() - started, 1),
            "trainable_without_grad": sorted(name for name in trainable if parameters[name].grad is None)[:10],
            "frozen_with_grad": sorted(
                name for name, p in parameters.items() if name not in trainable and p.grad is not None
            )[:10],
        }
        assert torch.isfinite(result.loss)
        assert not summary["train"]["trainable_without_grad"]
        assert not summary["train"]["frozen_with_grad"]
        net.zero_grad(set_to_none=True)

        started = time.monotonic()
        predict_samples = [
            Cosmos3Sample(
                canvas=sample.canvas[:1].expand(frames, -1, -1, -1).clone(),
                actions=sample.actions,
                states=torch.cat((sample.states[:history_states], torch.zeros_like(sample.states[history_states:]))),
                task=sample.task,
            )
            for sample in samples
        ]
        prediction = world_model.predict(
            world_model.build_predict_batch(predict_samples),
            num_inference_steps=args.inference_steps,
            seeds=list(range(args.batch_size)),
        )
        assert tuple(prediction.frames.shape) == (args.batch_size, contract.prediction_frames, *contract.canvas_size, 3)
        assert tuple(prediction.states.shape) == (args.batch_size, contract.prediction_frames, contract.state_dim)
        assert torch.isfinite(prediction.states).all()
        restored = world_model.split_canvas(prediction.frames, (args.view_size, args.view_size))
        assert tuple(restored.shape) == (
            args.batch_size,
            len(contract.views),
            contract.prediction_frames,
            args.view_size,
            args.view_size,
            3,
        )
        summary["predict"] = {
            "frames_shape": list(prediction.frames.shape),
            "states_shape": list(prediction.states.shape),
            "seconds": round(time.monotonic() - started, 1),
        }
    else:
        summary["train"] = summary["predict"] = "skipped: the attention backends need CUDA"

    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
