# Change log

## Unreleased

- New features
  - Add [`cosmos_framework.world_models`](./docs/world_models.md): Cosmos3-Edge forward dynamics as a host-managed world model with joint video and proprioceptive-state prediction (`libero_state` domain 31, explicit per-row vision frame indexes via `SequencePlan.action_frame_indexes`).
  - Install into host environments on torch 2.7.1 / transformers 5 / diffusers 0.33: relaxed dependency bounds, optional object-store SDKs, `typing_extensions.override`, torch < 2.8 import of `inference/model.py`.

- Breaking changes
  - `requires-python` is now `>=3.11`; `nvidia-cudnn-frontend` and `uv` moved to the `cudnn-attention` and `launcher` extras.

- Other changes
  - The fork carries no GitHub Actions workflows: upstream's `docker-build`, `gpu-tests` and `pre-commit` target NVIDIA's self-hosted GPU runners and container registry, and the fork does not ship upstream's `uv.lock` that the Docker build locks against.

## 1.2.2 (May 14, 2026)

- New features
  - Add action policy closed-loop evaluation.

## 1.2.1 (May 08, 2026)

- New features
  - Add [action policy post-training (SFT)](./docs/training.md).

## 1.2.0 (May 05, 2026)

- New features
  - Add action modalities (Forward Dynamics, Inverse Dynamics, Policy) for Cosmos3-Nano model.
  - Upgrade Cosmos3-Nano checkpoint to improve T2V, I2V quality.

## 1.1.1 (May 01, 2026)

- New features
  - Add DCP checkpoint conversion/inference.

## 1.1.0 (April 29, 2026)

- New features
  - Add [Post-Training (Supervised Fine-Tuning)](./docs/training.md).
