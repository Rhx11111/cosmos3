# Cosmos3 as a world model for model-based RL

`cosmos_framework.world_models` exposes Cosmos3 forward dynamics to a host
training framework that owns distribution, optimization, checkpoints and weight
exchange. The host wraps `Cosmos3WorldModel.net` (the MoT network) in its own
FSDP/optimizer; this package owns the model's flow-matching loss, its joint
video/state prediction, and the frozen runtime assets around the network (VAE
tokenizer, text processor, schedulers, samplers).

## Host environment

The package installs into a host environment with its own PyTorch stack:

```bash
pip install -e ".[world-model]"
```

Validated: Python 3.11, torch 2.10.0 (CUDA 12.8), transformers 5.8, diffusers
0.33, `flash-attn-3-nv` 1.0.3 (`+cu128.torch210`, Hopper GPUs), with
`I4_ATTN_BACKENDS=flash3`. Object-store SDKs (`boto3`, `multistorageclient`) are
optional; without them the disk and HTTP `easy_io` backends remain.

The MoT network packs variable-length sequences, so the attention frontend needs
a varlen backend, and only FlashAttention 3 and NATTEN qualify: flash2 varlen is
banned (`model/attention/flash2/checks.py`, which also accepts only flash-attn
2.7.0 to 2.7.4.post1), and the cuDNN backend has no varlen. The FlashAttention 3
and NATTEN wheels are built for torch 2.10 or later, so the CUDA path needs such
a host. The package still imports on torch 2.7 (Python 3.11, transformers 5,
diffusers 0.33), where the CPU covers tokenization, VAE encoding, prompt
rendering, sequence packing and noising, not the network forward.

The fork does not carry upstream's `uv.lock`: the relaxed `pyproject.toml`
dependencies resolve inside the host environment, and hosts install with pip as
above.

## Contract

`Cosmos3WorldModelContract` (default: LIBERO) fixes the layout both training
and prediction follow:

| Field | Default | Meaning |
| --- | --- | --- |
| `views` / `view_size` | `("agentview", "robot0_eye_in_hand")`, 256×256 | Each camera is letterboxed into a tile; tiles sit side by side on a 256×512 canvas (`compose_canvas` / `split_canvas`). |
| `fps` | 20 | Frame rate of observations and actions. |
| `action_domain` / `action_dim` | `libero`, 7 | Native controller input; always clean conditioning. |
| `state_domain` / `state_dim` | `libero_state`, 8 | Proprioception carried as extra action rows; the observed rows are clean, the future rows are noised and supervised. |
| `condition_frames` | 5 | Observed frames; the last one conditions the video, the 4 actions between them and all 5 states are clean rows. |
| `prediction_frames` | 8 | Generated frames, actions and states; the video has 1 + 8 frames (4k+1). |

Actions and states travel in **one** action payload. Rows 0..11 are actions
(domain `libero`), rows 12..24 are states (domain `libero_state`), and every row
carries an explicit vision frame index through `SequencePlan.action_frame_indexes`
(action row *k* produces frame *k − 3*, state row *j* observes frame *j − 4*), so
both streams share the video clock inside the mRoPE. `raw_action_dim` is the wider
of the two widths (8); action rows pad their eighth channel with zero. The row
layout is validated by `cosmos_framework/world_models/batch_test.py`.

`libero_state` is domain 31, the last free slot of the 32 Edge embodiment
domains. Its `action2llm`/`llm2action` rows start from the base checkpoint's
values for that slot and are trained by the host.

## API

```python
from cosmos_framework.world_models import (
    Cosmos3Sample,
    Cosmos3WorldModelContract,
    build_cosmos3_world_model,
)

contract = Cosmos3WorldModelContract()
world_model = build_cosmos3_world_model(
    checkpoint_dir="/path/to/Cosmos3-Edge",   # diffusers layout
    vae_path="/path/to/Wan2.2_VAE.pth",
    contract=contract,
)

# Training: the host wraps world_model (or world_model.net) in FSDP and steps an optimizer.
canvas = world_model.compose_canvas(views)        # uint8 [B, V, 9, h, w, 3] -> [B, 9, 256, 512, 3]
samples = [Cosmos3Sample(canvas=canvas[i], actions=actions[i], states=states[i], task=tasks[i]) for i in range(B)]
result = world_model(world_model.build_train_batch(samples))
result.loss.backward()                           # result.metrics: video_loss, state_loss

# Prediction: placeholder future frames repeat the conditioning frame, future states are zeros.
prediction = world_model.predict(world_model.build_predict_batch(samples), num_inference_steps=5, seeds=list(range(B)))
frames = world_model.split_canvas(prediction.frames, source_size=(h, w))  # uint8 [B, V, 8, h, w, 3]
states = prediction.states                                                 # float32 [B, 8, 8]
```

`build_cosmos3_world_model` loads `inference/configs/model/Cosmos3-Edge.yaml`
bound to the local checkpoint and VAE, disables EMA, compile and activation
checkpointing, builds the network unsharded in bfloat16 on `COSMOS_DEVICE`,
copies the diffusers safetensors into it (`world_models/checkpoint.py`), and
leaves only the forward-dynamics recipe's parameters trainable
(`TRAINABLE_PARAMETER_MARKERS`: the generation expert, `time_embedder`,
`vae2llm`/`llm2vae`, `action2llm`/`llm2action`, `action_modality_embed`).

`Cosmos3WorldModel` registers only `net`, so `state_dict()` holds `net.*` keys
and nothing else. `reload_frozen_assets(device)` / `offload_frozen_assets()`
move the VAE; the host moves `net` itself. Samples with a shorter history pack
fewer rows (`history_actions ≤ 4`, `1 ≤ history_states ≤ 5`).

## Acceptance

```bash
# CPU: build, load weights, tokenize, encode, pack, noise.
COSMOS_DEVICE=cpu CUDA_VISIBLE_DEVICES= python tools/mbrl_check.py \
    --checkpoint ~/run/models/Cosmos3-Edge --vae /path/to/Wan2.2_VAE.pth
# CUDA: additionally one training forward/backward and one prediction.
I4_ATTN_BACKENDS=flash3 python tools/mbrl_check.py --checkpoint ... --vae ...
```

The check asserts the forward-dynamics parameter set, finite weights, the
latent geometry, the clean/noised row split, and on CUDA a finite loss with
gradients on exactly the trainable parameters plus the prediction shapes.
