# VLM Distillation and Fine-Tuning

This repository trains [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) students in two main modes:

- standard supervised fine-tuning
- teacher-student distillation from one or more VLM teachers

The implemented training stack is centered on cache-backed distillation: teacher logits are precomputed once, then reused during student training. On top of that base workflow, the repo currently supports:

- single-teacher and multi-teacher KD
- cached teacher-logit training from local disk or streamed remote storage
- teacher weighting via `uniform_mean`, `routing`, or `reinforced_selection`
- `GRACE` routing refinement on top of the router
- logits-space KD losses including `uld_loss`, `trie_wasserstein_loss`, KL/JS variants, and `cka_loss`
- optional hidden-state / layer distillation in addition to cached-logit KD
- full SFT and LoRA SFT launchers
- a separate evaluation stack under `src/eval/`

## Documentation

- [Runtime and Training Guide](docs/runtime-guide.md)
  - comprehensive guide to the implemented training stack
  - covers entrypoints, cache layout, data format, routing, GRACE, and launcher behavior
- [PLAN.md](PLAN.md)
  - research notes and method planning
  - useful for context, but not the runtime source of truth
- [DEX-AR Demo](demo/DEX-AR/README.md)
  - separate explainability demo bundled in this repository

## Source of Truth

If you are trying to understand what currently runs, prioritize these directories:

- `src/train`
- `src/trainer`
- `src/dataset`
- `src/components`

The checked-in docs now describe the implemented runtime, not just the intended research direction. `PLAN.md` is still useful, but it should be treated as planning material rather than a strict runtime specification.

## Quick Start

### 1. Install dependencies

The repo currently targets:

- Ubuntu 22.04
- CUDA 12.x
- Python 3.11 or 3.12

`requirements.txt` path:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

`environment.yaml` path:

```bash
conda env create -f environment.yaml
conda activate train
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

### 2. Prepare LLaVA-style training data

Training expects JSON samples containing either `image` or `video`, plus `conversations`.

Example:

```json
[
  {
    "id": "sample1",
    "image": "sample1.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is shown here?"
      },
      {
        "from": "gpt",
        "value": "A receipt on a table."
      }
    ]
  }
]
```

Notes:

- image paths can be absolute or resolved relative to `--image_folder`
- video samples are decoded into frame lists at load time
- the loader converts LLaVA-style conversations into the processor-specific chat format during training

### 3. Cache teacher logits

The current distillation runtime expects cached teacher logits, either from a local cache root or a remote raw cache URI.

Example:

```bash
export PYTHONPATH=src:$PYTHONPATH

python scripts/analysis/cache_teacher_logits.py \
  --student-model-id HuggingFaceTB/SmolVLM-500M-Instruct \
  --teacher-model-ids '["Qwen/Qwen2.5-VL-3B-Instruct","Qwen/Qwen2-VL-2B-Instruct"]' \
  --data-path data/docvqa/train_llava.json \
  --image-folder data/docvqa/images \
  --output-dir /path/to/teacher-logits-cache
```

### 4. Run multi-teacher distillation

The most up-to-date checked-in launcher is:

```bash
TEACHER_LOGITS_CACHE_DIR=/path/to/teacher-logits-cache \
bash scripts/train/distill_multi_teachers.sh
```

The default script currently trains:

- student: `HuggingFaceTB/SmolVLM-500M-Instruct`
- teachers:
  - `Qwen/Qwen2.5-VL-3B-Instruct`
  - `Qwen/Qwen2-VL-2B-Instruct`
  - `ibm-granite/granite-vision-3.1-2b-preview`
  - `google/gemma-3-4b-it`
- dataset: `docvqa`
- weighting strategy: `routing`
- KD loss: `trie_wasserstein_loss`

Useful overrides:

```bash
TEACHER_WEIGHTING_STRATEGY=uniform_mean \
TEACHER_LOGITS_CACHE_DIR=/path/to/cache \
bash scripts/train/distill_multi_teachers.sh
```

```bash
TRIE_WASSERSTEIN_RHO=0.5 \
TRIE_WASSERSTEIN_TOPK=128 \
TEACHER_LOGITS_CACHE_DIR=/path/to/cache \
bash scripts/train/distill_multi_teachers.sh
```

```bash
NUM_TRAIN_EPOCHS=2 \
PER_DEVICE_TRAIN_BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=4 \
TEACHER_LOGITS_CACHE_DIR=/path/to/cache \
bash scripts/train/distill_multi_teachers.sh
```

### 5. Run standard SFT

The SFT launchers do not use teacher logits.

```bash
bash scripts/train/sft_full.sh
```

See the runtime guide for LoRA, vision LoRA, and video variants.

## Training Modes

| Mode | Entrypoint | Notes |
| --- | --- | --- |
| Multi-teacher distillation | [`scripts/train/distill_multi_teachers.sh`](scripts/train/distill_multi_teachers.sh) | Most up-to-date checked-in distillation launcher |
| Single-teacher distillation | [`src/train/train_distillation.py`](src/train/train_distillation.py) | Current distillation runtime is still cache-backed even for one teacher |
| Full SFT | [`scripts/train/sft_full.sh`](scripts/train/sft_full.sh) | Standard fine-tuning |
| LoRA SFT | [`scripts/train/sft_lora.sh`](scripts/train/sft_lora.sh) | LLM LoRA with explicit freeze settings |
| Vision LoRA SFT | [`scripts/train/sft_lora_vision.sh`](scripts/train/sft_lora_vision.sh) | Vision-tower LoRA path |
| Video SFT | [`scripts/train/sft_video.sh`](scripts/train/sft_video.sh) | Frame-based training with `max_num_frames` |

## Distillation At A Glance

The implemented distillation path is:

1. [`src/train/train_distillation.py`](src/train/train_distillation.py)
2. [`src/dataset/sft_data.py`](src/dataset/sft_data.py)
3. [`src/dataset/data_collator.py`](src/dataset/data_collator.py)
4. [`src/trainer/distillation_trainer.py`](src/trainer/distillation_trainer.py)
5. [`src/trainer/step_utils.py`](src/trainer/step_utils.py)
6. [`src/trainer/alignment_utils.py`](src/trainer/alignment_utils.py)
7. teacher weighting and KD losses from:
   - [`src/components/teacher_gate.py`](src/components/teacher_gate.py)
   - [`src/components/grace.py`](src/components/grace.py)
   - [`src/components/reinforced_teacher_selection.py`](src/components/reinforced_teacher_selection.py)
   - [`src/components/loss.py`](src/components/loss.py)

One practical consequence of the current design:

- distillation validates that a cache source is present
- if layer distillation is disabled, teacher model weights are not loaded during training
- if layer distillation is enabled, live teacher models are loaded in addition to cached teacher logits

## Repository Layout

- [`src/train`](src/train)
  - argument parsing, model loading, training entrypoints, trainer callback setup
- [`src/trainer`](src/trainer)
  - custom SFT trainer, distillation trainer, per-step loss orchestration, routing helpers
- [`src/dataset`](src/dataset)
  - dataset loading, conversation transforms, processor-specific encoding, cache readers, collator
- [`src/components`](src/components)
  - KD losses, router, GRACE, reinforced selection, trie-Wasserstein, CKA helpers
- [`src/eval`](src/eval)
  - separate evaluation stack
- [`scripts/train`](scripts/train)
  - shell launchers for distillation and SFT
- [`scripts/analysis`](scripts/analysis)
  - teacher-logit caching and offline analysis helpers
- [`demo/DEX-AR`](demo/DEX-AR)
  - separate explainability demo

The detailed file-by-file guide lives in [docs/runtime-guide.md](docs/runtime-guide.md).

## Notes and Caveats

- The distillation runtime is cache-first. Provide either `--teacher_logits_cache_dir` or `--teacher_logits_remote_uri`.
- `GRACE` is part of the `routing` path. It refines routed teacher weights; it is not a separate weighting strategy.
- `src/eval` is a separate stack and is not part of the core training loop.
- `modal_app.py` is remote execution glue. Its checked-in local entrypoint currently drops into the DEX-AR demo, not the main training path.
- Test coverage is currently narrow and mainly exercises the custom DeepSpeed fallback in [`tests/trainer/test_sft_trainer_deepspeed.py`](tests/trainer/test_sft_trainer_deepspeed.py).

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE).

## Acknowledgement

This project builds on:

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Mipha](https://github.com/zhuyiche/llava-phi)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
