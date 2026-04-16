# VLM Distillation and Fine-Tuning

This repository trains [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) in two modes:

- standard supervised fine-tuning
- teacher-student distillation with one or more VLM teachers

The current distillation stack supports:

- single-teacher and multi-teacher KD
- cached teacher-logit training, so teacher weights do not need to be loaded during training
- router-based teacher weighting
- reinforced teacher selection
- `GRACE` routing refinement on top of the router
- standard `CE + alpha * KD` optimization

## What This Repo Is For

The main research workflow in this repo is:

1. choose one or more teacher VLMs
2. optionally cache their logits on the training set
3. train a smaller student VLM with:
   - `uniform_mean`
   - `routing`
   - `reinforced_selection`
4. optionally evaluate checkpoints in the separate `src/eval/` stack

If you only care about the current training code, the important entrypoint is:

- [train_distillation.py](/home/automl/VLM-Distillation/src/train/train_distillation.py)

## Installation

### Environment

- Ubuntu 22.04
- CUDA 12.x
- PyTorch with CUDA

### `requirements.txt`

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

### `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

## Dataset Format

Training expects LLaVA-style JSON.

Each sample contains:

- `image` or `video`
- `conversations`

The dataset loader converts the conversation into the model-specific processor format at runtime.

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

## Main Training Commands

### Multi-teacher distillation

```bash
bash scripts/train/distill_multi_teachers.sh
```

### Single-teacher distillation

```bash
bash scripts/train/distill_single_teacher.sh
```

### Standard SFT

```bash
bash scripts/train/sft_full.sh
```

### LoRA SFT

```bash
bash scripts/train/sft_lora.sh
```

## Cached Teacher Logits Workflow

If you only use logits for distillation, you can precompute teacher outputs once and train without loading teacher model weights.

### 1. Cache teacher logits

```bash
python scripts/analysis/cache_teacher_logits.py ...
```

This produces a cache directory with:

- `metadata.json`
- one subdirectory per teacher
- one `.pt` file per dataset sample

### 2. Train from the cache

```bash
TEACHER_LOGITS_CACHE_DIR=/path/to/cache \
bash scripts/train/distill_multi_teachers.sh
```

When `teacher_logits_cache_dir` is set:

- the dataset loads cached logits from disk
- the collator pads cached logits/labels into the batch
- training uses those tensors directly for KD
- teacher model weights are not loaded during training

## Distillation Methods

### Teacher weighting

- `uniform_mean`
  - average all teacher KD losses
- `routing`
  - learned teacher gate over pooled student hidden states
  - optional soft routing, capacity control, entropy regularization, and `GRACE`
- `reinforced_selection`
  - policy-based teacher selection driven by CE and KD reward signals

### CE vs KD objective combination

- `fixed`

## Repository Guide

The sections below describe what each important file is responsible for.

### Root Files

- [modal_app.py](/home/automl/VLM-Distillation/modal_app.py)
  - Modal entrypoints for remote training and evaluation jobs.
- [PLAN.md](/home/automl/VLM-Distillation/PLAN.md)
  - research notes and method planning, not runtime code.
- [requirements.txt](/home/automl/VLM-Distillation/requirements.txt)
  - pip dependencies.
- [environment.yaml](/home/automl/VLM-Distillation/environment.yaml)
  - conda environment definition.

### `src/components`

- [forward_utils.py](/home/automl/VLM-Distillation/src/components/forward_utils.py)
  - safe forward helpers, including retry logic for models that reject unexpected kwargs.
- [grace.py](/home/automl/VLM-Distillation/src/components/grace.py)
  - `GRACE` routing refinement.
  - takes routed teacher weights plus teacher agreement scores and returns final teacher weights.
- [loss.py](/home/automl/VLM-Distillation/src/components/loss.py)
  - KD loss implementations and logit-gradient formulas.
  - includes `uld_loss` and the helper used for KD-gradient computations.
- [teacher_gate.py](/home/automl/VLM-Distillation/src/components/teacher_gate.py)
  - learned router for the `routing` strategy.
  - captures pooled student hidden states and outputs teacher scores.

### `src/dataset`

- [conversation_transforms.py](/home/automl/VLM-Distillation/src/dataset/conversation_transforms.py)
  - converts LLaVA-style conversations into the internal OpenAI-style message format.
- [conversation_encoders.py](/home/automl/VLM-Distillation/src/dataset/conversation_encoders.py)
  - high-level student/teacher encoding entrypoints.
- [processor_encoders.py](/home/automl/VLM-Distillation/src/dataset/processor_encoders.py)
  - model-specific tokenization and multimodal packing logic for SmolVLM, Qwen, InternVL, and similar processors.
- [data_collator.py](/home/automl/VLM-Distillation/src/dataset/data_collator.py)
  - pads student tensors and teacher tensors.
  - also pads cached teacher logits and labels.
- [sft_data.py](/home/automl/VLM-Distillation/src/dataset/sft_data.py)
  - main dataset class used by training.
  - loads images/videos, encodes student inputs, and either:
    - loads cached teacher logits, or
    - builds live teacher inputs.
- [teacher_logits_cache.py](/home/automl/VLM-Distillation/src/dataset/teacher_logits_cache.py)
  - reads the on-disk teacher-logit cache.
- [internvl_utils.py](/home/automl/VLM-Distillation/src/dataset/internvl_utils.py)
  - InternVL-specific preprocessing helpers.
- [vqa_loading.py](/home/automl/VLM-Distillation/src/dataset/vqa_loading.py)
  - utility loaders for VQA-style data.
- [data_utils.py](/home/automl/VLM-Distillation/src/dataset/data_utils.py)
  - lower-level dataset helpers.

### `src/train`

- [distillation_setup.py](/home/automl/VLM-Distillation/src/train/distillation_setup.py)
  - distillation-specific argument definitions, validation, and setup logging.
  - this is where most experiment knobs are exposed.
- [distillation_runtime.py](/home/automl/VLM-Distillation/src/train/distillation_runtime.py)
  - teacher model loading and trainer callback setup.
- [model_setup.py](/home/automl/VLM-Distillation/src/train/model_setup.py)
  - model construction and setup helpers used by training.
- [save_utils.py](/home/automl/VLM-Distillation/src/train/save_utils.py)
  - save/checkpoint utilities for training scripts.
- [train_utils.py](/home/automl/VLM-Distillation/src/train/train_utils.py)
  - shared training helpers and compatibility glue.
- [arg_utils.py](/home/automl/VLM-Distillation/src/train/arg_utils.py)
  - argument parsing helpers.
- [log_utils.py](/home/automl/VLM-Distillation/src/train/log_utils.py)
  - training/log formatting helpers.
- [train_distillation.py](/home/automl/VLM-Distillation/src/train/train_distillation.py)
  - main distillation entrypoint.
  - loads student model, optionally loads teachers, builds datasets, and starts `DistillationTrainer`.
- [train_sft.py](/home/automl/VLM-Distillation/src/train/train_sft.py)
  - main supervised fine-tuning entrypoint without teacher distillation.

### `src/trainer`

- [distillation_trainer.py](/home/automl/VLM-Distillation/src/trainer/distillation_trainer.py)
  - thin trainer shell on top of the SFT trainer.
  - owns training state, trainer configuration, and overall loss orchestration.
- [step_utils.py](/home/automl/VLM-Distillation/src/trainer/step_utils.py)
  - per-step training pipeline.
  - prepares teacher batches, runs the student forward pass, applies teacher weighting, computes KD, applies objective-conflict logic, and builds the total loss.
- [alignment_utils.py](/home/automl/VLM-Distillation/src/trainer/alignment_utils.py)
  - builds the per-teacher KD loss matrix and the per-teacher GRACE scores.
  - works with both cached teacher logits and live teacher forwards.
- [gradient_utils.py](/home/automl/VLM-Distillation/src/trainer/gradient_utils.py)
  - computes pooled CE and KD gradient vectors used by `GRACE` and objective-conflict logic.
- [kd_sequence_utils.py](/home/automl/VLM-Distillation/src/trainer/kd_sequence_utils.py)
  - aligns student and teacher token sequences for logits KD.
  - masks unsupervised positions and handles EOS trimming.
- [routing_utils.py](/home/automl/VLM-Distillation/src/trainer/routing_utils.py)
  - gate-specific math:
    - top-k/capacity constraints
    - load balancing
    - z-loss
    - entropy bonus
- [metrics_utils.py](/home/automl/VLM-Distillation/src/trainer/metrics_utils.py)
  - assembles the training metrics logged to W&B or the trainer logger.
- [setup_utils.py](/home/automl/VLM-Distillation/src/trainer/setup_utils.py)
  - validates trainer args and constructs the teacher gate when needed.
- [checkpoint_utils.py](/home/automl/VLM-Distillation/src/trainer/checkpoint_utils.py)
  - checkpoint bookkeeping, including best-checkpoint tracking by training CE.
- [distillation_utils.py](/home/automl/VLM-Distillation/src/trainer/distillation_utils.py)
  - generic teacher-batch helpers, cached-logit batch extraction, and eval-memory cleanup.
- [sft_trainer.py](/home/automl/VLM-Distillation/src/trainer/sft_trainer.py)
  - base trainer used by both SFT and distillation.

### `scripts/train`

- [distill_multi_teachers.sh](/home/automl/VLM-Distillation/scripts/train/distill_multi_teachers.sh)
  - main multi-teacher distillation launcher.
- [distill_single_teacher.sh](/home/automl/VLM-Distillation/scripts/train/distill_single_teacher.sh)
  - single-teacher variant.
- [sft_full.sh](/home/automl/VLM-Distillation/scripts/train/sft_full.sh)
  - full fine-tuning launcher.
- [sft_lora.sh](/home/automl/VLM-Distillation/scripts/train/sft_lora.sh)
  - LoRA fine-tuning launcher.
- [sft_lora_vision.sh](/home/automl/VLM-Distillation/scripts/train/sft_lora_vision.sh)
  - vision-aware LoRA launcher.
- [sft_video.sh](/home/automl/VLM-Distillation/scripts/train/sft_video.sh)
  - video-as-frames training launcher.

### `scripts/analysis`

- [cache_teacher_logits.py](/home/automl/VLM-Distillation/scripts/analysis/cache_teacher_logits.py)
  - precomputes and saves teacher logits for logits-only KD.
- [gradient_agreement.py](/home/automl/VLM-Distillation/scripts/analysis/gradient_agreement.py)
  - offline gradient-agreement analysis.
- [run_docvqa_gradient_agreement.sh](/home/automl/VLM-Distillation/scripts/analysis/run_docvqa_gradient_agreement.sh)
  - launcher for the gradient-agreement analysis.

### `src/eval`

The evaluation stack is separate from the training stack and is not described in detail here.
Use it for checkpoint evaluation after training, not for the core distillation loop.

## Training Flow

The distillation path is:

1. [train_distillation.py](/home/automl/VLM-Distillation/src/train/train_distillation.py)
2. [sft_data.py](/home/automl/VLM-Distillation/src/dataset/sft_data.py) and [data_collator.py](/home/automl/VLM-Distillation/src/dataset/data_collator.py)
3. [distillation_trainer.py](/home/automl/VLM-Distillation/src/trainer/distillation_trainer.py)
4. [step_utils.py](/home/automl/VLM-Distillation/src/trainer/step_utils.py)
5. [alignment_utils.py](/home/automl/VLM-Distillation/src/trainer/alignment_utils.py)
6. one of:
   - [teacher_gate.py](/home/automl/VLM-Distillation/src/components/teacher_gate.py) + [grace.py](/home/automl/VLM-Distillation/src/components/grace.py)

## Notes

- Cached-logit training is the cleanest way to reduce GPU memory when all distillation methods are logits-only.
- `GRACE` belongs to the `routing` strategy. It refines router weights; it is not a separate teacher-weighting strategy.
- Training-time eval is optional and the current distillation workflow is designed to run without online teacher loading when cached logits are available.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](/home/automl/VLM-Distillation/LICENSE).

## Acknowledgement

This project builds on:

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Mipha](https://github.com/zhuyiche/llava-phi)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
