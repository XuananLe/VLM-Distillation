# Fine-tuning SmolVLM

This repository contains a script for training [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) with only using HuggingFace.

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Molmo Finetune]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fixed error in LoRA.
- [2025/01/24] 🔥Supports mixed-modality data.

## Table of Contents

- [Fine-tuning SmolVLM](#fine-tuning-smolvlm)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
      - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yaml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install flash-attn --no-build-isolation
pip install pillow-avif-plugin
pip install num2words
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** SmolVLM uses a video as a sequential of images.

</details>

## Training

**Note:** With the mixed-dataset (e.g. some data in a batch have images while some don't) It only supports with zero2.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/train/sft_full.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/train/sft_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/train/sft_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/deepspeed/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the SmolVLM model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_connector` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--connector_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--min_pixels` (int): Option for minimum input tokens.
- `--max_pixles` (int): OPtion for maximum maxmimum tokens.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--use_dora` (bool): Option for using DoRA instead of LoRA. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 16).
- `--lora_alpha` (int): LoRA alpha (default: 16).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. However, SmolVLm processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/train/sft_video.sh
```

**Note:** When training with video, it just as multi-image so you should adjust the `max_pixels` for maximum resolution and `fps` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/deepspeed/zero3_offload.json) instead of [zero3](./scripts/deepspeed/zero3.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge/merge_lora_weights.sh
```

**Note:** Remember to replace the paths in `sft_full.sh` or `sft_lora.sh` with your specific paths. (Also in `merge_lora_weights.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [ ] Add feature for controlling image size.
- [ ] Add support smolvlm2.
- [ ] Add DPO Training.
- [x] Handle interleaved dataset.
- [x] Hadnle mixed-modality dataset.

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{SmolVLM-Finetuning,
  author = {Yuwon Lee},
  title = {SmolmVLM-Finetune},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/2U1/SmolVLM-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct): Awesome pretrained MLLM based on SmolLM2.


# Flexible Baseline Configuration Guide

This guide shows how to use the flexible baseline configuration system for VLM distillation experiments.

## Key Features

- **Optional parameters**: Only specify what you need (temperature, alpha, etc.)
- **Custom hyperparameters**: Add any experiment-specific parameters
- **Easy extension**: Create and register new baselines at runtime
- **Type safety**: All parameters are validated

## Basic Usage

### 1. Using Pre-defined Baselines

```python
from src.distillation import BaselineRegistry

# List all available baselines
BaselineRegistry.print_baselines()

# Get a specific baseline config
config = BaselineRegistry.get_baseline("vanilla_kd")
print(config.to_dict())
# Output: {'name': 'vanilla_kd', 'description': '...', 'loss_type': 'kl',
#          'temperature': 4.0, 'alpha': 0.5, 'learning_rate': 2e-5, ...}

# Access parameters flexibly
loss_type = config.loss_type  # Standard parameter
lr = config.get("learning_rate", 1e-5)  # From hyperparameters with default
```

### 2. Creating Custom Baselines

```python
from src.distillation import create_custom_baseline

# Example 1: KD with only temperature (no alpha)
simple_kd = create_custom_baseline(
    name="simple_kd",
    description="Simple KD without hard targets",
    loss_type="kl",
    temperature=3.0,
    learning_rate=2e-5
)

# Example 2: Vision-only distillation (no temperature/alpha)
vision_only = create_custom_baseline(
    name="vision_only",
    description="Distill vision features only",
    loss_type="mse",
    freeze_llm=True,
    vision_lr=1e-6,
    projection_layers=[12, 18, 24]
)

# Example 3: Multi-stage distillation
multi_stage = create_custom_baseline(
    name="multi_stage",
    description="Stage 1: Vision, Stage 2: Full model",
    stage_1_epochs=5,
    stage_2_epochs=10,
    stage_1_freeze_llm=True,
    stage_2_freeze_llm=False
)

# Example 4: Custom loss with many parameters
advanced = create_custom_baseline(
    name="advanced_kd",
    description="Advanced KD with all bells and whistles",
    loss_type="feature",
    temperature=4.0,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
    learning_rate=2e-5,
    use_ema_teacher=True,
    ema_decay=0.999,
    layer_wise_weights=[0.1, 0.2, 0.3, 0.4],
    adaptive_temperature=True
)
```

### 3. Training with Different Baseline Types

```bash
# Standard KD approach
python -m src.train.train_distillation \
    --baseline_method vanilla_kd \
    --teacher_model_id "teacher-model" \
    --model_id "student-model" \
    --data_path data.json

# Vision-only distillation (no teacher for LLM)
python -m src.train.train_distillation \
    --baseline_method vision_distill \
    --teacher_model_id "teacher-model" \
    --model_id "student-model" \
    --data_path data.json

# Feature-based (no temperature parameter)
python -m src.train.train_distillation \
    --baseline_method fitnets \
    --teacher_model_id "teacher-model" \
    --model_id "student-model" \
    --data_path data.json

# Self-distillation (no teacher at all!)
python -m src.train.train_distillation \
    --baseline_method self_distill \
    --model_id "student-model" \
    --data_path data.json
```

## Available Baseline Types

### 1. **vanilla_kd** - Standard Knowledge Distillation
- **Parameters**: `temperature`, `alpha`
- **Use case**: General-purpose logit distillation

### 2. **fitnets** - Feature Hints
- **Parameters**: `alpha` (no temperature)
- **Use case**: Intermediate layer matching

### 3. **attention_transfer** - Attention Matching
- **Parameters**: `alpha` (no temperature)
- **Use case**: Transfer attention patterns

### 4. **feature_distillation** - Multi-component
- **Parameters**: `temperature`, `alpha`, `beta`, `gamma`
- **Use case**: Logits + features + attention

### 5. **vision_distill** - Vision Tower Only
- **Parameters**: None (only hyperparameters)
- **Use case**: Distill vision encoder only

### 6. **mimicking** - Hidden State Matching
- **Parameters**: None (feature-based)
- **Use case**: Match all intermediate hidden states

### 7. **contrastive_kd** - Contrastive Learning
- **Parameters**: None (custom parameters only)
- **Use case**: Use contrastive loss for distillation

### 8. **self_distill** - Self-Distillation
- **Parameters**: `temperature`, `alpha`
- **Use case**: No teacher, use model's own predictions

## Runtime Registration

You can register new baselines during execution:

```python
from src.distillation import create_custom_baseline, register_custom_baseline

# Create your custom baseline
my_baseline = create_custom_baseline(
    name="my_experiment",
    description="My novel distillation approach",
    loss_type="custom",
    learning_rate=1e-5,
    custom_weight=0.7,
    use_importance_sampling=True
)

# Register it
register_custom_baseline(my_baseline)

# Now you can use it
python -m src.train.train_distillation --baseline_method my_experiment ...
```

## Accessing Parameters in Code

The flexible config works automatically in `train_distillation.py`:

```python
# In train_distillation.py
if distillation_args.baseline_method:
    baseline_config = BaselineRegistry.get_baseline(distillation_args.baseline_method)

    # Only override if parameter exists
    if baseline_config.loss_type is not None:
        distillation_args.distillation_loss_type = baseline_config.loss_type
    if baseline_config.temperature is not None:
        distillation_args.temperature = baseline_config.temperature
    # ... etc

    # All hyperparameters automatically applied
    for key, value in baseline_config.hyperparameters.items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
```

## Adding Your Own Baseline Types

1. **Edit** `src/distillation/baselines.py`
2. **Add to** `BaselineRegistry.list_baselines()`:

```python
"my_new_method": BaselineConfig(
    name="my_new_method",
    description="Description of my method",
    # Only add parameters you need
    temperature=5.0,  # Optional
    alpha=0.6,        # Optional
    # Custom hyperparameters
    hyperparameters={
        "learning_rate": 3e-5,
        "my_custom_param": 42,
        "use_special_trick": True,
    }
),
```

3. **Use it**:
```bash
python -m src.train.train_distillation --baseline_method my_new_method ...
```

## Examples

### Example 1: Two-Stage Training
```python
# Stage 1: Train vision tower only
stage1 = create_custom_baseline(
    name="stage1_vision",
    description="Stage 1: Vision tower only",
    loss_type="mse",
    freeze_llm=True,
    learning_rate=5e-6
)

# Stage 2: Train full model
stage2 = create_custom_baseline(
    name="stage2_full",
    description="Stage 2: Full model",
    loss_type="kl",
    temperature=3.0,
    alpha=0.7,
    learning_rate=2e-5
)
```

### Example 2: Progressive Distillation
```python
progressive = create_custom_baseline(
    name="progressive_kd",
    description="Progressive temperature annealing",
    loss_type="kl",
    initial_temperature=10.0,
    final_temperature=2.0,
    alpha=0.5,
    temperature_schedule="cosine"
)
```

### Example 3: Minimal Configuration
```python
# Just hyperparameters, no standard params
minimal = create_custom_baseline(
    name="minimal",
    description="Minimal config with only LR",
    learning_rate=1e-5
)
```
