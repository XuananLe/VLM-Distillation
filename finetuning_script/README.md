# VLM Fine-tuning Framework using Unsloth + Modal

A modular, research-oriented framework for fine-tuning Vision Language Models on OCR/VQA datasets.

---

## 🔑 Required Secrets (API Keys)

You need to set up these Modal secrets **before running**:

### 1. WandB Secret (Required)
```bash
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
```
Get your key from: https://wandb.ai/settings → API Keys

### 2. HuggingFace Secret (Required for gated models)
```bash
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
```
Get your token from: https://huggingface.co/settings/tokens

---

## 🚀 Quick Start

```bash
# 1. Set up secrets (one time)
modal secret create wandb-secret WANDB_API_KEY=<your-key>
modal secret create huggingface-secret HF_TOKEN=<your-token>

# 2. Run training
cd /home/automl/VLM-Distillation
modal run finetuning_script/train_vlm.py
```

---

## 📊 What Gets Logged to WandB

The framework logs comprehensive metrics for research:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `trainable_params` | Number of trainable parameters |
| `trainable_ratio` | % of model being trained |
| `dataset_size` | Number of training samples |
| `final_train_loss` | Final training loss |
| `train_runtime_seconds` | Total training time |
| `gpu_name` | GPU used for training |
| `gpu_memory_gb` | GPU memory available |

WandB also tracks:
- Full experiment config (model, dataset, hyperparameters)
- Training curves
- System metrics (GPU utilization, memory)

---

## ⚙️ Configuration

Edit the `CONFIG` object in [train_vlm.py](train_vlm.py):

```python
CONFIG = ExperimentConfig(
    # Experiment
    experiment_name="smolvlm500m_ocrbench_v1",
    wandb_project="VLM-Distillation",
    
    # Model - change this to try different VLMs
    model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
    load_in_4bit=True,
    
    # Dataset - change this to try different datasets
    dataset_name="echo840/OCRBench",
    
    # LoRA config
    lora_r=16,
    lora_alpha=16,
    
    # Training
    learning_rate=2e-4,
    max_steps=100,
    
    # Hardware
    gpu="A100",
)
```

---

## 🤖 Supported Models

| Model | ID | GPU Recommendation |
|-------|----|--------------------|
| SmolVLM-500M | `HuggingFaceTB/SmolVLM-500M-Instruct` | L4 (24GB) |
| SmolVLM-256M | `HuggingFaceTB/SmolVLM-256M-Instruct` | L4 (24GB) |
| Llama-3.2-11B-Vision | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` | A100 (40GB) |
| Qwen2-VL-7B | `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit` | A10G (24GB) |
| Qwen2.5-VL-7B | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` | A10G (24GB) |
| Pixtral-12B | `unsloth/Pixtral-12B-2409-bnb-4bit` | A100 (40GB) |

---

## 📚 Supported Datasets

| Dataset | HuggingFace ID | Samples | Task |
|---------|----------------|---------|------|
| OCRBench | `echo840/OCRBench` | 1,000 | OCR evaluation benchmark |
| TextVQA | `textvqa` | 45,336 | Scene text VQA |
| DocVQA | `lmms-lab/DocVQA` | 50,000 | Document VQA |
| ChartQA | `ahmed-masry/ChartQA` | 28,299 | Chart understanding |

---

## 📁 Output Structure

All outputs are saved to Modal's persistent volume at `/outputs/`:

```
/outputs/
└── {experiment_name}/
    └── {timestamp}/
        ├── config.json              # Full experiment config
        ├── training_stats.json      # Final metrics
        ├── lora_adapters/           # Saved LoRA weights
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   └── ...
        └── checkpoints/             # Intermediate checkpoints
```

---

## 🔬 Running Experiments for Research

### Experiment 1: Compare models on OCRBench
```python
# Run 1: SmolVLM-500M
CONFIG = ExperimentConfig(
    experiment_name="smolvlm500m_ocrbench",
    model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
    dataset_name="echo840/OCRBench",
)

# Run 2: Qwen2-VL-7B  
CONFIG = ExperimentConfig(
    experiment_name="qwen2vl7b_ocrbench",
    model_name="unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    dataset_name="echo840/OCRBench",
)
```

### Experiment 2: Compare LoRA ranks
```python
# Low rank
CONFIG = ExperimentConfig(experiment_name="smolvlm_r8", lora_r=8, lora_alpha=8)

# Medium rank
CONFIG = ExperimentConfig(experiment_name="smolvlm_r16", lora_r=16, lora_alpha=16)

# High rank
CONFIG = ExperimentConfig(experiment_name="smolvlm_r32", lora_r=32, lora_alpha=32)
```

### Experiment 3: Compare datasets
```python
datasets = ["echo840/OCRBench", "textvqa", "lmms-lab/DocVQA", "ahmed-masry/ChartQA"]
for ds in datasets:
    CONFIG = ExperimentConfig(
        experiment_name=f"smolvlm_{ds.split('/')[-1]}",
        dataset_name=ds,
    )
```

---

## 📈 Evaluation with VLMEvalKit

After training, evaluate on OCRBench:

```bash
cd VLMEvalKit
python run.py --model /outputs/{experiment}/lora_adapters --data OCRBench
```

---

## 🐛 Troubleshooting

### "Secret not found"
```bash
modal secret create wandb-secret WANDB_API_KEY=xxx
modal secret create huggingface-secret HF_TOKEN=xxx
```

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Use `gpu="A100"` instead of smaller GPUs

### Model not supported by Unsloth
Check Unsloth's supported models: https://unsloth.ai/docs/get-started/unsloth-model-catalog

---

## 📚 References

- [Unsloth Vision Fine-tuning Docs](https://unsloth.ai/docs/basics/vision-fine-tuning)
- [Unsloth Llama 3.2 Vision Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
- [OCRBench Paper](https://arxiv.org/abs/2305.07895)
- [SmolVLM Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
