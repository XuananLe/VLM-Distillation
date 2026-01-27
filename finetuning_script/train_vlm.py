"""
VLM Fine-tuning Framework using Unsloth + Modal

A modular, configurable framework for fine-tuning Vision Language Models (VLMs)
on various OCR/VQA datasets. Designed for research experimentation with proper
logging and reproducibility.

Supported Models:
- SmolVLM-500M-Instruct, SmolVLM-256M-Instruct (use load_in_4bit=False)
- Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct
- Llama-3.2-11B-Vision-Instruct
- Pixtral-12B-2409

NOTE: SmolVLM requires load_in_4bit=False to avoid Unsloth gradient bugs.

Supported Datasets:
- OCRBench (echo840/OCRBench)
- TextVQA (textvqa)
- DocVQA (lmms-lab/DocVQA)
- ChartQA (ahmed-masry/ChartQA)

Usage:
    modal run finetuning_script/train_vlm.py

Author: VLM-Distillation Research
"""

import modal
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import wandb


# ============================================================================
# PATHS - DON'T CHANGE THESE
# ============================================================================
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = Path("/outputs")
ROOT_DIR = Path("/root/VLM-Distillation")
EVAL_DIR = ROOT_DIR / "VLMEvalKit"
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)



@dataclass
class ExperimentConfig:
    # Experiment identification
    experiment_name: str = "smolvlm500m_ocrbench"
    run_number: int = 5  # Increment this for each new run (1, 2, 3, 4...)
    wandb_project: str = "VLM-Distillation"
    
    # Model configuration
    model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    
    # Dataset configuration
    dataset_name: str = "echo840/OCRBench"
    dataset_split: str = "test"  # OCRBench only has test split
    max_samples: Optional[int] = None  
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    
    # Training configuration (conservative batch for VLMs - images use lots of memory)
    per_device_train_batch_size: int = 4  # VLMs need smaller batches due to image memory
    gradient_accumulation_steps: int = 4  # Effective batch = 4 x 4 = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_steps: int = -1  # Use num_train_epochs instead
    num_train_epochs: int = 10  # Full training
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50  # Evaluate every 50 steps
    
    # Hardware
    train_gpu: str = "A100"  # Big GPU for training
    eval_gpu: str = "L4"     # Smaller GPU for evaluation
    
    # Random seed for reproducibility
    seed: int = 42
    
    @property
    def run_name(self) -> str:
        """Generate run name like: smolvlm500m_ocrbench_1"""
        return f"{self.experiment_name}_{self.run_number}"


# ============================================================================
# SUPPORTED MODELS REGISTRY
# ============================================================================
SUPPORTED_MODELS = {
    # SmolVLM models
    "HuggingFaceTB/SmolVLM-500M-Instruct": {
        "type": "smolvlm",
        "size": "500M",
        "recommended_gpu": "A100",
    },
    "HuggingFaceTB/SmolVLM-256M-Instruct": {
        "type": "smolvlm", 
        "size": "256M",
        "recommended_gpu": "L4",
    },
    # Llama Vision models
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit": {
        "type": "llama_vision",
        "size": "11B",
        "recommended_gpu": "A100",
    },
    "unsloth/Llama-3.2-11B-Vision-Instruct": {
        "type": "llama_vision",
        "size": "11B", 
        "recommended_gpu": "A100",
    },
    # Qwen VL models
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit": {
        "type": "qwen_vl",
        "size": "7B",
        "recommended_gpu": "A10G",
    },
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit": {
        "type": "qwen_vl",
        "size": "7B",
        "recommended_gpu": "A10G",
    },
}


# ============================================================================
# DATASET LOADERS REGISTRY
# ============================================================================
DATASET_CONFIGS = {
    "echo840/OCRBench": {
        "name": "OCRBench",
        "split": "test",
        "image_field": "image",
        "question_field": "question",
        "answer_field": "answer",
        "category_field": "category",
    },
    "textvqa": {
        "name": "TextVQA",
        "split": "train",
        "image_field": "image",
        "question_field": "question",
        "answer_field": "answers",  # List of answers
    },
    "lmms-lab/DocVQA": {
        "name": "DocVQA",
        "split": "train",
        "image_field": "image",
        "question_field": "question",
        "answer_field": "answers",
    },
    "ahmed-masry/ChartQA": {
        "name": "ChartQA",
        "split": "train",
        "image_field": "image",
        "question_field": "query",
        "answer_field": "label",
    },
}


# ============================================================================
# CREATE YOUR EXPERIMENT CONFIG HERE
# ============================================================================
CONFIG = ExperimentConfig(
    # Experiment identification
    experiment_name="smolvlm500m_ocrbench",
    run_number=5,  # <-- INCREMENT THIS FOR EACH NEW RUN (1, 2, 3, 4...)
    wandb_project="VLM-Distillation",
    
    # Model - SmolVLM-500M (MUST use load_in_4bit=False to avoid gradient bug)
    model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
    load_in_4bit=False,  
    max_seq_length=2048,
    
    # Dataset
    dataset_name="echo840/OCRBench",
    dataset_split="test",
    max_samples=None, 
    
    # LoRA
    lora_r=16,
    lora_alpha=16,
    finetune_vision_layers=False,  # Language-only LoRA for stability
    finetune_language_layers=True,
    
    # Training - VLMs need smaller batches due to image memory overhead
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # Effective batch = 16
    learning_rate=2e-4,
    num_train_epochs=5,  # Reduced for faster iteration (early stopping will kick in anyway)
    logging_steps=1,
    save_steps=10,  # Save very frequently for quick eval
    eval_steps=10,  # Eval frequently to enable early stopping
    
    # Hardware
    train_gpu="A100",
    eval_gpu="L4",
    seed=42,
)

# WandB project
WANDB_PROJECT = CONFIG.wandb_project


# ============================================================================
# MODAL IMAGE SETUP
# ============================================================================
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
    )
    # Install PyTorch first (required for unsloth)
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "torchaudio",
        "triton",
    )
    # Install Unsloth and dependencies
    .pip_install(
        "unsloth",
        "xformers",
        "trl>=0.8.0",
        "peft",
        "accelerate",
        "bitsandbytes",
    )
    # Install data processing libraries
    .pip_install(
        "datasets",
        "transformers>=4.45.0",
        "pillow",
        "wandb",
        "huggingface_hub",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App(
    name=f"VLM-Finetune-{CONFIG.run_name}",
    image=base_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),  
    ],
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
@app.function(
    gpu=CONFIG.train_gpu,
    timeout=60 * 60 * 12,  # 12 hours
)
def train_vlm(config_dict: Dict[str, Any], wandb_run_id: str):
    """
    Train a Vision Language Model using Unsloth.
    
    Uses Unsloth's FastVisionModel for 2x faster training.
    """
    import torch
    import wandb
    import json
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback
    
    # Reconstruct config
    config = ExperimentConfig(**config_dict)
    
    # Setup environment
    env = os.environ.copy()
    env["WANDB_PROJECT"] = config.wandb_project
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_DATASETS_CACHE"] = str(DATASET_DIR / ".hf_cache" / "datasets")
    env["HF_HUB_CACHE"] = str(MODEL_DIR / ".hf_cache" / "hub")
    os.environ.update(env)
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    
    # Create output directory for this experiment (using run_name)
    experiment_output_dir = OUTPUT_DIR / config.run_name
    os.makedirs(experiment_output_dir, exist_ok=True)
    
    # Save config
    config_path = experiment_output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[INFO] Config saved to {config_path}")
    
    # ========================================================================
    # Resume WandB run (connect to the run started locally)
    # ========================================================================
    print("=" * 60)
    print(f"Connecting to WandB run: {config.run_name}")
    print("=" * 60)
    
    wandb.init(
        project=config.wandb_project,
        id=wandb_run_id,
        resume="must",
    )
    
    # Log that training started
    wandb.log({"training_started": 1, "gpu": config.train_gpu})
    
    # ========================================================================
    # STEP 1: Load Model with Unsloth
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"STEP 1: Loading model: {config.model_name}")
    print(f"        load_in_4bit={config.load_in_4bit}")
    print("=" * 60)
    
    # For SmolVLM, use bfloat16 when not using 4-bit to avoid gradient issues
    model_dtype = None if config.load_in_4bit else torch.bfloat16
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=model_dtype,
    )
    
    print(f"[INFO] Model loaded successfully!")
    wandb.log({"model_loaded": 1, "load_in_4bit": config.load_in_4bit})
    
    # ========================================================================
    # STEP 2: Configure LoRA adapters
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Configuring LoRA adapters...")
    print("=" * 60)
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    wandb.log({
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
    })
    
    # ========================================================================
    # STEP 3: Load Dataset
    # ========================================================================
    print("\n" + "=" * 60)
    print(f"STEP 3: Loading dataset: {config.dataset_name}")
    print("=" * 60)
    
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
    )
    
    if config.max_samples is not None:
        dataset = dataset.shuffle(seed=config.seed).select(range(min(config.max_samples, len(dataset))))
    
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    print(f"[INFO] Features: {dataset.features}")
    wandb.log({"dataset_size": len(dataset)})
    
    # ========================================================================
    # STEP 4: Format Dataset for Unsloth Vision Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Converting dataset to conversation format...")
    print("=" * 60)
    
    dataset_config = DATASET_CONFIGS.get(config.dataset_name, {})
    
    def convert_to_conversation(sample):
        """
        Convert dataset sample to Unsloth conversation format.
        
        Format required by Unsloth:
        {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": <question>}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": <answer>}
                ]}
            ]
        }
        """
        try:
            # Get image
            image_field = dataset_config.get("image_field", "image")
            image = sample[image_field]
            if hasattr(image, 'convert') and image.mode != "RGB":
                image = image.convert("RGB")
            
            # Get question
            question_field = dataset_config.get("question_field", "question")
            question = sample[question_field]
            
            # Get answer (handle different formats)
            answer_field = dataset_config.get("answer_field", "answer")
            answer = sample[answer_field]
            
            # Handle list of answers (TextVQA, DocVQA)
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            # Handle string that's actually a list
            elif isinstance(answer, str):
                try:
                    parsed = eval(answer)
                    if isinstance(parsed, list):
                        answer = parsed[0] if parsed else ""
                except:
                    pass
            
            answer = str(answer)
            
            # Format as Unsloth conversation
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
            }
            
            return conversation
            
        except Exception as e:
            print(f"[WARNING] Error converting sample: {e}")
            return None
    
    # Convert dataset
    converted_data = [convert_to_conversation(sample) for sample in dataset]
    converted_data = [x for x in converted_data if x is not None]
    
    print(f"[INFO] Converted {len(converted_data)} samples successfully")
    
    # Split into train/val (90/10 split)
    import random
    random.seed(config.seed)
    random.shuffle(converted_data)
    val_size = max(1, int(len(converted_data) * 0.1))  # 10% for validation
    train_data = converted_data[val_size:]
    val_data = converted_data[:val_size]
    
    print(f"[INFO] Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    wandb.log({"train_samples": len(train_data), "val_samples": len(val_data)})
    
    # ========================================================================
    # STEP 5: Setup Unsloth Trainer
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Setting up SFTTrainer...")
    print("=" * 60)
    
    # Enable training mode
    FastVisionModel.for_training(model)
    
    # Create data collator (per Unsloth docs)
    data_collator = UnslothVisionDataCollator(model, tokenizer)
    
    # Training arguments
    # NOTE: auto_find_batch_size=False because it causes infinite restarts on OOM
    # Instead, use a conservative fixed batch size with gradient accumulation
    eval_batch_size = max(1, config.per_device_train_batch_size // 2)  # Smaller eval batch
    
    training_args = SFTConfig(
        output_dir=str(experiment_output_dir / "checkpoints"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=config.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        auto_find_batch_size=True,  # DISABLED - causes restarts and loses progress
        # Validation settings
        eval_strategy="steps",  # Evaluate every N steps
        eval_steps=getattr(config, 'eval_steps', 50),  # Evaluate periodically
        per_device_eval_batch_size=eval_batch_size,  # Smaller batch for eval
        load_best_model_at_end=True,  # Load best model based on val loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better
        # Gradient checkpointing to save memory
        gradient_checkpointing=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # Stop if no improvement for 3 evals (150 steps)
                early_stopping_threshold=0.01,  # Min improvement to count as "better"
            )
        ],
    )
    
    print("[INFO] Trainer configured!")
    
    # ========================================================================
    # STEP 6: Train
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Starting training...")
    print("=" * 60)
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB")
        wandb.log({"gpu_name": gpu_name, "gpu_memory_gb": gpu_memory})
    
    trainer_stats = trainer.train()
    
    print("\n[INFO] Training complete!")
    
    # Log final metrics (only if wandb is still active)
    try:
        if wandb.run is not None:
            wandb.log({
                "final_train_loss": trainer_stats.training_loss,
                "total_steps": trainer_stats.global_step,
                "train_runtime_seconds": trainer_stats.metrics.get("train_runtime", 0),
            })
    except Exception as e:
        print(f"[INFO] Could not log final metrics to wandb (already closed): {e}")
    
    # ========================================================================
    # STEP 7: Save Model
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Saving model...")
    print("=" * 60)
    
    # Save LoRA adapters
    lora_save_path = experiment_output_dir / "lora_adapters"
    model.save_pretrained(str(lora_save_path))
    tokenizer.save_pretrained(str(lora_save_path))
    print(f"[INFO] LoRA adapters saved to {lora_save_path}")
    
    # Save training stats
    stats_path = experiment_output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "run_name": config.run_name,
            "training_loss": trainer_stats.training_loss,
            "global_step": trainer_stats.global_step,
            "metrics": trainer_stats.metrics,
        }, f, indent=2, default=str)
    
    # Log artifact paths to wandb
    wandb.log({
        "output/lora_path": str(lora_save_path),
        "output/experiment_dir": str(experiment_output_dir),
    })
    
    # Commit volume changes
    output_volume.commit()
    print(f"[INFO] Results saved to {experiment_output_dir}")
    
    wandb.finish()
    
    return {
        "status": "success",
        "run_name": config.run_name,
        "output_dir": str(experiment_output_dir),
        "training_loss": trainer_stats.training_loss,
        "total_steps": trainer_stats.global_step,
    }


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================
@app.local_entrypoint()
def main():
    """Run the VLM fine-tuning experiment."""
    print("=" * 60)
    print("VLM Fine-tuning Framework")
    print("=" * 60)
    print(f"Run Name: {CONFIG.run_name}")
    print(f"Model: {CONFIG.model_name}")
    print(f"Dataset: {CONFIG.dataset_name}")
    print(f"Train GPU: {CONFIG.train_gpu}")
    print(f"Batch Size: {CONFIG.per_device_train_batch_size} x {CONFIG.gradient_accumulation_steps} = {CONFIG.per_device_train_batch_size * CONFIG.gradient_accumulation_steps}")
    print(f"Epochs: {CONFIG.num_train_epochs}")
    print("=" * 60)
    
    # Initialize WandB locally (like run.py pattern)
    RUN = wandb.init(
        project=WANDB_PROJECT,
        name=CONFIG.run_name,
        config=asdict(CONFIG),
        tags=[
            CONFIG.model_name.split("/")[-1],
            DATASET_CONFIGS.get(CONFIG.dataset_name, {}).get("name", CONFIG.dataset_name),
            f"lora_r{CONFIG.lora_r}",
            f"bs{CONFIG.per_device_train_batch_size}",
            f"run_{CONFIG.run_number}",
        ],
    )
    
    # Get WandB run ID to pass to remote
    wandb_run_id = RUN.id
    print(f"[INFO] WandB Run ID: {wandb_run_id}")
    print(f"[INFO] WandB URL: {RUN.url}")
    
    # Convert config to dict for serialization
    config_dict = asdict(CONFIG)
    
    # Run training
    result = train_vlm.remote(config_dict, wandb_run_id)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Run Name: {result['run_name']}")
    print(f"Output: {result['output_dir']}")
    print(f"Final Loss: {result['training_loss']:.4f}")
    print(f"Total Steps: {result['total_steps']}")
    
    # Final log
    wandb.log({"completed": 1})
    wandb.finish()


if __name__ == "__main__":
    print("Run with: modal run finetuning_script/train_vlm.py")
