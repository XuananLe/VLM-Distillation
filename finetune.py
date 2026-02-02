#!/usr/bin/env python3
"""
finetune.py - VLM fine-tuning script.

Usage:
  python finetune.py [options]

Examples:
  python finetune.py --model HuggingFaceTB/SmolVLM-500M-Instruct --dataset lmms-lab/textvqa
  python finetune.py --smoke  # Quick test with tiny model
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from utils import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    DEFAULT_MICROBATCH,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MAX_SEQ_LENGTH,
    TINY_MODEL,
    OUTPUT_DIR,
    UNSLOTH_AVAILABLE,
    get_device,
    get_gpu_memory_mb,
    normalize_answer,
    extract_sample_fields,
    format_prompt,
    load_dataset_with_fallback,
    load_model_and_processor,
    init_wandb,
)


def get_next_run_number(output_dir: Path, prefix: str) -> int:
    """Get the next sequential run number for a given prefix."""
    if not output_dir.exists():
        return 1
    
    existing_runs = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    max_num = 0
    for run_name in existing_runs:
        # Extract the number suffix (e.g., "train_SmolVLM-500M-Instruct_3" -> 3)
        suffix = run_name[len(prefix):]
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return max_num + 1


def cmd_doctor() -> int:
    """Check environment and dependencies."""
    print("=" * 60)
    print("VLM Fine-tuning Doctor")
    print("=" * 60)

    errors = []

    print(f"\nPython: {sys.version}")

    core_deps = ["torch", "transformers", "datasets", "peft", "trl", "wandb", "PIL"]
    print("\nCore Dependencies:")
    for dep in core_deps:
        try:
            if dep == "PIL":
                import PIL
                print(f"  ✓ PIL (Pillow): {PIL.__version__}")
            else:
                mod = __import__(dep)
                version = getattr(mod, "__version__", "unknown")
                print(f"  ✓ {dep}: {version}")
        except ImportError:
            print(f"  ✗ {dep}: NOT FOUND")
            errors.append(dep)

    print("\nCUDA Status:")
    try:
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  - CUDA not available (CPU mode)")
    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")

    print("\nOptional Dependencies:")
    print(f"  {'✓' if UNSLOTH_AVAILABLE else '-'} unsloth: {'available' if UNSLOTH_AVAILABLE else 'not installed'}")

    try:
        import groq
        print(f"  ✓ groq: available")
    except ImportError:
        print(f"  - groq: not installed")

    try:
        import modal
        print(f"  ✓ modal: available")
    except ImportError:
        print(f"  - modal: not installed")

    print("\nEnvironment Variables:")
    env_vars = ["WANDB_PROJECT", "WANDB_MODE", "GROQ_API_KEY", "HF_TOKEN", "OUTPUT_DIR"]
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            if "KEY" in var or "TOKEN" in var:
                print(f"  ✓ {var}: [set]")
            else:
                print(f"  ✓ {var}: {val}")
        else:
            print(f"  - {var}: not set")

    print("\n" + "=" * 60)
    if errors:
        print(f"ERRORS: Missing core dependencies: {', '.join(errors)}")
        return 1
    else:
        print("All core dependencies OK!")
        return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Validate configuration and dataset access."""
    print("=" * 60)
    print("VLM Fine-tuning Preflight Check")
    print("=" * 60)

    model_id = args.model or DEFAULT_MODEL
    dataset_id = args.dataset or DEFAULT_DATASET
    split = args.split or DEFAULT_SPLIT
    subset = getattr(args, "subset", None)

    print(f"\nModel: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Split: {split}")
    if subset:
        print(f"Subset: {subset}")

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    print(f"\nOutput directory: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("  ✓ Output directory accessible")
    except Exception as e:
        print(f"  ✗ Cannot create output directory: {e}")
        return 1

    print(f"\nLoading dataset...")
    try:
        ds, actual_split, dataset_info = load_dataset_with_fallback(
            dataset_id, split, subset, limit=2
        )
        print(f"  ✓ Loaded split '{actual_split}' with {len(ds)} samples (limited to 2)")

        print("\nSample fields:")
        for i, sample in enumerate(ds):
            fields = extract_sample_fields(sample, i, dataset_info)
            print(f"\n  Sample {i}:")
            print(f"    ID: {fields['sample_id']}")
            print(f"    Question: {fields['question'][:100]}..." if len(fields['question']) > 100 else f"    Question: {fields['question']}")
            print(f"    Ground truths: {fields['ground_truths'][:3]}")
            print(f"    Has image: {fields['image'] is not None}")

    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Preflight check PASSED!")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run smoke test with tiny model on CPU."""
    print("=" * 60)
    print("VLM Fine-tuning Smoke Test")
    print("=" * 60)

    device = "cpu"
    os.environ.setdefault("WANDB_MODE", "offline")

    # Generate sequential run name for smoke tests
    if args.run_name:
        run_name = args.run_name
    else:
        smoke_dir = Path("./outputs_smoke")
        run_num = get_next_run_number(smoke_dir, "smoke_")
        run_name = f"smoke_{run_num}"
    output_dir = Path(args.output) if args.output else Path("./outputs_smoke") / run_name

    print(f"\nDevice: {device}")
    print(f"Output: {output_dir}")
    print(f"Run name: {run_name}")

    wandb_run = init_wandb(
        run_name=run_name,
        config={"mode": "smoke", "model": TINY_MODEL, "device": device},
    )

    print(f"\nLoading tiny model: {TINY_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            TINY_MODEL, trust_remote_code=True, torch_dtype=torch.float32
        )
        model = model.to(device)
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        wandb_run.finish()
        return 1

    print("\nCreating dummy training data...")
    dummy_data = [
        {"text": "Question: What is 2+2?\nAnswer: 4"},
        {"text": "Question: What color is the sky?\nAnswer: blue"},
    ]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=64,
            padding="max_length", return_tensors=None,
        )

    ds = Dataset.from_list(dummy_data)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    ds = ds.map(add_labels)

    print("\nRunning training (2 steps)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,
        max_steps=2,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_steps=2,
        report_to="wandb",
        remove_unused_columns=False,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds, tokenizer=tokenizer,
    )

    train_result = trainer.train()
    print(f"  ✓ Training completed: {train_result.metrics}")

    wandb_run.log({"train_loss": train_result.metrics.get("train_loss", 0)})

    print("\nRunning evaluation (2 samples)...")
    model.eval()

    predictions = []
    for i, sample in enumerate(dummy_data):
        prompt = sample["text"].split("Answer:")[0] + "Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answer = pred_text[len(prompt):].strip()

        predictions.append({
            "sample_id": i,
            "question": prompt,
            "prediction": pred_answer,
            "ground_truth": sample["text"].split("Answer:")[-1].strip(),
        })
        print(f"  Sample {i}: pred='{pred_answer}'")

    print("\nWriting outputs...")
    eval_dir = output_dir / "eval" / "smoke" / "final"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "question", "prediction", "ground_truth"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  ✓ {csv_path}")

    metrics = {
        "num_samples": len(predictions),
        "exact_match": sum(1 for p in predictions if normalize_answer(p["prediction"]) == normalize_answer(p["ground_truth"])) / len(predictions),
    }
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {metrics_path}")

    wandb_run.log({"eval_exact_match": metrics["exact_match"]})
    wandb_run.finish()

    print("\n" + "=" * 60)
    print("Smoke test PASSED!")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run full training."""
    print("=" * 60)
    print("VLM Fine-tuning Training")
    print("=" * 60)

    model_id = args.model or DEFAULT_MODEL
    dataset_id = args.dataset or DEFAULT_DATASET
    split = args.split or DEFAULT_SPLIT
    subset = getattr(args, "subset", None)
    
    # Generate sequential run name (e.g., train_SmolVLM-500M-Instruct_textvqa_1, _2, _3...)
    if args.run_name:
        run_name = args.run_name
    else:
        model_short = model_id.split('/')[-1]
        dataset_short = dataset_id.split('/')[-1].lower().replace("_", "")
        prefix = f"train_{model_short}_{dataset_short}_"
        run_num = get_next_run_number(OUTPUT_DIR, prefix)
        run_name = f"{prefix}{run_num}"
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / run_name

    microbatch = args.microbatch if args.microbatch else DEFAULT_MICROBATCH
    grad_accum = args.grad_accum if args.grad_accum else DEFAULT_GRAD_ACCUM
    max_steps = args.max_steps if args.max_steps else DEFAULT_MAX_STEPS
    num_epochs = args.num_epochs if args.num_epochs else DEFAULT_NUM_EPOCHS
    lr = args.lr if args.lr else DEFAULT_LR
    load_in_4bit = args.load_in_4bit
    use_unsloth = not args.no_unsloth

    device = get_device()
    if args.cpu:
        device = "cpu"

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Dataset: {dataset_id} (split: {split})")
    print(f"  Device: {device}")
    print(f"  Microbatch: {microbatch}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch: {microbatch * grad_accum}")
    print(f"  Max steps: {max_steps if max_steps > 0 else 'auto'}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Load in 4bit: {load_in_4bit}")
    print(f"  Use Unsloth: {use_unsloth and UNSLOTH_AVAILABLE}")
    print(f"  Output: {output_dir}")

    config = {
        "model": model_id,
        "dataset": dataset_id,
        "split": split,
        "device": device,
        "microbatch": microbatch,
        "grad_accum": grad_accum,
        "effective_batch": microbatch * grad_accum,
        "max_steps": max_steps,
        "num_epochs": num_epochs,
        "lr": lr,
        "load_in_4bit": load_in_4bit,
        "unsloth": use_unsloth and UNSLOTH_AVAILABLE,
    }

    wandb_run = init_wandb(run_name=run_name, config=config)

    print(f"\nLoading model...")
    start_time = time.time()
    try:
        model, processor = load_model_and_processor(
            model_id, device=device, load_in_4bit=load_in_4bit,
            use_unsloth=use_unsloth, for_training=True,
        )
        print(f"  ✓ Model loaded in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        wandb_run.finish()
        return 1

    print(f"\nLoading dataset...")
    try:
        ds, actual_split, dataset_info = load_dataset_with_fallback(
            dataset_id, split, subset, limit=args.limit_samples
        )
        print(f"  ✓ Loaded {len(ds)} samples from split '{actual_split}'")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        wandb_run.finish()
        return 1

    print(f"\nPreparing training data...")

    tokenizer = processor if hasattr(processor, "tokenizer") else processor
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prepare_sample(sample, idx):
        fields = extract_sample_fields(sample, idx, dataset_info)
        prompt = format_prompt(fields["question"])
        answer = fields["ground_truths"][0] if fields["ground_truths"] else ""
        full_text = f"{prompt} {answer}"
        return {"text": full_text, "prompt": prompt, "answer": answer}

    train_data = [prepare_sample(sample, i) for i, sample in enumerate(ds)]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=DEFAULT_MAX_SEQ_LENGTH,
            padding="max_length", return_tensors=None,
        )

    train_ds = Dataset.from_list(train_data)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "prompt", "answer"])

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    train_ds = train_ds.map(add_labels)
    print(f"  ✓ Prepared {len(train_ds)} training samples")

    print(f"\nStarting training...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=device == "cuda" and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4 if device == "cuda" else 0,
        gradient_checkpointing=True if device == "cuda" else False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if device == "cuda" else None,
        ddp_find_unused_parameters=False,
    )

    try:
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model, args=training_args, train_dataset=train_ds, tokenizer=tokenizer,
        )
        print("  Using TRL SFTTrainer")
    except Exception:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_ds, tokenizer=tokenizer,
        )
        print("  Using HF Trainer")

    train_start = time.time()
    try:
        train_result = trainer.train()
        train_time = time.time() - train_start
        print(f"\n  ✓ Training completed in {train_time:.1f}s")
        print(f"  Final loss: {train_result.metrics.get('train_loss', 'N/A')}")

        wandb_run.log({
            "train_time_seconds": train_time,
            "final_train_loss": train_result.metrics.get("train_loss", 0),
            "gpu_memory_mb": get_gpu_memory_mb(),
        })

        final_checkpoint = checkpoint_dir / "final"
        trainer.save_model(str(final_checkpoint))
        print(f"  ✓ Saved checkpoint to {final_checkpoint}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n  ✗ OOM Error with batch size {microbatch}")
            print(f"    Try reducing --microbatch or enabling --load_in_4bit")
        else:
            print(f"\n  ✗ Training failed: {e}")
        wandb_run.finish()
        return 1

    wandb_run.finish()
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Doctor
    subparsers.add_parser("doctor", help="Check environment and dependencies")

    # Preflight
    p_preflight = subparsers.add_parser("preflight", help="Validate configuration and dataset")
    p_preflight.add_argument("--model", type=str, help="Model ID")
    p_preflight.add_argument("--dataset", type=str, help="Dataset ID")
    p_preflight.add_argument("--split", type=str, help="Dataset split")
    p_preflight.add_argument("--subset", type=str, help="Dataset subset")
    p_preflight.add_argument("--output", type=str, help="Output directory")

    # Smoke
    p_smoke = subparsers.add_parser("smoke", help="Run smoke test")
    p_smoke.add_argument("--run_name", type=str, help="Run name")
    p_smoke.add_argument("--output", type=str, help="Output directory")

    # Train
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("--model", type=str, help="Model ID")
    p_train.add_argument("--dataset", type=str, help="Dataset ID")
    p_train.add_argument("--split", type=str, default="train", help="Dataset split")
    p_train.add_argument("--subset", type=str, help="Dataset subset")
    p_train.add_argument("--run_name", type=str, help="Run name")
    p_train.add_argument("--output", type=str, help="Output directory")
    p_train.add_argument("--microbatch", type=int, help="Per-device batch size")
    p_train.add_argument("--grad_accum", type=int, help="Gradient accumulation steps")
    p_train.add_argument("--max_steps", type=int, help="Max training steps")
    p_train.add_argument("--num_epochs", type=int, help="Number of epochs")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--load_in_4bit", action="store_true", help="Load in 4-bit")
    p_train.add_argument("--no_unsloth", action="store_true", help="Disable Unsloth")
    p_train.add_argument("--cpu", action="store_true", help="Force CPU")
    p_train.add_argument("--limit_samples", type=int, help="Limit samples")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "doctor": lambda a: cmd_doctor(),
        "preflight": cmd_preflight,
        "smoke": cmd_smoke,
        "train": cmd_train,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
