#!/usr/bin/env python3
"""
eval.py - VLM evaluation script.

Usage:
  python eval.py [options]

Examples:
  python eval.py --model HuggingFaceTB/SmolVLM-500M-Instruct --dataset lmms-lab/textvqa
  python eval.py --checkpoint ./outputs/train_run/checkpoints/final --dataset lmms-lab/DocVQA
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

from utils import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
    GROQ_AVAILABLE,
    get_device,
    normalize_answer,
    compute_exact_match,
    compute_relaxed_match,
    extract_sample_fields,
    format_prompt,
    format_vlm_messages,
    load_dataset_with_fallback,
    load_model_and_processor,
    load_checkpoint,
    init_wandb,
)


def judge_with_groq(question: str, prediction: str, ground_truths: list) -> dict:
    """Use Groq LLM as judge for answer evaluation."""
    if not GROQ_AVAILABLE:
        return {"score": compute_exact_match(prediction, ground_truths), "judge": "exact_match", "reason": "groq not available"}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"score": compute_exact_match(prediction, ground_truths), "judge": "exact_match", "reason": "GROQ_API_KEY not set"}

    try:
        import groq
        client = groq.Groq(api_key=api_key)
        gt_str = " OR ".join(ground_truths[:3])
        prompt = f"""You are evaluating a VQA model's answer.

Question: {question}
Ground Truth Answer(s): {gt_str}
Model Prediction: {prediction}

Is the model's prediction correct or semantically equivalent to the ground truth?
Reply with ONLY "CORRECT" or "INCORRECT" followed by a brief reason."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
        is_correct = result.upper().startswith("CORRECT")
        return {"score": 1.0 if is_correct else 0.0, "judge": "groq", "reason": result}
    except Exception as e:
        return {"score": compute_exact_match(prediction, ground_truths), "judge": "exact_match", "reason": f"groq error: {e}"}


def cmd_eval(args: argparse.Namespace) -> int:
    """Run evaluation."""
    print("=" * 60)
    print("VLM Evaluation")
    print("=" * 60)

    model_id = args.model or DEFAULT_MODEL
    dataset_id = args.dataset or DEFAULT_DATASET
    split = args.split or "validation"
    subset = getattr(args, "subset", None)
    checkpoint = args.checkpoint
    run_name = args.run_name or f"eval_{model_id.split('/')[-1]}_{int(time.time())}"
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / run_name
    use_judge = args.use_judge

    device = get_device()
    if args.cpu:
        device = "cpu"

    print(f"\nConfiguration:")
    print(f"  Model: {model_id}")
    print(f"  Checkpoint: {checkpoint or 'base model'}")
    print(f"  Dataset: {dataset_id} (split: {split})")
    print(f"  Device: {device}")
    print(f"  Use judge: {use_judge}")
    print(f"  Output: {output_dir}")

    config = {
        "model": model_id,
        "checkpoint": checkpoint,
        "dataset": dataset_id,
        "split": split,
        "device": device,
        "use_judge": use_judge,
    }
    wandb_run = init_wandb(run_name=run_name, config=config)

    print(f"\nLoading model...")
    start_time = time.time()
    try:
        model, processor = load_model_and_processor(
            model_id, device=device, load_in_4bit=args.load_in_4bit,
            use_unsloth=False, for_training=False,
        )

        if checkpoint:
            if checkpoint == "latest":
                ckpt_dir = output_dir / "checkpoints"
                if ckpt_dir.exists():
                    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
                    if checkpoints:
                        checkpoint = str(checkpoints[-1])
                    elif (ckpt_dir / "final").exists():
                        checkpoint = str(ckpt_dir / "final")

            if checkpoint and checkpoint != "latest":
                model = load_checkpoint(checkpoint, model)

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

    print(f"\nRunning evaluation...")
    model.eval()

    tokenizer = processor if hasattr(processor, "tokenizer") else processor
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions = []
    exact_matches = []
    relaxed_matches = []
    judge_scores = []

    # Check if this is a VLM with chat template
    is_vlm = hasattr(processor, "apply_chat_template") or hasattr(processor, "image_processor")
    print(f"  VLM mode: {is_vlm}")

    for i, sample in enumerate(ds):
        fields = extract_sample_fields(sample, i, dataset_info)
        question = fields["question"]
        image = fields["image"]

        try:
            if is_vlm and image is not None:
                # Use VLM chat template
                messages = format_vlm_messages(question, has_image=True)
                prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    images=[image],
                    text=prompt_text,
                    return_tensors="pt",
                ).to(device)
            elif image is not None and hasattr(processor, "__call__"):
                # Fallback for other VLMs
                prompt = format_prompt(question)
                inputs = processor(
                    images=image, text=prompt, return_tensors="pt",
                ).to(device)
            else:
                # Text-only
                prompt = format_prompt(question)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
        except Exception as e:
            print(f"  [WARN] Sample {i}: processor error: {e}, using text-only")
            prompt = format_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0,
            )

        # Decode only the generated tokens (not the input)
        input_len = inputs.get("input_ids", inputs.get("input_token_ids", [[]])).shape[1]
        generated_ids = outputs[0][input_len:]
        pred_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up common artifacts
        pred_answer = pred_answer.split("\n")[0].strip()  # Take first line
        if pred_answer.lower().startswith("answer:"):
            pred_answer = pred_answer[7:].strip()

        # Show first few predictions for debugging
        if i < 5:
            gt_preview = fields["ground_truths"][0] if fields["ground_truths"] else "(none)"
            print(f"  [Sample {i}] Q: {question[:60]}...")
            print(f"             Pred: '{pred_answer}' | GT: '{gt_preview}'")

        em = compute_exact_match(pred_answer, fields["ground_truths"])
        rm = compute_relaxed_match(pred_answer, fields["ground_truths"])
        exact_matches.append(em)
        relaxed_matches.append(rm)

        judge_result = {"score": em, "judge": "exact_match", "reason": ""}
        if use_judge:
            judge_result = judge_with_groq(question, pred_answer, fields["ground_truths"])
        judge_scores.append(judge_result["score"])

        predictions.append({
            "sample_id": str(fields["sample_id"]),
            "question": question,
            "prediction": pred_answer,
            "ground_truth": "|".join(fields["ground_truths"]),
            "exact_match": em,
            "relaxed_match": rm,
            "judge_score": judge_result["score"],
            "judge_backend": judge_result["judge"],
        })

        if (i + 1) % 10 == 0:
            running_em = sum(exact_matches) / len(exact_matches)
            print(f"  Processed {i + 1}/{len(ds)} samples... (EM: {running_em:.1%})")

    print(f"  ✓ Evaluation completed")

    metrics = {
        "num_samples": len(predictions),
        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
        "relaxed_match": sum(relaxed_matches) / len(relaxed_matches) if relaxed_matches else 0,
        "judge_accuracy": sum(judge_scores) / len(judge_scores) if judge_scores else 0,
        "backend": "groq" if use_judge and GROQ_AVAILABLE else "exact_match",
        "model": model_id,
        "checkpoint": checkpoint or "base",
        "dataset": dataset_id,
        "split": actual_split,
    }

    print(f"\nWriting outputs...")
    dataset_name = dataset_id.split("/")[-1]
    ckpt_tag = Path(checkpoint).name if checkpoint else "base"
    eval_dir = output_dir / "eval" / dataset_name / ckpt_tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["sample_id", "question", "prediction", "ground_truth", "exact_match", "relaxed_match", "judge_score", "judge_backend"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  ✓ {csv_path}")

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {metrics_path}")

    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Checkpoint: {checkpoint or 'base'}\n")
        f.write(f"Dataset: {dataset_id} ({actual_split})\n")
        f.write(f"Samples: {len(predictions)}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"  Exact Match: {metrics['exact_match']:.2%}\n")
        f.write(f"  Relaxed Match: {metrics['relaxed_match']:.2%}\n")
        f.write(f"  Judge Accuracy: {metrics['judge_accuracy']:.2%}\n")
        f.write(f"  Backend: {metrics['backend']}\n")
    print(f"  ✓ {summary_path}")

    wandb_run.log({
        "eval_exact_match": metrics["exact_match"],
        "eval_relaxed_match": metrics["relaxed_match"],
        "eval_judge_accuracy": metrics["judge_accuracy"],
        "eval_num_samples": metrics["num_samples"],
    })
    wandb_run.finish()

    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Exact Match:    {metrics['exact_match']:.2%}")
    print(f"  Relaxed Match:  {metrics['relaxed_match']:.2%}")
    print(f"  Judge Accuracy: {metrics['judge_accuracy']:.2%}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="Model ID")
    parser.add_argument("--dataset", type=str, help="Dataset ID")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--subset", type=str, help="Dataset subset")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path or 'latest'")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load in 4-bit")
    parser.add_argument("--use_judge", action="store_true", help="Use Groq LLM judge")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--limit_samples", type=int, help="Limit samples")

    args = parser.parse_args()
    return cmd_eval(args)


if __name__ == "__main__":
    sys.exit(main())
