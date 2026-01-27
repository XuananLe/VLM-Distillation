"""
Standalone OCRBench evaluation for finetuned SmolVLM checkpoint.

This script directly loads the model with LoRA adapters and evaluates on OCRBench
without VLMEvalKit dependencies.

Usage:
    modal run --detach finetuning_script/eval_ocrbench.py
"""

import modal
import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = Path("/outputs")

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHECKPOINT_PATH = "/outputs/smolvlm500m_ocrbench_5/checkpoints/checkpoint-75"
EVAL_GPU = "L4"

# ============================================================================
# MODAL IMAGE - Minimal dependencies
# ============================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "transformers>=4.45.0",
        "peft",
        "accelerate",
        "datasets",
        "pillow",
        "pandas",
        "openpyxl",
        "huggingface_hub",
        "tqdm",
        "openai",  # For Grok API (OpenAI-compatible)
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(
    name="OCRBench-Eval",
    image=image,
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
    secrets=[modal.Secret.from_name("xai-api-key")],
)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
@app.function(
    gpu=EVAL_GPU,
    timeout=60 * 60 * 4,  # 4 hours
)
def evaluate_ocrbench(
    base_model: str = BASE_MODEL,
    checkpoint_path: str = CHECKPOINT_PATH,
):
    """
    Evaluate finetuned SmolVLM on OCRBench.
    """
    import torch
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    from peft import PeftModel
    from datasets import load_dataset
    from PIL import Image
    from tqdm import tqdm
    import pandas as pd
    import re
    from pathlib import Path
    
    print("=" * 60)
    print("OCRBench Evaluation")
    print("=" * 60)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    # Find checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        run_dir = Path("/outputs/smolvlm500m_ocrbench_5")
        checkpoints = sorted(run_dir.glob("checkpoints/checkpoint-*"))
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
        else:
            raise FileNotFoundError(f"No checkpoints found")
    
    print(f"[INFO] Using checkpoint: {checkpoint_dir}")
    
    # Load model with LoRA
    print(f"\n[INFO] Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    
    base = Idefics3ForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    
    print(f"[INFO] Loading LoRA adapters from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model = model.merge_and_unload()
    model.eval()
    
    torch.cuda.empty_cache()
    print(f"[INFO] Model loaded!")
    
    # Load OCRBench dataset
    print(f"\n[INFO] Loading OCRBench dataset...")
    dataset = load_dataset("echo840/OCRBench", split="test")
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    print(f"\n[INFO] Running evaluation with Grok LLM Judge...")
    print("-" * 80)
    results = []
    correct = 0
    total = 0
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            image = item['image']
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            
            question = item['question']
            ground_truth = item['answer']
            
            # Create prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
            # Decode
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Use Grok LLM as judge
            judge_result = grok_judge(question, str(ground_truth), response)
            is_correct = judge_result['correct']
            judge_reason = judge_result['reason']
            
            if is_correct:
                correct += 1
            total += 1
            
            # Log each sample to terminal
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"\n[{idx+1}/{len(dataset)}] {status}")
            print(f"  Q: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"  GT: {ground_truth}")
            print(f"  Pred: {response[:150]}{'...' if len(response) > 150 else ''}")
            print(f"  Judge: {judge_reason}")
            print(f"  Running Acc: {correct}/{total} ({correct/total*100:.1f}%)")
            
            results.append({
                'index': idx,
                'question': question,
                'ground_truth': ground_truth,
                'prediction': response,
                'correct': is_correct,
                'judge_reason': judge_reason,
            })
                
        except Exception as e:
            print(f"\n[{idx+1}/{len(dataset)}] ✗ ERROR: {e}")
            results.append({
                'index': idx,
                'question': item.get('question', ''),
                'ground_truth': item.get('answer', ''),
                'prediction': f"ERROR: {e}",
                'correct': False,
                'judge_reason': f"Error: {e}",
            })
            total += 1
    
    # Calculate final metrics
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    # Save results in VLMEvalKit-compatible format
    model_name = "SmolVLM-500M-FT"
    benchmark = "OCRBench"
    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save {model}_{benchmark}_acc.csv - accuracy score file
    acc_file = output_dir / f"{model_name}_{benchmark}_acc.csv"
    acc_df = pd.DataFrame([{
        'model': model_name,
        'benchmark': benchmark,
        'accuracy': round(accuracy, 2),
        'correct': correct,
        'total': total,
    }])
    acc_df.to_csv(acc_file, index=False)
    print(f"\n[INFO] Accuracy file saved: {acc_file}")
    
    # 2. Save {model}_{benchmark}.xlsx - raw predictions
    raw_file = output_dir / f"{model_name}_{benchmark}.xlsx"
    raw_df = pd.DataFrame([{
        'index': r['index'],
        'question': r['question'],
        'answer': r['ground_truth'],
        'prediction': r['prediction'],
    } for r in results])
    raw_df.to_excel(raw_file, index=False)
    print(f"[INFO] Predictions file saved: {raw_file}")
    
    # 3. Save {model}_{benchmark}_results.xlsx - detailed results with scoring
    results_file = output_dir / f"{model_name}_{benchmark}_results.xlsx"
    results_df = pd.DataFrame(results)
    results_df.to_excel(results_file, index=False)
    print(f"[INFO] Results file saved: {results_file}")
    
    # Also save a summary text file
    summary = {
        'model': model_name,
        'checkpoint': str(checkpoint_dir),
        'benchmark': benchmark,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
    }
    
    summary_file = output_dir / f"{model_name}_{benchmark}_summary.txt"
    with open(summary_file, 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n[INFO] All outputs saved to: {output_dir}")
    print(f"  - {model_name}_{benchmark}_acc.csv")
    print(f"  - {model_name}_{benchmark}.xlsx")
    print(f"  - {model_name}_{benchmark}_results.xlsx")
    print(f"  - {model_name}_{benchmark}_summary.txt")
    
    # Commit volume
    output_volume.commit()
    
    return summary


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    import re
    # Lowercase
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def grok_judge(question: str, ground_truth: str, prediction: str) -> dict:
    """
    Use Groq (Llama) as LLM judge to evaluate if prediction matches ground truth.
    Returns dict with 'correct' (bool) and 'reason' (str).
    """
    import os
    from openai import OpenAI
    
    # Use Groq API (OpenAI-compatible) with Llama model
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),  # Actually Groq key
        base_url="https://api.groq.com/openai/v1",
    )
    
    prompt = f"""You are evaluating an OCR/document understanding model's response.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Prediction: {prediction}

Judge if the model's prediction is CORRECT or INCORRECT.
- CORRECT: The prediction conveys the same meaning/answer as ground truth (minor formatting differences OK)
- INCORRECT: The prediction is wrong, missing key info, or doesn't answer the question

Respond with ONLY a JSON object:
{{"correct": true/false, "reason": "brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Groq's Llama model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        # Handle potential markdown code blocks
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        return {
            'correct': result.get('correct', False),
            'reason': result.get('reason', 'No reason provided'),
        }
    except Exception as e:
        # Fallback to string matching if API fails
        pred_norm = normalize_answer(prediction)
        gt_norm = normalize_answer(str(ground_truth))
        is_correct = pred_norm == gt_norm or gt_norm in pred_norm
        return {
            'correct': is_correct,
            'reason': f'API error ({e}), used string matching fallback',
        }


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================
@app.local_entrypoint()
def main(
    checkpoint_path: str = CHECKPOINT_PATH,
):
    """Run OCRBench evaluation on a checkpoint."""
    print("Starting OCRBench evaluation...")
    
    result = evaluate_ocrbench.remote(
        base_model=BASE_MODEL,
        checkpoint_path=checkpoint_path,
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Correct: {result['correct']}/{result['total']}")


if __name__ == "__main__":
    print("Run with: modal run --detach finetuning_script/eval_ocrbench.py")
