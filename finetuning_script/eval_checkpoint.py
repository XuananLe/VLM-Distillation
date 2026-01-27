"""
Evaluate a finetuned SmolVLM checkpoint using VLMEvalKit.

Uses VLMEvalKit's run.py command to evaluate the model properly.

Usage:
    modal run --detach finetuning_script/eval_checkpoint.py
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
ROOT_DIR = Path("/root/VLM-Distillation")
EVAL_DIR = ROOT_DIR / "VLMEvalKit"

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHECKPOINT_PATH = "/outputs/smolvlm500m_ocrbench_5/checkpoints/checkpoint-75"
BENCHMARK = "OCRBench"  # Single benchmark (no VAL/TEST split)
MODEL_NAME = "SmolVLM-500M-FT"  # Custom name for our finetuned model
EVAL_GPU = "L4"

# LLM Judge Configuration (using Grok)
USE_LLM_JUDGE = True
JUDGE_MODEL = "grok-2-vision-1212"  # Options: grok-vision-beta, grok-2-vision-1212, grok-4-0709

# ============================================================================
# MODAL IMAGE
# ============================================================================
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision", 
        "torchaudio",
    )
    # Install transformers with SmolVLM support
    .pip_install("transformers>=4.51.0")
    .pip_install(
        "peft",
        "accelerate",
        "datasets",
        "pillow",
        "pandas",
        "openpyxl",
        "huggingface_hub",
        "tqdm",
        # VLMEvalKit dependencies
        "validators",
        "xlsxwriter",
        "portalocker",
        "opencv-python",
        "python-dotenv",
        "rich",
        "tabulate",
        "omegaconf",
        "imageio",
        "einops",
        "decord",
        "sty",
        "matplotlib",
        "tiktoken",
        "sentencepiece",
        "openai",
        "timeout-decorator",
        "json-repair",
        "ipdb",
        "nltk",
        "scikit-learn",
        "pylatexenc==2.10",
        "sympy",
        "apted>=1.0.3",
        "distance>=0.1.3",
        "lxml>=6.0.2",
        "levenshtein>=0.27.1",
        "jieba>=0.42.1",
        "editdistance>=0.8.1",
        "anls>=0.0.2",
        "antlr4-python3-runtime==4.11.1",
        "math-verify",
        "qwen_vl_utils",
        "timm",
        # SArena dependencies
        "torchmetrics",
        "scikit-image",
        "lpips",
        "cairosvg",
        # UniSVG dependencies
        "sentence_transformers",
        "bert_score",
        # Additional VLMEvalKit dependencies
        "google-genai",
        "protobuf",
        "requests",
        "typing_extensions",
        "numpy",
        "setuptools",
        "colormath>=3.0.0",
        "pdf2image>=1.17.0",
        "zss>=1.2.0",
        "polygon3>=3.0.9.1",
        "openai-clip",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("./VLMEvalKit", remote_path=str(EVAL_DIR))
)

app = modal.App(
    name="VLM-Eval-Checkpoint",
    image=base_image,
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
@app.function(
    gpu=EVAL_GPU,
    timeout=60 * 60 * 4,  # 4 hours
    secrets=[modal.Secret.from_name("xai-api-key")],
)
def evaluate_checkpoint(
    base_model: str = BASE_MODEL,
    checkpoint_path: str = CHECKPOINT_PATH,
    benchmark: str = BENCHMARK,
    model_name: str = MODEL_NAME,
    use_llm_judge: bool = USE_LLM_JUDGE,
    judge_model: str = JUDGE_MODEL,
):
    """
    Evaluate finetuned checkpoint using VLMEvalKit API directly.
    """
    import sys
    import os
    
    # Add VLMEvalKit to path
    vlmeval_path = str(EVAL_DIR)
    sys.path.insert(0, vlmeval_path)
    os.chdir(vlmeval_path)
    
    print("=" * 60)
    print("VLM Checkpoint Evaluation via VLMEvalKit")
    print("=" * 60)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Benchmark: {benchmark}")
    print(f"Model Name: {model_name}")
    print(f"LLM Judge: {judge_model if use_llm_judge else 'disabled'}")
    print("=" * 60)
    
    # Check for XAI API key
    xai_key = os.environ.get('XAI_API_KEY', '')
    if use_llm_judge and not xai_key:
        print("[WARNING] XAI_API_KEY not set, LLM judge may not work")
        print("[INFO] Set it with: modal secret create xai-api-key XAI_API_KEY=your_key")
    else:
        print(f"[INFO] XAI API key configured: {xai_key[:10]}..." if xai_key else "")
    
    # Find checkpoint
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        run_dir = Path("/outputs/smolvlm500m_ocrbench_5")
        checkpoints = sorted(run_dir.glob("checkpoints/checkpoint-*"))
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
        elif (run_dir / "lora_adapters").exists():
            checkpoint_dir = run_dir / "lora_adapters"
        else:
            raise FileNotFoundError(f"No checkpoints found")
    
    print(f"[INFO] Using checkpoint: {checkpoint_dir}")
    
    # Set output directory
    work_dir = OUTPUT_DIR / "eval_results" / model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the finetuned model directly
    print(f"\n[INFO] Loading finetuned model...")
    model = load_finetuned_model(base_model, str(checkpoint_dir))
    
    # Run evaluation using VLMEvalKit API
    print(f"\n[INFO] Running VLMEvalKit evaluation on {benchmark}...")
    
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data
    from vlmeval.evaluate import evaluate
    from vlmeval.utils import TSVDataset
    import pandas as pd
    
    # Build the dataset
    dataset = build_dataset(benchmark)
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")
    
    # Output file paths
    pred_file = work_dir / f"{model_name}_{benchmark}.xlsx"
    result_file = work_dir / f"{model_name}_{benchmark}_result.xlsx"
    
    # Run inference
    print(f"[INFO] Running inference...")
    predictions = []
    
    from tqdm import tqdm
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        item = dataset.data.iloc[idx]
        
        # Get image path and question
        image_path = item.get('image_path', item.get('image', ''))
        question = item.get('question', '')
        
        # Generate prediction
        response = model.generate(image_path, question)
        
        predictions.append({
            'index': idx,
            'question': question,
            'answer': item.get('answer', ''),
            'prediction': response,
        })
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_excel(pred_file, index=False)
    print(f"[INFO] Predictions saved to: {pred_file}")
    
    # Run evaluation
    print(f"[INFO] Running evaluation metrics...")
    result = dataset.evaluate(pred_file, result_file)
    
    # Read and display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if result_file.exists():
        result_df = pd.read_excel(result_file)
        print(result_df.to_string())
    else:
        print(f"Result: {result}")
    
    # Commit volume
    output_volume.commit()
    
    return {"status": "complete", "work_dir": str(work_dir), "result": result}


def load_finetuned_model(base_model_path: str, lora_path: str):
    """
    Load the finetuned model with LoRA adapters.
    Returns a wrapper with a generate method compatible with VLMEvalKit.
    """
    import torch
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    from peft import PeftModel
    from PIL import Image
    
    print(f"[INFO] Loading base model: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # Load base model
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    
    # Load and merge LoRA adapters
    print(f"[INFO] Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.eval()
    
    torch.cuda.empty_cache()
    print(f"[INFO] Finetuned model loaded!")
    
    class ModelWrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
            self.device = "cuda"
        
        def generate(self, image_path, question, max_new_tokens=2048):
            """Generate a response for the given image and question."""
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Create messages in SmolVLM format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            # Decode
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
    
    return ModelWrapper(model, processor)


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================
@app.local_entrypoint()
def main(
    checkpoint_path: str = CHECKPOINT_PATH,
    benchmark: str = BENCHMARK,
    model_name: str = MODEL_NAME,
    use_judge: bool = USE_LLM_JUDGE,
    judge_model: str = JUDGE_MODEL,
):
    """Run VLMEvalKit evaluation on a checkpoint."""
    print("Starting VLMEvalKit evaluation...")
    print(f"Using Grok LLM judge: {judge_model}" if use_judge else "LLM judge disabled")
    
    result = evaluate_checkpoint.remote(
        base_model=BASE_MODEL,
        checkpoint_path=checkpoint_path,
        benchmark=benchmark,
        model_name=model_name,
        use_llm_judge=use_judge,
        judge_model=judge_model,
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {result['work_dir']}")


if __name__ == "__main__":
    print("Run with: modal run --detach finetuning_script/eval_checkpoint.py")
