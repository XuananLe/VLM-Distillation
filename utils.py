#!/usr/bin/env python3
"""
utils.py - Shared utilities for VLM fine-tuning and evaluation.
"""

import os
import re
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# OPTIONAL IMPORTS (guarded)
# =============================================================================

UNSLOTH_AVAILABLE = False
try:
    import unsloth
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

GROQ_AVAILABLE = False
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_DATASET = "lmms-lab/textvqa"
DEFAULT_SPLIT = "train"
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

DATASET_REGISTRY = {
    "textvqa": {
        "hf_path": "lmms-lab/textvqa",
        "splits": ["train", "validation", "test"],
        "question_field": "question",
        "answer_field": "answers",
        "image_field": "image",
        "id_field": "question_id",
    },
    "docvqa": {
        "hf_path": "lmms-lab/DocVQA",
        "subset": "DocVQA",  # Required config name
        "splits": ["train", "validation", "test"],
        "question_field": "question",
        "answer_field": "answers",
        "image_field": "image",
        "id_field": "questionId",
    },
    "chartqa": {
        "hf_path": "lmms-lab/ChartQA",
        # No subset needed - uses default config
        "splits": ["train", "val", "test"],
        "question_field": "query",
        "answer_field": "label",
        "image_field": "image",
        "id_field": "id",
    },
    "ocrbench": {
        "hf_path": "echo840/OCRBench",
        "splits": ["test"],
        "question_field": "question",
        "answer_field": "answer",
        "image_field": "image",
        "id_field": "id",
    },
}

# LoRA defaults
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training defaults
DEFAULT_MICROBATCH = 16  # Small batch per device for VLM (images use lots of memory)
DEFAULT_GRAD_ACCUM = 8  # Effective batch = 4 * 32 = 128
DEFAULT_MAX_STEPS = -1
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LR = 2e-4
DEFAULT_MAX_SEQ_LENGTH = 2048

# Generation defaults
MAX_NEW_TOKENS = 64

# Output directory
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_device() -> str:
    """Get available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_gpu_memory_mb() -> Optional[float]:
    """Get GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = " ".join(answer.split())
    return answer


def compute_exact_match(pred: str, ground_truths: List[str]) -> float:
    """Compute normalized exact match score."""
    pred_norm = normalize_answer(pred)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


def compute_relaxed_match(pred: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """Compute relaxed match using character-level similarity."""
    pred_norm = normalize_answer(pred)
    best_score = 0.0
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        ratio = SequenceMatcher(None, pred_norm, gt_norm).ratio()
        best_score = max(best_score, ratio)
    return 1.0 if best_score >= threshold else 0.0


def extract_sample_fields(sample: Dict, idx: int, dataset_info: Dict) -> Dict[str, Any]:
    """Extract standardized fields from a dataset sample."""
    sample_id = None
    for id_field in [dataset_info.get("id_field"), "questionId", "question_id", "id", "image_id"]:
        if id_field and id_field in sample:
            sample_id = sample[id_field]
            break
    if sample_id is None:
        sample_id = str(idx)

    q_field = dataset_info.get("question_field", "question")
    question = sample.get(q_field, sample.get("question", ""))

    a_field = dataset_info.get("answer_field", "answers")
    raw_answer = sample.get(a_field, sample.get("answers", sample.get("answer", "")))
    if isinstance(raw_answer, list):
        ground_truths = [str(a) for a in raw_answer]
    elif isinstance(raw_answer, str):
        ground_truths = [raw_answer]
    else:
        ground_truths = [str(raw_answer)] if raw_answer else [""]

    img_field = dataset_info.get("image_field", "image")
    image = sample.get(img_field)

    return {
        "sample_id": sample_id,
        "question": question,
        "ground_truths": ground_truths,
        "image": image,
    }


def format_prompt(question: str, use_chat_template: bool = False) -> str:
    """Format question into prompt."""
    if use_chat_template:
        # For VLMs that use chat templates (like SmolVLM)
        return question
    return f"Question: {question}\nAnswer:"


def format_vlm_messages(question: str, has_image: bool = True) -> list:
    """Format messages for VLM chat template."""
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"Answer briefly. {question}"})
    return [{"role": "user", "content": content}]


def load_dataset_with_fallback(
    dataset_path: str,
    split: str,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Any, str, Dict]:
    """Load dataset with fallback to available splits."""
    from datasets import load_dataset

    dataset_key = dataset_path.split("/")[-1].lower().replace("_", "").replace("-", "")
    dataset_info = None
    for key, info in DATASET_REGISTRY.items():
        if key in dataset_key or dataset_key in info["hf_path"].lower():
            dataset_info = info
            break
    if dataset_info is None:
        dataset_info = {
            "hf_path": dataset_path,
            "splits": ["train", "validation", "test"],
            "question_field": "question",
            "answer_field": "answers",
            "image_field": "image",
            "id_field": "id",
        }

    hf_path = dataset_info["hf_path"]
    available_splits = dataset_info.get("splits", ["train", "validation", "test"])
    # Use subset from registry if not explicitly provided
    if subset is None:
        subset = dataset_info.get("subset", None)

    actual_split = split
    try:
        if subset:
            ds = load_dataset(hf_path, subset, split=split)
        else:
            ds = load_dataset(hf_path, split=split)
    except Exception as e:
        print(f"[WARN] Split '{split}' not available: {e}")
        ds = None
        for fallback in available_splits:
            if fallback == split:
                continue
            try:
                if subset:
                    ds = load_dataset(hf_path, subset, split=fallback)
                else:
                    ds = load_dataset(hf_path, split=fallback)
                actual_split = fallback
                print(f"[INFO] Using fallback split: {fallback}")
                break
            except Exception:
                continue
        if ds is None:
            raise RuntimeError(f"Could not load any split from {hf_path}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    return ds, actual_split, dataset_info


def init_wandb(
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    mode: Optional[str] = None,
) -> Any:
    """Initialize W&B with proper settings."""
    import wandb

    project = project or os.environ.get("WANDB_PROJECT", "VLM-Distillation")
    mode = mode or os.environ.get("WANDB_MODE", "offline")

    run = wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        mode=mode,
        reinit=True,
    )
    return run


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_processor(
    model_id: str,
    device: str = "cpu",
    load_in_4bit: bool = False,
    use_unsloth: bool = True,
    for_training: bool = False,
) -> Tuple[Any, Any]:
    """Load model and processor with optional Unsloth/LoRA."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    tokenizer = None

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        pass

    if processor is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Could not load processor or tokenizer for {model_id}: {e}")

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    # Try Unsloth first
    if use_unsloth and UNSLOTH_AVAILABLE and for_training and device == "cuda":
        try:
            from unsloth import FastVisionModel
            model, tokenizer_or_proc = FastVisionModel.from_pretrained(
                model_id,
                load_in_4bit=load_in_4bit,
                dtype=dtype,
            )
            model = FastVisionModel.get_peft_model(
                model,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
            )
            print(f"[INFO] Loaded model with Unsloth FastVisionModel + LoRA")
            return model, processor or tokenizer_or_proc
        except Exception as e:
            print(f"[WARN] Unsloth load failed: {e}, falling back to HF")

    # Standard HF loading
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, BitsAndBytesConfig

    bnb_config = None
    if load_in_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    model = None
    model_classes = [AutoModelForVision2Seq, AutoModelForCausalLM]

    for model_cls in model_classes:
        try:
            if bnb_config:
                model = model_cls.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            else:
                model = model_cls.from_pretrained(
                    model_id,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=dtype if device == "cuda" else torch.float32,
                )
            print(f"[INFO] Loaded model with {model_cls.__name__}")
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Could not load model {model_id}")

    # Apply LoRA for training
    if for_training:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # Prepare model for training (handles gradient checkpointing compatibility)
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            # Enable input gradients for gradient checkpointing
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if device != "cuda":
        model = model.to(device)

    return model, processor or tokenizer


def load_checkpoint(checkpoint_path: str, model: Any) -> Any:
    """Load LoRA weights from checkpoint."""
    from peft import PeftModel

    if os.path.isdir(checkpoint_path):
        adapter_path = checkpoint_path
    else:
        adapter_path = os.path.dirname(checkpoint_path)

    if hasattr(model, "load_adapter"):
        model.load_adapter(adapter_path)
    else:
        model = PeftModel.from_pretrained(model, adapter_path)

    print(f"[INFO] Loaded checkpoint from {adapter_path}")
    return model
