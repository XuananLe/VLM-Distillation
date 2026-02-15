import torch
from transformers import PreTrainedModel
from transformers import AutoModelForImageTextToText
import accelerate

def load_model(
    model_id: str,
) -> PreTrainedModel:
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model: {model_id}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flash_attn_available = accelerate.utils.import_utils.is_flash_attention_2_available()
    
    if not flash_attn_available:
        print("Warning: Flash Attention is not available. Using eager attention implementation which may be slower.")
    if device != 'cuda':
        print("Warning: CUDA is not available. The model will be loaded on CPU which may be very slow.")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch_dtype,
        _attn_implementation="flash_attention_2" if flash_attn_available else "eager",
        device_map=device,
    )

    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model
