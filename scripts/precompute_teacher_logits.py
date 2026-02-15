import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse

def prepare_sample(sample, image_folder):
    """Prepare a single sample for batching."""
    sample_id = sample['id']
    image_file = sample['image']
    conversations = sample['conversations']
    
    # Load image
    if not os.path.exists(image_file):
        image_file = os.path.join(image_folder, image_file)
    image = Image.open(image_file).convert("RGB")
    
    # Format conversation
    prompt_parts = []
    for turn_idx in range(0, len(conversations), 2):
        user_msg = conversations[turn_idx]
        assistant_msg = conversations[turn_idx + 1] if turn_idx + 1 < len(conversations) else None
        
        if user_msg['value'].startswith('<image>'):
            user_prompt = f"User:{user_msg['value']}<end_of_utterance>\nAssistant: "
        else:
            user_prompt = f"User: {user_msg['value']}<end_of_utterance>\nAssistant: "
        
        prompt_parts.append(user_prompt)
        
        if assistant_msg:
            is_last = (turn_idx + 2 >= len(conversations))
            if is_last:
                assistant_prompt = f"{assistant_msg['value']}<end_of_utterance>"
            else:
                assistant_prompt = f"{assistant_msg['value']}<end_of_utterance>\n"
            prompt_parts.append(assistant_prompt)
    
    full_prompt = "".join(prompt_parts)
    
    return {
        'sample_id': sample_id,
        'image': image,
        'prompt': full_prompt
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading teacher model: {args.teacher_model_id}")
    processor = AutoProcessor.from_pretrained(
        args.teacher_model_id,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    print(f"Loading dataset from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    logits_metadata = []
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing teacher logits"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch_samples = dataset[start_idx:end_idx]
        
        # Prepare batch
        prepared_samples = [prepare_sample(s, args.image_folder) for s in batch_samples]
        
        batch_images = [s['image'] for s in prepared_samples]
        batch_prompts = [s['prompt'] for s in prepared_samples]
        batch_ids = [s['sample_id'] for s in prepared_samples]
        
        # Process batch
        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            teacher_logits = outputs.logits
        
        # Save each sample's logits
        for i, sample_id in enumerate(batch_ids):
            logits_file = os.path.join(args.output_dir, f"logits_{sample_id}.pt")
            torch.save({
                'logits': teacher_logits[i].cpu(),
                'input_ids': inputs['input_ids'][i].cpu(),
                'attention_mask': inputs['attention_mask'][i].cpu(),
                'sample_id': sample_id
            }, logits_file)
            
            logits_metadata.append({
                'sample_id': sample_id,
                'logits_file': logits_file,
                'seq_length': teacher_logits[i].shape[0]
            })
        
        del teacher_logits, outputs, inputs
        if batch_idx % 25 == 0:
            torch.cuda.empty_cache()
    
    metadata_file = os.path.join(args.output_dir, "logits_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(logits_metadata, f, indent=2)
    
    print(f"\nSaved {len(logits_metadata)} teacher logits to {args.output_dir}")
    print(f"Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    main()