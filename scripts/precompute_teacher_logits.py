import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse

def prepare_sample(sample, image_folder):
    sample_id = sample['id']
    image_file = sample['image']
    conversations = sample['conversations']
    
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
    parser.add_argument("--teacher_model_id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="/workspace/VLM-Distillation/data/textvqa/train_llava.json")
    parser.add_argument("--image_folder", type=str,
                        default="/workspace/VLM-Distillation/data/textvqa/images")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/VLM-Distillation/data/textvqa/teacher_logits")
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
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_id,
        # always load in bfloat16
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    
    print(f"Loading dataset from: {args.data_path}")
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset)}")

    logits_metadata = []

    for idx, sample in enumerate(tqdm(dataset, desc="Computing teacher logits")):
        # Prepare single sample
        prepared = prepare_sample(sample, args.image_folder)

        # Process single sample
        inputs = processor(
            text=prepared['prompt'],
            images=prepared['image'],
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            teacher_logits = outputs.logits[0]

        # Save logits
        logits_file = os.path.join(args.output_dir, f"logits_{prepared['sample_id']}.pt")
        torch.save({
            'logits': teacher_logits.cpu(),
            'input_ids': inputs['input_ids'][0].cpu(),
            'attention_mask': inputs['attention_mask'][0].cpu(),
            'sample_id': prepared['sample_id']
        }, logits_file)

        logits_metadata.append({
            'sample_id': prepared['sample_id'],
            'logits_file': logits_file,
            'seq_length': teacher_logits.shape[0]
        })

        del teacher_logits, outputs, inputs
        if idx % 100 == 0:
            torch.cuda.empty_cache()
    
    print(f"Teacher {args.teacher_model_id} logits computation completed.")

    metadata_file = os.path.join(args.output_dir, "logits_metadata.json")

    with open(metadata_file, 'w') as f:
        json.dump(logits_metadata, f, indent=2)
    
    print(f"Saved {len(logits_metadata)} teacher logits to {args.output_dir}")
    print(f"Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    main()