"""Run DEX-AR on a small DocVQA subset.

This script loads a supported VLM once, samples a small number of valid
DocVQA examples, optionally generates the model answer for each question,
then computes and saves filtered/unfiltered DEX-AR visualizations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dexar import DexarWrapper, visualize, visualize_multi
from src.dataset.vqa_loading import (
    canonical_dataset_name,
    extract_image_as_pil,
    infer_schema,
    load_dataset_split,
    pick_first_text,
)


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEX-AR on a small DocVQA subset.")
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Model to load with DexarWrapper.",
    )
    parser.add_argument(
        "--dataset",
        default="docvqa",
        help="Dataset alias/name understood by src.dataset.vqa_loading.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Number of successful samples to process.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting row offset in the dataset.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Starting DEX-AR layer index.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Model device. Use auto/cuda/cpu or an explicit device id.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum generated answer length when target-mode=generated.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("generated", "ground_truth"),
        default="generated",
        help="Whether to explain the model answer or the first ground-truth answer.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("vqa", "paper"),
        default="vqa",
        help="Use dataset-question VQA prompts or the paper caption/classification prompts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./docvqa_subset"),
        help="Directory to write per-sample outputs and summary.json.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=30,
        help="Abort after this many consecutive processable rows fail.",
    )
    return parser.parse_args()


def sanitize_for_filename(raw_value: Any) -> str:
    sanitized = SAFE_FILENAME_RE.sub("_", str(raw_value)).strip("._")
    return sanitized or "sample"


def pick_first_answer(answer_value: Any) -> str | None:
    values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
    return next((text for text in (pick_first_text(value) for value in values) if text), None)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_vqa_prompt(
    dataset_name: str,
    model_family: str,
    question: str,
    prompt_style: str = "vqa",
) -> str:
    if prompt_style == "paper":
        if model_family == "llava":
            return "USER: <image>\nClassify the image.\nASSISTANT:"
        if model_family == "paligemma":
            return "cap en\n"
        if model_family == "florence2":
            return "<DETAILED_CAPTION>"

    if model_family == "paligemma":
        return f"answer en {question}\n"
    if model_family == "florence2":
        return question.strip()
    if model_family == "smolvlm":
        if dataset_name == "chartqa":
            return (
                "<|im_start|>User:<image>For the question below, follow the following instructions:\n"
                "-The answer should contain as few words as possible.\n"
                "-Don’t paraphrase or reformat the text you see in the image.\n"
                "-Answer a binary question with Yes or No.\n"
                "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                "-Don’t include any units in the answer.\n"
                "-Do not include any full stops at the end of the answer.\n"
                "-Try to include the full label from the graph when asked about an entity.\n"
                f"Question: {question}<end_of_utterance>\nAssistant:"
            )
        if dataset_name == "docvqa":
            return (
                "<|im_start|>User:<image>Give a short and terse answer to the following question. "
                "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
                f"Just give the answer without additional explanation. Question: {question}"
                "<end_of_utterance>\nAssistant:"
            )
        if dataset_name == "textvqa":
            return (
                "<|im_start|>User:<image>Answer the following question about the image using as few words as possible. "
                "Follow these additional instructions:\n"
                "-Always answer a binary question with Yes or No.\n"
                "-When asked what time it is, reply with the time seen in the image.\n"
                "-Do not put any full stops at the end of the answer.\n"
                "-Do not put quotation marks around the answer.\n"
                "-An answer with one or two words is favorable.\n"
                "-Do not apply common sense knowledge. The answer can be found in the image.\n"
                f"Question: {question}<end_of_utterance>\nAssistant:"
            )
        instruction = (
            "Answer the question using evidence from the image. Keep the answer short.\n"
            f"Question: {question}"
        )
        return (
            "<|im_start|>User:<image>\n"
            f"{instruction}\n"
            "<end_of_utterance>\nAssistant:"
        )
    if dataset_name == "docvqa":
        instruction = (
            "Answer the question using a short text span from the document.\n"
            f"Question: {question}"
        )
    elif dataset_name == "textvqa":
        instruction = (
            "Answer the question using the text visible in the image when possible. "
            "Keep the answer short.\n"
            f"Question: {question}"
        )
    elif dataset_name == "chartqa":
        instruction = (
            "Answer the question using the chart. Return a short answer, number or phrase.\n"
            f"Question: {question}"
        )
    else:
        instruction = (
            "Answer the question using evidence from the image. Keep the answer short.\n"
            f"Question: {question}"
        )
    if model_family == "llava":
        return f"USER: <image>\n{instruction}\nASSISTANT:"
    if model_family == "internvl":
        return f"<image>\n{instruction}"
    return f"<image>\n{instruction}\nAnswer:"


def candidate_images(model: DexarWrapper, image: Image.Image):
    rgb_image = image.convert("RGB")
    yielded: set[tuple[str, tuple[int, int]]] = set()

    for prepared_name, prepared_image in (("original", rgb_image),):
        yielded_key = (prepared_name, prepared_image.size)
        if yielded_key not in yielded:
            yielded.add(yielded_key)
            yield prepared_name, prepared_image

    contained_sizes = [model.recommended_image_size]
    if model.model_family == "qwen2vl":
        contained_sizes.extend([384, 336, 280, 224, 168])

    for contained_size in contained_sizes:
        contained = ImageOps.contain(
            rgb_image,
            (contained_size, contained_size),
        )
        contained_key = (f"contained_{contained_size}", contained.size)
        if contained_key not in yielded:
            yielded.add(contained_key)
            yield contained_key[0], contained

    if model.model_family != "qwen2vl":
        square = rgb_image.resize(
            (model.recommended_image_size, model.recommended_image_size)
        )
        square_key = (f"square_{model.recommended_image_size}", square.size)
        if square_key not in yielded:
            yield square_key[0], square


def cleanup_cuda(model: DexarWrapper) -> None:
    model.model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_answer(
    model: DexarWrapper,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    device = next(model.model.parameters()).device
    custom_generate_answer = getattr(model.backend, "custom_generate_answer", None)
    if custom_generate_answer is not None:
        return custom_generate_answer(
            model.backend,
            image,
            prompt,
            max_new_tokens,
            device,
        )

    encoded_prompt = model.backend.encode_prompt(prompt=prompt, image=image, device=device)
    generation_inputs = dict(encoded_prompt.model_inputs)

    config = model.model.config
    old_output_attentions = getattr(config, "output_attentions", None)
    old_output_hidden_states = getattr(config, "output_hidden_states", None)
    if hasattr(config, "output_attentions"):
        config.output_attentions = False
    if hasattr(config, "output_hidden_states"):
        config.output_hidden_states = False

    try:
        with torch.inference_mode():
            generated_ids = model.model.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        if hasattr(config, "output_attentions"):
            config.output_attentions = old_output_attentions
        if hasattr(config, "output_hidden_states"):
            config.output_hidden_states = old_output_hidden_states

    prompt_length = generation_inputs["input_ids"].shape[1]
    new_token_ids = generated_ids[:, prompt_length:]
    generated_text = model.processor.tokenizer.decode(
        new_token_ids[0],
        skip_special_tokens=True,
    ).strip()
    return generated_text


def save_visualizations(
    *,
    image: Image.Image,
    result,
    output_dir: Path,
) -> None:
    filtered_dir = output_dir / "filtered"
    unfiltered_dir = output_dir / "unfiltered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    unfiltered_dir.mkdir(parents=True, exist_ok=True)

    visualize(
        image=image,
        heatmap=result.sentence_heatmap,
        title="Sentence heatmap (filtered)",
        save_path=str(filtered_dir / "sentence_heatmap.png"),
    )

    visualize_multi(
        image=image,
        heatmaps=result.per_token_heatmaps,
        tokens=result.tokens,
        save_path=str(filtered_dir / "per_token_heatmaps.png"),
    )

    visualize(
        image=image,
        heatmap=result.sentence_heatmap_unfiltered,
        title="Sentence heatmap (unfiltered)",
        save_path=str(unfiltered_dir / "sentence_heatmap.png"),
    )

    visualize_multi(
        image=image,
        heatmaps=result.per_token_heatmaps_unfiltered,
        tokens=result.tokens,
        save_path=str(unfiltered_dir / "per_token_heatmaps.png"),
    )

    torch.save(
        {
            "per_token_heatmaps": result.per_token_heatmaps.detach().cpu(),
            "per_token_heatmaps_unfiltered": result.per_token_heatmaps_unfiltered.detach().cpu(),
            "token_weights": result.token_weights.detach().cpu(),
            "sentence_heatmap": result.sentence_heatmap.detach().cpu(),
            "sentence_heatmap_unfiltered": result.sentence_heatmap_unfiltered.detach().cpu(),
            "tokens": result.tokens,
        },
        output_dir / "heatmaps.pt",
    )


def main() -> None:
    args = parse_args()
    dataset_name = canonical_dataset_name(args.dataset)
    device = resolve_device(args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(
        {
            "model_name": args.model_name,
            "dataset": dataset_name,
            "split": args.split,
            "subset_size": args.subset_size,
            "offset": args.offset,
            "layer_index": args.layer_index,
            "device": device,
            "target_mode": args.target_mode,
            "output_root": str(args.output_root),
        }
    )

    model = DexarWrapper.from_pretrained(
        args.model_name,
        device=device,
        layer_index=args.layer_index,
    )
    dataset, loaded_from = load_dataset_split(dataset_name, args.split, log_fallback=True)
    schema = infer_schema(dataset_name, dataset, require_answer_field=True)

    print(f"Loaded dataset from {loaded_from} with {len(dataset)} rows.")
    print(f"Fixed schema: {schema}")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    consecutive_failures = 0

    for row_index in range(args.offset, len(dataset)):
        if len(successes) >= args.subset_size:
            break
        if (
            args.max_consecutive_failures > 0
            and consecutive_failures >= args.max_consecutive_failures
        ):
            print(
                "Stopping early after "
                f"{consecutive_failures} consecutive failed processable rows."
            )
            break

        sample = dataset[row_index]
        question = pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
        ground_truth = pick_first_answer(sample.get(schema["answer_field"])) if schema["answer_field"] else None
        if not question or not ground_truth:
            continue

        sample_id = sample.get(schema["id_field"]) if schema["id_field"] else row_index
        output_dir = args.output_root / (
            f"sample_{len(successes):02d}_row_{row_index:06d}_{sanitize_for_filename(sample_id)}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_vqa_prompt(
            dataset_name,
            model.model_family,
            question,
            prompt_style=args.prompt_style,
        )

        print(
            f"[sample {len(successes) + 1}/{args.subset_size}] "
            f"row={row_index} id={sample_id} question={question!r}"
        )

        try:
            source_image = extract_image_as_pil(sample.get(schema["image_field"]))
            source_image.save(output_dir / "original_image.png")

            last_error: Exception | None = None
            for image_prep, prepared_image in candidate_images(model, source_image):
                try:
                    generated_answer = None
                    if args.target_mode == "generated":
                        generated_answer = generate_answer(
                            model,
                            prepared_image,
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                        )

                    target_sentence = generated_answer or ground_truth
                    if not target_sentence:
                        raise ValueError("Target sentence is empty after generation fallback.")

                    result = model.compute_dexar(
                        image=prepared_image,
                        target_sentence=target_sentence,
                        prompt=prompt,
                    )

                    prepared_image.save(output_dir / "input_image.png")
                    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    save_visualizations(
                        image=prepared_image,
                        result=result,
                        output_dir=output_dir,
                    )

                    metadata = {
                        "dataset": dataset_name,
                        "loaded_from": loaded_from,
                        "split": args.split,
                        "row_index": row_index,
                        "sample_id": str(sample_id),
                        "question": question,
                        "ground_truth_answer": ground_truth,
                        "generated_answer": generated_answer,
                        "target_sentence": target_sentence,
                        "target_mode": args.target_mode,
                        "prompt_style": args.prompt_style,
                        "image_preparation": image_prep,
                        "original_image_size": list(source_image.size),
                        "input_image_size": list(prepared_image.size),
                        "tokens": result.tokens,
                        "token_weights": result.token_weights.detach().cpu().tolist(),
                        "model_name": args.model_name,
                        "model_family": model.model_family,
                        "layer_index": args.layer_index,
                    }
                    (output_dir / "metadata.json").write_text(
                        json.dumps(metadata, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    successes.append(metadata | {"output_dir": str(output_dir)})
                    consecutive_failures = 0
                    print(
                        f"  saved -> {output_dir} "
                        f"(target={target_sentence!r}, prep={image_prep})"
                    )

                    del result
                    cleanup_cuda(model)
                    break
                except (OSError, RuntimeError, TypeError, ValueError) as sample_exc:  # noqa: PERF203
                    last_error = sample_exc
                    print(f"  retry after {image_prep} failed: {sample_exc}")
                    cleanup_cuda(model)
            else:
                assert last_error is not None
                raise last_error
        except (OSError, RuntimeError, TypeError, ValueError) as exc:  # noqa: PERF203
            failure = {
                "row_index": row_index,
                "sample_id": str(sample_id),
                "question": question,
                "error": str(exc),
            }
            failures.append(failure)
            consecutive_failures += 1
            print(f"  failed -> {failure}")

    summary = {
        "model_name": args.model_name,
        "dataset": dataset_name,
        "loaded_from": loaded_from,
        "split": args.split,
        "subset_size_requested": args.subset_size,
        "subset_size_completed": len(successes),
        "offset": args.offset,
        "layer_index": args.layer_index,
        "device": device,
        "target_mode": args.target_mode,
        "prompt_style": args.prompt_style,
        "output_root": str(args.output_root),
        "samples": successes,
        "failures": failures,
        "consecutive_failures_at_end": consecutive_failures,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if len(successes) < args.subset_size:
        raise RuntimeError(
            f"Completed only {len(successes)} successful samples out of "
            f"{args.subset_size}. See {args.output_root / 'summary.json'}."
        )

    print(f"Wrote {len(successes)} samples to {args.output_root}")


if __name__ == "__main__":
    main()
