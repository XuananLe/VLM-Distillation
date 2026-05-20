from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

DEFAULT_DATASETS = ("textvqa", "docvqa", "chartqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank DEX-AR VQA samples by heatmap clarity and build a contact sheet."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing one DEX-AR output directory per dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked samples to keep per dataset in JSON.",
    )
    parser.add_argument(
        "--sheet-k",
        type=int,
        default=3,
        help="Number of samples per dataset to render into the contact sheet.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories to rank under --root.",
    )
    return parser.parse_args()


def output_dir_from_summary(remote_path: str) -> Path:
    path = Path(remote_path)
    if path.is_absolute() and len(path.parts) >= 3 and path.parts[1] == "output":
        return Path("/output", *path.parts[2:])
    return path


def zero_metrics() -> dict[str, float]:
    return {
        "score": 0.0,
        "concentration": 0.0,
        "top10_mass": 0.0,
        "area_gt_50": 1.0,
        "area_gt_70": 1.0,
        "token_signal": 0.0,
        "max_token_weight": 0.0,
    }


def score_sample(sample_dir: Path) -> dict[str, Any]:
    heatmaps = torch.load(sample_dir / "heatmaps.pt", map_location="cpu")
    heatmap = torch.nan_to_num(
        heatmaps["sentence_heatmap"].float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    if not torch.isfinite(heatmap_max - heatmap_min) or float(heatmap_max - heatmap_min) <= 1e-8:
        return zero_metrics() | {"heatmap_shape": list(heatmap.shape)}

    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    flat = heatmap.flatten()
    total = flat.sum()
    if float(total) <= 1e-8:
        return zero_metrics() | {"heatmap_shape": list(heatmap.shape)}

    probability = (flat + 1e-8) / (total + flat.numel() * 1e-8)
    entropy = -(probability * probability.log()).sum() / math.log(flat.numel())
    concentration = float(1.0 - entropy)
    top_k = max(1, int(math.ceil(flat.numel() * 0.10)))
    top10_mass = float(torch.topk(flat, k=top_k).values.sum() / total)
    area_gt_50 = float((flat > 0.50).float().mean())
    area_gt_70 = float((flat > 0.70).float().mean())

    weights = torch.nan_to_num(
        heatmaps["token_weights"].float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_weight = float(weights.max()) if weights.numel() else 0.0
    token_signal = min(1.0, math.log1p(max_weight) / math.log1p(10.0))

    score = (
        0.42 * concentration
        + 0.25 * top10_mass
        + 0.12 * max(0.0, 1.0 - area_gt_50)
        + 0.06 * max(0.0, 1.0 - area_gt_70)
        + 0.15 * token_signal
    )
    return {
        "score": float(score),
        "concentration": concentration,
        "top10_mass": top10_mass,
        "area_gt_50": area_gt_50,
        "area_gt_70": area_gt_70,
        "token_signal": token_signal,
        "max_token_weight": max_weight,
        "heatmap_shape": list(heatmap.shape),
    }


def rank_samples(root: Path, datasets: list[str]) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        summary_path = root / dataset / "summary.json"
        if not summary_path.exists():
            print(f"Skipping {dataset}: missing {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = []
        for sample in summary["samples"]:
            sample_dir = output_dir_from_summary(sample["output_dir"])
            metrics = score_sample(sample_dir)
            rows.append(
                {
                    "dataset": dataset,
                    "score": metrics["score"],
                    "metrics": metrics,
                    "row_index": sample["row_index"],
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "ground_truth_answer": sample["ground_truth_answer"],
                    "generated_answer": sample["generated_answer"],
                    "target_sentence": sample["target_sentence"],
                    "image_preparation": sample["image_preparation"],
                    "output_dir": str(sample_dir),
                    "input_image": str(sample_dir / "input_image.png"),
                    "filtered_sentence_heatmap": str(sample_dir / "filtered" / "sentence_heatmap.png"),
                    "unfiltered_sentence_heatmap": str(sample_dir / "unfiltered" / "sentence_heatmap.png"),
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        rankings[dataset] = rows
    return rankings


def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def fit_image(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def build_contact_sheet(
    selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    bold = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

    thumb_w, thumb_h = 390, 270
    label_h = 118
    row_h = label_h + thumb_h + 24
    sheet_w = 2 * thumb_w + 60
    sheet_h = len(selected) * row_h + 40
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(selected):
        y = 20 + index * row_h
        title = f"{row['dataset']} #{index % 3 + 1} | row {row['row_index']} | score {row['score']:.3f}"
        draw.text((20, y), title, fill="black", font=bold)
        label = f"Q: {row['question']}\nTarget: {row['target_sentence']}\nGT: {row['ground_truth_answer']}"
        wrapped = []
        for line in label.splitlines():
            wrapped.extend(textwrap.wrap(line, width=92) or [""])
        draw.multiline_text(
            (20, y + 26),
            "\n".join(wrapped[:5]),
            fill=(35, 35, 35),
            font=small,
            spacing=3,
        )

        image_y = y + label_h
        sheet.paste(fit_image(row["input_image"], (thumb_w, thumb_h)), (20, image_y))
        sheet.paste(
            fit_image(row["filtered_sentence_heatmap"], (thumb_w, thumb_h)),
            (40 + thumb_w, image_y),
        )
        draw.text((20, image_y + thumb_h + 4), "input", fill=(80, 80, 80), font=small)
        draw.text(
            (40 + thumb_w, image_y + thumb_h + 4),
            "filtered sentence heatmap",
            fill=(80, 80, 80),
            font=small,
        )

    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    rankings = rank_samples(args.root, args.datasets)
    analysis = {
        "selection_method": (
            "Ranked by raw filtered sentence heatmap concentration/top-10 mass, "
            "penalized diffuse maps, with a small boost for nonzero visual token relevance."
        ),
        "top_by_dataset": {dataset: rows[: args.top_k] for dataset, rows in rankings.items()},
        "all_rankings": rankings,
    }
    analysis_path = args.root / "clearest_samples.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    selected = []
    for dataset in args.datasets:
        selected.extend(rankings.get(dataset, [])[: args.sheet_k])
    sheet_path = args.root / "clearest_contact_sheet.jpg"
    if selected:
        build_contact_sheet(selected, sheet_path)
    else:
        sheet_path = None

    print(
        json.dumps(
            {
                "analysis_path": str(analysis_path),
                "sheet_path": str(sheet_path) if sheet_path is not None else None,
                "selected": [
                    {
                        "dataset": row["dataset"],
                        "row_index": row["row_index"],
                        "score": round(row["score"], 4),
                        "target": row["target_sentence"],
                    }
                    for row in selected
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
