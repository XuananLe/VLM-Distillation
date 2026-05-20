from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a VQA attribution suite report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--title", default="VQA Attribution Suite Top Samples")
    parser.add_argument("--heatmap-label", default="heatmap")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    root = args.root
    report = args.report_root
    report.mkdir(parents=True, exist_ok=True)
    (report / "contact_sheets").mkdir(exist_ok=True)

    for name in ("suite_summary.json", "suite_clearest_samples.json"):
        src = root / name
        if src.exists():
            shutil.copy2(src, report / name)

    analysis = json.loads((root / "suite_clearest_samples.json").read_text(encoding="utf-8"))
    summary_rows = []
    contact_sheet_count = 0
    for model_slug, payload in analysis.get("top_by_model_dataset", {}).items():
        model_name = payload.get("model_name", model_slug)
        safe_slug = model_slug.replace("/", "__")
        model_root = root / model_slug
        for name in ("clearest_samples.json", "clearest_contact_sheet.jpg"):
            src = model_root / name
            if src.exists():
                dst = report / "contact_sheets" / f"{safe_slug}_{name}"
                shutil.copy2(src, dst)
                if name.endswith(".jpg"):
                    contact_sheet_count += 1
        for dataset, rows in payload.get("top_by_dataset", {}).items():
            if not rows:
                continue
            row = rows[0]
            summary_rows.append(
                {
                    "model_name": model_name,
                    "model_slug": model_slug,
                    "dataset": dataset,
                    "row_index": row.get("row_index"),
                    "sample_id": row.get("sample_id"),
                    "score": row.get("score"),
                    "question": row.get("question"),
                    "ground_truth_answer": row.get("ground_truth_answer"),
                    "generated_answer": row.get("generated_answer"),
                    "input_image": row.get("input_image"),
                    "filtered_sentence_heatmap": row.get("filtered_sentence_heatmap"),
                }
            )

    summary_rows.sort(key=lambda item: (item["model_name"], item["dataset"]))
    top1_json = report / "top1_by_model_dataset.json"
    top1_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# {args.title}",
        "",
        "| Model | Dataset | Row | Score | Generated | Question |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        question = str(row.get("question") or "").replace("|", "\\|")
        generated = str(row.get("generated_answer") or "").replace("|", "\\|").replace("\n", " ")
        md_lines.append(
            f"| {row['model_name']} | {row['dataset']} | {row['row_index']} | "
            f"{float(row['score'] or 0):.4f} | {generated[:80]} | {question[:100]} |"
        )
    top1_md = report / "top1_by_model_dataset.md"
    top1_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    title_font = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    small_font = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)

    thumb_w, thumb_h = 330, 220
    label_h = 104
    row_h = label_h + thumb_h + 18
    sheet_w = 2 * thumb_w + 60
    sheet_h = max(1, len(summary_rows)) * row_h + 30
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(summary_rows):
        y = 15 + idx * row_h
        title = (
            f"{row['model_name']} | {row['dataset']} | row {row['row_index']} | "
            f"score {float(row['score'] or 0):.3f}"
        )
        draw.text((18, y), title[:110], fill="black", font=title_font)
        label = (
            f"Q: {row.get('question') or ''}\n"
            f"Gen: {row.get('generated_answer') or ''}\n"
            f"GT: {row.get('ground_truth_answer') or ''}"
        )
        wrapped = []
        for line in label.splitlines():
            wrapped.extend(textwrap.wrap(line, width=92) or [""])
        draw.multiline_text(
            (18, y + 24),
            "\n".join(wrapped[:5]),
            fill=(35, 35, 35),
            font=small_font,
            spacing=2,
        )
        image_y = y + label_h
        sheet.paste(fit_image(row["input_image"], (thumb_w, thumb_h)), (18, image_y))
        sheet.paste(
            fit_image(row["filtered_sentence_heatmap"], (thumb_w, thumb_h)),
            (36 + thumb_w, image_y),
        )
        draw.text((18, image_y + thumb_h + 2), "input", fill=(80, 80, 80), font=small_font)
        draw.text(
            (36 + thumb_w, image_y + thumb_h + 2),
            args.heatmap_label,
            fill=(80, 80, 80),
            font=small_font,
        )

    overview = report / "overview_contact_sheet.jpg"
    sheet.save(overview, quality=92)
    print(
        json.dumps(
            {
                "report_dir": str(report),
                "top1_json": str(top1_json),
                "top1_md": str(top1_md),
                "overview_contact_sheet": str(overview),
                "contact_sheets": contact_sheet_count,
                "rows": len(summary_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
