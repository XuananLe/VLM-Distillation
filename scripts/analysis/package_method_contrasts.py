from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_METHODS = (
    (
        "forward_kl",
        "Forward KL",
        "__tmp__dexar_smolvlm500m_methods_docvqa_ckpts__forward_kl",
    ),
    (
        "finetuning",
        "Finetuning",
        "__tmp__dexar_smolvlm500m_methods_docvqa_ckpts__finetuning",
    ),
    (
        "reverse_kl",
        "Reverse KL",
        "__tmp__dexar_smolvlm500m_methods_docvqa_ckpts__reverse_kl",
    ),
    (
        "uld",
        "ULD",
        "__tmp__dexar_smolvlm500m_methods_docvqa_ckpts__uld",
    ),
)

DEFAULT_LFM_LABEL = "LFM2.5-VL-450M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and package same-row method contrasts from DEX-AR VQA outputs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--dataset", default="docvqa")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--lfm-root",
        type=Path,
        default=None,
        help="Optional DEX-AR suite root for LFM, e.g. /output/dexar_lfm25_vqa30.",
    )
    parser.add_argument(
        "--title",
        default="DEX-AR Method Contrasts on DocVQA (SmolVLM-500M latest checkpoints)",
    )
    return parser.parse_args()


def normalize_answer(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def markdown_cell(value: object, limit: int | None = None) -> str:
    text = str(value or "").replace("|", "\\|").replace("\n", " ")
    if limit is not None:
        return text[:limit]
    return text


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


def build_methods(root: Path, lfm_root: Path | None) -> list[tuple[str, str, Path]]:
    methods = [
        (key, label, root / slug)
        for key, label, slug in DEFAULT_METHODS
    ]
    if lfm_root is not None:
        methods.append(("lfm25", DEFAULT_LFM_LABEL, lfm_root))
    return methods


def collect_contrasts(
    methods: list[tuple[str, str, Path]],
    dataset: str,
) -> list[dict[str, object]]:
    by_row: dict[int, dict[str, dict[str, object]]] = {}
    for key, _label, method_root in methods:
        analysis_path = method_root / "clearest_samples.json"
        data = json.loads(analysis_path.read_text(encoding="utf-8"))
        for row in data["all_rankings"][dataset]:
            by_row.setdefault(int(row["row_index"]), {})[key] = row

    contrast_rows: list[dict[str, object]] = []
    for row_index, payload in sorted(by_row.items()):
        if len(payload) != len(methods):
            continue

        answers = {
            key: normalize_answer(payload[key].get("generated_answer"))
            for key, _label, _method_root in methods
        }
        scores = {
            key: float(payload[key].get("score") or 0.0)
            for key, _label, _method_root in methods
        }
        top10 = {
            key: float(payload[key].get("metrics", {}).get("top10_mass") or 0.0)
            for key, _label, _method_root in methods
        }
        nonzero_methods = sum(1 for value in scores.values() if value > 1e-6)
        unique_answers = len(set(answers.values()))
        score_spread = max(scores.values()) - min(scores.values())
        top10_spread = max(top10.values()) - min(top10.values())
        contrast_score = (
            unique_answers * 2.0
            + score_spread
            + top10_spread
            + nonzero_methods * 0.05
        )

        exemplar = next(iter(payload.values()))
        contrast_rows.append(
            {
                "row_index": row_index,
                "sample_id": exemplar.get("sample_id"),
                "question": exemplar.get("question"),
                "ground_truth_answer": exemplar.get("ground_truth_answer"),
                "unique_answers": unique_answers,
                "nonzero_methods": nonzero_methods,
                "score_spread": score_spread,
                "top10_spread": top10_spread,
                "contrast_score": contrast_score,
                "methods": {
                    key: {
                        "label": label,
                        "generated_answer": payload[key].get("generated_answer"),
                        "score": float(payload[key].get("score") or 0.0),
                        "top10_mass": float(
                            payload[key].get("metrics", {}).get("top10_mass") or 0.0
                        ),
                        "input_image": payload[key].get("input_image"),
                        "filtered_sentence_heatmap": payload[key].get(
                            "filtered_sentence_heatmap"
                        ),
                        "output_dir": payload[key].get("output_dir"),
                    }
                    for key, label, _method_root in methods
                },
            }
        )

    contrast_rows.sort(key=lambda row: float(row["contrast_score"]), reverse=True)
    return contrast_rows


def write_markdown(
    rows: list[dict[str, object]],
    output_path: Path,
    method_specs: list[tuple[str, str, Path]],
) -> None:
    method_headers = " | ".join(label for _key, label, _root in method_specs)
    lines = [
        "# DEX-AR SmolVLM-500M DocVQA Method Contrasts",
        "",
        f"| Row | Contrast | Spread | Question | {method_headers} |",
        "|---:|---:|---:|---|" + "|".join("---" for _ in method_specs) + "|",
    ]
    for row in rows:
        values = []
        method_payloads = row["methods"]
        assert isinstance(method_payloads, dict)
        for key, _label, _method_root in method_specs:
            item = method_payloads[key]
            values.append(
                markdown_cell(
                    f"{item['generated_answer']} ({float(item['score']):.3f})"
                )
            )
        lines.append(
            f"| {row['row_index']} | {float(row['contrast_score']):.3f} | "
            f"{float(row['score_spread']):.3f} | "
            f"{markdown_cell(row['question'], 90)} | "
            + " | ".join(values)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
    method_specs: list[tuple[str, str, Path]],
) -> None:
    title_font = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    font = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    font_small = load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

    cell_w, img_h = 260, 170
    left_w = 260
    header_h = 44
    row_label_h = 104
    row_h = row_label_h + img_h + 20
    sheet_w = left_w + len(method_specs) * cell_w + 30
    sheet_h = header_h + max(1, len(rows)) * row_h + 20
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill="black", font=title_font)

    for index, (_key, label, _method_root) in enumerate(method_specs):
        x = left_w + index * cell_w + 10
        draw.text((x, header_h - 24), label, fill="black", font=title_font)

    for row_index, row in enumerate(rows):
        y = header_h + row_index * row_h
        q_lines = textwrap.wrap(
            f"row {row['row_index']} | spread {float(row['score_spread']):.3f} | "
            f"{row['question']}",
            width=38,
        )
        gt_lines = textwrap.wrap(f"GT: {row.get('ground_truth_answer') or ''}", width=38)
        draw.multiline_text(
            (12, y + 8),
            "\n".join((q_lines + gt_lines)[:5]),
            fill=(20, 20, 20),
            font=font,
            spacing=2,
        )

        method_payloads = row["methods"]
        assert isinstance(method_payloads, dict)
        for method_index, (key, _label, _method_root) in enumerate(method_specs):
            item = method_payloads[key]
            x = left_w + method_index * cell_w + 10
            text = f"score {float(item['score']):.3f}\n{item['generated_answer'] or ''}"
            wrapped = []
            for line in text.splitlines():
                wrapped.extend(textwrap.wrap(line, width=30) or [""])
            draw.multiline_text(
                (x, y + 8),
                "\n".join(wrapped[:4]),
                fill=(20, 20, 20),
                font=font_small,
                spacing=1,
            )
            sheet.paste(
                fit_image(str(item["filtered_sentence_heatmap"]), (cell_w - 20, img_h)),
                (x, y + row_label_h),
            )

    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.report_root.exists():
        shutil.rmtree(args.report_root)
    args.report_root.mkdir(parents=True)

    methods = build_methods(args.root, args.lfm_root)
    selected_rows = collect_contrasts(methods, args.dataset)[: args.top_k]
    contrast_json = args.report_root / "contrast_samples.json"
    contrast_md = args.report_root / "contrast_samples.md"
    contact_sheet = args.report_root / "contrast_contact_sheet.jpg"

    contrast_json.write_text(
        json.dumps(selected_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(selected_rows, contrast_md, methods)
    write_contact_sheet(selected_rows, contact_sheet, args.title, methods)

    print(
        json.dumps(
            {
                "report": str(args.report_root),
                "contrast_json": str(contrast_json),
                "contrast_md": str(contrast_md),
                "contact_sheet": str(contact_sheet),
                "selected_rows": [row["row_index"] for row in selected_rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
