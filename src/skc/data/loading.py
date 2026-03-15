from src.dataset.vqa_loading import (
    canonical_dataset_name,
    extract_image_as_pil,
    infer_schema,
    load_dataset_split,
    load_hf_dataset,
    pick_first_text,
)


def load_probe_dataset(dataset_name: str, split: str, config: str | None):
    try:
        canonical_name = canonical_dataset_name(dataset_name)
    except ValueError:
        dataset = load_hf_dataset(dataset_name, config, split)
        loaded_from = dataset_name
    else:
        dataset, loaded_from = load_dataset_split(canonical_name, split)

    schema = infer_schema(dataset, require_answer_field=False)
    return dataset, schema, loaded_from


def build_probe_samples(dataset, schema, n: int):
    probe_samples = []
    for row in dataset:
        question = pick_first_text(row.get(schema["question_field"])) if schema["question_field"] else None
        if not question:
            continue

        try:
            image = extract_image_as_pil(row.get(schema["image_field"]))
        except Exception:
            continue

        probe_samples.append({"question": question, "image": image})
        if len(probe_samples) >= n:
            break

    if not probe_samples:
        raise ValueError("No valid probe samples found in the requested dataset split.")

    return probe_samples
