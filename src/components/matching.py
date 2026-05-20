import json
from pathlib import Path

import numpy as np


def load_cka_json(json_path: str):
    json_path = Path(json_path)

    with open(json_path, "r") as file_handle:
        cka_payload = json.load(file_handle)

    cka_matrix = np.asarray(cka_payload["cka_matrix"], dtype=np.float64)

    model_a_layers = [item["layer_name"] for item in cka_payload["model_a"]["selected_layers"]]
    model_b_layers = [item["layer_name"] for item in cka_payload["model_b"]["selected_layers"]]

    return {
        "raw": cka_payload,
        "cka_matrix": cka_matrix,
        "model_a_name": cka_payload["model_a"]["model_name"],
        "model_b_name": cka_payload["model_b"]["model_name"],
        "model_a_layers": model_a_layers,
        "model_b_layers": model_b_layers,
        "model_a_selected_layer_indices": cka_payload["model_a"]["selected_layer_indices"],
        "model_b_selected_layer_indices": cka_payload["model_b"]["selected_layer_indices"],
    }


def resolve_student_teacher_similarity(
    payload,
    student_key: str = "model_b",
    teacher_key: str = "model_a",
):
    cka = payload["cka_matrix"]
    row_key = "model_a"
    col_key = "model_b"

    if student_key == row_key and teacher_key == col_key:
        return {
            "sim": cka,
            "student_name": payload["model_a_name"],
            "teacher_name": payload["model_b_name"],
            "student_layers": payload["model_a_layers"],
            "teacher_layers": payload["model_b_layers"],
            "student_selected_layer_indices": payload["model_a_selected_layer_indices"],
            "teacher_selected_layer_indices": payload["model_b_selected_layer_indices"],
        }

    if student_key == col_key and teacher_key == row_key:
        # Stored CKA matrix is model_a x model_b; transpose when student/teacher are swapped.
        return {
            "sim": cka.T,
            "student_name": payload["model_b_name"],
            "teacher_name": payload["model_a_name"],
            "student_layers": payload["model_b_layers"],
            "teacher_layers": payload["model_a_layers"],
            "student_selected_layer_indices": payload["model_b_selected_layer_indices"],
            "teacher_selected_layer_indices": payload["model_a_selected_layer_indices"],
        }

    raise ValueError(
        f"Unsupported keys: student_key={student_key}, teacher_key={teacher_key}. Use 'model_a' or 'model_b'."
    )


def topk_soft_match_student_teacher(
    json_path: str,
    student_key: str = "model_b",
    teacher_key: str = "model_a",
    topk: int = 3,
):
    """Keep the top-k teacher layers per student layer and normalize them into soft weights."""
    payload = load_cka_json(json_path)
    resolved = resolve_student_teacher_similarity(
        payload,
        student_key=student_key,
        teacher_key=teacher_key,
    )
    sim = resolved["sim"]
    student_name = resolved["student_name"]
    teacher_name = resolved["teacher_name"]
    student_layers = resolved["student_layers"]
    teacher_layers = resolved["teacher_layers"]
    student_selected_layer_indices = resolved["student_selected_layer_indices"]
    teacher_selected_layer_indices = resolved["teacher_selected_layer_indices"]

    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}.")

    matches = []
    mean_top1_cka = []
    k = min(topk, sim.shape[1])

    for student_pos, row in enumerate(sim):
        # Each student layer keeps the top-k teacher layers and normalizes their
        # CKA scores into a soft mixture over teachers.
        topk_unsorted = np.argpartition(row, -k)[-k:]
        teacher_positions = topk_unsorted[np.argsort(row[topk_unsorted])[::-1]]
        teacher_scores = np.asarray(row[teacher_positions], dtype=np.float64)
        teacher_weights = teacher_scores / np.clip(teacher_scores.sum(), a_min=1e-12, a_max=None)
        mean_top1_cka.append(float(teacher_scores[0]))
        matches.append(
            {
                "student_layer_index": int(student_selected_layer_indices[student_pos]),
                "student_layer_name": student_layers[student_pos],
                "teacher_layer_indices": [
                    int(teacher_selected_layer_indices[pos]) for pos in teacher_positions.tolist()
                ],
                "teacher_layer_names": [teacher_layers[pos] for pos in teacher_positions.tolist()],
                "teacher_layer_weights": teacher_weights.tolist(),
                "teacher_layer_ckas": teacher_scores.tolist(),
            }
        )

    summary = {
        "student_model": student_name,
        "teacher_model": teacher_name,
        "num_student_layers": len(student_layers),
        "num_teacher_layers": len(teacher_layers),
        "topk": k,
        "mean_top1_cka": float(np.mean(mean_top1_cka)) if mean_top1_cka else None,
    }
    return matches, summary
