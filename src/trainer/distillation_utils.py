def build_cached_teacher_target_batches(inputs, num_teachers: int):
    prefixes = []
    if "teacher_cached_logits" in inputs:
        prefixes.append("teacher")
    prefixes.extend(f"teacher_{index}" for index in range(num_teachers) if f"teacher_{index}_cached_logits" in inputs)

    if not prefixes:
        return None

    return [
        (
            inputs[f"{prefix}_cached_logits"],
            inputs[f"{prefix}_cached_labels"],
        )
        for prefix in prefixes
    ]
