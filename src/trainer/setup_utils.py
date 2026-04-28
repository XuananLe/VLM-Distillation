from src.components.reinforced_teacher_selection import ReinforcedTeacherSelectionPolicy


def normalize_teacher_models(teacher_model, teacher_count: int | None):
    """Normalize teacher input into a frozen teacher-model list and resolved count.

    Input: a teacher model, list/tuple of teacher models, or None, plus optional
    teacher_count. Output: (teacher_models, teacher_count). Exists so the trainer
    can accept one or many live teachers through one code path.
    """
    if teacher_model is None:
        teacher_models = []
    else:
        teacher_models = (
            list(teacher_model)
            if isinstance(teacher_model, (list, tuple))
            else [teacher_model]
        )

    if teacher_count is None:
        teacher_count = len(teacher_models)
    if teacher_count < 1:
        raise ValueError("DistillationTrainer requires at least one teacher.")
    if teacher_models and len(teacher_models) != teacher_count:
        raise ValueError(
            "teacher_count must match the number of teacher models when both are provided."
        )

    # Live teachers are inference-only in the trainer: keep them in eval mode and
    # freeze gradients so only the student path participates in optimization.
    for model in teacher_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return teacher_models, int(teacher_count)


def maybe_create_reinforced_teacher_selector(
    *,
    model,
    num_teachers: int,
    teacher_weighting_strategy: str,
):
    """Build the REINFORCE selector only for reinforced-selection runs.

    Input: student model, teacher count, and weighting strategy. Output:
    ReinforcedTeacherSelectionPolicy or None. Exists to keep optional policy
    setup out of the main trainer flow.
    """
    if num_teachers <= 1 or teacher_weighting_strategy != "reinforced_selection":
        return None

    # The reinforced selector follows the same pattern as the gate: register once
    # on the student and let the training step query it when that strategy is active.
    selector = ReinforcedTeacherSelectionPolicy(
        model,
        num_teachers,
    )
    model.reinforced_teacher_selector = selector
    return selector


def log_distillation_trainer_setup(
    *,
    num_teachers: int,
    teacher_weighting_strategy: str,
    loss_function: str,
    layer_distillation_enabled: bool,
    layer_distill_source: str,
    layer_distill_weight: float,
    layer_match_json_path: str | None,
    layer_match_topk: int,
    student_layer_indices: list[int],
    teacher_layer_soft_matches: list[list[dict]],
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
    alpha: float,
    teacher_gate,
    teacher_gate_top_k: int,
    teacher_gate_temperature: float,
    teacher_gate_entropy_alpha: float,
    teacher_gate_router_z_loss_alpha: float,
    teacher_gate_hard_routing_warmup_ratio: float,
    grace_threshold: float,
    grace_warmup_ratio: float,
    grace_epsilon: float,
    grace_softmax_beta: float,
    grace_router_blend_lambda: float,
    grace_ema_decay: float,
    reinforced_selection_warmup_ratio: float,
    reinforced_selection_reward_type: str,
    reinforced_selection_reward_ema_decay: float,
    reinforced_selection_policy_alpha: float,
    trie_wasserstein_rho: float,
    trie_wasserstein_topk: int,
    trie_tail_depth: int,
    trie_tail_weight: float,
) -> None:
    """Print the trainer-side distillation config summary.

    Input: resolved trainer knobs after model/layer setup. Output: None.
    Exists so one place reports which optional distillation path is actually
    active after constructor normalization.
    """
    print("Distillation Trainer initialized:")
    print(f"  - Teachers: {num_teachers}")
    if num_teachers > 1 and teacher_weighting_strategy == "routing":
        print("  - Teacher weighting: learned deep gate + GRACE routing")
    elif num_teachers > 1 and teacher_weighting_strategy == "reinforced_selection":
        print("  - Teacher weighting: reinforced teacher selection")
    elif num_teachers == 1:
        print("  - Teacher weighting: single teacher")
    else:
        print("  - Teacher weighting: uniform mean")
    print(f"  - Loss function: {loss_function}")
    if loss_function == "trie_wasserstein_loss":
        print(f"  - Trie Wasserstein rho: {trie_wasserstein_rho}")
        print(f"  - Trie Wasserstein top-k: {trie_wasserstein_topk}")
        print(f"  - Trie tail depth: {trie_tail_depth}")
        print(f"  - Trie tail weight: {trie_tail_weight}")
    print(f"  - Student temperature: {student_temperature}")
    print(f"  - Teacher temperature: {teacher_temperature}")
    print(f"  - Skip student EOS: {skip_student_eos}")
    print(f"  - Skip teacher EOS: {skip_teacher_eos}")
    print(f"  - Alpha: {alpha}")
    print(f"  - KD weight: {alpha}")
    print("  - CE weight: 1.0")
    if layer_distillation_enabled:
        print("  - Layer distillation: enabled")
        print(f"  - Layer distill source: {layer_distill_source}")
        print(f"  - Layer distill weight: {layer_distill_weight}")
        print(f"  - Layer match JSON: {layer_match_json_path}")
        print(f"  - Layer match top-k: {layer_match_topk}")
        print(f"  - Student layer indices: {student_layer_indices}")
        for teacher_index, soft_matches in enumerate(teacher_layer_soft_matches):
            if layer_match_json_path:
                print(f"  - Teacher {teacher_index} soft matches:")
                for match in soft_matches:
                    teacher_terms = ", ".join(
                        f"{layer_idx}:{weight:.4f}"
                        for layer_idx, weight in zip(
                            match["teacher_layer_indices"],
                            match["teacher_layer_weights"],
                        )
                    )
                    print(f"    - student {match['student_layer_index']} -> {teacher_terms}")
            else:
                layer_pairs = [
                    (match["student_layer_index"], match["teacher_layer_indices"][0])
                    for match in soft_matches
                ]
                print(f"  - Teacher {teacher_index} layer pairs: {layer_pairs}")
    else:
        print("  - Layer distillation: disabled")
    if teacher_gate is not None:
        print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
        print(f"  - Teacher gate temperature: {teacher_gate_temperature}")
        print(f"  - Teacher gate entropy alpha: {teacher_gate_entropy_alpha}")
        print(f"  - Teacher gate router z-loss alpha: {teacher_gate_router_z_loss_alpha}")
        print(
            "  - Teacher gate hard routing warmup ratio: "
            f"{teacher_gate_hard_routing_warmup_ratio}"
        )
        print(f"  - GRACE threshold: {grace_threshold}")
        print(f"  - GRACE warmup ratio: {grace_warmup_ratio}")
        print(f"  - GRACE epsilon: {grace_epsilon}")
        print(f"  - GRACE softmax beta: {grace_softmax_beta}")
        print(
            "  - GRACE router blend lambda: "
            f"{grace_router_blend_lambda}"
        )
        print(f"  - GRACE EMA decay: {grace_ema_decay}")
    elif num_teachers > 1 and teacher_weighting_strategy == "reinforced_selection":
        print(f"  - Reinforced selection warmup ratio: {reinforced_selection_warmup_ratio}")
        print(f"  - Reinforced selection reward type: {reinforced_selection_reward_type}")
        print(
            f"  - Reinforced selection reward EMA decay: "
            f"{reinforced_selection_reward_ema_decay}"
        )
        print(f"  - Reinforced selection policy alpha: {reinforced_selection_policy_alpha}")
    print("  - Loss weighting: CE + alpha * KD")


__all__ = [
    "log_distillation_trainer_setup",
    "maybe_create_reinforced_teacher_selector",
    "normalize_teacher_models",
]
