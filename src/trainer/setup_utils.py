from src.components.teacher_gate import Gate
from src.components.reinforced_teacher_selection import ReinforcedTeacherSelectionPolicy


def validate_distillation_trainer_args(
    *,
    alpha: float,
    layer_distill_source: str,
    layer_distill_weight: float,
    layer_match_topk: int,
    teacher_gate_balance_alpha: float,
    teacher_gate_top_k: int,
    teacher_gate_capacity_factor: float,
    teacher_gate_bias_update_rate: float,
    teacher_gate_temperature: float,
    teacher_gate_noise_std: float,
    teacher_gate_entropy_alpha: float,
    teacher_gate_router_z_loss_alpha: float,
    teacher_gate_hard_routing_warmup_ratio: float,
    grace_warmup_ratio: float,
    grace_epsilon: float,
    grace_softmax_beta: float,
    grace_router_blend_lambda: float,
    grace_ema_decay: float,
    reinforced_selection_warmup_ratio: float,
    reinforced_selection_reward_type: str,
    reinforced_selection_reward_ema_decay: float,
    reinforced_selection_policy_alpha: float,
    teacher_weighting_strategy: str,
    trie_wasserstein_rho: float,
    trie_wasserstein_topk: int,
    loss_function: str,
    distillation_loss_module,
) -> None:
    if alpha < 0.0:
        raise ValueError("DistillationTrainer requires `alpha >= 0`.")
    if layer_distill_source not in {"none", "vision", "model"}:
        raise ValueError(
            "DistillationTrainer requires `layer_distill_source` to be "
            "`none`, `vision`, or `model`."
        )
    if layer_distill_weight < 0.0:
        raise ValueError("DistillationTrainer requires `layer_distill_weight >= 0`.")
    if layer_match_topk < 1:
        raise ValueError("DistillationTrainer requires `layer_match_topk >= 1`.")
    if teacher_gate_balance_alpha < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_balance_alpha >= 0`.")
    if teacher_gate_top_k < 1:
        raise ValueError("DistillationTrainer requires `teacher_gate_top_k >= 1`.")
    if teacher_gate_capacity_factor <= 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_capacity_factor > 0`.")
    if teacher_gate_bias_update_rate < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_bias_update_rate >= 0`.")
    if teacher_gate_temperature <= 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_temperature > 0`.")
    if teacher_gate_noise_std < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_noise_std >= 0`.")
    if teacher_gate_entropy_alpha < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_entropy_alpha >= 0`.")
    if teacher_gate_router_z_loss_alpha < 0.0:
        raise ValueError(
            "DistillationTrainer requires `teacher_gate_router_z_loss_alpha >= 0`."
        )
    if teacher_gate_hard_routing_warmup_ratio < 0.0:
        raise ValueError(
            "DistillationTrainer requires `teacher_gate_hard_routing_warmup_ratio >= 0`."
        )
    if grace_warmup_ratio < 0.0:
        raise ValueError("DistillationTrainer requires `grace_warmup_ratio >= 0`.")
    if grace_epsilon < 0.0:
        raise ValueError("DistillationTrainer requires `grace_epsilon >= 0`.")
    if grace_softmax_beta <= 0.0:
        raise ValueError("DistillationTrainer requires `grace_softmax_beta > 0`.")
    if not 0.0 <= grace_router_blend_lambda <= 1.0:
        raise ValueError(
            "DistillationTrainer requires `0 <= grace_router_blend_lambda <= 1`."
        )
    if not 0.0 <= grace_ema_decay < 1.0:
        raise ValueError("DistillationTrainer requires `0 <= grace_ema_decay < 1`.")
    if reinforced_selection_warmup_ratio < 0.0:
        raise ValueError("DistillationTrainer requires `reinforced_selection_warmup_ratio >= 0`.")
    if reinforced_selection_reward_type not in {"reward1", "reward2"}:
        raise ValueError(
            "DistillationTrainer requires `reinforced_selection_reward_type` to be "
            "`reward1` or `reward2`."
        )
    if not 0.0 <= reinforced_selection_reward_ema_decay < 1.0:
        raise ValueError(
            "DistillationTrainer requires `0 <= reinforced_selection_reward_ema_decay < 1`."
        )
    if reinforced_selection_policy_alpha < 0.0:
        raise ValueError("DistillationTrainer requires `reinforced_selection_policy_alpha >= 0`.")
    if trie_wasserstein_rho <= 0.0 or trie_wasserstein_rho >= 1.0:
        raise ValueError("DistillationTrainer requires `0 < trie_wasserstein_rho < 1`.")
    if trie_wasserstein_topk < 1:
        raise ValueError("DistillationTrainer requires `trie_wasserstein_topk >= 1`.")
    if loss_function != "trie_wasserstein_loss" and not hasattr(distillation_loss_module, loss_function):
        raise ValueError(f"Unknown distillation loss: {loss_function!r}")
    if teacher_weighting_strategy not in {"routing", "uniform_mean", "reinforced_selection"}:
        raise ValueError(
            "DistillationTrainer requires `teacher_weighting_strategy` to be "
            "`routing`, `uniform_mean`, or `reinforced_selection`."
        )


def normalize_teacher_models(teacher_model, teacher_count: int | None):
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

    for model in teacher_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return teacher_models, int(teacher_count)


def maybe_create_teacher_gate(
    *,
    model,
    num_teachers: int,
    teacher_weighting_strategy: str,
    teacher_gate_bias_update_rate: float,
    teacher_gate_temperature: float,
    teacher_gate_noise_std: float,
):
    if num_teachers <= 1 or teacher_weighting_strategy != "routing":
        return None

    teacher_gate = Gate(
        model,
        num_teachers,
        bias_update_rate=teacher_gate_bias_update_rate,
        router_temperature=teacher_gate_temperature,
        router_noise_std=teacher_gate_noise_std,
    )
    model.teacher_gate = teacher_gate
    return teacher_gate


def maybe_create_reinforced_teacher_selector(
    *,
    model,
    num_teachers: int,
    teacher_weighting_strategy: str,
):
    if num_teachers <= 1 or teacher_weighting_strategy != "reinforced_selection":
        return None

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
    teacher_gate_balance_alpha: float,
    teacher_gate_top_k: int,
    teacher_gate_capacity_factor: float,
    teacher_gate_bias_update_rate: float,
    teacher_gate_temperature: float,
    teacher_gate_noise_std: float,
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
) -> None:
    print("Distillation Trainer initialized:")
    print(f"  - Teachers: {num_teachers}")
    if num_teachers > 1 and teacher_weighting_strategy == "routing":
        print("  - Teacher weighting: learned deep gate + balancing + GRACE routing")
    elif num_teachers > 1 and teacher_weighting_strategy == "reinforced_selection":
        print("  - Teacher weighting: reinforced teacher selection")
    else:
        print("  - Teacher weighting: uniform mean")
    print(f"  - Loss function: {loss_function}")
    if loss_function == "trie_wasserstein_loss":
        print(f"  - Trie Wasserstein rho: {trie_wasserstein_rho}")
        print(f"  - Trie Wasserstein top-k: {trie_wasserstein_topk}")
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
        print(f"  - Teacher gate balance alpha: {teacher_gate_balance_alpha}")
        print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
        print(f"  - Teacher gate capacity factor: {teacher_gate_capacity_factor}")
        print(f"  - Teacher gate bias update rate: {teacher_gate_bias_update_rate}")
        print(f"  - Teacher gate temperature: {teacher_gate_temperature}")
        print(f"  - Teacher gate noise std: {teacher_gate_noise_std}")
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
    "maybe_create_teacher_gate",
    "maybe_create_reinforced_teacher_selector",
    "normalize_teacher_models",
    "validate_distillation_trainer_args",
]
