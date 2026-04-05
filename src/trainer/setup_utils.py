from src.components.teacher_gate import Gate


def validate_distillation_trainer_args(
    *,
    alpha: float,
    teacher_gate_balance_alpha: float,
    teacher_gate_top_k: int,
    teacher_gate_capacity_factor: float,
    teacher_gate_bias_update_rate: float,
    teacher_gate_router_z_loss_alpha: float,
    gradient_alignment_warmup_ratio: float,
    gradient_alignment_epsilon: float,
    gradient_alignment_softmax_beta: float,
    gradient_alignment_router_blend_lambda: float,
    gradient_alignment_ema_decay: float,
    teacher_weighting_strategy: str,
    loss_function: str,
    distillation_loss_module,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("DistillationTrainer requires `0 <= alpha <= 1`.")
    if teacher_gate_balance_alpha < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_balance_alpha >= 0`.")
    if teacher_gate_top_k < 1:
        raise ValueError("DistillationTrainer requires `teacher_gate_top_k >= 1`.")
    if teacher_gate_capacity_factor <= 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_capacity_factor > 0`.")
    if teacher_gate_bias_update_rate < 0.0:
        raise ValueError("DistillationTrainer requires `teacher_gate_bias_update_rate >= 0`.")
    if teacher_gate_router_z_loss_alpha < 0.0:
        raise ValueError(
            "DistillationTrainer requires `teacher_gate_router_z_loss_alpha >= 0`."
        )
    if gradient_alignment_warmup_ratio < 0.0:
        raise ValueError("DistillationTrainer requires `gradient_alignment_warmup_ratio >= 0`.")
    if gradient_alignment_epsilon < 0.0:
        raise ValueError("DistillationTrainer requires `gradient_alignment_epsilon >= 0`.")
    if gradient_alignment_softmax_beta <= 0.0:
        raise ValueError("DistillationTrainer requires `gradient_alignment_softmax_beta > 0`.")
    if not 0.0 <= gradient_alignment_router_blend_lambda <= 1.0:
        raise ValueError(
            "DistillationTrainer requires `0 <= gradient_alignment_router_blend_lambda <= 1`."
        )
    if not 0.0 <= gradient_alignment_ema_decay < 1.0:
        raise ValueError("DistillationTrainer requires `0 <= gradient_alignment_ema_decay < 1`.")
    if not hasattr(distillation_loss_module, loss_function):
        raise ValueError(f"Unknown distillation loss: {loss_function!r}")
    if teacher_weighting_strategy not in {"routing", "uniform_mean"}:
        raise ValueError(
            "DistillationTrainer requires `teacher_weighting_strategy` to be "
            "`routing` or `uniform_mean`."
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
):
    if num_teachers <= 1 or teacher_weighting_strategy != "routing":
        return None

    teacher_gate = Gate(
        model,
        num_teachers,
        bias_update_rate=teacher_gate_bias_update_rate,
    )
    model.teacher_gate = teacher_gate
    return teacher_gate


def log_distillation_trainer_setup(
    *,
    num_teachers: int,
    teacher_weighting_strategy: str,
    loss_function: str,
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
    teacher_gate_router_z_loss_alpha: float,
    gradient_alignment_threshold: float,
    gradient_alignment_warmup_ratio: float,
    gradient_alignment_epsilon: float,
    gradient_alignment_softmax_beta: float,
    gradient_alignment_router_blend_lambda: float,
    gradient_alignment_ema_decay: float,
) -> None:
    print("Distillation Trainer initialized:")
    print(f"  - Teachers: {num_teachers}")
    if num_teachers > 1 and teacher_weighting_strategy == "routing":
        print("  - Teacher weighting: learned deep gate + balancing + gradient alignment")
    else:
        print("  - Teacher weighting: uniform mean")
    print(f"  - Loss function: {loss_function}")
    print(f"  - Student temperature: {student_temperature}")
    print(f"  - Teacher temperature: {teacher_temperature}")
    print(f"  - Skip student EOS: {skip_student_eos}")
    print(f"  - Skip teacher EOS: {skip_teacher_eos}")
    print(f"  - KD weight (alpha): {alpha}")
    print(f"  - CE weight: {1.0 - alpha}")
    if teacher_gate is not None:
        print(f"  - Teacher gate balance alpha: {teacher_gate_balance_alpha}")
        print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
        print(f"  - Teacher gate capacity factor: {teacher_gate_capacity_factor}")
        print(f"  - Teacher gate bias update rate: {teacher_gate_bias_update_rate}")
        print(f"  - Teacher gate router z-loss alpha: {teacher_gate_router_z_loss_alpha}")
        print(f"  - Gradient alignment threshold: {gradient_alignment_threshold}")
        print(f"  - Gradient alignment warmup ratio: {gradient_alignment_warmup_ratio}")
        print(f"  - Gradient alignment epsilon: {gradient_alignment_epsilon}")
        print(f"  - Gradient alignment softmax beta: {gradient_alignment_softmax_beta}")
        print(
            "  - Gradient alignment router blend lambda: "
            f"{gradient_alignment_router_blend_lambda}"
        )
        print(f"  - Gradient alignment EMA decay: {gradient_alignment_ema_decay}")
    print("  - Loss weighting: (1 - alpha) * CE + alpha * KD")


__all__ = [
    "log_distillation_trainer_setup",
    "maybe_create_teacher_gate",
    "normalize_teacher_models",
    "validate_distillation_trainer_args",
]
