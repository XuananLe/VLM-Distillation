def log_distillation_trainer_setup(
    *,
    num_teachers: int,
    teacher_weighting_strategy: str,
    loss_function: str,
    student_temperature: float,
    teacher_temperature: float,
    alpha: float,
    teacher_gate,
    teacher_gate_top_k: int,
    teacher_gate_entropy_alpha: float,
    teacher_gate_router_z_loss_alpha: float,
    grace_threshold: float,
    grace_warmup_ratio: float,
    grace_epsilon: float,
    grace_softmax_beta: float,
    grace_router_blend_lambda: float,
    grace_ema_decay: float,
    trie_wasserstein_rho: float,
    trie_wasserstein_topk: int,
) -> None:
    """Print the trainer-side distillation config summary.

    Input: resolved trainer knobs after model/layer setup. Output: None.
    Exists so one place reports which optional distillation path is actually
    active after constructor normalization.
    """
    print("Distillation Trainer initialized:")
    print(f"  - Teachers: {num_teachers}")
    if alpha == 0.0:
        print("  - Teacher weighting: disabled because alpha is 0")
    elif num_teachers > 1 and teacher_weighting_strategy == "routing":
        print("  - Teacher weighting: learned deep gate + GRACE routing")
    elif num_teachers == 1:
        print("  - Teacher weighting: single teacher")
    else:
        print("  - Teacher weighting: uniform mean")
    print(f"  - Loss function: {loss_function}")
    if loss_function == "trie_wasserstein_loss":
        print(f"  - Trie Wasserstein rho: {trie_wasserstein_rho}")
        print(f"  - Trie Wasserstein top-k: {trie_wasserstein_topk}")
    print(f"  - Student temperature: {student_temperature}")
    print(f"  - Teacher temperature: {teacher_temperature}")
    print("  - Drop final supervised token for KD: True")
    print(f"  - Alpha: {alpha}")
    print(f"  - KD weight: {alpha}")
    print("  - CE weight: 1.0")
    if teacher_gate is not None:
        print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
        print(f"  - Teacher gate entropy alpha: {teacher_gate_entropy_alpha}")
        print(f"  - Teacher gate router z-loss alpha: {teacher_gate_router_z_loss_alpha}")
        print(f"  - GRACE threshold: {grace_threshold}")
        print(f"  - GRACE warmup ratio: {grace_warmup_ratio}")
        print(f"  - GRACE epsilon: {grace_epsilon}")
        print(f"  - GRACE softmax beta: {grace_softmax_beta}")
        print(f"  - GRACE router blend lambda: {grace_router_blend_lambda}")
        print(f"  - GRACE EMA decay: {grace_ema_decay}")
    print("  - Loss weighting: CE + alpha * KD")


__all__ = [
    "log_distillation_trainer_setup",
]
