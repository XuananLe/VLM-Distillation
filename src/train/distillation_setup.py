from dataclasses import dataclass, field

@dataclass
class DistillationArguments:
    """CLI arguments that control teacher loading, KD loss, routing, and layer distillation."""
    student_model_id: str = field(
        metadata={"help": "Student model ID or path."}
    )

    teacher_model_ids: list[str] = field(
        default_factory=list,
        metadata={"help": "Teacher model IDs. Pass as repeated values after --teacher_model_ids."}
    )

    teacher_logits_cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Local teacher-logits cache root. Use /cache for mounted local caches, or a writable /tmp path when downloading remote teacher logits on demand."
        },
    )

    teacher_logits_remote_uri: str | None = field(
        default=None,
        metadata={
            "help": "Optional s3:// base prefix for remote raw teacher logits. Leave unset to read directly from a local cache root such as /cache."
        },
    )

    teacher_weighting_strategy: str = field(
        default="routing",
        metadata={"help": "Teacher weighting strategy: `routing`, `uniform_mean`, or `reinforced_selection`."},
    )

    distillation_loss: str = field(
        default="uld_loss",
        metadata={
            "help": "KD loss to use. Supported by src/components/loss.py, e.g. uld_loss, trie_wasserstein_loss, cka_loss, forward_kl, reverse_kl, jensen_shannon_divergence."
        },
    )

    layer_distill_source: str = field(
        default="none",
        metadata={"help": "Optional hidden-state distillation source. Supported: none, vision, model."},
    )

    layer_distill_weight: float = field(
        default=0.0,
        metadata={"help": "Extra weight applied to the auxiliary soft layer-matching CKA loss."},
    )

    layer_match_json_path: str | None = field(
        default=None,
        metadata={"help": "Optional CKA matrix.json path used to derive top-k soft teacher matches."},
    )

    layer_match_topk: int = field(
        default=1,
        metadata={"help": "Number of teacher layers to soft-match per student layer from the CKA matrix."},
    )

    student_layer_indices: list[int] = field(
        default_factory=list,
        metadata={"help": "Student layer indices. Pass as repeated integer values after --student_layer_indices."},
    )

    teacher_layer_indices: list[int] = field(
        default_factory=list,
        metadata={"help": "Teacher layer indices. Pass as repeated integer values after --teacher_layer_indices."},
    )

    trie_wasserstein_rho: float = field(
        default=0.7,
        metadata={"help": "Edge-decay factor rho used by trie_wasserstein_loss."},
    )

    trie_wasserstein_topk: int = field(
        default=64,
        metadata={"help": "Sparse top-k used by trie_wasserstein_loss before routing leftover mass to the TAIL edge."},
    )

    student_temperature: float = field(
        default=2.0,
        metadata={"help": "Student softmax temperature for KD."}
    )

    teacher_temperature: float = field(
        default=2.0,
        metadata={"help": "Teacher softmax temperature for KD."}
    )

    skip_student_eos: bool = field(
        default=False,
        metadata={"help": "Optionally drop the last supervised student token from KD."}
    )

    skip_teacher_eos: bool = field(
        default=False,
        metadata={"help": "Optionally drop the last supervised teacher token from KD."}
    )

    alpha: float = field(
        default=1.0,
        metadata={"help": "KD scaling factor in `ce_loss + alpha * kd_loss`."},
    )

    teacher_gate_balance_alpha: float = field(
        default=1e-2,
        metadata={"help": "Weight on the teacher-gate balancing loss."},
    )

    teacher_gate_top_k: int = field(
        default=1,
        metadata={"help": "Top-k teachers retained per sample before applying capacity constraints."},
    )

    teacher_gate_capacity_factor: float = field(
        default=1.25,
        metadata={"help": "Capacity multiplier used by the teacher-gate routing constraints."},
    )

    teacher_gate_bias_update_rate: float = field(
        default=1e-3,
        metadata={"help": "Feedback update rate for the teacher-gate expert bias."},
    )

    teacher_gate_temperature: float = field(
        default=1.5,
        metadata={"help": "Softmax temperature applied to router scores before teacher-gate weighting."},
    )

    teacher_gate_noise_std: float = field(
        default=0.01,
        metadata={"help": "Gaussian noise std added to router scores during training before teacher-gate softmax."},
    )

    teacher_gate_entropy_alpha: float = field(
        default=1e-3,
        metadata={"help": "Weight on the entropy bonus applied to teacher-gate probabilities."},
    )

    teacher_gate_router_z_loss_alpha: float = field(
        default=1e-3,
        metadata={"help": "Weight on router z-loss for stabilizing teacher-gate logits."},
    )

    teacher_gate_hard_routing_warmup_ratio: float = field(
        default=0.2,
        metadata={"help": "Fraction of training steps to keep routing fully soft before enforcing the configured top-k."},
    )

    grace_threshold: float = field(
        default=0.0,
        metadata={"help": "GRACE threshold: keep a routed teacher active only when its agreement score exceeds this threshold."},
    )

    grace_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of training steps to wait before enabling GRACE routing refinement."},
    )

    grace_epsilon: float = field(
        default=0.01,
        metadata={
            "help": "If the spread between teacher agreement scores is below this epsilon, GRACE falls back to uniform-mean teacher weights."
        },
    )

    grace_softmax_beta: float = field(
        default=20.0,
        metadata={
            "help": "Inverse temperature used by GRACE to convert agreement scores into softmax gradient weights."
        },
    )

    grace_router_blend_lambda: float = field(
        default=0.5,
        metadata={
            "help": "GRACE blend factor between router weights and gradient-derived weights. 1.0 keeps only router weights; 0.0 keeps only gradient weights."
        },
    )

    grace_ema_decay: float = field(
        default=0.9,
        metadata={"help": "EMA decay applied to teacher agreement scores before GRACE computes gradient weights."},
    )

    reinforced_selection_warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Fraction of training steps to pretrain reinforced teacher selection with all teachers active."},
    )

    reinforced_selection_reward_type: str = field(
        default="reward2",
        metadata={"help": "Reinforced teacher selection reward: `reward1` uses `-CE`, `reward2` uses `-CE-KD`."},
    )

    reinforced_selection_reward_ema_decay: float = field(
        default=0.9,
        metadata={"help": "EMA decay for the reinforced teacher-selection reward baseline."},
    )

    reinforced_selection_policy_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight on the reinforced teacher-selection policy loss."},
    )
def validate_distillation_args(distillation_args) -> None:
    """Validate distillation CLI arguments before any heavy model or cache loading starts."""
    if not distillation_args.teacher_model_ids:
        raise ValueError("At least one teacher model ID must be provided via --teacher_model_ids.")
    if (
        distillation_args.teacher_logits_cache_dir is None
        and distillation_args.teacher_logits_remote_uri is None
    ):
        raise ValueError(
            "Teacher logits require either --teacher_logits_cache_dir (for example /cache "
            "or a writable /tmp path) or --teacher_logits_remote_uri (remote raw cache root)."
        )
    if distillation_args.teacher_weighting_strategy not in {"routing", "uniform_mean", "reinforced_selection"}:
        raise ValueError("--teacher_weighting_strategy must be `routing`, `uniform_mean`, or `reinforced_selection`.")
    if not 0.0 < distillation_args.trie_wasserstein_rho < 1.0:
        raise ValueError("--trie_wasserstein_rho must be in (0, 1).")
    if distillation_args.trie_wasserstein_topk < 1:
        raise ValueError("--trie_wasserstein_topk must be >= 1.")
    if distillation_args.layer_distill_source not in {"none", "vision", "model"}:
        raise ValueError("--layer_distill_source must be `none`, `vision`, or `model`.")
    if distillation_args.layer_distill_weight < 0.0:
        raise ValueError("--layer_distill_weight must be >= 0.")
    if distillation_args.layer_match_topk < 1:
        raise ValueError("--layer_match_topk must be >= 1.")
    if distillation_args.alpha < 0.0:
        raise ValueError("--alpha must be >= 0.")
    if distillation_args.teacher_gate_balance_alpha < 0.0:
        raise ValueError("--teacher_gate_balance_alpha must be >= 0.")
    if distillation_args.teacher_gate_top_k < 1:
        raise ValueError("--teacher_gate_top_k must be >= 1.")
    if distillation_args.teacher_gate_capacity_factor <= 0.0:
        raise ValueError("--teacher_gate_capacity_factor must be > 0.")
    if distillation_args.teacher_gate_bias_update_rate < 0.0:
        raise ValueError("--teacher_gate_bias_update_rate must be >= 0.")
    if distillation_args.teacher_gate_temperature <= 0.0:
        raise ValueError("--teacher_gate_temperature must be > 0.")
    if distillation_args.teacher_gate_noise_std < 0.0:
        raise ValueError("--teacher_gate_noise_std must be >= 0.")
    if distillation_args.teacher_gate_entropy_alpha < 0.0:
        raise ValueError("--teacher_gate_entropy_alpha must be >= 0.")
    if distillation_args.teacher_gate_router_z_loss_alpha < 0.0:
        raise ValueError("--teacher_gate_router_z_loss_alpha must be >= 0.")
    if distillation_args.teacher_gate_hard_routing_warmup_ratio < 0.0:
        raise ValueError("--teacher_gate_hard_routing_warmup_ratio must be >= 0.")
    if distillation_args.grace_warmup_ratio < 0.0:
        raise ValueError("--grace_warmup_ratio must be >= 0.")
    if distillation_args.grace_epsilon < 0.0:
        raise ValueError("--grace_epsilon must be >= 0.")
    if distillation_args.grace_softmax_beta <= 0.0:
        raise ValueError("--grace_softmax_beta must be > 0.")
    if not 0.0 <= distillation_args.grace_router_blend_lambda <= 1.0:
        raise ValueError("--grace_router_blend_lambda must be between 0 and 1.")
    if not 0.0 <= distillation_args.grace_ema_decay < 1.0:
        raise ValueError("--grace_ema_decay must be in [0, 1).")
    if distillation_args.reinforced_selection_warmup_ratio < 0.0:
        raise ValueError("--reinforced_selection_warmup_ratio must be >= 0.")
    if distillation_args.reinforced_selection_reward_type not in {"reward1", "reward2"}:
        raise ValueError("--reinforced_selection_reward_type must be `reward1` or `reward2`.")
    if not 0.0 <= distillation_args.reinforced_selection_reward_ema_decay < 1.0:
        raise ValueError("--reinforced_selection_reward_ema_decay must be in [0, 1).")
    if distillation_args.reinforced_selection_policy_alpha < 0.0:
        raise ValueError("--reinforced_selection_policy_alpha must be >= 0.")
    if distillation_args.student_temperature <= 0:
        raise ValueError("--student_temperature must be > 0.")
    if distillation_args.teacher_temperature <= 0:
        raise ValueError("--teacher_temperature must be > 0.")
    if (
        distillation_args.layer_distill_source in {"vision", "model"}
        and distillation_args.layer_distill_weight > 0.0
        and (
            distillation_args.student_layer_indices
            or distillation_args.layer_match_json_path
        )
    ):
        if (
            not distillation_args.teacher_layer_indices
            and not distillation_args.layer_match_json_path
        ):
            raise ValueError("--teacher_layer_indices must be provided when layer distillation is enabled.")
        if (
            distillation_args.student_layer_indices
            and distillation_args.teacher_layer_indices
            and len(distillation_args.student_layer_indices) != len(distillation_args.teacher_layer_indices)
        ):
            raise ValueError(
                "--student_layer_indices and --teacher_layer_indices must have the same length."
            )


def log_distillation_setup(
    *,
    teacher_ids,
    student_layer_indices,
    teacher_layer_indices,
    data_args,
    training_args,
    distillation_args,
    gradient_checkpointing_kwargs,
) -> None:
    """Print the resolved distillation configuration once at startup for reproducibility."""
    print("=" * 80)
    print("Logits Distillation Training")
    print("=" * 80)
    print(f"Student Model: {distillation_args.student_model_id}")
    print(f"Teacher Model(s): {teacher_ids}")
    if distillation_args.teacher_logits_cache_dir:
        print(f"Teacher Logits Cache: {distillation_args.teacher_logits_cache_dir}")
    if distillation_args.teacher_logits_remote_uri:
        print(f"Teacher Logits Remote URI: {distillation_args.teacher_logits_remote_uri}")
    print(
        "Teacher Weighting: learned deep gate + balancing + GRACE routing"
        if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "routing"
        else (
            "Teacher Weighting: reinforced teacher selection"
            if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "reinforced_selection"
            else "Teacher Weighting: uniform mean"
        )
    )
    print("Objective: CE + alpha * KD")
    print(f"KD Weight: {distillation_args.alpha}")
    print(f"KD Function: {distillation_args.distillation_loss}")
    print(f"Layer Distill Source: {distillation_args.layer_distill_source}")
    print(f"Layer Distill Weight: {distillation_args.layer_distill_weight}")
    print(f"Layer Match JSON: {distillation_args.layer_match_json_path}")
    print(f"Layer Match Top-k: {distillation_args.layer_match_topk}")
    print(f"Student Layer Indices: {student_layer_indices}")
    print(f"Teacher Layer Indices: {teacher_layer_indices}")
    if distillation_args.distillation_loss == "trie_wasserstein_loss":
        print(f"Trie Wasserstein Rho: {distillation_args.trie_wasserstein_rho}")
        print(f"Trie Wasserstein Top-k: {distillation_args.trie_wasserstein_topk}")
    print(f"Alpha: {distillation_args.alpha}")
    print("CE Weight: 1.0")
    print(f"Student Temperature: {distillation_args.student_temperature}")
    print(f"Teacher Temperature: {distillation_args.teacher_temperature}")
    print(f"Skip Student EOS: {distillation_args.skip_student_eos}")
    print(f"Skip Teacher EOS: {distillation_args.skip_teacher_eos}")
    if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "routing":
        print(f"Teacher Gate Balance Alpha: {distillation_args.teacher_gate_balance_alpha}")
        print(f"Teacher Gate Top-k: {distillation_args.teacher_gate_top_k}")
        print(f"Teacher Gate Capacity Factor: {distillation_args.teacher_gate_capacity_factor}")
        print(f"Teacher Gate Bias Update Rate: {distillation_args.teacher_gate_bias_update_rate}")
        print(f"Teacher Gate Temperature: {distillation_args.teacher_gate_temperature}")
        print(f"Teacher Gate Noise Std: {distillation_args.teacher_gate_noise_std}")
        print(f"Teacher Gate Entropy Alpha: {distillation_args.teacher_gate_entropy_alpha}")
        print(
            f"Teacher Gate Router Z-Loss Alpha: "
            f"{distillation_args.teacher_gate_router_z_loss_alpha}"
        )
        print(
            f"Teacher Gate Hard Routing Warmup Ratio: "
            f"{distillation_args.teacher_gate_hard_routing_warmup_ratio}"
        )
        print(f"GRACE Threshold: {distillation_args.grace_threshold}")
        print(f"GRACE Warmup Ratio: {distillation_args.grace_warmup_ratio}")
        print(
            f"GRACE Epsilon: "
            f"{distillation_args.grace_epsilon}"
        )
        print(
            f"GRACE Softmax Beta: "
            f"{distillation_args.grace_softmax_beta}"
        )
        print(
            f"GRACE Router Blend Lambda: "
            f"{distillation_args.grace_router_blend_lambda}"
        )
        print(
            f"GRACE EMA Decay: "
            f"{distillation_args.grace_ema_decay}"
        )
    elif len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "reinforced_selection":
        print(
            f"Reinforced Selection Warmup Ratio: "
            f"{distillation_args.reinforced_selection_warmup_ratio}"
        )
        print(
            f"Reinforced Selection Reward Type: "
            f"{distillation_args.reinforced_selection_reward_type}"
        )
        print(
            f"Reinforced Selection Reward EMA Decay: "
            f"{distillation_args.reinforced_selection_reward_ema_decay}"
        )
        print(
            f"Reinforced Selection Policy Alpha: "
            f"{distillation_args.reinforced_selection_policy_alpha}"
        )
    if training_args.gradient_checkpointing:
        print(f"Gradient Checkpointing Kwargs: {gradient_checkpointing_kwargs}")
    print("=" * 80)
