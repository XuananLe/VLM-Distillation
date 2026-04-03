from dataclasses import dataclass, field

from src.train.train_utils import rank0_print


@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation."""

    student_model_id: str = field(
        metadata={"help": "Student model ID or path."}
    )

    teacher_model_ids: str = field(
        metadata={"help": "Teacher model IDs as a Python list literal or comma-separated string."}
    )

    teacher_weighting_strategy: str = field(
        default="routing",
        metadata={"help": "Teacher weighting strategy: `routing` or `uniform_mean`."},
    )

    distillation_loss: str = field(
        default="uld_loss",
        metadata={
            "help": "KD loss to use. Supported by src/components/loss.py, e.g. uld_loss, forward_kl, reverse_kl, jensen_shannon_divergence."
        },
    )

    temperature: float = field(
        default=2.0,
        metadata={"help": "Legacy shorthand temperature. Used for both student and teacher if separate temperatures are not set."}
    )

    student_temperature: float | None = field(
        default=None,
        metadata={"help": "Student softmax temperature for KD. Defaults to --temperature when omitted."}
    )

    teacher_temperature: float | None = field(
        default=None,
        metadata={"help": "Teacher softmax temperature for KD. Defaults to --temperature when omitted."}
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
        metadata={"help": "KD mixing weight in `(1 - alpha) * ce_loss + alpha * kd_loss`."},
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

    gradient_alignment_threshold: float = field(
        default=0.0,
        metadata={"help": "Keep a routed teacher active only when its gradient alignment exceeds this threshold."},
    )

    gradient_alignment_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of training steps to wait before enabling gradient-alignment filtering."},
    )

    gradient_alignment_sigmoid_temperature: float = field(
        default=0.02,
        metadata={"help": "Sigmoid temperature used to convert alignment scores into soft weights."},
    )


def validate_distillation_args(distillation_args) -> None:
    if distillation_args.teacher_weighting_strategy not in {"routing", "uniform_mean"}:
        raise ValueError("--teacher_weighting_strategy must be `routing` or `uniform_mean`.")
    if not 0.0 <= distillation_args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if distillation_args.teacher_gate_balance_alpha < 0.0:
        raise ValueError("--teacher_gate_balance_alpha must be >= 0.")
    if distillation_args.teacher_gate_top_k < 1:
        raise ValueError("--teacher_gate_top_k must be >= 1.")
    if distillation_args.teacher_gate_capacity_factor <= 0.0:
        raise ValueError("--teacher_gate_capacity_factor must be > 0.")
    if distillation_args.teacher_gate_bias_update_rate < 0.0:
        raise ValueError("--teacher_gate_bias_update_rate must be >= 0.")
    if distillation_args.gradient_alignment_warmup_ratio < 0.0:
        raise ValueError("--gradient_alignment_warmup_ratio must be >= 0.")
    if distillation_args.gradient_alignment_sigmoid_temperature <= 0.0:
        raise ValueError("--gradient_alignment_sigmoid_temperature must be > 0.")
    if distillation_args.student_temperature is not None and distillation_args.student_temperature <= 0:
        raise ValueError("--student_temperature must be > 0.")
    if distillation_args.teacher_temperature is not None and distillation_args.teacher_temperature <= 0:
        raise ValueError("--teacher_temperature must be > 0.")
    if distillation_args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")


def log_distillation_setup(
    *,
    teacher_ids,
    data_args,
    training_args,
    distillation_args,
    gradient_checkpointing_kwargs,
) -> None:
    rank0_print("=" * 80)
    rank0_print("Logits Distillation Training")
    rank0_print("=" * 80)
    rank0_print(f"Student Model: {distillation_args.student_model_id}")
    rank0_print(f"Teacher Model(s): {teacher_ids}")
    rank0_print(
        "Teacher Weighting: learned deep gate + balancing + gradient alignment"
        if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "routing"
        else "Teacher Weighting: uniform mean"
    )
    rank0_print("Objective: (1 - alpha) * CE + alpha * KD")
    rank0_print(f"KD Function: {distillation_args.distillation_loss}")
    rank0_print(f"KD Weight (Alpha): {distillation_args.alpha}")
    rank0_print(f"CE Weight: {1.0 - distillation_args.alpha}")
    resolved_student_temperature = (
        distillation_args.temperature
        if distillation_args.student_temperature is None
        else distillation_args.student_temperature
    )
    resolved_teacher_temperature = (
        distillation_args.temperature
        if distillation_args.teacher_temperature is None
        else distillation_args.teacher_temperature
    )
    rank0_print(f"Student Temperature: {resolved_student_temperature}")
    rank0_print(f"Teacher Temperature: {resolved_teacher_temperature}")
    rank0_print(f"Skip Student EOS: {distillation_args.skip_student_eos}")
    rank0_print(f"Skip Teacher EOS: {distillation_args.skip_teacher_eos}")
    if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "routing":
        rank0_print(f"Teacher Gate Balance Alpha: {distillation_args.teacher_gate_balance_alpha}")
        rank0_print(f"Teacher Gate Top-k: {distillation_args.teacher_gate_top_k}")
        rank0_print(f"Teacher Gate Capacity Factor: {distillation_args.teacher_gate_capacity_factor}")
        rank0_print(f"Teacher Gate Bias Update Rate: {distillation_args.teacher_gate_bias_update_rate}")
        rank0_print(f"Gradient Alignment Threshold: {distillation_args.gradient_alignment_threshold}")
        rank0_print(f"Gradient Alignment Warmup Ratio: {distillation_args.gradient_alignment_warmup_ratio}")
        rank0_print(
            f"Gradient Alignment Sigmoid Temperature: "
            f"{distillation_args.gradient_alignment_sigmoid_temperature}"
        )
    if training_args.gradient_checkpointing:
        rank0_print(f"Gradient Checkpointing Kwargs: {gradient_checkpointing_kwargs}")
    rank0_print("=" * 80)
