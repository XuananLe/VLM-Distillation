from dataclasses import field

from pydantic import model_validator
from pydantic.dataclasses import dataclass


@dataclass
class DistillationArguments:
    student_model_id: str = field(metadata={"help": "Student model ID or path."})

    teacher_model_ids: list[str] = field(
        default_factory=list, metadata={"help": "Teacher model IDs. Pass as repeated values after --teacher_model_ids."}
    )

    teacher_logits_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Local teacher-logits cache root, for example /workspace/cache."},
    )

    teacher_weighting_strategy: str = field(
        default="routing",
        metadata={"help": "Teacher weighting strategy: `routing`, `uniform_mean` or `reinforced_selection`."},
    )

    distillation_loss: str = field(
        default="uld_loss",
        metadata={
            "help": "KD loss to use. Supported by src/components/loss.py, e.g. uld_loss, trie_wasserstein_loss, cka_loss, forward_kl, reverse_kl, jensen_shannon_divergence."
        },
    )

    trie_wasserstein_rho: float = field(
        default=0.7,
        metadata={"help": "Edge-decay factor rho used by trie_wasserstein_loss."},
    )

    trie_wasserstein_topk: int = field(
        default=64,
        metadata={
            "help": "Sparse top-k used by trie_wasserstein_loss before routing leftover probability mass to the residual tail edge."
        },
    )

    student_temperature: float = field(default=2.0, metadata={"help": "Student softmax temperature for KD."})

    teacher_temperature: float = field(default=2.0, metadata={"help": "Teacher softmax temperature for KD."})

    skip_student_eos: bool = field(
        default=False, metadata={"help": "Optionally drop the last supervised student token from KD."}
    )

    skip_teacher_eos: bool = field(
        default=False, metadata={"help": "Optionally drop the last supervised teacher token from KD."}
    )

    alpha: float = field(
        default=1.0,
        metadata={"help": "KD scaling factor in `ce_loss + alpha * kd_loss`."},
    )

    teacher_gate_top_k: int = field(
        default=1,
        metadata={"help": "Top-k teachers retained per sample by the teacher gate."},
    )

    teacher_gate_entropy_alpha: float = field(
        default=1e-3,
        metadata={"help": "Weight on the entropy bonus applied to teacher-gate probabilities."},
    )

    teacher_gate_router_z_loss_alpha: float = field(
        default=1e-3,
        metadata={"help": "Weight on router z-loss for stabilizing teacher-gate logits."},
    )

    grace_threshold: float = field(
        default=0.0,
        metadata={
            "help": "GRACE threshold: keep a routed teacher active only when its agreement score exceeds this threshold."
        },
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
        metadata={
            "help": "Fraction of training steps to pretrain reinforced teacher selection with all teachers active."
        },
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

    @model_validator(mode="after")
    def validate(self):
        if not self.teacher_model_ids:
            raise ValueError("At least one teacher model ID must be provided via --teacher_model_ids.")
        if self.alpha > 0.0 and self.teacher_logits_cache_dir is None:
            raise ValueError("Teacher logits require --teacher_logits_cache_dir when --alpha > 0.")

        allowed_values = (
            (
                "--teacher_weighting_strategy",
                self.teacher_weighting_strategy,
                {"routing", "uniform_mean", "reinforced_selection"},
            ),
            ("--reinforced_selection_reward_type", self.reinforced_selection_reward_type, {"reward1", "reward2"}),
        )
        for arg_name, value, allowed in allowed_values:
            if value not in allowed:
                choices = ", ".join(f"`{choice}`" for choice in sorted(allowed))
                raise ValueError(f"{arg_name} must be one of: {choices}.")

        if len(self.teacher_model_ids) == 1 and self.teacher_weighting_strategy == "routing":
            raise ValueError(
                "Teacher routing requires at least two teachers. "
                "For one teacher, use single-teacher distillation without `--teacher_weighting_strategy routing`."
            )

        if not 0.0 < self.trie_wasserstein_rho < 1.0:
            raise ValueError("--trie_wasserstein_rho must be in (0, 1).")
        for arg_name, value in (
            ("--trie_wasserstein_topk", self.trie_wasserstein_topk),
            ("--teacher_gate_top_k", self.teacher_gate_top_k),
        ):
            if value < 1:
                raise ValueError(f"{arg_name} must be >= 1.")

        for arg_name, value in (
            ("--grace_softmax_beta", self.grace_softmax_beta),
            ("--student_temperature", self.student_temperature),
            ("--teacher_temperature", self.teacher_temperature),
        ):
            if value <= 0.0:
                raise ValueError(f"{arg_name} must be > 0.")

        for arg_name, value in (
            ("--alpha", self.alpha),
            ("--teacher_gate_entropy_alpha", self.teacher_gate_entropy_alpha),
            ("--teacher_gate_router_z_loss_alpha", self.teacher_gate_router_z_loss_alpha),
            ("--grace_warmup_ratio", self.grace_warmup_ratio),
            ("--grace_epsilon", self.grace_epsilon),
            ("--reinforced_selection_warmup_ratio", self.reinforced_selection_warmup_ratio),
            ("--reinforced_selection_policy_alpha", self.reinforced_selection_policy_alpha),
        ):
            if value < 0.0:
                raise ValueError(f"{arg_name} must be >= 0.")

        if not 0.0 <= self.grace_router_blend_lambda <= 1.0:
            raise ValueError("--grace_router_blend_lambda must be between 0 and 1.")
        if not 0.0 <= self.grace_ema_decay < 1.0:
            raise ValueError("--grace_ema_decay must be in [0, 1).")
        if not 0.0 <= self.reinforced_selection_reward_ema_decay < 1.0:
            raise ValueError("--reinforced_selection_reward_ema_decay must be in [0, 1).")
        return self


def log_distillation_setup(
    *,
    teacher_ids,
    data_args,
    training_args,
    distillation_args,
    gradient_checkpointing_kwargs,
) -> None:
    print("=" * 80)
    print("Logits Distillation Training")
    print("=" * 80)
    print(f"Student Model: {distillation_args.student_model_id}")
    print(f"Teacher Model(s): {teacher_ids}")
    if distillation_args.teacher_logits_cache_dir:
        print(f"Teacher Logits Cache: {distillation_args.teacher_logits_cache_dir}")
    if distillation_args.alpha == 0.0:
        print("Teacher Weighting: disabled because alpha is 0")
    else:
        print(
            "Teacher Weighting: learned deep gate + GRACE routing"
            if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "routing"
            else (
                "Teacher Weighting: reinforced teacher selection"
                if len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "reinforced_selection"
                else "Teacher Weighting: single teacher"
                if len(teacher_ids) == 1
                else "Teacher Weighting: uniform mean"
            )
        )
    print("Objective: CE + alpha * KD")
    print(f"KD Weight: {distillation_args.alpha}")
    print(f"KD Function: {distillation_args.distillation_loss}")
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
        print(f"Teacher Gate Top-k: {distillation_args.teacher_gate_top_k}")
        print(f"Teacher Gate Entropy Alpha: {distillation_args.teacher_gate_entropy_alpha}")
        print(f"Teacher Gate Router Z-Loss Alpha: {distillation_args.teacher_gate_router_z_loss_alpha}")
        print(f"GRACE Threshold: {distillation_args.grace_threshold}")
        print(f"GRACE Warmup Ratio: {distillation_args.grace_warmup_ratio}")
        print(f"GRACE Epsilon: {distillation_args.grace_epsilon}")
        print(f"GRACE Softmax Beta: {distillation_args.grace_softmax_beta}")
        print(f"GRACE Router Blend Lambda: {distillation_args.grace_router_blend_lambda}")
        print(f"GRACE EMA Decay: {distillation_args.grace_ema_decay}")
    elif len(teacher_ids) > 1 and distillation_args.teacher_weighting_strategy == "reinforced_selection":
        print(f"Reinforced Selection Warmup Ratio: {distillation_args.reinforced_selection_warmup_ratio}")
        print(f"Reinforced Selection Reward Type: {distillation_args.reinforced_selection_reward_type}")
        print(f"Reinforced Selection Reward EMA Decay: {distillation_args.reinforced_selection_reward_ema_decay}")
        print(f"Reinforced Selection Policy Alpha: {distillation_args.reinforced_selection_policy_alpha}")
    if training_args.gradient_checkpointing:
        print(f"Gradient Checkpointing Kwargs: {gradient_checkpointing_kwargs}")
    print("=" * 80)
