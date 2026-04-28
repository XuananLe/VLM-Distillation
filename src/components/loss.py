import torch
import torch.nn.functional as F

from src.components.cka import linear_cka_loss
from src.components.trie_wasserstein import TrieWassersteinLoss


def build_distillation_loss(
    *,
    loss_function: str,
    student_tokenizer=None,
    teacher_tokenizers=None,
    trie_wasserstein_rho: float = 0.7,
    trie_wasserstein_topk: int = 64,
    trie_tail_depth: int = 1,
    trie_tail_weight: float = 0.5,
):
    """Return prepare and loss callables for the configured KD loss."""
    if loss_function == "trie_wasserstein_loss":
        if student_tokenizer is None:
            raise ValueError("Trie Wasserstein loss requires a student tokenizer.")
        if not teacher_tokenizers:
            raise ValueError("Trie Wasserstein loss requires teacher tokenizers.")
        loss_modules = [
            TrieWassersteinLoss(
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                rho=trie_wasserstein_rho,
                topk=trie_wasserstein_topk,
                tail_depth=trie_tail_depth,
                tail_weight=trie_tail_weight,
            )
            for teacher_tokenizer in teacher_tokenizers
        ]
        prepared_vocab_shapes: dict[int, tuple[int, int]] = {}

        def select_loss_module(teacher_index: int | None) -> TrieWassersteinLoss:
            """Return the trie loss module associated with one teacher index."""
            if teacher_index is None:
                raise ValueError("Trie Wasserstein loss requires a teacher index.")
            return loss_modules[teacher_index]

        def prepare_teacher_batch(
            *,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            teacher_labels: torch.Tensor | None = None,
            teacher_index: int | None = None,
        ) -> None:
            """Prepare the selected trie module for the current teacher batch."""
            loss_module = select_loss_module(teacher_index)
            teacher_key = int(teacher_index)
            student_vocab_size = student_logits.size(-1)
            teacher_vocab_size = teacher_logits.size(-1)
            loss_module.prepare_runtime_state(
                student_vocab_size=student_vocab_size,
                teacher_vocab_size=teacher_vocab_size,
                teacher_labels=teacher_labels,
            )
            prepared_vocab_shapes[teacher_key] = (student_vocab_size, teacher_vocab_size)

        def compute_loss(
            *,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            student_temperature: float = 1.0,
            teacher_temperature: float = 1.0,
            teacher_index: int | None = None,
        ) -> torch.Tensor:
            """Compute trie-Wasserstein KD against the selected teacher tokenizer."""
            loss_module = select_loss_module(teacher_index)
            teacher_key = int(teacher_index)
            student_vocab_size = student_logits.size(-1)
            teacher_vocab_size = teacher_logits.size(-1)
            vocab_shape = (student_vocab_size, teacher_vocab_size)
            if prepared_vocab_shapes.get(teacher_key) != vocab_shape:
                loss_module.prepare_runtime_state(
                    student_vocab_size=student_vocab_size,
                    teacher_vocab_size=teacher_vocab_size,
                )
                prepared_vocab_shapes[teacher_key] = vocab_shape
            return loss_module(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
            )

        def trie_metrics() -> dict[str, float]:
            """Return aggregate trie prefix-tail diagnostics from the latest logged step."""
            metric_values: dict[str, list[torch.Tensor]] = {
                "trie_exact_mass_mean": [],
                "trie_tail_mass_mean": [],
                "trie_tail_bucket_entropy": [],
                "trie_tail_top_bucket_mass": [],
            }
            tail_bucket_counts = []
            for loss_module in loss_modules:
                for side_stats in loss_module.last_prefix_tail_stats.values():
                    metric_values["trie_exact_mass_mean"].append(side_stats.exact_mass_mean)
                    metric_values["trie_tail_mass_mean"].append(side_stats.tail_mass_mean)
                    metric_values["trie_tail_bucket_entropy"].append(side_stats.tail_bucket_entropy)
                    metric_values["trie_tail_top_bucket_mass"].append(side_stats.tail_top_bucket_mass)
                    tail_bucket_counts.append(float(side_stats.num_tail_buckets))
            metrics = {
                key: torch.stack(values).mean().item()
                for key, values in metric_values.items()
                if values
            }
            if tail_bucket_counts:
                metrics["trie_num_tail_buckets"] = sum(tail_bucket_counts) / len(tail_bucket_counts)
            return metrics

        compute_loss.trie_metrics = trie_metrics
        return prepare_teacher_batch, compute_loss

    if loss_function not in DISTILLATION_LOSSES:
        raise ValueError(f"Unknown distillation loss: {loss_function!r}")
    loss_fn = DISTILLATION_LOSSES[loss_function]

    def prepare_teacher_batch(
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor | None = None,
        teacher_index: int | None = None,
    ) -> None:
        """Accept the shared loss interface even though function losses keep no batch state."""
        del student_logits, teacher_logits, teacher_labels, teacher_index

    def compute_loss(
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_index: int | None = None,
    ) -> torch.Tensor:
        """Compute one function-based KD loss for aligned student and teacher logits."""
        del teacher_index
        return loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )

    return prepare_teacher_batch, compute_loss


def cka_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Logits-space linear CKA loss using softened token distributions.

    Unlike token-wise KL losses, CKA only requires the same number of aligned
    samples, so student and teacher vocabulary sizes may differ.
    """
    student_temperature = float(student_temperature)
    teacher_temperature = float(teacher_temperature)
    student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    # L_CKA = 1 - sqrt(CKA(P_student, P_teacher)).
    return linear_cka_loss(student_probs, teacher_probs)


def uld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Universal Logit Distillation (ULD) loss via Wasserstein-1 distance.
    https://arxiv.org/abs/2402.12030

    Supports different vocabulary sizes between student and teacher by sorting
    each distribution in descending order and computing the L1 distance between
    the aligned sorted probability masses (closed-form W1, Eq. 5 in the paper).

    student_logits: (N, V_s) — N tokens, student vocab size V_s
    teacher_logits: (N, V_t) — N tokens, teacher vocab size V_t
    student_temperature / teacher_temperature: softmax temperatures for the two distributions
    returns scalar Wasserstein-1 loss averaged over tokens
    """
    student_temperature = float(student_temperature)
    teacher_temperature = float(teacher_temperature)
    student_probs = F.softmax(student_logits / student_temperature, dim=-1)
    student_sorted = student_probs.sort(dim=-1, descending=True).values

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / teacher_temperature, dim=-1)
        teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    # Pad the smaller vocabulary to match sizes with zeros so that
    # the remaining mass is implicitly assigned to "phantom" tokens.
    V_s, V_t = student_sorted.size(dim=-1), teacher_sorted.size(-1)
    if V_s < V_t:
        student_sorted = F.pad(student_sorted, (0, V_t - V_s))
    elif V_t < V_s:
        teacher_sorted = F.pad(teacher_sorted, (0, V_s - V_t))

    # Closed-form 1D Wasserstein-1 on sorted probability masses:
    # L_ULD = sum_i |sort(p_s)_i - sort(p_t)_i|.
    return (student_sorted - teacher_sorted).abs().sum(dim=-1).mean()


def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute forward KL distillation when student and teacher share a vocab."""
    student_temperature = float(student_temperature)
    teacher_temperature = float(teacher_temperature)
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    # L = T^2 * KL(p_teacher || p_student) with p_student = softmax(z_s / T_s).
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    return F.kl_div(
        F.log_softmax(student_logits.float() / student_temperature, dim=-1),
        teacher_probs,
        reduction="batchmean",
    ) * student_temperature ** 2

def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute reverse KL distillation when student and teacher share a vocab."""
    student_temperature = float(student_temperature)
    teacher_temperature = float(teacher_temperature)
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    # L = T^2 * KL(p_student || p_teacher).
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    return F.kl_div(
        teacher_log_probs,
        F.softmax(student_logits.float() / student_temperature, dim=-1),
        reduction="batchmean",
    ) * student_temperature ** 2


def jensen_shannon_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between matched student and teacher distributions."""
    student_temperature = float(student_temperature)
    teacher_temperature = float(teacher_temperature)
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    s = F.softmax(student_logits.float() / student_temperature, dim=-1)
    with torch.no_grad():
        t = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    m = 0.5 * (s + t)
    # JSD(s, t) = 0.5 * KL(s || m) + 0.5 * KL(t || m), where m = 0.5 * (s + t).
    return 0.5 * (
        F.kl_div(s.log(), m, reduction="batchmean") +
        F.kl_div(t.log(), m, reduction="batchmean")
    ) * student_temperature ** 2


DISTILLATION_LOSSES = {
    "cka_loss": cka_loss,
    "uld_loss": uld_loss,
    "forward_kl": forward_kl,
    "reverse_kl": reverse_kl,
    "jensen_shannon_divergence": jensen_shannon_divergence,
}
