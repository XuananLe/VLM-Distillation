import torch
import torch.nn.functional as F
from einops import einsum

from src.components.cka import linear_cka_loss
from src.components.trie_wasserstein import TrieWassersteinLoss


def resolve_temperatures(
    *,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> tuple[float, float]:
    """Clamp the configured student and teacher temperatures away from zero."""
    resolved_student_temperature = max(float(student_temperature), torch.finfo(torch.float32).eps)
    resolved_teacher_temperature = max(float(teacher_temperature), torch.finfo(torch.float32).eps)
    return resolved_student_temperature, resolved_teacher_temperature


class FunctionalDistillationLoss:
    """Wrap a function-based KD loss in the trainer's object interface."""
    def __init__(self, loss_function: str):
        """Bind one global loss function name so trainers can call a uniform API."""
        self.loss_function = loss_function
        self.loss_fn = globals()[loss_function]

    def compute_loss(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_index: int | None = None,
    ) -> torch.Tensor:
        """Compute one function-based KD loss for aligned student and teacher logits."""
        del teacher_index
        return self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )

    def compute_logit_grad(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_index: int | None = None,
        loss_function: str | None = None,
    ) -> torch.Tensor:
        """Return the student-logit gradient for a function-based KD loss."""
        del teacher_index, loss_function
        return distillation_logit_grad(
            self.loss_function,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )

    def prepare_teacher_batch(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor | None = None,
        teacher_index: int | None = None,
    ) -> None:
        """Accept the shared loss interface even though function losses keep no batch state."""
        del student_logits, teacher_logits, teacher_labels, teacher_index


class TrieWassersteinDistillationLoss:
    """Own one trie-Wasserstein module per teacher tokenizer."""
    def __init__(
        self,
        *,
        student_tokenizer,
        teacher_tokenizers,
        trie_wasserstein_rho: float,
        trie_wasserstein_topk: int,
    ):
        """Build the per-teacher trie loss modules used by cross-tokenizer KD."""
        if student_tokenizer is None:
            raise ValueError("Trie Wasserstein loss requires a student tokenizer.")
        if not teacher_tokenizers:
            raise ValueError("Trie Wasserstein loss requires teacher tokenizers.")
        self.loss_modules = [
            TrieWassersteinLoss(
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                rho=trie_wasserstein_rho,
                topk=trie_wasserstein_topk,
            )
            for teacher_tokenizer in teacher_tokenizers
        ]

    def _select_loss_module(self, teacher_index: int | None) -> TrieWassersteinLoss:
        """Return the trie loss module associated with one teacher index."""
        if teacher_index is None:
            raise ValueError("Trie Wasserstein loss requires a teacher index.")
        return self.loss_modules[teacher_index]

    def prepare_teacher_batch(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor | None = None,
        teacher_index: int | None = None,
    ) -> None:
        """Prepare the selected trie module for the current teacher batch."""
        self._select_loss_module(teacher_index).prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
            teacher_labels=teacher_labels,
        )

    def compute_loss(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_index: int | None = None,
    ) -> torch.Tensor:
        """Compute trie-Wasserstein KD against the selected teacher tokenizer."""
        loss_module = self._select_loss_module(teacher_index)
        loss_module.prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
        )
        return loss_module(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )

    def compute_logit_grad(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        teacher_index: int | None = None,
        loss_function: str | None = None,
    ) -> torch.Tensor:
        """Return dL/d(student_logits) for the selected trie-Wasserstein module."""
        del loss_function
        loss_module = self._select_loss_module(teacher_index)
        loss_module.prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
        )
        return loss_module.compute_logit_grad(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )


def build_distillation_loss(
    *,
    loss_function: str,
    student_tokenizer=None,
    teacher_tokenizers=None,
    trie_wasserstein_rho: float = 0.7,
    trie_wasserstein_topk: int = 64,
):
    """Construct the configured KD loss object from CLI settings and tokenizers."""
    if loss_function == "trie_wasserstein_loss":
        return TrieWassersteinDistillationLoss(
            student_tokenizer=student_tokenizer,
            teacher_tokenizers=teacher_tokenizers,
            trie_wasserstein_rho=trie_wasserstein_rho,
            trie_wasserstein_topk=trie_wasserstein_topk,
        )
    if loss_function not in globals():
        raise ValueError(f"Unknown distillation loss: {loss_function!r}")
    return FunctionalDistillationLoss(loss_function)


def distillation_logit_grad(
    loss_function: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Return dL/d(student_logits) for the configured KD loss in aligned logit space."""
    if loss_function == "cka_loss":
        with torch.enable_grad():
            logits_for_grad = student_logits.detach().clone().requires_grad_(True)
            loss = cka_loss(
                student_logits=logits_for_grad,
                teacher_logits=teacher_logits,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
            )
            return torch.autograd.grad(loss, logits_for_grad, retain_graph=False)[0].float()

    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
    student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)

    if loss_function == "uld_loss":
        # ULD compares sorted probability masses, so its logit gradient is built in
        # sorted-probability space and then mapped back through the softmax Jacobian.
        student_sorted, sort_idx = student_probs.sort(dim=-1, descending=True)
        teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)
        teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

        student_vocab = student_sorted.size(-1)
        teacher_vocab = teacher_sorted.size(-1)
        if student_vocab < teacher_vocab:
            padded_student_sorted = F.pad(student_sorted, (0, teacher_vocab - student_vocab))
            grad_sorted = (padded_student_sorted - teacher_sorted).sign()[..., :student_vocab]
        else:
            if teacher_vocab < student_vocab:
                teacher_sorted = F.pad(teacher_sorted, (0, student_vocab - teacher_vocab))
            grad_sorted = (student_sorted - teacher_sorted).sign()

        grad_probs = torch.zeros_like(student_probs)
        grad_probs.scatter_(dim=-1, index=sort_idx, src=grad_sorted)
        grad_dot = einsum(
            grad_probs,
            student_probs,
            "... vocab, ... vocab -> ...",
        ).unsqueeze(-1)
        # For softmax p = softmax(z / T), dL/dz = p * (dL/dp - <dL/dp, p>) / T.
        return student_probs * (grad_probs - grad_dot) / student_temperature

    if student_logits.size(-1) != teacher_logits.size(-1):
        return torch.zeros_like(student_logits, dtype=torch.float32)

    teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)
    # For KL-style losses in matched vocab space, the pooled logit gradient reduces to
    # (p_student - p_teacher) / T.
    return (student_probs - teacher_probs) / student_temperature


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
    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
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
    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
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
    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    # L = T^2 * KL(p_teacher || p_student) with p_student = softmax(z_s / T_s).
    return F.kl_div(
        F.log_softmax(student_logits / student_temperature, dim=-1),
        F.softmax(teacher_logits / teacher_temperature, dim=-1),
        reduction='batchmean'
    ) * student_temperature ** 2

def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute reverse KL distillation when student and teacher share a vocab."""
    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    # L = T^2 * KL(p_student || p_teacher).
    return F.kl_div(
        F.log_softmax(teacher_logits / teacher_temperature, dim=-1),
        F.softmax(student_logits / student_temperature, dim=-1),
        reduction='batchmean'
    ) * student_temperature ** 2


def jensen_shannon_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between matched student and teacher distributions."""
    student_temperature, teacher_temperature = resolve_temperatures(
        student_temperature=student_temperature,
        teacher_temperature=teacher_temperature,
    )
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    s = F.softmax(student_logits / student_temperature, dim=-1)
    t = F.softmax(teacher_logits / teacher_temperature, dim=-1)
    m = 0.5 * (s + t)
    # JSD(s, t) = 0.5 * KL(s || m) + 0.5 * KL(t || m), where m = 0.5 * (s + t).
    return 0.5 * (
        F.kl_div(s.log(), m, reduction='batchmean') +
        F.kl_div(t.log(), m, reduction='batchmean')
    ) * student_temperature ** 2
