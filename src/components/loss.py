import torch
import torch.nn.functional as F


def uld_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Universal Logit Distillation (ULD) loss via Wasserstein-1 distance.
    https://arxiv.org/abs/2402.12030

    Supports different vocabulary sizes between student and teacher by sorting
    each distribution in descending order and computing the L1 distance between
    the aligned sorted probability masses (closed-form W1, Eq. 5 in the paper).

    student_logits: (N, V_s) — N tokens, student vocab size V_s
    teacher_logits: (N, V_t) — N tokens, teacher vocab size V_t
    temperature:    softmax temperature (default 1.0 per paper)
    returns scalar Wasserstein-1 loss averaged over tokens
    """
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Sort both distributions in descending order to obtain the
    # rank-aligned coupling (uniform-cost OT closed-form solution).
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    # Pad the smaller vocabulary to match sizes with zeros so that
    # the remaining mass is implicitly assigned to "phantom" tokens.
    V_s, V_t = student_sorted.size(dim=-1), teacher_sorted.size(-1)
    if V_s < V_t:
        student_sorted = F.pad(student_sorted, (0, V_t - V_s))
    elif V_t < V_s:
        teacher_sorted = F.pad(teacher_sorted, (0, V_s - V_t))

    return (student_sorted - teacher_sorted).abs().sum(dim=-1).mean()


def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    student_logits: (batch_size, ...)
    teacher_logits: (batch_size, ...)
    temperature: scaling factor for the logits
    returns (batch_size, ...): Forward KL divergence loss
    """
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    return F.kl_div(
        F.log_softmax(student_logits_scaled, dim=-1),
        F.softmax(teacher_logits_scaled, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    student_logits: (batch_size, ...)
    teacher_logits: (batch_size, ...)
    temperature: scaling factor for the logits
    returns (batch_size, ...): Reverse KL divergence loss
    """
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    return F.kl_div(
        F.log_softmax(teacher_logits_scaled, dim=-1),
        F.softmax(student_logits_scaled, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)


def jensen_shannon_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    student_logits: (batch_size, ...)
    teacher_logits: (batch_size, ...)
    temperature: scaling factor for the logits
    returns (batch_size, ...): Jensen-Shannon divergence loss
    """
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature
    student_probs = F.softmax(student_logits_scaled, dim=-1)
    teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
    mean_probs = 0.5 * (student_probs + teacher_probs)
    kl_student = F.kl_div(F.log_softmax(student_logits_scaled, dim=-1), mean_probs, reduction='batchmean')
    kl_teacher = F.kl_div(F.log_softmax(teacher_logits_scaled, dim=-1), mean_probs, reduction='batchmean')
    return 0.5 * (kl_student + kl_teacher) * (temperature ** 2)

