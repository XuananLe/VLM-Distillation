import torch
import torch.nn.functional as F


def distillation_logit_grad(
    loss_function: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    temperature = max(float(temperature), torch.finfo(torch.float32).eps)
    student_probs = F.softmax(student_logits.float() / temperature, dim=-1)

    if loss_function == "uld_loss":
        student_sorted, sort_idx = student_probs.sort(dim=-1, descending=True)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
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
        grad_dot = (grad_probs * student_probs).sum(dim=-1, keepdim=True)
        return student_probs * (grad_probs - grad_dot) / temperature

    if student_logits.size(-1) != teacher_logits.size(-1):
        return torch.zeros_like(student_logits, dtype=torch.float32)

    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return (student_probs - teacher_probs) / temperature


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
    student_sorted = student_probs.sort(dim=-1, descending=True).values

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
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
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * temperature ** 2

def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    return F.kl_div(
        F.log_softmax(teacher_logits / temperature, dim=-1),
        F.softmax(student_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * temperature ** 2


def jensen_shannon_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    assert student_logits.shape == teacher_logits.shape, "student_logits and teacher_logits must have the same shape"
    s = F.softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    m = 0.5 * (s + t)
    return 0.5 * (
        F.kl_div(s.log(), m, reduction='batchmean') +
        F.kl_div(t.log(), m, reduction='batchmean')
    ) * temperature ** 2
