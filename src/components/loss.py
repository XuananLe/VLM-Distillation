import torch
import torch.nn.functional as F

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

