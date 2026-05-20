import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.loss import uld_loss


def test_uld_loss_input_output_example():
    student_logits = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
        ]
    )
    teacher_logits = torch.tensor(
        [
            [1.0, 0.0, 2.0, -1.0, -2.0],
            [3.0, 0.0, 1.0, 2.0, -1.0],
        ]
    )

    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values
    student_sorted_padded = F.pad(
        student_sorted,
        (0, teacher_sorted.size(-1) - student_sorted.size(-1)),
    )
    per_token_loss = (student_sorted_padded - teacher_sorted).abs().sum(dim=-1)
    loss = uld_loss(student_logits, teacher_logits)

    print("student_logits:", student_logits)
    print("teacher_logits:", teacher_logits)
    print("student_probs:", student_probs)
    print("teacher_probs:", teacher_probs)
    print("student_sorted:", student_sorted)
    print("teacher_sorted:", teacher_sorted)
    print("student_sorted_padded:", student_sorted_padded)
    print("per_token_uld_loss:", per_token_loss)
    print("mean_uld_loss:", float(loss))

    torch.testing.assert_close(loss, per_token_loss.mean())


if __name__ == "__main__":
    test_uld_loss_input_output_example()
