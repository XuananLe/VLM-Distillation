import torch.nn as nn


class TeacherOutputAdapter(nn.Module):
    """Project teacher hidden states into the student hidden space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        teacher_id: str,
    ):
        super().__init__()
        self.teacher_id = teacher_id
        hidden_dim = self._default_hidden_dim(
            teacher_id=teacher_id,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_states):
        return self.proj(hidden_states)

    @staticmethod
    def _default_hidden_dim(
        teacher_id: str,
        input_dim: int,
        output_dim: int,
    ) -> int:
        teacher_name = teacher_id.lower()

        if "qwen/qwen2-vl-2b-instruct" in teacher_name or "qwen2-vl-2b-instruct" in teacher_name:
            return 1024

        if "qwen/qwen2.5-vl-3b-instruct" in teacher_name or "qwen2.5-vl-3b-instruct" in teacher_name:
            return 1536

        return max(input_dim, output_dim)
