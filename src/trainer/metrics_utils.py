import torch


def build_distillation_train_metrics(
    *,
    loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    ce_loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": loss.item(),
        "ce_loss": ce_loss.item(),
        "kd_loss": distillation_loss.item(),
    }


__all__ = ["build_distillation_train_metrics"]
