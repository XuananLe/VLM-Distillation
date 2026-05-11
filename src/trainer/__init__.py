__all__ = ["DistillationTrainer", "SmolVLMSFTTrainer"]


def __getattr__(name: str):
    if name == "DistillationTrainer":
        from .distillation_trainer import DistillationTrainer

        return DistillationTrainer
    if name == "SmolVLMSFTTrainer":
        from .sft_trainer import SmolVLMSFTTrainer

        return SmolVLMSFTTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
