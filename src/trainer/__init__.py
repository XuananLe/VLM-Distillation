__all__ = ["DistillationTrainer"]


def __getattr__(name: str):
    if name == "DistillationTrainer":
        from .distillation_trainer import DistillationTrainer

        return DistillationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
