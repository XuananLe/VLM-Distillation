import os

import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME


def update_eval_ce_logs(trainer, logs: dict[str, float]) -> dict[str, float]:
    if trainer.eval_ce_loss_count <= 0:
        return logs

    eval_prefixes = [
        key[:-5]
        for key in logs
        if key.startswith("eval") and key.endswith("_loss") and not key.endswith("_ce_loss")
    ]
    if not eval_prefixes:
        return logs

    stats = torch.tensor(
        [trainer.eval_ce_loss_sum, float(trainer.eval_ce_loss_count)],
        device=trainer.args.device,
        dtype=torch.float64,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    updated_logs = dict(logs)
    eval_ce_loss = (stats[0] / stats[1]).item()
    for prefix in eval_prefixes:
        updated_logs[f"{prefix}_ce_loss"] = eval_ce_loss

    trainer.eval_ce_loss_sum = 0.0
    trainer.eval_ce_loss_count = 0
    return updated_logs


def update_best_checkpoint_by_train_ce(trainer, trial) -> None:
    if not trainer.tracks_best_checkpoint_by_train_ce() or trainer.latest_train_ce_loss is None:
        return

    current_best = trainer.state.best_metric
    if current_best is not None and trainer.latest_train_ce_loss >= current_best:
        return

    checkpoint_dir = os.path.join(
        trainer._get_output_dir(trial=trial),
        f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}",
    )
    trainer.state.best_metric = trainer.latest_train_ce_loss
    trainer.state.best_model_checkpoint = checkpoint_dir
    trainer.state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))
    print(
        "Updated best checkpoint by train_ce_loss: "
        f"{checkpoint_dir} (train_ce_loss={trainer.latest_train_ce_loss:.6f})"
    )


def load_best_non_lora_weights(trainer) -> None:
    if not getattr(trainer.args, "lora_enable", False):
        return

    checkpoint_dir = trainer.state.best_model_checkpoint
    if not checkpoint_dir:
        return

    non_lora_path = os.path.join(checkpoint_dir, "non_lora_state_dict.bin")
    if not os.path.exists(non_lora_path):
        return

    non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
    _, unexpected_keys = trainer.model.load_state_dict(non_lora_state_dict, strict=False)
    if unexpected_keys:
        print(f"Loaded best non-LoRA weights with unexpected keys: {unexpected_keys}")


__all__ = [
    "load_best_non_lora_weights",
    "update_best_checkpoint_by_train_ce",
    "update_eval_ce_logs",
]
