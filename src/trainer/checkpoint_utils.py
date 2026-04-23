import os

from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME

def update_best_checkpoint_by_train_ce(trainer, trial) -> None:
    """Update TrainerState best-checkpoint metadata using train CE when eval is disabled."""
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

__all__ = [
    "update_best_checkpoint_by_train_ce",
]
