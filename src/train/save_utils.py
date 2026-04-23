import logging

import torch
import transformers


def maybe_zero_3(param, ignore_status=False, name=None):
    """Materialize a possibly ZeRO-sharded parameter on CPU before saving or inspection."""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collect the model state and dump it to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        _save_processing_assets(trainer, output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)
        _save_processing_assets(trainer, output_dir)


def _save_processing_assets(trainer: transformers.Trainer, output_dir: str) -> None:
    """Save one processor/tokenizer-like asset alongside the model when the main process owns I/O."""
    if not getattr(trainer.args, "should_save", False):
        return
    if hasattr(trainer, "is_world_process_zero") and not trainer.is_world_process_zero():
        return

    saved = set()
    for asset in (
        getattr(trainer, "processing_class", None),
        getattr(trainer, "processor", None),
        getattr(trainer, "tokenizer", None),
    ):
        if asset is None or id(asset) in saved:
            continue
        if hasattr(asset, "save_pretrained"):
            asset.save_pretrained(output_dir)
            saved.add(id(asset))
            break


__all__ = [
    "_save_processing_assets",
    "maybe_zero_3",
    "safe_save_model_for_hf_trainer",
]
