"""Compatibility facade for training utilities."""

from src.train.model_setup import (
    load_model,
    load_model_and_processor,
    load_processor_and_tokenizer,
    resolve_model_type,
)

__all__ = [
    "load_model",
    "load_model_and_processor",
    "load_processor_and_tokenizer",
    "resolve_model_type",
]
