from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import torch

from src.trainer.distillation_utils import (
    capture_layer_outputs,
    compute_student_representations,
    compute_teacher_layer_distillation_loss,
    setup_layer_matching,
)


class LayerDistiller(Protocol):
    enabled: bool
    source: str | None
    weight: float
    student_layer_indices: list[int]
    teacher_layer_soft_matches: list[list[dict]]

    def student_forward_kwargs(self) -> dict:
        ...

    def capture_student(self, model):
        ...

    def student_representations(
        self,
        *,
        student_inputs,
        student_outputs,
        captured_outputs,
    ):
        ...

    def compute_loss(
        self,
        *,
        teacher_models,
        live_teacher_batches,
        student_layer_representations,
    ):
        ...


@dataclass(slots=True)
class NoLayerDistiller:
    enabled: bool = False
    source: str | None = None
    weight: float = 0.0
    student_layer_indices: list[int] = field(default_factory=list)
    teacher_layer_soft_matches: list[list[dict]] = field(default_factory=list)

    def student_forward_kwargs(self) -> dict:
        return {}

    def capture_student(self, model):
        del model
        return nullcontext(None)

    def student_representations(
        self,
        *,
        student_inputs,
        student_outputs,
        captured_outputs,
    ):
        del student_inputs, student_outputs, captured_outputs
        return None

    def compute_loss(
        self,
        *,
        teacher_models,
        live_teacher_batches,
        student_layer_representations,
    ):
        del teacher_models, live_teacher_batches, student_layer_representations
        return None


@dataclass(slots=True)
class ModelLayerDistiller:
    weight: float
    student_layer_indices: list[int]
    teacher_layer_soft_matches: list[list[dict]]
    enabled: bool = True
    source: str = "model"

    def student_forward_kwargs(self) -> dict:
        return {"output_hidden_states": True}

    def capture_student(self, model):
        del model
        return nullcontext(None)

    def student_representations(
        self,
        *,
        student_inputs,
        student_outputs,
        captured_outputs,
    ):
        del captured_outputs
        return compute_student_representations(
            self.source,
            self.student_layer_indices,
            student_inputs,
            None,
            student_outputs,
        )

    def compute_loss(
        self,
        *,
        teacher_models,
        live_teacher_batches,
        student_layer_representations,
    ):
        return compute_active_layer_distillation_loss(
            source=self.source,
            output_hidden_states=True,
            teacher_models=teacher_models,
            live_teacher_batches=live_teacher_batches,
            teacher_layer_soft_matches=self.teacher_layer_soft_matches,
            student_layer_representations=student_layer_representations,
        )


@dataclass(slots=True)
class VisionLayerDistiller:
    weight: float
    student_layer_indices: list[int]
    teacher_layer_soft_matches: list[list[dict]]
    enabled: bool = True
    source: str = "vision"

    def student_forward_kwargs(self) -> dict:
        return {"output_hidden_states": False}

    def capture_student(self, model):
        return capture_layer_outputs(model, self.student_layer_indices)

    def student_representations(
        self,
        *,
        student_inputs,
        student_outputs,
        captured_outputs,
    ):
        del student_outputs
        return compute_student_representations(
            self.source,
            self.student_layer_indices,
            student_inputs,
            captured_outputs,
            None,
        )

    def compute_loss(
        self,
        *,
        teacher_models,
        live_teacher_batches,
        student_layer_representations,
    ):
        return compute_active_layer_distillation_loss(
            source=self.source,
            output_hidden_states=False,
            teacher_models=teacher_models,
            live_teacher_batches=live_teacher_batches,
            teacher_layer_soft_matches=self.teacher_layer_soft_matches,
            student_layer_representations=student_layer_representations,
        )


def create_layer_distiller(
    *,
    model,
    teacher_models,
    layer_distill_source: str,
    layer_distill_weight: float,
    layer_match_json_path: str | None,
    layer_match_topk: int,
    student_layer_indices: list[int],
    teacher_layer_indices: list[int],
) -> LayerDistiller:
    if layer_distill_source == "none":
        if (
            layer_distill_weight > 0.0
            or layer_match_json_path
            or student_layer_indices
            or teacher_layer_indices
        ):
            raise ValueError(
                "Layer distillation arguments were provided while --layer_distill_source is `none`."
            )
        return NoLayerDistiller()

    if layer_distill_source not in {"vision", "model"}:
        raise ValueError(
            f"--layer_distill_source must be `none`, `vision`, or `model`, got {layer_distill_source!r}."
        )
    if layer_distill_weight <= 0.0:
        raise ValueError(
            "--layer_distill_weight must be > 0 when layer distillation is enabled."
        )
    if not teacher_models:
        raise ValueError("Layer distillation requires live teacher models to be loaded.")
    if not student_layer_indices and not layer_match_json_path:
        raise ValueError(
            "Layer distillation requires --student_layer_indices or --layer_match_json_path."
        )
    if not teacher_layer_indices and not layer_match_json_path:
        raise ValueError(
            "Layer distillation requires --teacher_layer_indices or --layer_match_json_path."
        )

    resolved_student_indices, teacher_layer_soft_matches = setup_layer_matching(
        model,
        teacher_models,
        layer_match_json_path,
        layer_match_topk,
        layer_distill_source,
        student_layer_indices,
        teacher_layer_indices,
    )
    if layer_distill_source == "vision":
        return VisionLayerDistiller(
            weight=layer_distill_weight,
            student_layer_indices=resolved_student_indices,
            teacher_layer_soft_matches=teacher_layer_soft_matches,
        )
    return ModelLayerDistiller(
        weight=layer_distill_weight,
        student_layer_indices=resolved_student_indices,
        teacher_layer_soft_matches=teacher_layer_soft_matches,
    )


def compute_active_layer_distillation_loss(
    *,
    source: str,
    output_hidden_states: bool,
    teacher_models,
    live_teacher_batches,
    teacher_layer_soft_matches: list[list[dict]],
    student_layer_representations,
):
    if live_teacher_batches is None:
        raise ValueError(
            "Layer distillation requires live teacher inputs. "
            "Provide teacher processors alongside cached teacher logits."
        )
    if student_layer_representations is None:
        raise ValueError("Layer distillation is enabled but student representations are missing.")
    if len(teacher_models) != len(live_teacher_batches):
        raise ValueError(
            "Layer distillation teacher model count does not match teacher batch count: "
            f"{len(teacher_models)} != {len(live_teacher_batches)}."
        )
    if len(teacher_models) != len(teacher_layer_soft_matches):
        raise ValueError(
            "Layer distillation teacher model count does not match layer-match count: "
            f"{len(teacher_models)} != {len(teacher_layer_soft_matches)}."
        )

    layer_distillation_losses = []
    for teacher_model, (teacher_inputs, _teacher_labels), soft_matches in zip(
        teacher_models,
        live_teacher_batches,
        teacher_layer_soft_matches,
        strict=True,
    ):
        layer_loss = compute_teacher_layer_distillation_loss(
            teacher_model=teacher_model,
            teacher_inputs=teacher_inputs,
            teacher_layer_soft_matches=soft_matches,
            layer_distill_source=source,
            student_layer_representations=student_layer_representations,
            output_hidden_states=output_hidden_states,
        )
        layer_distillation_losses.append(layer_loss)

    if not layer_distillation_losses:
        raise ValueError("Layer distillation is enabled but no layer losses were produced.")
    return torch.stack(layer_distillation_losses).mean()
