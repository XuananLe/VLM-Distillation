import contextlib

import torch

from src.components.forward_utils import unwrap_tensor
from src.components.vision_forward import infer_vision_group_counts, pool_vision_features
from src.utils import find_vision_layer_indices, get_specific_layer


REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags")


def get_base_model(model):
    """Extract the base model, handling PEFT-wrapped models."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def resolve_layer_indices(total_layers: int, layer_indices: list[int], label: str) -> list[int]:
    resolved = []
    for layer_index in layer_indices:
        normalized = total_layers + layer_index if layer_index < 0 else layer_index
        if normalized < 0 or normalized >= total_layers:
            raise IndexError(
                f"{label} layer index {layer_index} resolved to {normalized}, "
                f"but valid range is 0-{total_layers - 1}."
            )
        resolved.append(int(normalized))
    return resolved


def prepare_vision_layer_distillation(
    student_model,
    teacher_models,
    student_layer_indices: list[int],
    teacher_layer_indices: list[int],
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    student_vision_info = find_vision_layer_indices(get_base_model(student_model))
    resolved_student_layer_indices = resolve_layer_indices(
        student_vision_info["total_layers"],
        student_layer_indices,
        "Student vision",
    )

    teacher_layer_pairs = []
    for teacher_index, teacher_model in enumerate(teacher_models):
        teacher_vision_info = find_vision_layer_indices(get_base_model(teacher_model))
        resolved_teacher_layer_indices = resolve_layer_indices(
            teacher_vision_info["total_layers"],
            teacher_layer_indices,
            f"Teacher {teacher_index} vision",
        )
        teacher_layer_pairs.append(list(zip(resolved_student_layer_indices, resolved_teacher_layer_indices)))

    return resolved_student_layer_indices, teacher_layer_pairs

@contextlib.contextmanager
def capture_layer_outputs(model, layer_indices: list[int]):
    raw_outputs = {}
    layer_model = get_base_model(model)

    with contextlib.ExitStack() as stack:
        for layer_index in layer_indices:
            layer, _ = get_specific_layer(layer_model, layer_index)

            def make_hook(index: int):
                def hook(module, hook_inputs, output):
                    del module, hook_inputs
                    raw_outputs[index] = unwrap_tensor(output)
                return hook

            handle = layer.register_forward_hook(make_hook(layer_index))
            stack.callback(handle.remove)
        yield raw_outputs


def pool_vision_representations(
    raw_outputs: dict[int, torch.Tensor],
    layer_indices: list[int],
    batch_size: int,
    model_inputs,
) -> dict[int, torch.Tensor]:
    group_counts = infer_vision_group_counts(model_inputs, batch_size)
    return {
        layer_index: pool_vision_features(
            raw_outputs[layer_index],
            batch_size,
            group_counts=group_counts,
        )
        for layer_index in layer_indices
    }


def build_teacher_batches(inputs, student_inputs, num_teachers: int):
    prefixes = []
    if "teacher_input_ids" in inputs:
        prefixes.append("teacher")
    prefixes.extend(
        f"teacher_{index}"
        for index in range(num_teachers)
        if f"teacher_{index}_input_ids" in inputs
    )

    if not prefixes:
        return [
            (
                {key: value for key, value in student_inputs.items() if key != "labels"},
                student_inputs.get("labels"),
            )
        ]

    batches = []
    for prefix in prefixes:
        teacher_inputs = {}
        for suffix in REQUIRED_TEACHER_INPUTS:
            teacher_inputs[suffix] = inputs[f"{prefix}_{suffix}"]
        for suffix in OPTIONAL_TEACHER_INPUTS:
            key = f"{prefix}_{suffix}"
            if key in inputs:
                teacher_inputs[suffix] = inputs[key]
        batches.append((teacher_inputs, inputs[f"{prefix}_labels"]))
    return batches


@contextlib.contextmanager
def suspend_deepspeed_backward_hooks(model):
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None or not hasattr(optimizer, "_grad_acc_post_hooks"):
        yield
        return

    saved_hooks = list(optimizer._grad_acc_post_hooks)
    saved_enable_backward_allreduce = getattr(model, "enable_backward_allreduce", None)
    optimizer.unregister_grad_acc_post_hooks()
    if saved_enable_backward_allreduce is not None:
        model.enable_backward_allreduce = False
    try:
        yield
    finally:
        optimizer._grad_acc_post_hooks = saved_hooks
        if saved_enable_backward_allreduce is not None:
            model.enable_backward_allreduce = saved_enable_backward_allreduce
        if hasattr(optimizer, "reset_for_new_step"):
            optimizer.reset_for_new_step()


def update_gradnorm_weights(
    model,
    aux_losses: dict[str, torch.Tensor],
    reference_tensor: torch.Tensor,
    gradnorm_weights: dict[str, float],
    gradnorm_initial_losses: dict[str, float],
    *,
    gradnorm_active: bool,
    gradnorm_eps: float,
    gradnorm_alpha: float,
    gradnorm_lr: float,
) -> None:
    if not gradnorm_active or not aux_losses or not torch.is_grad_enabled():
        return

    if reference_tensor is None or not reference_tensor.requires_grad:
        return

    weight_names = list(aux_losses.keys())

    current_losses = {}
    for name in weight_names:
        current_loss = aux_losses[name].detach().float().clamp_min(gradnorm_eps)
        current_losses[name] = current_loss
        gradnorm_initial_losses.setdefault(name, current_loss.item())

    with suspend_deepspeed_backward_hooks(model):
        base_grad_norms = []
        for name in weight_names:
            grad = torch.autograd.grad(
                aux_losses[name],
                reference_tensor,
                retain_graph=True,
                create_graph=False,
            )[0]
            base_grad_norms.append(grad.detach().norm(p=2))

        base_grad_norm_tensor = torch.stack(base_grad_norms)
        current_weight_tensor = torch.stack(
            [reference_tensor.new_tensor(gradnorm_weights[name]) for name in weight_names]
        )
        grad_norm_tensor = current_weight_tensor * base_grad_norm_tensor
        loss_ratio_tensor = torch.stack(
            [
                current_losses[name]
                / current_losses[name].new_tensor(gradnorm_initial_losses[name]).clamp_min(gradnorm_eps)
                for name in weight_names
            ]
        )
        inverse_train_rate = loss_ratio_tensor / loss_ratio_tensor.mean().clamp_min(gradnorm_eps)
        grad_norm_target = grad_norm_tensor.detach().mean() * inverse_train_rate.pow(gradnorm_alpha)
        weight_grads = torch.sign(grad_norm_tensor - grad_norm_target.detach()) * base_grad_norm_tensor

    with torch.no_grad():
        for name, grad in zip(weight_names, weight_grads):
            gradnorm_weights[name] = max(
                gradnorm_weights[name] - gradnorm_lr * grad.item(),
                gradnorm_eps,
            )

        weight_sum = sum(gradnorm_weights[name] for name in weight_names)
        renorm = len(weight_names) / weight_sum
        for name in weight_names:
            gradnorm_weights[name] *= renorm

    weight_tensor = torch.tensor(
        [gradnorm_weights[name] for name in weight_names],
        device=reference_tensor.device,
        dtype=reference_tensor.dtype,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(weight_tensor, op=torch.distributed.ReduceOp.SUM)
        weight_tensor /= torch.distributed.get_world_size()
    for name, value in zip(weight_names, weight_tensor.tolist()):
        gradnorm_weights[name] = float(max(value, gradnorm_eps))
