import contextlib

import torch


def initialize_gradnorm_weights(
    task_names: list[str],
    *,
    gradnorm_eps: float,
) -> dict[str, float]:
    """Initialize all active GradNorm tasks equally, matching the paper's w_i(0)=1."""
    gradnorm_weights = {name: 1.0 for name in task_names}
    normalize_gradnorm_weights(gradnorm_weights, gradnorm_eps=gradnorm_eps)
    return gradnorm_weights


def normalize_gradnorm_weights(
    gradnorm_weights: dict[str, float],
    *,
    gradnorm_eps: float,
) -> None:
    """Clamp weights positive and renormalize them to sum to the active task count."""
    if not gradnorm_weights:
        return

    weight_names = list(gradnorm_weights.keys())
    for name in weight_names:
        gradnorm_weights[name] = float(max(gradnorm_weights[name], gradnorm_eps))

    weight_sum = sum(gradnorm_weights[name] for name in weight_names)
    renorm = len(weight_names) / max(weight_sum, gradnorm_eps)
    for name in weight_names:
        gradnorm_weights[name] *= renorm


def _distributed_average_(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
    return tensor


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
    task_losses: dict[str, torch.Tensor],
    reference_params: list[torch.nn.Parameter],
    gradnorm_weights: dict[str, float],
    gradnorm_initial_losses: dict[str, float],
    *,
    gradnorm_active: bool,
    gradnorm_eps: float,
    gradnorm_alpha: float,
    gradnorm_lr: float,
) -> None:
    """
    Update task weights using the GradNorm objective from Chen et al. (2018).

    The target gradient norms are treated as constants during differentiation, as
    described in Equation 2 / Algorithm 1 of the paper.
    """
    if not gradnorm_active or not task_losses or not torch.is_grad_enabled():
        return

    if not reference_params:
        return

    weight_names = list(task_losses.keys())
    current_losses = {}
    for name in weight_names:
        current_loss = task_losses[name].detach().float().clamp_min(gradnorm_eps)
        if name not in gradnorm_initial_losses:
            initial_loss = _distributed_average_(current_loss.clone())
            gradnorm_initial_losses[name] = float(initial_loss.item())
        current_losses[name] = current_loss

    with suspend_deepspeed_backward_hooks(model):
        base_grad_norms = []
        for name in weight_names:
            grads = torch.autograd.grad(
                task_losses[name],
                reference_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grad_sq_norm = None
            for grad in grads:
                if grad is None:
                    continue
                term = grad.detach().float().pow(2).sum()
                grad_sq_norm = term if grad_sq_norm is None else grad_sq_norm + term
            if grad_sq_norm is None:
                grad_sq_norm = torch.zeros((), device=reference_params[0].device, dtype=torch.float32)
            base_grad_norms.append(grad_sq_norm.sqrt())

        base_grad_norm_tensor = torch.stack(base_grad_norms)
        current_weight_tensor = torch.tensor(
            [gradnorm_weights[name] for name in weight_names],
            device=reference_params[0].device,
            dtype=torch.float32,
            requires_grad=True,
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
        grad_loss = torch.abs(grad_norm_tensor - grad_norm_target.detach()).sum()
        weight_grads = torch.autograd.grad(grad_loss, current_weight_tensor)[0]

    with torch.no_grad():
        updated_weight_tensor = (current_weight_tensor - gradnorm_lr * weight_grads).clamp_min(gradnorm_eps)
        _distributed_average_(updated_weight_tensor)
        for name, value in zip(weight_names, updated_weight_tensor.tolist()):
            gradnorm_weights[name] = float(value)
        normalize_gradnorm_weights(gradnorm_weights, gradnorm_eps=gradnorm_eps)

    # Clean up intermediate tensors to prevent GPU memory leaks
    del base_grad_norms, base_grad_norm_tensor, grad_norm_tensor, weight_grads, current_weight_tensor
