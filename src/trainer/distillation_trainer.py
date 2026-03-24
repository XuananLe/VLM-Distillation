import io
import os
import inspect
from contextlib import nullcontext, redirect_stdout

import torch
from transformers import PreTrainedModel

from src.components.forward_utils import (
    forward_with_kwarg_retry,
    get_decoder_hidden_states,
    infer_batch_size,
    pool_model_hidden_states,
)
from src.components.skc import linear_cka_loss
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.trainer.distillation_utils import (
    build_teacher_batches,
    capture_layer_outputs,
    pool_vision_representations,
    resolve_gradnorm_reference_params,
    setup_layer_matching,
    update_gradnorm_weights,
)


class DistillationTrainer(SmolVLMSFTTrainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        loss_weighting: str = "fixed",
        gradnorm_alpha: float = 1.5,
        gradnorm_lr: float = 0.025,
        layer_distill_source: str = "none",
        layer_distill_weight: float = 0.0,
        layer_match_json_path: str | None = None,
        layer_match_topk: int = 1,
        student_layer_indices: list[int] | None = None,
        teacher_layer_indices: list[int] | None = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        self.loss_function = loss_function
        self.distillation_loss_params = inspect.signature(self.distillation_loss_fn).parameters

        if teacher_model is None:
            raise ValueError("teacher_model must be provided for distillation.")
        if isinstance(teacher_model, (list, tuple)):
            self.teacher_models = list(teacher_model)
        else:
            self.teacher_models = [teacher_model]
        for model in self.teacher_models:
            model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.loss_weighting = loss_weighting
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self.non_lora_require_grad_only = True
        self.layer_distill_source = layer_distill_source
        self.layer_distill_weight = layer_distill_weight
        self.layer_match_json_path = layer_match_json_path
        self.layer_match_topk = layer_match_topk
        self.gradnorm_eps = 1e-8
        self.gradnorm_weights = {}
        self.gradnorm_initial_losses = {}
        self.gradnorm_active = False
        self.gradnorm_reference_params = None
        self.gradnorm_reference_desc = None
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0

        self.layer_distillation_enabled = (
            layer_distill_source in {"vision", "model"}
            and layer_distill_weight > 0.0
            and (bool(student_layer_indices) or bool(layer_match_json_path))
        )
        if self.layer_distillation_enabled and teacher_layer_indices is None and not self.layer_match_json_path:
            raise ValueError("--teacher_layer_indices must be provided when layer distillation is enabled.")

        self.student_layer_indices = []
        self.teacher_layer_indices = teacher_layer_indices
        self.teacher_layer_soft_matches = []

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
        print(f"  - Loss weighting: {loss_weighting}")
        if self.layer_distillation_enabled:
            self.student_layer_indices, self.teacher_layer_soft_matches = setup_layer_matching(
                self.model,
                self.teacher_models,
                self.layer_match_json_path,
                self.layer_match_topk,
                self.layer_distill_source,
                student_layer_indices,
                self.teacher_layer_indices,
            )
            print("  - Layer distillation: enabled")
            print(f"  - Layer distill source: {self.layer_distill_source}")
            print(f"  - Layer distill weight: {self.layer_distill_weight}")
            print(f"  - Student {self.layer_distill_source} layers: {self.student_layer_indices}")
            for teacher_index, soft_matches in enumerate(self.teacher_layer_soft_matches):
                layer_pairs = [
                    (match["student_layer_index"], match["teacher_layer_indices"][0])
                    for match in soft_matches
                ]
                print(f"  - Teacher {teacher_index} layer pairs: {layer_pairs}")
        else:
            print("  - Layer distillation: disabled")
        if self.loss_weighting == "gradnorm":
            if self.alpha > 0.0:
                self.gradnorm_weights["distillation"] = float(self.alpha)
            if self.layer_distillation_enabled:
                self.gradnorm_weights[self.layer_distill_source] = float(self.layer_distill_weight)
            self.gradnorm_active = len(self.gradnorm_weights) > 1
            print("  - GradNorm: enabled")
            print(f"  - GradNorm alpha: {self.gradnorm_alpha}")
            print(f"  - GradNorm lr: {self.gradnorm_lr}")
            print(f"  - Initial auxiliary weights: {self.gradnorm_weights}")
            if self.gradnorm_active:
                print("  - GradNorm mode: adaptive auxiliary weighting with fixed CE weight = 1.0")
            else:
                print("  - GradNorm mode: inactive (<2 auxiliary losses); using fixed CE=1.0 + auxiliary weights")
        else:
            print("  - GradNorm: disabled")

    def _compute_single_teacher_loss(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        student_mask = student_labels != -100
        teacher_mask = teacher_labels != -100

        if self.loss_function == "uld_loss":
            sample_losses = []
            for i in range(student_logits.size(0)):
                student_logits_masked = student_logits[i][student_mask[i]]
                teacher_logits_masked = teacher_logits[i][teacher_mask[i]]
                min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
                if min_len > 0:
                    sample_losses.append(
                        self.distillation_loss_fn(
                            student_logits=student_logits_masked[:min_len],
                            teacher_logits=teacher_logits_masked[:min_len],
                            temperature=self.temperature,
                        )
                    )
            return torch.stack(sample_losses).mean() if sample_losses else student_logits.new_zeros(())

        student_logits_masked = student_logits.view(-1, student_logits.size(-1))[student_mask.view(-1)]
        teacher_logits_masked = teacher_logits.view(-1, teacher_logits.size(-1))[teacher_mask.view(-1)]
        student_labels_masked = student_labels.view(-1)[student_mask.view(-1)]
        min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
        loss_kwargs = dict(
            student_logits=student_logits_masked[:min_len],
            teacher_logits=teacher_logits_masked[:min_len],
            temperature=self.temperature,
        )
        if "labels" in self.distillation_loss_params:
            loss_kwargs["labels"] = student_labels_masked[:min_len]
        return self.distillation_loss_fn(**loss_kwargs)

    def _compute_student_representations(self, model, student_inputs, student_layer_outputs, student_outputs):
        """Extract and pool student model representations for layer distillation."""
        if not self.layer_distillation_enabled:
            return None

        if self.layer_distill_source == "vision":
            student_batch_size = infer_batch_size(student_inputs)
            return pool_vision_representations(
                student_layer_outputs,
                self.student_layer_indices,
                student_batch_size,
                student_inputs,
            )
        else:
            student_hidden_states = get_decoder_hidden_states(student_outputs)
            student_attention_mask = student_inputs.get("attention_mask")
            return {
                layer_index: pool_model_hidden_states(student_hidden_states[layer_index], student_attention_mask)
                for layer_index in self.student_layer_indices
            }

    def _compute_teacher_forward_and_layer_distillation(
        self,
        teacher_index,
        teacher_model,
        teacher_inputs,
        teacher_labels,
        student_logits,
        student_labels,
        student_layer_representations,
        output_hidden_states,
    ):
        """Forward pass through teacher and compute both distillation loss and layer distillation loss."""
        teacher_hook_context = nullcontext(None)
        soft_matches = None
        teacher_layer_indices = []

        if self.layer_distillation_enabled:
            soft_matches = self.teacher_layer_soft_matches[teacher_index]
            teacher_layer_indices = sorted(
                {
                    teacher_layer_index
                    for match in soft_matches
                    for teacher_layer_index in match["teacher_layer_indices"]
                }
            )
            if self.layer_distill_source == "vision":
                teacher_hook_context = capture_layer_outputs(teacher_model, teacher_layer_indices)

        with teacher_hook_context as teacher_layer_outputs:
            with torch.no_grad():
                stdout_context = (
                    redirect_stdout(io.StringIO())
                    if getattr(teacher_model, "_suppress_forward_stdout", False)
                    else nullcontext()
                )
                with stdout_context:
                    teacher_outputs = forward_with_kwarg_retry(
                        teacher_model,
                        {**teacher_inputs, "return_dict": True, "output_hidden_states": output_hidden_states},
                    )

        # Compute distillation loss
        teacher_loss = self._compute_single_teacher_loss(
            student_logits=student_logits,
            student_labels=student_labels,
            teacher_logits=teacher_outputs.logits.detach(),
            teacher_labels=teacher_labels,
        )

        # Compute layer distillation loss
        layer_loss = None
        if self.layer_distillation_enabled:
            if self.layer_distill_source == "vision":
                teacher_batch_size = infer_batch_size(teacher_inputs)
                teacher_layer_representations = pool_vision_representations(
                    teacher_layer_outputs,
                    teacher_layer_indices,
                    teacher_batch_size,
                    teacher_inputs,
                )
            else:
                teacher_hidden_states = get_decoder_hidden_states(teacher_outputs)
                teacher_attention_mask = teacher_inputs.get("attention_mask")
                teacher_layer_representations = {
                    layer_index: pool_model_hidden_states(teacher_hidden_states[layer_index], teacher_attention_mask)
                    for layer_index in teacher_layer_indices
                }

            soft_match_losses = []
            for match in soft_matches:
                weighted_losses = [
                    weight * linear_cka_loss(
                        student_layer_representations[match["student_layer_index"]],
                        teacher_layer_representations[teacher_layer_index],
                    )
                    for teacher_layer_index, weight in zip(
                        match["teacher_layer_indices"],
                        match["teacher_layer_weights"],
                    )
                ]
                if weighted_losses:
                    soft_match_losses.append(torch.stack(weighted_losses).sum())
            if soft_match_losses:
                layer_loss = torch.stack(soft_match_losses).mean()

        return teacher_loss, layer_loss


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        target_device = inputs["input_ids"].device
        for index, teacher_model in enumerate(self.teacher_models):
            teacher_device = next(teacher_model.parameters()).device
            if teacher_device != target_device:
                self.teacher_models[index] = teacher_model.to(target_device)

        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}
        teacher_batches = build_teacher_batches(
            inputs,
            student_inputs,
            len(self.teacher_models),
        )

        # Student forward pass
        output_hidden_states = self.layer_distillation_enabled and self.layer_distill_source == "model"
        student_hook_context = (
            capture_layer_outputs(model, self.student_layer_indices)
            if self.layer_distillation_enabled and self.layer_distill_source == "vision"
            else nullcontext(None)
        )
        with student_hook_context as student_layer_outputs:
            student_outputs = forward_with_kwarg_retry(
                model,
                {**student_inputs, "return_dict": True, "output_hidden_states": output_hidden_states},
            )
        student_logits = student_outputs.logits

        # Extract student representations for layer distillation
        student_layer_representations = self._compute_student_representations(
            model, student_inputs, student_layer_outputs, student_outputs
        )

        # Teacher forward passes and loss computation
        teacher_losses = []
        layer_distillation_losses = []
        for teacher_index, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(
            zip(self.teacher_models, teacher_batches)
        ):
            teacher_loss, layer_loss = self._compute_teacher_forward_and_layer_distillation(
                teacher_index,
                teacher_model,
                teacher_inputs,
                teacher_labels,
                student_logits,
                student_inputs["labels"],
                student_layer_representations,
                output_hidden_states,
            )
            teacher_losses.append(teacher_loss)
            if layer_loss is not None:
                layer_distillation_losses.append(layer_loss)

        distillation_loss = torch.stack(teacher_losses).mean()
        layer_distillation_loss = (
            torch.stack(layer_distillation_losses).mean()
            if layer_distillation_losses
            else distillation_loss.new_zeros(())
        )

        ce_loss = student_outputs.loss
        if not model.training:
            batch_size = infer_batch_size(student_inputs)
            self.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
            self.eval_ce_loss_count += batch_size
        if self.loss_weighting == "gradnorm":
            aux_losses = {}
            if "distillation" in self.gradnorm_weights:
                aux_losses["distillation"] = distillation_loss
            if self.layer_distill_source in self.gradnorm_weights:
                aux_losses[self.layer_distill_source] = layer_distillation_loss

            if model.training and self.gradnorm_active and self.gradnorm_reference_params is None:
                (
                    self.gradnorm_reference_params,
                    self.gradnorm_reference_desc,
                ) = resolve_gradnorm_reference_params(
                    model,
                    layer_distill_source=self.layer_distill_source,
                    student_layer_indices=self.student_layer_indices,
                )
                print(f"  - GradNorm reference params: {self.gradnorm_reference_desc}")

            update_gradnorm_weights(
                model,
                aux_losses,
                self.gradnorm_reference_params,
                self.gradnorm_weights,
                self.gradnorm_initial_losses,
                gradnorm_active=self.gradnorm_active,
                gradnorm_eps=self.gradnorm_eps,
                gradnorm_alpha=self.gradnorm_alpha,
                gradnorm_lr=self.gradnorm_lr,
            )

            loss = ce_loss
            for name, aux_loss in aux_losses.items():
                loss = loss + aux_loss.new_tensor(self.gradnorm_weights[name]) * aux_loss
        else:
            loss = (
                self.alpha * distillation_loss
                + (1 - self.alpha) * ce_loss
                + self.layer_distill_weight * layer_distillation_loss
            )

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if self.layer_distillation_enabled:
                metrics[f"{self.layer_distill_source}_layer_distill_loss"] = layer_distillation_loss.item()
            if self.loss_weighting == "gradnorm":
                for name, value in self.gradnorm_weights.items():
                    metrics[f"gradnorm_w_{name}"] = value
            self.log(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def log(self, logs: dict[str, float], start_time=None) -> None:
        if self.eval_ce_loss_count > 0:
            eval_prefixes = [
                key[:-5]
                for key in logs
                if key.startswith("eval") and key.endswith("_loss") and not key.endswith("_ce_loss")
            ]
            if eval_prefixes:
                stats = torch.tensor(
                    [self.eval_ce_loss_sum, float(self.eval_ce_loss_count)],
                    device=self.args.device,
                    dtype=torch.float64,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                logs = dict(logs)
                eval_ce_loss = (stats[0] / stats[1]).item()
                for prefix in eval_prefixes:
                    logs[f"{prefix}_ce_loss"] = eval_ce_loss
                self.eval_ce_loss_sum = 0.0
                self.eval_ce_loss_count = 0

        super().log(logs, start_time=start_time)

    def _load_best_model(self):
        super()._load_best_model()

        if not getattr(self.args, "lora_enable", False):
            return

        checkpoint_dir = self.state.best_model_checkpoint
        if not checkpoint_dir:
            return

        non_lora_path = os.path.join(checkpoint_dir, "non_lora_state_dict.bin")
        if not os.path.exists(non_lora_path):
            return

        non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
        _, unexpected_keys = self.model.load_state_dict(non_lora_state_dict, strict=False)
        if unexpected_keys:
            print(f"Loaded best non-LoRA weights with unexpected keys: {unexpected_keys}")
