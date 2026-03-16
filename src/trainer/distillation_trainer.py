from transformers import Trainer, PreTrainedModel
import inspect
import torch

_TEACHER_KEYS = frozenset({
    "teacher_input_ids",
    "teacher_labels",
    "teacher_attention_mask",
    "teacher_pixel_values",
    "teacher_pixel_attention_mask",
    "teacher_image_grid_thw",
})


class LogitsDistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        ofa_eps: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        self.loss_function = loss_function
        self.distillation_loss_params = inspect.signature(self.distillation_loss_fn).parameters

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.ofa_eps = ofa_eps

        print(f"Distillation Trainer initialized:")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
        if "eps" in self.distillation_loss_params:
            print(f"  - OFA eps: {ofa_eps}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Use the device of the input tensors as the reference, not model.device.
        # With DeepSpeed ZeRO-2/3 the student model's .device property can return
        # an unexpected value because parameters are sharded across ranks.
        target_device = inputs["input_ids"].device
        if next(self.teacher_model.parameters()).device != target_device:
            self.teacher_model = self.teacher_model.to(target_device)

        has_teacher_inputs = "teacher_input_ids" in inputs
        student_inputs = {k: v for k, v in inputs.items() if k not in _TEACHER_KEYS}

        if has_teacher_inputs:
            teacher_inputs = {
                "input_ids": inputs["teacher_input_ids"],
                "attention_mask": inputs["teacher_attention_mask"],
                "pixel_values": inputs["teacher_pixel_values"],
            }
            if "teacher_pixel_attention_mask" in inputs:
                teacher_inputs["pixel_attention_mask"] = inputs["teacher_pixel_attention_mask"]
            if "teacher_image_grid_thw" in inputs:
                teacher_inputs["image_grid_thw"] = inputs["teacher_image_grid_thw"]
            teacher_labels = inputs["teacher_labels"]
        else:
            teacher_inputs = {k: v for k, v in student_inputs.items() if k != "labels"}
            teacher_labels = student_inputs.get("labels")

        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits

        student_labels = student_inputs.get("labels")
        assert student_labels is not None, "Labels must be provided for distillation loss masking"

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.detach()

        student_mask = student_labels != -100
        teacher_mask = teacher_labels != -100

        if student_mask.sum() > 0 and teacher_mask.sum() > 0:
            if self.loss_function == "uld_loss":
                sample_losses = []
                for i in range(student_logits.size(0)):
                    student_logits_masked = student_logits[i][student_mask[i]]
                    teacher_logits_masked = teacher_logits[i][teacher_mask[i]]
                    if student_logits_masked.size(0) == 0 or teacher_logits_masked.size(0) == 0:
                        continue
                    min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
                    sample_losses.append(
                        self.distillation_loss_fn(
                            student_logits=student_logits_masked[:min_len],
                            teacher_logits=teacher_logits_masked[:min_len],
                            temperature=self.temperature,
                        )
                    )
                if sample_losses:
                    distillation_loss = torch.stack(sample_losses).mean()
                else:
                    distillation_loss = torch.tensor(0.0, device=student_logits.device)
            else:
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
                if "eps" in self.distillation_loss_params:
                    loss_kwargs["eps"] = self.ofa_eps
                distillation_loss = self.distillation_loss_fn(**loss_kwargs)
        else:
            distillation_loss = torch.tensor(0.0, device=student_logits.device)

        ce_loss = student_outputs.loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            })

        return (loss, student_outputs) if return_outputs else loss
