from transformers import Trainer, PreTrainedModel
import torch
import torch.nn.functional as F

class LogitsDistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        from src.components import loss as distillation_loss_module
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha
        
        print(f"Distillation Trainer initialized:")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.teacher_model.device != model.device:
            self.teacher_model = self.teacher_model.to(model.device)
        
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        labels = inputs.get("labels", None)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        assert labels is not None, "Labels must be provided in the inputs for distillation loss masking"

        mask = (labels != -100)
        
        if mask.sum() > 0:
            student_logits_masked = student_logits.view(-1, student_logits.size(-1))[mask.view(-1)]
            teacher_logits_masked = teacher_logits.view(-1, teacher_logits.size(-1))[mask.view(-1)]
            
            distillation_loss = self.distillation_loss_fn(
                student_logits=student_logits_masked,
                teacher_logits=teacher_logits_masked,
                temperature=self.temperature
            )
        else:
            distillation_loss = torch.tensor(0.0, device=student_logits.device)

        ce_loss = student_outputs.loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
        
        # Log individual loss components
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            })
        
        return (loss, student_outputs) if return_outputs else loss
