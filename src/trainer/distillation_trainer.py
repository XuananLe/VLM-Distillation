import torch
from transformers import Trainer, PreTrainedModel

class LogitsDistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel,
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
        # Forward pass through student model
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        teacher_outputs = self.teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits
        
        distillation_loss = self.distillation_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=self.temperature
        )
        
        # Combine distillation loss with standard cross-entropy loss
        ce_loss = student_outputs.loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
        
        # Log individual loss components
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            })
        
        return (loss, student_outputs) if return_outputs else loss
