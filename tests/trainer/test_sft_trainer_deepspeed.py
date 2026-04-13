import contextlib
from types import SimpleNamespace

import torch

from src.trainer.sft_trainer import VisionLanguageSFTTrainer


class FakeOptimizer:
    def __init__(self):
        self.train_called = False

    def train(self):
        self.train_called = True


class FakeEngine:
    def __init__(self):
        self.calls = []

    def set_gradient_accumulation_boundary(self, is_boundary):
        self.calls.append(("boundary", is_boundary))

    def backward(self, loss, scale_wrt_gas=True):
        self.calls.append(("backward", float(loss.detach().item()), scale_wrt_gas))

    def step(self):
        self.calls.append(("step",))


def build_trainer(*, engine, sync_gradients, grad_accumulation_steps=4):
    trainer = object.__new__(VisionLanguageSFTTrainer)
    trainer.optimizer = FakeOptimizer()
    trainer.args = SimpleNamespace(n_gpu=1, gradient_accumulation_steps=grad_accumulation_steps)
    trainer.accelerator = SimpleNamespace(
        sync_gradients=sync_gradients,
        distributed_type="DistributedType.DEEPSPEED",
        deepspeed_engine_wrapped=None,
        _models=[engine],
    )
    trainer.model = SimpleNamespace(module=SimpleNamespace())
    trainer.model_wrapped = SimpleNamespace(module=SimpleNamespace())
    trainer.deepspeed = None
    trainer.current_gradient_accumulation_steps = grad_accumulation_steps
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = contextlib.nullcontext
    trainer.compute_loss = lambda model, inputs, num_items_in_batch=None: torch.tensor(8.0, requires_grad=True)
    return trainer


def test_resolve_deepspeed_engine_recovers_engine_from_accelerator_models():
    engine = FakeEngine()
    trainer = build_trainer(engine=engine, sync_gradients=True)

    resolved_engine = trainer.resolve_deepspeed_engine(model=SimpleNamespace(module=SimpleNamespace()))

    assert resolved_engine is engine
    assert trainer.accelerator.deepspeed_engine_wrapped is not None


def test_manual_deepspeed_training_step_respects_gradient_boundary():
    engine = FakeEngine()
    trainer = build_trainer(engine=engine, sync_gradients=False)

    loss = trainer.manual_deepspeed_training_step(
        model=SimpleNamespace(train=lambda: None),
        inputs={"input_ids": torch.tensor([[1, 2, 3]])},
    )

    assert trainer.optimizer.train_called is True
    assert engine.calls == [
        ("boundary", False),
        ("backward", 8.0, False),
    ]
    assert loss.item() == 2.0
