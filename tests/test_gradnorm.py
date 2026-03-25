import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.components.gradnorm import (
    initialize_gradnorm_weights,
    normalize_gradnorm_weights,
    update_gradnorm_weights,
)
from src.trainer.distillation_utils import is_layer_distillation_enabled
from src.trainer.distillation_trainer import DistillationTrainer


class _ToyModel(torch.nn.Module):
    def __init__(self, initial_value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))


class GradNormTests(unittest.TestCase):
    def test_initialize_gradnorm_weights_starts_all_tasks_equally(self):
        weights = initialize_gradnorm_weights(
            ["ce", "distillation", "layer_distill"],
            gradnorm_eps=1e-8,
        )

        self.assertDictEqual(
            weights,
            {
                "ce": 1.0,
                "distillation": 1.0,
                "layer_distill": 1.0,
            },
        )

    def test_normalize_gradnorm_weights_preserves_relative_scale(self):
        weights = {
            "ce": 0.5,
            "distillation": 0.5,
            "layer_distill": 0.2,
        }

        normalize_gradnorm_weights(weights, gradnorm_eps=1e-8)

        self.assertAlmostEqual(sum(weights.values()), 3.0, places=6)
        self.assertAlmostEqual(weights["ce"], weights["distillation"], places=6)
        self.assertGreater(weights["ce"], weights["layer_distill"])

    def test_update_gradnorm_weights_downweights_dominant_gradient(self):
        model = _ToyModel(initial_value=1.0)
        task_losses = {
            "ce": 0.5 * model.weight.pow(2),
            "distillation": 50.0 * model.weight.pow(2),
        }
        gradnorm_weights = {
            "ce": 1.0,
            "distillation": 1.0,
        }
        gradnorm_initial_losses = {}

        update_gradnorm_weights(
            model,
            task_losses,
            [model.weight],
            gradnorm_weights,
            gradnorm_initial_losses,
            gradnorm_active=True,
            gradnorm_eps=1e-8,
            gradnorm_alpha=1.5,
            gradnorm_lr=0.01,
        )

        self.assertAlmostEqual(sum(gradnorm_weights.values()), 2.0, places=6)
        self.assertGreater(gradnorm_weights["ce"], 1.0)
        self.assertLess(gradnorm_weights["distillation"], 1.0)
        self.assertSetEqual(set(gradnorm_initial_losses.keys()), {"ce", "distillation"})

    def test_prediction_step_uses_loss_only_path_without_returning_outputs(self):
        trainer = object.__new__(DistillationTrainer)
        trainer.label_names = ["labels"]
        trainer.can_return_loss = False
        trainer.args = SimpleNamespace(device=torch.device("cpu"))
        trainer._prepare_inputs = lambda inputs: inputs
        trainer.compute_loss_context_manager = nullcontext
        trainer._get_num_items_in_batch = lambda batches, device: None

        called = {}

        def fake_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            called["return_outputs"] = return_outputs
            called["num_items_in_batch"] = num_items_in_batch
            return torch.tensor(2.5)

        trainer.compute_loss = fake_compute_loss

        loss, logits, labels = DistillationTrainer.prediction_step(
            trainer,
            model=object(),
            inputs={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[1, 2, 3]]),
            },
            prediction_loss_only=True,
            ignore_keys=None,
        )

        self.assertEqual(loss.item(), 2.5)
        self.assertIsNone(logits)
        self.assertIsNone(labels)
        self.assertFalse(called["return_outputs"])

    def test_layer_distillation_enabled_under_gradnorm_without_positive_weight(self):
        self.assertTrue(
            is_layer_distillation_enabled(
                loss_weighting="gradnorm",
                layer_distill_source="model",
                layer_distill_weight=0.0,
                student_layer_indices=None,
                layer_match_json_path="artifacts/matrix.json",
            )
        )

    def test_layer_distillation_stays_disabled_in_fixed_mode_without_weight(self):
        self.assertFalse(
            is_layer_distillation_enabled(
                loss_weighting="fixed",
                layer_distill_source="model",
                layer_distill_weight=0.0,
                student_layer_indices=None,
                layer_match_json_path="artifacts/matrix.json",
            )
        )


if __name__ == "__main__":
    unittest.main()
