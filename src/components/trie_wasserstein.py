from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F


EOS_SENTINEL = 256


def extract_tokenizer_backend(processor_or_tokenizer):
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    return tokenizer if tokenizer is not None else processor_or_tokenizer


def resolve_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "__len__"):
        return int(len(tokenizer))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Tokenizer must define either __len__() or vocab_size.")
    return int(vocab_size)


def token_piece_to_bytes(tokenizer, token_id: int) -> bytes:
    token_id = int(token_id)
    if hasattr(tokenizer, "decode"):
        try:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            text = tokenizer.decode([token_id])
    else:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if hasattr(tokenizer, "convert_tokens_to_string"):
            text = tokenizer.convert_tokens_to_string([token])
        else:
            text = token

    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8")


def default_ignored_token_ids(tokenizer) -> set[int]:
    ignored = set()
    for attr_name in ("pad_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            ignored.add(int(token_id))
    return ignored


@dataclass(slots=True)
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    edge_id: int | None = None


@dataclass(slots=True)
class TrieBuildResult:
    path_flat: torch.Tensor
    path_offsets: torch.Tensor
    ignored_mask: torch.Tensor


class TrieWassersteinLoss(nn.Module):
    """
    Tree-Wasserstein distillation on a shared byte trie.

    This module restores token identity across mismatched vocabularies by
    placing student and teacher token pieces on a shared UTF-8 byte trie and
    computing the weighted subtree-mass imbalance. The implementation uses a
    sparse top-k approximation with a dedicated tail edge, which keeps the
    forward pass proportional to the active trie paths rather than a dense
    cost matrix.
    """

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        rho: float = 0.7,
        topk: int = 64,
        ignored_student_token_ids: set[int] | None = None,
        ignored_teacher_token_ids: set[int] | None = None,
    ):
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

        self.student_tokenizer = extract_tokenizer_backend(student_tokenizer)
        self.teacher_tokenizer = extract_tokenizer_backend(teacher_tokenizer)
        self.student_tokenizer_vocab_size = resolve_vocab_size(self.student_tokenizer)
        self.teacher_tokenizer_vocab_size = resolve_vocab_size(self.teacher_tokenizer)
        self.student_vocab_size = self.student_tokenizer_vocab_size
        self.teacher_vocab_size = self.teacher_tokenizer_vocab_size
        self.rho = float(rho)
        self.topk = int(topk)

        ignored_student = (
            default_ignored_token_ids(self.student_tokenizer)
            if ignored_student_token_ids is None
            else {int(token_id) for token_id in ignored_student_token_ids}
        )
        ignored_teacher = (
            default_ignored_token_ids(self.teacher_tokenizer)
            if ignored_teacher_token_ids is None
            else {int(token_id) for token_id in ignored_teacher_token_ids}
        )

        root = TrieNode()
        edge_weights: list[float] = []

        student_paths = self.build_paths(
            tokenizer=self.student_tokenizer,
            vocab_size=self.student_vocab_size,
            ignored_token_ids=ignored_student,
            root=root,
            edge_weights=edge_weights,
        )
        teacher_paths = self.build_paths(
            tokenizer=self.teacher_tokenizer,
            vocab_size=self.teacher_vocab_size,
            ignored_token_ids=ignored_teacher,
            root=root,
            edge_weights=edge_weights,
        )

        self.tail_edge_id = len(edge_weights)
        edge_weights.append(1.0)

        self.register_buffer(
            "edge_weights_cpu",
            torch.tensor(edge_weights, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("student_path_flat_cpu", student_paths.path_flat, persistent=True)
        self.register_buffer("student_path_offsets_cpu", student_paths.path_offsets, persistent=True)
        self.register_buffer("student_ignored_mask_cpu", student_paths.ignored_mask, persistent=True)
        self.register_buffer("teacher_path_flat_cpu", teacher_paths.path_flat, persistent=True)
        self.register_buffer("teacher_path_offsets_cpu", teacher_paths.path_offsets, persistent=True)
        self.register_buffer("teacher_ignored_mask_cpu", teacher_paths.ignored_mask, persistent=True)

        self.cached_device: torch.device | None = None
        self.edge_weights_device: torch.Tensor | None = None
        self.student_path_flat_device: torch.Tensor | None = None
        self.student_path_offsets_device: torch.Tensor | None = None
        self.student_ignored_mask_device: torch.Tensor | None = None
        self.teacher_path_flat_device: torch.Tensor | None = None
        self.teacher_path_offsets_device: torch.Tensor | None = None
        self.teacher_ignored_mask_device: torch.Tensor | None = None

    def invalidate_device_cache(self) -> None:
        self.cached_device = None
        self.edge_weights_device = None
        self.student_path_flat_device = None
        self.student_path_offsets_device = None
        self.student_ignored_mask_device = None
        self.teacher_path_flat_device = None
        self.teacher_path_offsets_device = None
        self.teacher_ignored_mask_device = None

    def extend_vocab_state_with_ignored_tokens(
        self,
        *,
        side: str,
        target_vocab_size: int,
    ) -> None:
        if side == "student":
            current_vocab_size = self.student_vocab_size
            if target_vocab_size <= current_vocab_size:
                return
            path_flat = self.student_path_flat_cpu
            path_offsets = self.student_path_offsets_cpu
            ignored_mask = self.student_ignored_mask_cpu
        elif side == "teacher":
            current_vocab_size = self.teacher_vocab_size
            if target_vocab_size <= current_vocab_size:
                return
            path_flat = self.teacher_path_flat_cpu
            path_offsets = self.teacher_path_offsets_cpu
            ignored_mask = self.teacher_ignored_mask_cpu
        else:
            raise ValueError(f"Unknown trie side: {side!r}")

        extra_tokens = target_vocab_size - current_vocab_size
        repeated_offset = path_offsets[-1].repeat(extra_tokens)
        extended_offsets = torch.cat([path_offsets[:-1], repeated_offset, path_offsets[-1:]], dim=0)
        extended_ignored_mask = torch.cat(
            [ignored_mask, torch.ones(extra_tokens, dtype=torch.bool)],
            dim=0,
        )

        if side == "student":
            self.student_vocab_size = target_vocab_size
            self.student_path_flat_cpu = path_flat
            self.student_path_offsets_cpu = extended_offsets
            self.student_ignored_mask_cpu = extended_ignored_mask
        else:
            self.teacher_vocab_size = target_vocab_size
            self.teacher_path_flat_cpu = path_flat
            self.teacher_path_offsets_cpu = extended_offsets
            self.teacher_ignored_mask_cpu = extended_ignored_mask

        self.invalidate_device_cache()

    def build_paths(
        self,
        *,
        tokenizer,
        vocab_size: int,
        ignored_token_ids: set[int],
        root: TrieNode,
        edge_weights: list[float],
    ) -> TrieBuildResult:
        path_flat: list[int] = []
        path_offsets = [0]
        ignored_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for token_id in range(vocab_size):
            if token_id in ignored_token_ids:
                ignored_mask[token_id] = True
                path_offsets.append(len(path_flat))
                continue

            token_bytes = list(token_piece_to_bytes(tokenizer, token_id))
            token_bytes.append(EOS_SENTINEL)
            path = self.insert_bytes(
                token_bytes=token_bytes,
                root=root,
                edge_weights=edge_weights,
            )
            path_flat.extend(path)
            path_offsets.append(len(path_flat))

        return TrieBuildResult(
            path_flat=torch.tensor(path_flat, dtype=torch.long),
            path_offsets=torch.tensor(path_offsets, dtype=torch.long),
            ignored_mask=ignored_mask,
        )

    def insert_bytes(
        self,
        *,
        token_bytes: list[int],
        root: TrieNode,
        edge_weights: list[float],
    ) -> list[int]:
        node = root
        path: list[int] = []
        for depth, byte_value in enumerate(token_bytes, start=1):
            child = node.children.get(byte_value)
            if child is None:
                child = TrieNode()
                child.edge_id = len(edge_weights)
                edge_weights.append(self.rho ** (depth - 1))
                node.children[byte_value] = child
            path.append(child.edge_id)
            node = child
        return path

    def ensure_device_tensors(self, device: torch.device) -> None:
        if self.cached_device == device:
            return
        self.cached_device = device
        self.edge_weights_device = self.edge_weights_cpu.to(device=device, non_blocking=True)
        self.student_path_flat_device = self.student_path_flat_cpu.to(device=device, non_blocking=True)
        self.student_path_offsets_device = self.student_path_offsets_cpu.to(device=device, non_blocking=True)
        self.student_ignored_mask_device = self.student_ignored_mask_cpu.to(device=device, non_blocking=True)
        self.teacher_path_flat_device = self.teacher_path_flat_cpu.to(device=device, non_blocking=True)
        self.teacher_path_offsets_device = self.teacher_path_offsets_cpu.to(device=device, non_blocking=True)
        self.teacher_ignored_mask_device = self.teacher_ignored_mask_cpu.to(device=device, non_blocking=True)

    def build_signed_edge_contributions(
        self,
        *,
        probs: torch.Tensor,
        path_flat: torch.Tensor,
        path_offsets: torch.Tensor,
        ignored_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = ~ignored_mask
        num_valid = int(valid_mask.sum().item())
        tail_edge = torch.tensor([self.tail_edge_id], device=probs.device, dtype=torch.long)

        if num_valid == 0:
            tail_mass = probs.sum().unsqueeze(0)
            return tail_edge, tail_mass

        k = min(self.topk, num_valid)
        masked_probs = probs.masked_fill(ignored_mask, float("-inf"))
        kept_values, kept_token_ids = torch.topk(masked_probs, k=k, dim=-1)
        tail_mass = (1.0 - kept_values.sum()).clamp_min(0.0)

        edge_chunks: list[torch.Tensor] = []
        mass_chunks: list[torch.Tensor] = []

        for token_id, prob in zip(kept_token_ids.tolist(), kept_values.unbind(0)):
            start = int(path_offsets[token_id].item())
            end = int(path_offsets[token_id + 1].item())
            if end <= start:
                tail_mass = tail_mass + prob
                continue

            token_edge_ids = path_flat[start:end]
            edge_chunks.append(token_edge_ids)
            mass_chunks.append(prob.expand(token_edge_ids.numel()))

        tail_mass_tensor = tail_mass.unsqueeze(0)

        if edge_chunks:
            return (
                torch.cat([*edge_chunks, tail_edge], dim=0),
                torch.cat([*mass_chunks, tail_mass_tensor], dim=0),
            )
        return tail_edge, tail_mass_tensor

    def prepare_runtime_state(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        teacher_labels: torch.Tensor | None = None,
    ) -> None:
        if student_vocab_size > self.student_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="student",
                target_vocab_size=student_vocab_size,
            )
        elif student_vocab_size < self.student_vocab_size:
            raise ValueError(
                "student logits vocab size does not match the trie state: "
                f"{student_vocab_size} != {self.student_vocab_size}"
            )

        if teacher_vocab_size > self.teacher_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="teacher",
                target_vocab_size=teacher_vocab_size,
            )
        elif teacher_vocab_size < self.teacher_vocab_size:
            raise ValueError(
                "teacher logits vocab size does not match the trie state: "
                f"{teacher_vocab_size} != {self.teacher_vocab_size}"
            )

        if teacher_labels is None:
            return

        valid_labels = teacher_labels[teacher_labels != -100]
        if valid_labels.numel() == 0:
            return

        max_label = int(valid_labels.max().item())
        if max_label >= self.teacher_tokenizer_vocab_size:
            raise ValueError(
                "teacher labels contain ids outside tokenizer space: "
                f"max label {max_label} >= tokenizer vocab {self.teacher_tokenizer_vocab_size}"
            )

    def single_step_loss(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        student_edges, student_masses = self.build_signed_edge_contributions(
            probs=student_probs,
            path_flat=self.student_path_flat_device,
            path_offsets=self.student_path_offsets_device,
            ignored_mask=self.student_ignored_mask_device,
        )
        teacher_edges, teacher_masses = self.build_signed_edge_contributions(
            probs=teacher_probs,
            path_flat=self.teacher_path_flat_device,
            path_offsets=self.teacher_path_offsets_device,
            ignored_mask=self.teacher_ignored_mask_device,
        )

        all_edges = torch.cat([student_edges, teacher_edges], dim=0)
        signed_masses = torch.cat([student_masses, -teacher_masses], dim=0)
        active_edges, inverse = torch.unique(all_edges, sorted=False, return_inverse=True)
        signed_edge_balance = signed_masses.new_zeros(active_edges.size(0))
        signed_edge_balance.index_add_(0, inverse, signed_masses)
        return (
            self.edge_weights_device.index_select(0, active_edges) * signed_edge_balance.abs()
        ).sum()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
        student_temperature: float | None = None,
        teacher_temperature: float | None = None,
    ) -> torch.Tensor:
        from src.components.loss import resolve_temperatures

        if student_logits.ndim != 2:
            raise ValueError(
                f"student_logits must have shape (N, V_s), got {tuple(student_logits.shape)}"
            )
        if teacher_logits.ndim != 2:
            raise ValueError(
                f"teacher_logits must have shape (N, V_t), got {tuple(teacher_logits.shape)}"
            )
        if student_logits.size(0) != teacher_logits.size(0):
            raise ValueError(
                "student and teacher must have the same token dimension, got "
                f"{student_logits.size(0)} and {teacher_logits.size(0)}"
            )
        if student_logits.size(-1) != self.student_vocab_size:
            raise ValueError(
                "student logits vocab size does not match the trie state: "
                f"{student_logits.size(-1)} != {self.student_vocab_size}"
            )
        if teacher_logits.size(-1) != self.teacher_vocab_size:
            raise ValueError(
                "teacher logits vocab size does not match the trie state: "
                f"{teacher_logits.size(-1)} != {self.teacher_vocab_size}"
            )

        self.ensure_device_tensors(student_logits.device)
        student_temperature, teacher_temperature = resolve_temperatures(
            temperature=temperature,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )

        student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)

        step_losses = [
            self.single_step_loss(student_probs[index], teacher_probs[index])
            for index in range(student_probs.size(0))
        ]
        if not step_losses:
            return student_logits.new_zeros((), dtype=torch.float32)
        return torch.stack(step_losses, dim=0).mean()

    def compute_logit_grad(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0,
        student_temperature: float | None = None,
        teacher_temperature: float | None = None,
    ) -> torch.Tensor:
        self.prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
        )
        with torch.enable_grad():
            detached_student_logits = student_logits.detach().clone().requires_grad_(True)
            loss = self.forward(
                detached_student_logits,
                teacher_logits=teacher_logits.detach(),
                temperature=temperature,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
            )
            grad, = torch.autograd.grad(loss, detached_student_logits, create_graph=False)
        return grad.detach()


__all__ = [
    "TrieWassersteinLoss",
    "extract_tokenizer_backend",
]
