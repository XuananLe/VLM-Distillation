from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from joblib import Memory
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from src.constants import TOKENIZER_VOCAB_SIZES


EOS_SENTINEL = 256
# Byte values live in [0, 255], so 256 is a clean end-of-token marker for the trie.
TRIE_CACHE_VERSION = 1
DEFAULT_TRIE_CACHE_DIR = "/dev/shm/vlm_distillation_trie_cache"


def normalize_model_id(model_id) -> str | None:
    """Return a canonical model id string when the tokenizer exposes one."""
    if not isinstance(model_id, str) or not model_id:
        return None
    return model_id.rstrip("/")


def resolve_vocab_size(tokenizer) -> int:
    """Return the fixed tokenizer size for one supported VLM tokenizer."""
    model_id = normalize_model_id(getattr(tokenizer, "name_or_path", None))
    try:
        return TOKENIZER_VOCAB_SIZES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tokenizer for trie OT: {model_id!r}. "
            f"Supported models: {sorted(TOKENIZER_VOCAB_SIZES)}"
        ) from exc


def token_piece_to_bytes(tokenizer, token_id: int) -> bytes:
    """Convert one token id to UTF-8 bytes; input is tokenizer plus token id, output is bytes, and this exists so student and teacher can share one byte-level trie."""
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
    """Return special token ids ignored by trie OT; input is a tokenizer, output is a set of ids, and this exists to skip non-semantic pad/bos tokens."""
    ignored = set()
    for attr_name in ("pad_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            ignored.add(int(token_id))
    return ignored


@dataclass(slots=True)
class TrieNode:
    """One byte-trie node; it stores outgoing byte children and the edge id that reaches this node."""
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    edge_id: int | None = None


@dataclass(slots=True)
class TrieBuildResult:
    """Packed trie-path tables for one tokenizer; it stores flattened paths, path offsets, and ignored-token mask used at runtime."""
    path_flat: torch.Tensor
    path_offsets: torch.Tensor
    ignored_mask: torch.Tensor


@dataclass(slots=True)
class TrieRuntimeState:
    """Static CPU trie state shared by every forward pass for one student/teacher tokenizer pair."""
    edge_weights_cpu: torch.Tensor
    student_path_flat_cpu: torch.Tensor
    student_path_offsets_cpu: torch.Tensor
    student_ignored_mask_cpu: torch.Tensor
    teacher_path_flat_cpu: torch.Tensor
    teacher_path_offsets_cpu: torch.Tensor
    teacher_ignored_mask_cpu: torch.Tensor
    tail_edge_id: int
    num_edges: int
    student_valid_count: int
    teacher_valid_count: int
    student_max_path_len: int
    teacher_max_path_len: int


@lru_cache(maxsize=4)
def trie_memory(cache_dir: str) -> Memory:
    """Return one joblib Memory handle per cache directory."""
    return Memory(location=cache_dir, verbose=0)


@lru_cache(maxsize=4)
def cached_trie_state_builder(cache_dir: str):
    """Return a joblib-cached trie builder that does not hash tokenizer objects."""
    return trie_memory(cache_dir).cache(
        build_trie_state_from_tokenizers,
        ignore=["student_tokenizer", "teacher_tokenizer"],
    )


def tokenizer_class_name(tokenizer) -> str:
    """Return a stable tokenizer class name used as part of the trie cache key."""
    tokenizer_type = type(tokenizer)
    return f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"


def can_cache_trie_state(student_tokenizer, teacher_tokenizer) -> bool:
    """Return whether tokenizers are real Hugging Face tokenizers safe for persistent cache keys."""
    return isinstance(student_tokenizer, PreTrainedTokenizerBase) and isinstance(
        teacher_tokenizer,
        PreTrainedTokenizerBase,
    )


def insert_token_bytes(
    *,
    token_bytes: list[int],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
) -> list[int]:
    """Insert one token byte sequence into the shared trie and return its edge-id path."""
    node = root
    path: list[int] = []
    for depth, byte_value in enumerate(token_bytes, start=1):
        child = node.children.get(byte_value)
        if child is None:
            child = TrieNode()
            child.edge_id = len(edge_weights)
            edge_weights.append(float(rho) ** (depth - 1))
            node.children[byte_value] = child
        path.append(child.edge_id)
        node = child
    return path


def build_tokenizer_paths(
    *,
    tokenizer,
    vocab_size: int,
    ignored_token_ids: tuple[int, ...],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
) -> TrieBuildResult:
    """Build flattened trie paths for one tokenizer so runtime path lookup is cheap."""
    ignored_token_ids_set = set(ignored_token_ids)
    path_flat: list[int] = []
    path_offsets = [0]
    ignored_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_id in range(vocab_size):
        if token_id in ignored_token_ids_set:
            ignored_mask[token_id] = True
            path_offsets.append(len(path_flat))
            continue

        token_bytes = list(token_piece_to_bytes(tokenizer, token_id))
        # Prefix tokens and longer tokens need distinct terminal paths.
        token_bytes.append(EOS_SENTINEL)
        path = insert_token_bytes(
            token_bytes=token_bytes,
            root=root,
            edge_weights=edge_weights,
            rho=rho,
        )
        path_flat.extend(path)
        path_offsets.append(len(path_flat))

    return TrieBuildResult(
        path_flat=torch.tensor(path_flat, dtype=torch.long),
        path_offsets=torch.tensor(path_offsets, dtype=torch.long),
        ignored_mask=ignored_mask,
    )


def build_trie_state_from_tokenizers(
    *,
    cache_version: int,
    student_model_id: str,
    teacher_model_id: str,
    student_tokenizer_class: str,
    teacher_tokenizer_class: str,
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: tuple[int, ...],
    teacher_ignored_token_ids: tuple[int, ...],
    rho: float,
    student_tokenizer,
    teacher_tokenizer,
) -> TrieRuntimeState:
    """Build static CPU trie state for one student/teacher tokenizer pair."""
    del cache_version, student_model_id, teacher_model_id
    del student_tokenizer_class, teacher_tokenizer_class

    root = TrieNode()
    edge_weights: list[float] = []
    student_paths = build_tokenizer_paths(
        tokenizer=student_tokenizer,
        vocab_size=student_vocab_size,
        ignored_token_ids=student_ignored_token_ids,
        root=root,
        edge_weights=edge_weights,
        rho=rho,
    )
    teacher_paths = build_tokenizer_paths(
        tokenizer=teacher_tokenizer,
        vocab_size=teacher_vocab_size,
        ignored_token_ids=teacher_ignored_token_ids,
        root=root,
        edge_weights=edge_weights,
        rho=rho,
    )

    tail_edge_id = len(edge_weights)
    edge_weights.append(1.0)
    num_edges = len(edge_weights)

    student_path_lengths = student_paths.path_offsets[1:] - student_paths.path_offsets[:-1]
    teacher_path_lengths = teacher_paths.path_offsets[1:] - teacher_paths.path_offsets[:-1]

    return TrieRuntimeState(
        edge_weights_cpu=torch.tensor(edge_weights, dtype=torch.float32),
        student_path_flat_cpu=student_paths.path_flat,
        student_path_offsets_cpu=student_paths.path_offsets,
        student_ignored_mask_cpu=student_paths.ignored_mask,
        teacher_path_flat_cpu=teacher_paths.path_flat,
        teacher_path_offsets_cpu=teacher_paths.path_offsets,
        teacher_ignored_mask_cpu=teacher_paths.ignored_mask,
        tail_edge_id=tail_edge_id,
        num_edges=num_edges,
        student_valid_count=int((~student_paths.ignored_mask).sum().item()),
        teacher_valid_count=int((~teacher_paths.ignored_mask).sum().item()),
        student_max_path_len=(
            int(student_path_lengths.max().item()) if student_path_lengths.numel() else 0
        ),
        teacher_max_path_len=(
            int(teacher_path_lengths.max().item()) if teacher_path_lengths.numel() else 0
        ),
    )


def load_or_build_trie_state(
    *,
    student_tokenizer,
    teacher_tokenizer,
    student_model_id: str,
    teacher_model_id: str,
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: tuple[int, ...],
    teacher_ignored_token_ids: tuple[int, ...],
    rho: float,
) -> TrieRuntimeState:
    """Load static trie state from joblib cache when possible, otherwise build it directly."""
    builder = build_trie_state_from_tokenizers
    if (
        student_model_id is not None
        and teacher_model_id is not None
        and can_cache_trie_state(student_tokenizer, teacher_tokenizer)
    ):
        builder = cached_trie_state_builder(DEFAULT_TRIE_CACHE_DIR)

    return builder(
        cache_version=TRIE_CACHE_VERSION,
        student_model_id=student_model_id,
        teacher_model_id=teacher_model_id,
        student_tokenizer_class=tokenizer_class_name(student_tokenizer),
        teacher_tokenizer_class=tokenizer_class_name(teacher_tokenizer),
        student_vocab_size=student_vocab_size,
        teacher_vocab_size=teacher_vocab_size,
        student_ignored_token_ids=student_ignored_token_ids,
        teacher_ignored_token_ids=teacher_ignored_token_ids,
        rho=float(rho),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )


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
        """Build the shared byte-trie loss state; input is student/teacher tokenizers plus trie hyperparameters, output is an initialized loss module, and this exists to precompute cross-tokenizer path structure once."""
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

        self.student_tokenizer = getattr(student_tokenizer, "tokenizer", None) or student_tokenizer
        self.teacher_tokenizer = getattr(teacher_tokenizer, "tokenizer", None) or teacher_tokenizer
        self.student_model_id = normalize_model_id(getattr(self.student_tokenizer, "name_or_path", None))
        self.teacher_model_id = normalize_model_id(getattr(self.teacher_tokenizer, "name_or_path", None))
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

        trie_state = load_or_build_trie_state(
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            student_model_id=self.student_model_id,
            teacher_model_id=self.teacher_model_id,
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=self.teacher_vocab_size,
            student_ignored_token_ids=tuple(sorted(ignored_student)),
            teacher_ignored_token_ids=tuple(sorted(ignored_teacher)),
            rho=self.rho,
        )

        self.tail_edge_id = trie_state.tail_edge_id
        self.num_edges = trie_state.num_edges
        self.student_valid_count = trie_state.student_valid_count
        self.teacher_valid_count = trie_state.teacher_valid_count
        self.student_max_path_len = trie_state.student_max_path_len
        self.teacher_max_path_len = trie_state.teacher_max_path_len

        self.register_buffer("edge_weights_cpu", trie_state.edge_weights_cpu, persistent=True)
        self.register_buffer("student_path_flat_cpu", trie_state.student_path_flat_cpu, persistent=True)
        self.register_buffer(
            "student_path_offsets_cpu",
            trie_state.student_path_offsets_cpu,
            persistent=True,
        )
        self.register_buffer(
            "student_ignored_mask_cpu",
            trie_state.student_ignored_mask_cpu,
            persistent=True,
        )
        self.register_buffer("teacher_path_flat_cpu", trie_state.teacher_path_flat_cpu, persistent=True)
        self.register_buffer(
            "teacher_path_offsets_cpu",
            trie_state.teacher_path_offsets_cpu,
            persistent=True,
        )
        self.register_buffer(
            "teacher_ignored_mask_cpu",
            trie_state.teacher_ignored_mask_cpu,
            persistent=True,
        )

        self.cached_device: torch.device | None = None
        self.edge_weights_device: torch.Tensor | None = None
        self.student_path_flat_device: torch.Tensor | None = None
        self.student_path_offsets_device: torch.Tensor | None = None
        self.student_ignored_mask_device: torch.Tensor | None = None
        self.teacher_path_flat_device: torch.Tensor | None = None
        self.teacher_path_offsets_device: torch.Tensor | None = None
        self.teacher_ignored_mask_device: torch.Tensor | None = None
        self.student_path_arange_device: torch.Tensor | None = None
        self.teacher_path_arange_device: torch.Tensor | None = None

    def invalidate_device_cache(self) -> None:
        """Clear cached device-side trie tensors; input/output are None, and this exists because vocab extension invalidates earlier device copies."""
        self.cached_device = None
        self.edge_weights_device = None
        self.student_path_flat_device = None
        self.student_path_offsets_device = None
        self.student_ignored_mask_device = None
        self.teacher_path_flat_device = None
        self.teacher_path_offsets_device = None
        self.teacher_ignored_mask_device = None
        self.student_path_arange_device = None
        self.teacher_path_arange_device = None

    def extend_vocab_state_with_ignored_tokens(
        self,
        *,
        side: str,
        target_vocab_size: int,
    ) -> None:
        """Extend one trie side with ignored extra tokens; input is side name and target vocab size, output is None, and this exists to tolerate runtime vocab growth without rebuilding the trie."""
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

    def ensure_device_tensors(self, device: torch.device) -> None:
        """Materialize cached trie tensors on a target device; input is a torch device, output is None, and this exists because the trie is built on CPU but used during GPU loss computation."""
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
        self.student_path_arange_device = torch.arange(
            self.student_max_path_len,
            device=device,
            dtype=torch.long,
        )
        self.teacher_path_arange_device = torch.arange(
            self.teacher_max_path_len,
            device=device,
            dtype=torch.long,
        )

    def build_batched_signed_edge_contributions(
        self,
        *,
        scaled_logits: torch.Tensor,
        path_flat: torch.Tensor,
        path_offsets: torch.Tensor,
        ignored_mask: torch.Tensor,
        valid_count: int,
        max_path_len: int,
        path_arange: torch.Tensor,
        sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand a batch of token distributions into signed row-edge masses."""
        num_rows = scaled_logits.size(0)
        device = scaled_logits.device
        dtype = scaled_logits.dtype
        row_base = torch.arange(num_rows, device=device, dtype=torch.long) * self.num_edges
        tail_keys = row_base + self.tail_edge_id

        if valid_count == 0 or max_path_len == 0:
            tail_masses = scaled_logits.new_full((num_rows,), float(sign))
            return tail_keys, tail_masses

        k = min(self.topk, valid_count)
        masked_logits = scaled_logits.masked_fill(ignored_mask, float("-inf"))
        kept_logits, kept_token_ids = torch.topk(
            masked_logits,
            k=k,
            dim=-1,
            sorted=False,
        )

        log_z = torch.logsumexp(scaled_logits, dim=-1, keepdim=True)
        kept_masses = (kept_logits - log_z).exp()
        tail_masses = (1.0 - kept_masses.sum(dim=-1)).clamp_min(0.0)

        starts = path_offsets[kept_token_ids]
        ends = path_offsets[kept_token_ids + 1]
        lengths = ends - starts
        rel = path_arange.view(1, 1, max_path_len)
        valid_path = rel < lengths.unsqueeze(-1)

        flat_positions = (starts.unsqueeze(-1) + rel).clamp_max(path_flat.numel() - 1)
        edge_ids = path_flat[flat_positions]
        keys = row_base.view(num_rows, 1, 1) + edge_ids
        masses = kept_masses.unsqueeze(-1).expand(num_rows, k, max_path_len)
        masses = masses * valid_path.to(dtype) * float(sign)

        return (
            torch.cat([keys.reshape(-1), tail_keys], dim=0),
            torch.cat([masses.reshape(-1), tail_masses * float(sign)], dim=0),
        )

    def prepare_runtime_state(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        teacher_labels: torch.Tensor | None = None,
    ) -> None:
        """Validate or extend trie runtime state for one batch; input is current vocab sizes and optional teacher labels, output is None, and this exists to keep cached trie buffers aligned with runtime tensors."""
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
            raise ValueError("Teacher labels contain no supervised answer tokens for trie OT.")

        max_label = int(valid_labels.max().item())
        if max_label >= self.teacher_tokenizer_vocab_size:
            raise ValueError(
                "teacher labels contain ids outside tokenizer space: "
                f"max label {max_label} >= tokenizer vocab {self.teacher_tokenizer_vocab_size}"
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute mean trie-Wasserstein KD over aligned positions; input is student/teacher logits, output is a scalar loss tensor, and this exists as the main cross-tokenizer KD objective."""
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
        if student_logits.size(0) == 0:
            raise ValueError("Trie Wasserstein loss received no aligned supervised token positions.")
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
        student_temperature = float(student_temperature)
        teacher_temperature = float(teacher_temperature)

        student_scaled_logits = student_logits.float() / student_temperature
        teacher_scaled_logits = teacher_logits.detach().float() / teacher_temperature

        student_keys, student_masses = self.build_batched_signed_edge_contributions(
            scaled_logits=student_scaled_logits,
            path_flat=self.student_path_flat_device,
            path_offsets=self.student_path_offsets_device,
            ignored_mask=self.student_ignored_mask_device,
            valid_count=self.student_valid_count,
            max_path_len=self.student_max_path_len,
            path_arange=self.student_path_arange_device,
            sign=1.0,
        )
        teacher_keys, teacher_masses = self.build_batched_signed_edge_contributions(
            scaled_logits=teacher_scaled_logits,
            path_flat=self.teacher_path_flat_device,
            path_offsets=self.teacher_path_offsets_device,
            ignored_mask=self.teacher_ignored_mask_device,
            valid_count=self.teacher_valid_count,
            max_path_len=self.teacher_max_path_len,
            path_arange=self.teacher_path_arange_device,
            sign=-1.0,
        )

        all_keys = torch.cat([student_keys, teacher_keys], dim=0)
        signed_masses = torch.cat([student_masses, teacher_masses], dim=0)
        active_keys, inverse = torch.unique(all_keys, return_inverse=True)
        signed_edge_balance = signed_masses.new_zeros(active_keys.numel())
        signed_edge_balance.index_add_(0, inverse, signed_masses)
        active_edges = active_keys.remainder(self.num_edges)
        total_loss = (
            self.edge_weights_device.index_select(0, active_edges) * signed_edge_balance.abs()
        ).sum()
        return total_loss / student_logits.size(0)

__all__ = [
    "TrieWassersteinLoss",
]
