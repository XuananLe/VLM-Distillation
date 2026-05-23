from __future__ import annotations

import torch

from src.constants import EOS_SENTINEL

from .canonicalization import canonicalize_token_piece
from .types import BoundaryKind, TrieNode, TrieRuntimeState

BOUNDARY_EDGE_SYMBOLS: dict[BoundaryKind, int] = {
    "none": 0,
    "space": -10_000_001,
    "continuation": -10_000_002,
    "end_word": -10_000_003,
}

if EOS_SENTINEL in {symbol for kind, symbol in BOUNDARY_EDGE_SYMBOLS.items() if kind != "none"}:
    raise ValueError(f"EOS_SENTINEL={EOS_SENTINEL!r} collides with boundary edge sentinels")


def build_tokenizer_paths(
    *,
    tokenizer,
    vocab_size: int,
    ignored_token_ids: tuple[int, ...],
    non_text_token_ids: tuple[int, ...],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
    boundary_weight: float,
    underscore_is_boundary_marker: bool,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    ignored_token_ids_set = set(ignored_token_ids)
    non_text_token_ids_set = set(non_text_token_ids)
    token_paths: list[list[int]] = []
    ignored_mask = torch.zeros(vocab_size, dtype=torch.bool)
    non_text_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_id in range(vocab_size):
        if token_id in ignored_token_ids_set:
            ignored_mask[token_id] = True
            token_paths.append([])
            continue

        if token_id in non_text_token_ids_set:
            non_text_mask[token_id] = True
            token_paths.append([])
            continue

        content_bytes, boundary_kind = canonicalize_token_piece(
            tokenizer,
            token_id,
            underscore_is_boundary_marker=underscore_is_boundary_marker,
        )

        # Boundary markers are attached after the terminal edge so tokenizer
        # whitespace conventions do not fork the shared content path at ROOT.
        node = root
        path: list[int] = []
        for depth, byte_value in enumerate([*content_bytes, EOS_SENTINEL], start=1):
            child = node.children.get(byte_value)
            if child is None:
                child = TrieNode()
                child.edge_id = len(edge_weights)
                edge_weights.append(float(rho) ** (depth - 1))
                node.children[byte_value] = child
            if child.edge_id is None:
                raise RuntimeError("Trie child edge_id was not initialized")
            path.append(child.edge_id)
            node = child

        if boundary_kind != "none":
            boundary_key = BOUNDARY_EDGE_SYMBOLS[boundary_kind]
            boundary_child = node.children.get(boundary_key)
            if boundary_child is None:
                boundary_child = TrieNode()
                boundary_child.edge_id = len(edge_weights)
                edge_weights.append(float(boundary_weight))
                node.children[boundary_key] = boundary_child
            if boundary_child.edge_id is None:
                raise RuntimeError("Boundary child edge_id was not initialized")
            path.append(boundary_child.edge_id)

        token_paths.append(path)

    return token_paths, ignored_mask, non_text_mask


def build_trie_state_from_tokenizers(
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: tuple[int, ...],
    teacher_ignored_token_ids: tuple[int, ...],
    rho: float,
    student_tokenizer,
    teacher_tokenizer,
    student_non_text_token_ids: tuple[int, ...] = (),
    teacher_non_text_token_ids: tuple[int, ...] = (),
    boundary_weight: float = 0.05,
    student_underscore_is_boundary_marker: bool = False,
    teacher_underscore_is_boundary_marker: bool = False,
) -> TrieRuntimeState:
    if boundary_weight < 0.0:
        raise ValueError(f"boundary_weight must be >= 0, got {boundary_weight}")

    root = TrieNode()
    edge_weights: list[float] = []

    # Student and teacher paths intentionally share one trie so matching byte
    # prefixes can cancel even when the two tokenizers use different vocabularies.
    student_token_paths, student_ignored_mask, student_non_text_mask = build_tokenizer_paths(
        tokenizer=student_tokenizer,
        vocab_size=student_vocab_size,
        ignored_token_ids=student_ignored_token_ids,
        non_text_token_ids=student_non_text_token_ids,
        root=root,
        edge_weights=edge_weights,
        rho=rho,
        boundary_weight=boundary_weight,
        underscore_is_boundary_marker=student_underscore_is_boundary_marker,
    )
    teacher_token_paths, teacher_ignored_mask, teacher_non_text_mask = build_tokenizer_paths(
        tokenizer=teacher_tokenizer,
        vocab_size=teacher_vocab_size,
        ignored_token_ids=teacher_ignored_token_ids,
        non_text_token_ids=teacher_non_text_token_ids,
        root=root,
        edge_weights=edge_weights,
        rho=rho,
        boundary_weight=boundary_weight,
        underscore_is_boundary_marker=teacher_underscore_is_boundary_marker,
    )

    tail_edge_id = len(edge_weights)
    edge_weights.append(1.0)

    return TrieRuntimeState(
        edge_weights=torch.tensor(edge_weights, dtype=torch.float32),
        student_token_paths=student_token_paths,
        student_ignored_mask=student_ignored_mask,
        student_non_text_mask=student_non_text_mask,
        teacher_token_paths=teacher_token_paths,
        teacher_ignored_mask=teacher_ignored_mask,
        teacher_non_text_mask=teacher_non_text_mask,
        tail_edge_id=tail_edge_id,
    )
