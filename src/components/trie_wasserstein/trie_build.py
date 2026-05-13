from __future__ import annotations

import torch

from src.constants import EOS_SENTINEL
from .canonicalization import canonicalize_token_piece
from .types import BoundaryKind, TrieBuildResult, TrieNode, TrieRuntimeState


BOUNDARY_EDGE_SYMBOLS: dict[BoundaryKind, int] = {
    "none": 0,
    "space": -10_000_001,
    "continuation": -10_000_002,
    "end_word": -10_000_003,
}


def assert_no_sentinel_collision() -> None:
    boundary_symbols = {
        symbol for kind, symbol in BOUNDARY_EDGE_SYMBOLS.items() if kind != "none"
    }
    if EOS_SENTINEL in boundary_symbols:
        raise ValueError(
            f"EOS_SENTINEL={EOS_SENTINEL!r} collides with boundary edge sentinels"
        )


def insert_child_edge(
    *,
    node: TrieNode,
    edge_key: int,
    edge_weights: list[float],
    edge_weight: float,
) -> TrieNode:
    child = node.children.get(edge_key)
    if child is None:
        child = TrieNode()
        child.edge_id = len(edge_weights)
        edge_weights.append(float(edge_weight))
        node.children[edge_key] = child
    return child


def insert_token_bytes(
    *,
    token_bytes: list[int],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
    boundary_kind: BoundaryKind = "none",
    boundary_weight: float = 0.05,
) -> list[int]:
    assert_no_sentinel_collision()

    node = root
    path: list[int] = []
    for depth, byte_value in enumerate(token_bytes, start=1):
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
        boundary_child = insert_child_edge(
            node=node,
            edge_key=BOUNDARY_EDGE_SYMBOLS[boundary_kind],
            edge_weights=edge_weights,
            edge_weight=boundary_weight,
        )
        if boundary_child.edge_id is None:
            raise RuntimeError("Boundary child edge_id was not initialized")
        path.append(boundary_child.edge_id)

    return path


def build_tokenizer_paths(
    *,
    tokenizer,
    vocab_size: int,
    ignored_token_ids: tuple[int, ...],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
    boundary_weight: float,
    underscore_is_boundary_marker: bool,
) -> TrieBuildResult:
    ignored_token_ids_set = set(ignored_token_ids)
    token_paths: list[list[int]] = []
    ignored_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_id in range(vocab_size):
        if token_id in ignored_token_ids_set:
            ignored_mask[token_id] = True
            token_paths.append([])
            continue

        piece = canonicalize_token_piece(
            tokenizer,
            token_id,
            underscore_is_boundary_marker=underscore_is_boundary_marker,
        )

        # Boundary markers are attached after the terminal edge so tokenizer
        # whitespace conventions do not fork the shared content path at ROOT.
        full_token_bytes = list(piece.content_bytes) + [EOS_SENTINEL]
        path = insert_token_bytes(
            token_bytes=full_token_bytes,
            root=root,
            edge_weights=edge_weights,
            rho=rho,
            boundary_kind=piece.boundary_kind,
            boundary_weight=boundary_weight,
        )
        token_paths.append(path)

    return TrieBuildResult(
        token_paths=token_paths,
        ignored_mask=ignored_mask,
    )


def build_trie_state_from_tokenizers(
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: tuple[int, ...],
    teacher_ignored_token_ids: tuple[int, ...],
    rho: float,
    student_tokenizer,
    teacher_tokenizer,
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
    student_paths = build_tokenizer_paths(
        tokenizer=student_tokenizer,
        vocab_size=student_vocab_size,
        ignored_token_ids=student_ignored_token_ids,
        root=root,
        edge_weights=edge_weights,
        rho=rho,
        boundary_weight=boundary_weight,
        underscore_is_boundary_marker=student_underscore_is_boundary_marker,
    )
    teacher_paths = build_tokenizer_paths(
        tokenizer=teacher_tokenizer,
        vocab_size=teacher_vocab_size,
        ignored_token_ids=teacher_ignored_token_ids,
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
        student_token_paths=student_paths.token_paths,
        student_ignored_mask=student_paths.ignored_mask,
        teacher_token_paths=teacher_paths.token_paths,
        teacher_ignored_mask=teacher_paths.ignored_mask,
        tail_edge_id=tail_edge_id,
    )
