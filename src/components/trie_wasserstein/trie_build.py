from __future__ import annotations

from collections.abc import Collection

import torch

from src.constants import EOS_SENTINEL

from .canonicalization import canonicalize_token_piece
from .types import TrieNode, TrieRuntimeState

BOUNDARY_EDGE_WEIGHT = 0.05
BOUNDARY_EDGE_SYMBOLS = {
    "space": -10_000_001,
}

# token_id = vocabulary item id
# edge_id = shared trie structure id

def build_tokenizer_paths(
    *,
    tokenizer,
    vocab_size: int,
    ignored_token_ids: Collection[int],
    non_text_token_ids: Collection[int],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    token_paths: list[list[int]] = []
    ignored_mask = torch.zeros(vocab_size, dtype=torch.bool)
    non_text_mask = torch.zeros(vocab_size, dtype=torch.bool)

    for token_id in range(vocab_size):
        if token_id in ignored_token_ids:
            ignored_mask[token_id] = True
            token_paths.append([])
            continue

        if token_id in non_text_token_ids:
            non_text_mask[token_id] = True
            token_paths.append([])
            continue

        content_bytes, boundary_kind = canonicalize_token_piece(
            tokenizer, token_id
        )

        # Boundary markers are attached after the terminal edge so tokenizer
        # whitespace conventions do not fork the shared content path at ROOT.
        node = root
        path: list[int] = []
        for depth, byte_value in enumerate(
            [*content_bytes, EOS_SENTINEL], start=1
        ):
            child = node.children.get(byte_value)
            if child is None:
                child = TrieNode(edge_id=len(edge_weights))
                edge_weights.append(float(rho) ** (depth - 1))
                node.children[byte_value] = child
            path.append(child.edge_id)
            node = child

        if boundary_kind != "none":
            boundary_key = BOUNDARY_EDGE_SYMBOLS[boundary_kind]
            boundary_child = node.children.get(boundary_key)
            if boundary_child is None:
                boundary_child = TrieNode(edge_id=len(edge_weights))
                edge_weights.append(BOUNDARY_EDGE_WEIGHT)
                node.children[boundary_key] = boundary_child
            path.append(boundary_child.edge_id)

        token_paths.append(path)

    return token_paths, ignored_mask, non_text_mask


def build_trie_state_from_tokenizers(
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: Collection[int],
    teacher_ignored_token_ids: Collection[int],
    rho: float,
    student_tokenizer,
    teacher_tokenizer,
    student_non_text_token_ids: Collection[int] = (),
    teacher_non_text_token_ids: Collection[int] = (),
) -> TrieRuntimeState:
    root = TrieNode()
    edge_weights: list[float] = []

    # Student and teacher paths intentionally share one trie so matching byte
    # prefixes can cancel even when the two tokenizers use different vocabularies.
    student_token_paths, student_ignored_mask, student_non_text_mask = (
        build_tokenizer_paths(
                tokenizer=student_tokenizer,
            vocab_size=student_vocab_size,
            ignored_token_ids=student_ignored_token_ids,
            non_text_token_ids=student_non_text_token_ids,
            root=root,
            edge_weights=edge_weights,
            rho=rho,
        )
    )
    teacher_token_paths, teacher_ignored_mask, teacher_non_text_mask = (
        build_tokenizer_paths(
            tokenizer=teacher_tokenizer,
            vocab_size=teacher_vocab_size,
            ignored_token_ids=teacher_ignored_token_ids,
            non_text_token_ids=teacher_non_text_token_ids,
            root=root,
            edge_weights=edge_weights,
            rho=rho,
        )
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
