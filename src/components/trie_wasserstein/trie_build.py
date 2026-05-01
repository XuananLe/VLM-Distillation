from __future__ import annotations

import torch

from src.tokenizer_utils import token_piece_to_bytes
from .constants import EOS_SENTINEL
from .types import TrieBuildResult, TrieNode, TrieRuntimeState


def insert_token_bytes(
    *,
    token_bytes: list[int],
    root: TrieNode,
    edge_weights: list[float],
    rho: float,
) -> list[int]:
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

        # The terminal marker distinguishes an exact token from a prefix of a
        # longer token, so "a" and "apple" do not share the same full-token path.
        full_token_bytes = token_bytes + [EOS_SENTINEL]
        path = insert_token_bytes(
            token_bytes=full_token_bytes,
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
    student_vocab_size: int,
    teacher_vocab_size: int,
    student_ignored_token_ids: tuple[int, ...],
    teacher_ignored_token_ids: tuple[int, ...],
    rho: float,
    student_tokenizer,
    teacher_tokenizer,
) -> TrieRuntimeState:
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

    return TrieRuntimeState(
        edge_weights=torch.tensor(edge_weights, dtype=torch.float32),
        student_path_flat=student_paths.path_flat,
        student_path_offsets=student_paths.path_offsets,
        student_ignored_mask=student_paths.ignored_mask,
        teacher_path_flat=teacher_paths.path_flat,
        teacher_path_offsets=teacher_paths.path_offsets,
        teacher_ignored_mask=teacher_paths.ignored_mask,
        num_edges=len(edge_weights),
        tail_edge_id=tail_edge_id,
        student_valid_count=int((~student_paths.ignored_mask).sum().item()),
        teacher_valid_count=int((~teacher_paths.ignored_mask).sum().item()),
    )
