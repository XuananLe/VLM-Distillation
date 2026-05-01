from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.trie_wasserstein.contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from src.components.trie_wasserstein.trie_build import build_trie_state_from_tokenizers


class TinyTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.pieces[int(token_id)] for token_id in ids)


def test_trie_wasserstein_input_output_example():
    student_tokenizer = TinyTokenizer(
        {
            0: "a",
            1: "ab",
            2: "b",
        }
    )
    teacher_tokenizer = TinyTokenizer(
        {
            0: "a",
            1: "ab",
            2: "b",
            3: "c",
        }
    )
    trie_state = build_trie_state_from_tokenizers(
        student_vocab_size=3,
        teacher_vocab_size=4,
        student_ignored_token_ids=(),
        teacher_ignored_token_ids=(),
        rho=0.5,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    print("student_token_paths:", trie_state.student_token_paths)
    print("teacher_token_paths:", trie_state.teacher_token_paths)
    print("edge_weights:", trie_state.edge_weights.tolist())
    print("tail_edge_id:", trie_state.tail_edge_id)

    student_logits_raw = torch.log(
        torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
            ]
        )
    )
    teacher_logits_raw = torch.log(
        torch.tensor(
            [
                [0.2, 0.6, 0.1, 0.1],
                [0.1, 0.3, 0.5, 0.1],
                [0.7, 0.1, 0.1, 0.1],
            ]
        )
    )

    aligned_length = min(student_logits_raw.size(0), teacher_logits_raw.size(0))
    student_logits = student_logits_raw[:aligned_length]
    teacher_logits = teacher_logits_raw[:aligned_length]

    print("student_logits_raw_shape:", tuple(student_logits_raw.shape))
    print("teacher_logits_raw_shape:", tuple(teacher_logits_raw.shape))
    print("aligned_length:", aligned_length)
    print("student_logits_aligned_shape:", tuple(student_logits.shape))
    print("teacher_logits_aligned_shape:", tuple(teacher_logits.shape))

    student_edge_masses = build_signed_edge_contributions(
        scaled_logits=student_logits,
        token_paths=trie_state.student_token_paths,
        ignored_mask=trie_state.student_ignored_mask,
        tail_edge_id=trie_state.tail_edge_id,
        topk=4,
        sign=1.0,
    )
    teacher_edge_masses = build_signed_edge_contributions(
        scaled_logits=teacher_logits,
        token_paths=trie_state.teacher_token_paths,
        ignored_mask=trie_state.teacher_ignored_mask,
        tail_edge_id=trie_state.tail_edge_id,
        topk=4,
        sign=-1.0,
    )

    print("student_edge_masses:", student_edge_masses)
    print("teacher_edge_masses:", teacher_edge_masses)

    trie_loss = reduce_signed_edge_contributions_to_tree_loss(
        student_edge_masses=student_edge_masses,
        teacher_edge_masses=teacher_edge_masses,
        edge_weights=trie_state.edge_weights,
    )
    print("trie_loss:", float(trie_loss))

    assert trie_state.student_token_paths == [[0, 1], [0, 2, 3], [4, 5]]
    assert trie_state.teacher_token_paths == [[0, 1], [0, 2, 3], [4, 5], [6, 7]]
    assert tuple(student_logits_raw.shape) == (2, 3)
    assert tuple(teacher_logits_raw.shape) == (3, 4)
    assert tuple(student_logits.shape) == (2, 3)
    assert tuple(teacher_logits.shape) == (2, 4)
    torch.testing.assert_close(trie_loss, torch.tensor(0.8125), atol=1e-6, rtol=0)


if __name__ == "__main__":
    test_trie_wasserstein_input_output_example()
