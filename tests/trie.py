import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.trie_wasserstein.contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from src.components.trie_wasserstein.trie_build import build_trie_state_from_tokenizers
from src.tokenizer_utils import (
    SUPPORTED_MODEL_NON_TEXT_TOKEN_STRINGS,
    VLM_NON_TEXT_TOKEN_STRINGS,
    collect_non_text_token_ids,
    default_ignored_token_ids,
)


class TinyTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.pieces[int(token_id)] for token_id in ids)

    def convert_tokens_to_ids(self, token):
        for token_id, piece in self.pieces.items():
            if piece == token:
                return token_id
        return None


class VisionSpecialTokenizer(TinyTokenizer):
    def __init__(self, pieces):
        super().__init__(pieces)
        self.additional_special_tokens = [
            token for token in pieces.values() if token.startswith("<") and token.endswith(">")
        ]
        self.special_tokens_map = {
            "additional_special_tokens": self.additional_special_tokens,
        }
        self.model_specific_special_tokens = {
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "image_token": "<image_soft_token>",
        }

    def get_added_vocab(self):
        return {
            token: token_id for token_id, token in self.pieces.items() if token.startswith("<") and token.endswith(">")
        }


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

    student_edge_masses, student_non_text_masses = build_signed_edge_contributions(
        scaled_logits=student_logits,
        token_paths=trie_state.student_token_paths,
        ignored_mask=trie_state.student_ignored_mask,
        non_text_mask=trie_state.student_non_text_mask,
        tail_edge_id=trie_state.tail_edge_id,
        topk=4,
        sign=1.0,
    )
    teacher_edge_masses, teacher_non_text_masses = build_signed_edge_contributions(
        scaled_logits=teacher_logits,
        token_paths=trie_state.teacher_token_paths,
        ignored_mask=trie_state.teacher_ignored_mask,
        non_text_mask=trie_state.teacher_non_text_mask,
        tail_edge_id=trie_state.tail_edge_id,
        topk=4,
        sign=-1.0,
    )

    print("student_edge_masses:", student_edge_masses)
    print("teacher_edge_masses:", teacher_edge_masses)
    print("student_non_text_masses:", student_non_text_masses)
    print("teacher_non_text_masses:", teacher_non_text_masses)

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
    torch.testing.assert_close(student_non_text_masses, torch.zeros(2))
    torch.testing.assert_close(teacher_non_text_masses, torch.zeros(2))
    torch.testing.assert_close(trie_loss, torch.tensor(0.8125), atol=1e-6, rtol=0)


def test_vlm_non_text_tokens_stay_out_of_text_trie():
    non_text_piece_map = {
        token_id: token for token_id, token in enumerate(("cat", *VLM_NON_TEXT_TOKEN_STRINGS, "<end_of_utterance>"))
    }
    tokenizer = VisionSpecialTokenizer(non_text_piece_map)

    non_text_ids = collect_non_text_token_ids(tokenizer)
    ignored_ids = default_ignored_token_ids(tokenizer)
    expected_non_text_ids = {
        token_id for token_id, token in non_text_piece_map.items() if token in VLM_NON_TEXT_TOKEN_STRINGS
    }
    control_token_id = tokenizer.convert_tokens_to_ids("<end_of_utterance>")

    assert expected_non_text_ids.issubset(non_text_ids)
    assert control_token_id in ignored_ids
    assert not (expected_non_text_ids & ignored_ids)


def test_supported_pool_non_text_token_list_is_complete():
    supported_tokens = {token for tokens in SUPPORTED_MODEL_NON_TEXT_TOKEN_STRINGS.values() for token in tokens}
    assert supported_tokens == set(VLM_NON_TEXT_TOKEN_STRINGS)
    assert "<|fim_prefix|>" not in supported_tokens
    assert "<fim_prefix>" not in supported_tokens


def test_non_text_probability_mass_is_visible():
    student_tokenizer = TinyTokenizer(
        {
            0: "cat",
            1: "<image>",
            2: "dog",
        }
    )
    teacher_tokenizer = TinyTokenizer(
        {
            0: "cat",
            1: "dog",
        }
    )
    trie_state = build_trie_state_from_tokenizers(
        student_vocab_size=3,
        teacher_vocab_size=2,
        student_ignored_token_ids=(),
        teacher_ignored_token_ids=(),
        student_non_text_token_ids=(1,),
        teacher_non_text_token_ids=(),
        rho=0.5,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    student_logits = torch.log(torch.tensor([[0.5, 0.4, 0.1]]))
    student_edge_masses, student_non_text_masses = build_signed_edge_contributions(
        scaled_logits=student_logits,
        token_paths=trie_state.student_token_paths,
        ignored_mask=trie_state.student_ignored_mask,
        non_text_mask=trie_state.student_non_text_mask,
        tail_edge_id=trie_state.tail_edge_id,
        topk=3,
        sign=1.0,
    )

    assert trie_state.student_token_paths[1] == []
    assert bool(trie_state.student_non_text_mask[1])
    assert not bool(trie_state.student_ignored_mask[1])
    torch.testing.assert_close(student_non_text_masses, torch.tensor([0.4]))
    torch.testing.assert_close(student_edge_masses[0][trie_state.tail_edge_id], torch.tensor(0.0))


if __name__ == "__main__":
    test_trie_wasserstein_input_output_example()
    test_vlm_non_text_tokens_stay_out_of_text_trie()
    test_supported_pool_non_text_token_list_is_complete()
    test_non_text_probability_mass_is_visible()
