import torch

from src.components.trie_wasserstein import TrieWassersteinLoss
from src.tokenizer_utils import TOKENIZER_VOCAB_SIZES


class TinyTokenizer:
    """Tokenizer stub with deterministic byte pieces and configurable special ids."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = None
    unk_token_id = None
    all_special_ids = [0, 1]

    def __init__(self, model_id: str, pieces: dict[int, str]):
        self.name_or_path = model_id
        self.pieces = pieces

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) -> str:
        return self.pieces[int(ids[0])]

    def convert_tokens_to_ids(self, token: str):
        return None


def build_loss(
    *,
    model_id: str = "tiny/shared",
    topk: int = 1,
    tail_depth: int = 2,
) -> TrieWassersteinLoss:
    pieces = {
        0: "<pad>",
        1: "<bos>",
        2: "cat",
        3: "car",
        4: "cap",
        5: "the",
        6: "then",
        7: "they",
    }
    TOKENIZER_VOCAB_SIZES[model_id] = len(pieces)
    tokenizer = TinyTokenizer(model_id, pieces)
    return TrieWassersteinLoss(
        student_tokenizer=tokenizer,
        teacher_tokenizer=tokenizer,
        rho=0.5,
        topk=topk,
        tail_depth=tail_depth,
        tail_weight=0.5,
    )


def test_topk_equal_vocab_leaves_no_prefix_tail_mass():
    loss = build_loss(topk=99)
    logits = torch.randn(2, loss.student_vocab_size)
    _ = loss(logits, logits)

    student_tail_mass = loss.last_prefix_tail_stats["student"].tail_mass_mean
    teacher_tail_mass = loss.last_prefix_tail_stats["teacher"].tail_mass_mean
    assert torch.allclose(student_tail_mass, torch.zeros_like(student_tail_mass), atol=1e-6)
    assert torch.allclose(teacher_tail_mass, torch.zeros_like(teacher_tail_mass), atol=1e-6)


def test_same_tokenizer_same_logits_has_zero_loss():
    loss = build_loss(topk=1)
    logits = torch.randn(3, loss.student_vocab_size)
    value = loss(logits, logits)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_same_prefix_residual_costs_less_than_different_prefix_residual():
    loss = build_loss(topk=1, tail_depth=2)
    student_logits = torch.full((1, loss.student_vocab_size), -10.0)
    same_prefix_teacher_logits = torch.full_like(student_logits, -10.0)
    different_prefix_teacher_logits = torch.full_like(student_logits, -10.0)

    student_logits[0, 2] = 4.0
    same_prefix_teacher_logits[0, 2] = 4.0
    different_prefix_teacher_logits[0, 2] = 4.0

    student_logits[0, 3] = 3.0
    student_logits[0, 4] = 3.0
    same_prefix_teacher_logits[0, 3] = 2.5
    same_prefix_teacher_logits[0, 4] = 3.5
    different_prefix_teacher_logits[0, 5] = 3.0
    different_prefix_teacher_logits[0, 6] = 3.0

    same_prefix_loss = loss(student_logits, same_prefix_teacher_logits)
    different_prefix_loss = loss(student_logits, different_prefix_teacher_logits)
    assert same_prefix_loss < different_prefix_loss


def test_ignored_tokens_do_not_consume_probability_mass():
    loss = build_loss(topk=1)
    teacher_logits = torch.zeros(1, loss.teacher_vocab_size)
    student_logits_high_ignored = torch.zeros(1, loss.student_vocab_size)
    student_logits_low_ignored = student_logits_high_ignored.clone()

    student_logits_high_ignored[0, 0] = 100.0
    student_logits_high_ignored[0, 1] = 100.0
    student_logits_low_ignored[0, 0] = -100.0
    student_logits_low_ignored[0, 1] = -100.0

    high_ignored_loss = loss(student_logits_high_ignored, teacher_logits)
    low_ignored_loss = loss(student_logits_low_ignored, teacher_logits)
    torch.testing.assert_close(high_ignored_loss, low_ignored_loss, atol=1e-6, rtol=1e-6)
