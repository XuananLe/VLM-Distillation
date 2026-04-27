import torch
import time

from src.components.trie_wasserstein import TrieWassersteinLoss


STUDENT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
TEACHER_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "ibm-granite/granite-vision-3.1-2b-preview",
    "google/gemma-3-4b-it",
)


class DeterministicFakeTokenizer:
    """Tokenizer stub with real vocab size and deterministic byte pieces."""

    pad_token_id = 0
    bos_token_id = 1

    def __init__(self, model_id: str):
        self.name_or_path = model_id

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) -> str:
        token_id = int(ids[0])
        if token_id in {self.pad_token_id, self.bos_token_id}:
            return f"<special_{token_id}>"
        # Keep pieces short so trie construction is fast, while still producing
        # shared prefixes and nontrivial paths across real vocab-sized logits.
        return f"tok_{token_id % 4096:04x}_{token_id // 4096:03x}"


def old_build_signed_edge_contributions(
    loss: TrieWassersteinLoss,
    *,
    probs: torch.Tensor,
    path_flat: torch.Tensor,
    path_offsets: torch.Tensor,
    ignored_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = ~ignored_mask
    num_valid = int(valid_mask.sum().item())
    tail_edge = torch.tensor([loss.tail_edge_id], device=probs.device, dtype=torch.long)

    if num_valid == 0:
        tail_mass = probs.sum().unsqueeze(0)
        return tail_edge, tail_mass

    k = min(loss.topk, num_valid)
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


def old_single_step_loss(
    loss: TrieWassersteinLoss,
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    student_edges, student_masses = old_build_signed_edge_contributions(
        loss,
        probs=student_probs,
        path_flat=loss.student_path_flat_device,
        path_offsets=loss.student_path_offsets_device,
        ignored_mask=loss.student_ignored_mask_device,
    )
    teacher_edges, teacher_masses = old_build_signed_edge_contributions(
        loss,
        probs=teacher_probs,
        path_flat=loss.teacher_path_flat_device,
        path_offsets=loss.teacher_path_offsets_device,
        ignored_mask=loss.teacher_ignored_mask_device,
    )

    all_edges = torch.cat([student_edges, teacher_edges], dim=0)
    signed_masses = torch.cat([student_masses, -teacher_masses], dim=0)
    active_edges, inverse = torch.unique(all_edges, sorted=False, return_inverse=True)
    signed_edge_balance = signed_masses.new_zeros(active_edges.size(0))
    signed_edge_balance.index_add_(0, inverse, signed_masses)
    return (
        loss.edge_weights_device.index_select(0, active_edges) * signed_edge_balance.abs()
    ).sum()


def old_trie_forward(
    loss: TrieWassersteinLoss,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    loss.ensure_device_tensors(student_logits.device)
    student_probs = torch.softmax(student_logits.float(), dim=-1)
    teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
    step_losses = [
        old_single_step_loss(loss, student_probs[index], teacher_probs[index])
        for index in range(student_probs.size(0))
    ]
    return torch.stack(step_losses, dim=0).mean()


def time_cuda_call(fn, *, warmup: int = 5, repeat: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return timings


def mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean().item()), float(tensor.std(unbiased=True).item())


def test_batched_trie_matches_old_reference_for_training_vocab_sizes():
    if not torch.cuda.is_available():
        raise RuntimeError("This regression test requires CUDA because training uses CUDA logits.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    student_tokenizer = DeterministicFakeTokenizer(STUDENT_MODEL)

    for teacher_model in TEACHER_MODELS:
        loss = TrieWassersteinLoss(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=DeterministicFakeTokenizer(teacher_model),
            rho=0.9,
            topk=64,
        ).to(device)

        student_logits = torch.randn(
            8,
            loss.student_vocab_size,
            device=device,
            requires_grad=True,
        )
        teacher_logits = torch.randn(
            8,
            loss.teacher_vocab_size,
            device=device,
        )

        old_value = old_trie_forward(loss, student_logits, teacher_logits)
        new_value = loss(student_logits, teacher_logits)
        old_grad, = torch.autograd.grad(old_value, student_logits, retain_graph=True)
        new_grad, = torch.autograd.grad(new_value, student_logits)
        old_times = time_cuda_call(lambda: old_trie_forward(loss, student_logits, teacher_logits))
        new_times = time_cuda_call(lambda: loss(student_logits, teacher_logits))
        old_time_mean, old_time_std = mean_std(old_times)
        new_time_mean, new_time_std = mean_std(new_times)
        speedups = [old_time / new_time for old_time, new_time in zip(old_times, new_times)]
        speedup_mean, speedup_std = mean_std(speedups)

        print(f"teacher: {teacher_model}")
        print("old:", old_value.item())
        print("new:", new_value.item())
        print("abs diff:", (old_value - new_value).abs().item())
        print("grad max diff:", (old_grad - new_grad).abs().max().item())
        print("old forward mean s:", old_time_mean)
        print("old forward std s:", old_time_std)
        print("new forward mean s:", new_time_mean)
        print("new forward std s:", new_time_std)
        print("speedup mean:", speedup_mean)
        print("speedup std:", speedup_std)

        torch.testing.assert_close(new_value, old_value, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(new_grad, old_grad, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_batched_trie_matches_old_reference_for_training_vocab_sizes()
