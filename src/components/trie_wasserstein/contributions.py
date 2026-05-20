from __future__ import annotations

import torch


def build_signed_edge_contributions(
    *,
    scaled_logits: torch.Tensor,
    token_paths: list[list[int]],
    ignored_mask: torch.Tensor,
    non_text_mask: torch.Tensor,
    tail_edge_id: int,
    topk: int,
    sign: float,
) -> tuple[list[dict[int, torch.Tensor]], torch.Tensor]:
    pathable_mask = (~ignored_mask) & (~non_text_mask)
    pathable_count = int(pathable_mask.sum().item())
    if pathable_count <= 0:
        raise ValueError("Trie loss has no text tokens with byte paths.")

    k = min(topk, pathable_count)
    row_edge_masses: list[dict[int, torch.Tensor]] = []
    row_non_text_masses: list[torch.Tensor] = []

    for row_logits in scaled_logits:
        active_logits = row_logits.masked_fill(ignored_mask, float("-inf"))
        pathable_logits = row_logits.masked_fill(~pathable_mask, float("-inf"))
        kept_logits, kept_token_ids = torch.topk(pathable_logits, k=k, dim=-1, sorted=False)

        log_z = torch.logsumexp(active_logits, dim=-1)
        kept_masses = (kept_logits - log_z).exp()
        pathable_mass = (pathable_logits - log_z).exp().sum()
        tail_mass = (pathable_mass - kept_masses.sum()).clamp_min(0.0)
        non_text_logits = row_logits.masked_fill(~(non_text_mask & ~ignored_mask), float("-inf"))
        non_text_mass = torch.nan_to_num((non_text_logits - log_z).exp().sum())

        edge_masses: dict[int, torch.Tensor] = {}
        for token_id, mass in zip(kept_token_ids, kept_masses):
            edge_path = token_paths[int(token_id.item())]
            if not edge_path:
                continue
            signed_mass = mass * float(sign)
            for edge_id in edge_path:
                edge_masses[edge_id] = edge_masses.get(edge_id, 0.0) + signed_mass

        edge_masses[tail_edge_id] = tail_mass * float(sign)
        row_edge_masses.append(edge_masses)
        row_non_text_masses.append(non_text_mass)

    return row_edge_masses, torch.stack(row_non_text_masses)


def reduce_signed_edge_contributions_to_tree_loss(
    *,
    student_edge_masses: list[dict[int, torch.Tensor]],
    teacher_edge_masses: list[dict[int, torch.Tensor]],
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    total_loss = edge_weights.new_zeros(())
    for student_edges, teacher_edges in zip(
        student_edge_masses,
        teacher_edge_masses,
        strict=True,
    ):
        edge_ids = student_edges.keys() | teacher_edges.keys()
        for edge_id in edge_ids:
            edge_balance = edge_weights.new_zeros(())
            if edge_id in student_edges:
                edge_balance = edge_balance + student_edges[edge_id]
            if edge_id in teacher_edges:
                edge_balance = edge_balance + teacher_edges[edge_id]
            total_loss = total_loss + edge_weights[edge_id] * edge_balance.abs()
    return total_loss / len(student_edge_masses)
