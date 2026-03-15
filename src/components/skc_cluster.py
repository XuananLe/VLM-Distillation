"""
SKC Matrix Analysis Pipeline
==============================

Given a precomputed SKC matrix, this module provides:
    1. Validation    — sanity checks before doing anything
    2. Diagnostics   — flag near-identical models early
    3. Clustering    — group similar models
    4. Selection     — pick one representative per cluster
"""

import itertools
from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Step 1: Validate the matrix
# ---------------------------------------------------------------------------

def validate_skc_matrix(skc_matrix: np.ndarray, model_names: List[str]) -> None:
    """
    Sanity checks before doing anything else.
    Raises AssertionError with a clear message if something is wrong.
    """
    k = len(model_names)
    assert skc_matrix.shape == (k, k), (
        f"Matrix shape {skc_matrix.shape} does not match {k} models."
    )
    assert np.allclose(skc_matrix, skc_matrix.T, atol=1e-4), (
        "SKC matrix is not symmetric. Check that skc_score(A,B) == skc_score(B,A)."
    )
    assert np.allclose(np.diag(skc_matrix), 0.0, atol=1e-4), (
        "Diagonal should be 0 by convention; self-similarity is not stored."
    )
    assert skc_matrix.min() >= 0.0 and skc_matrix.max() <= 1.0, (
        f"SKC values should be in [0, 1]. "
        f"Got [{skc_matrix.min():.4f}, {skc_matrix.max():.4f}]."
    )
    print(f"Matrix OK: {k} models, {k*(k-1)//2} pairs.")
    positive = skc_matrix[skc_matrix > 0]
    if positive.size > 0:
        print(f"  SKC range : [{positive.min():.4f}, {skc_matrix.max():.4f}]")
    else:
        print(f"  SKC range : [0.0000, {skc_matrix.max():.4f}]")
    print(f"  Mean SKC  : {skc_matrix[np.triu_indices(k, k=1)].mean():.4f}")


# ---------------------------------------------------------------------------
# Step 2: Diagnostics — flag problems before selection
# ---------------------------------------------------------------------------

def diagnose(
    skc_matrix: np.ndarray,
    model_names: List[str],
    threshold: float = 0.90,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Flag models that are:
        - Universally redundant: min similarity to any other model > threshold
          → this model is near-identical to the entire candidate pool
        - Clone pairs: a specific pair with similarity > threshold
          → near-identical models wasting a slot

    Returns:
        redundant   : list of universally redundant model names
        clone_pairs : list of (name_a, name_b, skc) tuples
    """
    k = len(model_names)
    print("\n--- Diagnostics ---")

    redundant = []
    for i in range(k):
        row = [skc_matrix[i, j] for j in range(k) if j != i]
        if min(row) > threshold:
            redundant.append(model_names[i])
            print(f"  WARNING: {model_names[i]} is universally redundant "
                  f"(min similarity={min(row):.4f} > {threshold})")

    clone_pairs = []
    for i, j in itertools.combinations(range(k), 2):
        if skc_matrix[i, j] > threshold:
            clone_pairs.append((model_names[i], model_names[j], skc_matrix[i, j]))
            print(f"  WARNING: {model_names[i]} ↔ {model_names[j]} are near-identical "
                  f"(similarity={skc_matrix[i, j]:.4f})")

    if not redundant and not clone_pairs:
        print("  All models pass. No redundant pairs detected.")

    return redundant, clone_pairs


# ---------------------------------------------------------------------------
# Step 3: Cluster — find natural family groups
# ---------------------------------------------------------------------------

def cluster_models(
    skc_matrix: np.ndarray,
    model_names: List[str],
    n_clusters: int,
    linkage_method: str = "average",
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Hierarchical clustering on the SKC distance matrix (1 - SKC).

    Why "average" linkage:
        SKC is not guaranteed to satisfy the triangle inequality, so the
        distance matrix may not be Euclidean. Ward linkage assumes Euclidean
        geometry and can produce unstable clusters. Average linkage (UPGMA)
        only requires ultrametric consistency and is more robust here.

    Returns:
        labels   : (k,) cluster assignment per model, 1-indexed
        clusters : list of lists, each containing model names in that cluster
    """
    dist_matrix = 1.0 - skc_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    clusters = []
    for cid in range(1, n_clusters + 1):
        members = [model_names[i] for i, l in enumerate(labels) if l == cid]
        clusters.append(members)

    print(f"\n--- Clusters (n={n_clusters}, linkage={linkage_method}) ---")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster}")

    return labels, clusters


# ---------------------------------------------------------------------------
# Step 4: Select — one representative per cluster
# ---------------------------------------------------------------------------

def select_via_clustering(
    skc_matrix: np.ndarray,
    model_names: List[str],
    n_teachers: int,
    linkage_method: str = "average",
) -> List[str]:
    """
    Cluster-then-select: pick one representative per cluster.

    Representative = model with the highest mean similarity to other members
    of its cluster. This picks a central representative for that similarity
    group.
    """
    labels, clusters = cluster_models(
        skc_matrix, model_names, n_teachers, linkage_method
    )

    k = len(model_names)
    selected = []

    for cid in range(1, n_teachers + 1):
        member_indices = [i for i, l in enumerate(labels) if l == cid]
        if not member_indices:
            continue

        # Pick the most central member within the cluster.
        mean_scores = [
            np.mean([skc_matrix[i, j] for j in member_indices if j != i])
            for i in member_indices
        ]
        best = member_indices[np.argmax(mean_scores)]
        selected.append(model_names[best])

    print(f"\nSelected ({n_teachers} teachers): {selected}")
    return selected


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    skc_matrix: np.ndarray,
    model_names: List[str],
    n_teachers: int = 2,
    redundancy_threshold: float = 0.90,
    linkage_method: str = "average",
) -> List[str]:
    """
    Full SKC matrix → teacher selection pipeline.

        1. Validate matrix
        2. Diagnose near-identical / clone pairs
        3. Cluster and select one representative per cluster

    Example:
        selected = run_pipeline(
            skc_matrix,
            model_names=["Qwen2-VL-2B", "Qwen2.5-VL-3B", "LLaVA-1.6-Mistral"],
            n_teachers=2,
        )
        # → ["Qwen2-VL-2B", "LLaVA-1.6-Mistral"]
    """
    validate_skc_matrix(skc_matrix, model_names)
    diagnose(skc_matrix, model_names, threshold=redundancy_threshold)
    return select_via_clustering(skc_matrix, model_names, n_teachers, linkage_method)
