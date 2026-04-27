#!/usr/bin/env python3
"""Reproduce the PCGrad 2D multi-task optimization toy landscape.

The loss functions come from Appendix D of "Gradient Surgery for Multi-Task
Learning" (Yu et al., NeurIPS 2020). The Adam run shows the phenomenon from
Figure 1: one task has a deep curved valley, the task gradients conflict, and
vanilla Adam can spend many updates moving along the wrong valley.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


LOSS_CLIP = 5e-6
ADAM_EPS = 1e-8


@dataclass(frozen=True)
class OptimRun:
    path: np.ndarray
    final_theta: np.ndarray
    final_losses: np.ndarray


def task_losses_xy(theta1: np.ndarray, theta2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Task losses from the PCGrad appendix, evaluated on scalars or grids."""
    tanh_theta2 = np.tanh(theta2)
    task1_residual = 0.5 * theta1 + tanh_theta2
    task2_residual = 0.5 * theta1 - tanh_theta2 + 2.0

    loss1 = 20.0 * np.log(np.maximum(np.abs(task1_residual), LOSS_CLIP))
    loss2 = 25.0 * np.log(np.maximum(np.abs(task2_residual), LOSS_CLIP))
    return loss1, loss2


def task_losses(theta: np.ndarray) -> np.ndarray:
    loss1, loss2 = task_losses_xy(theta[0], theta[1])
    return np.array([float(loss1), float(loss2)])


def task_gradients(theta: np.ndarray) -> np.ndarray:
    """Analytic gradients for both task losses at a single theta point."""
    theta1, theta2 = float(theta[0]), float(theta[1])
    tanh_theta2 = np.tanh(theta2)
    sech2_theta2 = 1.0 - tanh_theta2 * tanh_theta2

    task1_residual = 0.5 * theta1 + tanh_theta2
    task2_residual = 0.5 * theta1 - tanh_theta2 + 2.0

    grad1 = np.zeros(2, dtype=np.float64)
    grad2 = np.zeros(2, dtype=np.float64)

    if abs(task1_residual) > LOSS_CLIP:
        grad1 = (
            20.0
            * np.sign(task1_residual)
            / abs(task1_residual)
            * np.array([0.5, sech2_theta2], dtype=np.float64)
        )

    if abs(task2_residual) > LOSS_CLIP:
        grad2 = (
            25.0
            * np.sign(task2_residual)
            / abs(task2_residual)
            * np.array([0.5, -sech2_theta2], dtype=np.float64)
        )

    return np.stack([grad1, grad2], axis=0)


def pcgrad_two_task(task_grads: np.ndarray) -> np.ndarray:
    """Symmetric two-task PCGrad projection used for the optional comparison."""
    grad1, grad2 = task_grads
    dot = float(np.dot(grad1, grad2))
    if dot >= 0.0:
        return task_grads.copy()

    grad1_sq = float(np.dot(grad1, grad1)) + 1e-12
    grad2_sq = float(np.dot(grad2, grad2)) + 1e-12
    projected1 = grad1 - dot / grad2_sq * grad2
    projected2 = grad2 - dot / grad1_sq * grad1
    return np.stack([projected1, projected2], axis=0)


def run_adam(
    *,
    start: np.ndarray,
    steps: int,
    learning_rate: float,
    keep_points: int,
    use_pcgrad: bool = False,
) -> OptimRun:
    """Run Adam on the summed objective and keep a downsampled trajectory."""
    beta1 = 0.9
    beta2 = 0.999
    theta = start.astype(np.float64).copy()
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)

    sample_every = max(1, steps // keep_points)
    path = [theta.copy()]

    for step in range(1, steps + 1):
        task_grads = task_gradients(theta)
        if use_pcgrad:
            task_grads = pcgrad_two_task(task_grads)
        grad = task_grads.sum(axis=0)

        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * grad * grad
        first_hat = first_moment / (1.0 - beta1**step)
        second_hat = second_moment / (1.0 - beta2**step)

        theta = theta - learning_rate * first_hat / (np.sqrt(second_hat) + ADAM_EPS)

        if step % sample_every == 0 or step == steps:
            path.append(theta.copy())

    return OptimRun(path=np.asarray(path), final_theta=theta, final_losses=task_losses(theta))


def task_conflict_score(theta: np.ndarray) -> tuple[float, float, float]:
    """Return a score that favors strong opposing and imbalanced gradients."""
    grad1, grad2 = task_gradients(theta)
    norm1 = float(np.linalg.norm(grad1))
    norm2 = float(np.linalg.norm(grad2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0, 0.0, 1.0

    cosine = float(np.dot(grad1, grad2) / (norm1 * norm2))
    imbalance = max(norm1, norm2) / (min(norm1, norm2) + 1e-12)
    total_loss = float(task_losses(theta).sum())
    score = max(0.0, -cosine) * np.log10(imbalance + 1.0) * max(1.0, -total_loss)
    return score, imbalance, cosine


def choose_annotation_point(path: np.ndarray) -> tuple[int, float, float]:
    """Pick a trajectory point where the gradient conflict is visually clear."""
    warmup = max(1, len(path) // 20)
    best_idx = warmup
    best_score = -np.inf
    best_imbalance = 0.0
    best_cosine = 1.0

    for idx in range(warmup, len(path)):
        score, imbalance, cosine = task_conflict_score(path[idx])
        if score > best_score:
            best_idx = idx
            best_score = score
            best_imbalance = imbalance
            best_cosine = cosine

    return best_idx, best_imbalance, best_cosine


def add_trajectory(ax: plt.Axes, path: np.ndarray, title: str) -> None:
    if len(path) < 2:
        return

    segments = np.stack([path[:-1], path[1:]], axis=1)
    path_cmap = LinearSegmentedColormap.from_list(
        "adam_path",
        ["#111111", "#8b6100", "#ffd84d"],
    )
    line_collection = LineCollection(
        segments,
        cmap=path_cmap,
        norm=Normalize(0.0, 1.0),
        linewidths=2.8,
        zorder=4,
        capstyle="round",
    )
    line_collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(line_collection)
    ax.scatter(path[0, 0], path[0, 1], s=26, c="#111111", edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], s=32, c="#ffd84d", edgecolors="#111111", linewidths=0.6, zorder=5)
    ax.set_title(title)


def add_task_descent_arrows(ax: plt.Axes, theta: np.ndarray) -> tuple[float, float]:
    """Draw task descent directions at theta with readable visual lengths."""
    grad1, grad2 = task_gradients(theta)
    descents = [
        (-grad1, "#d62728", "Task 1 descent", 1.75),
        (-grad2, "#0050ff", "Task 2 descent", 0.95),
    ]

    ax.scatter(
        theta[0],
        theta[1],
        s=30,
        c="#ffd84d",
        edgecolors="#111111",
        linewidths=0.6,
        zorder=7,
    )

    for vec, color, label, length in descents:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            continue
        arrow = vec / norm * length
        ax.quiver(
            theta[0],
            theta[1],
            arrow[0],
            arrow[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.007,
            headwidth=4.2,
            headlength=5.0,
            headaxislength=4.2,
            color=color,
            zorder=6,
            label=label,
        )

    _, imbalance, cosine = task_conflict_score(theta)
    return imbalance, cosine


def configure_contour_axis(ax: plt.Axes, *, ylabel: bool = False) -> None:
    ax.set_xlim(-4.0, 2.0)
    ax.set_ylim(-4.0, 7.0)
    ax.set_xlabel(r"$\theta_1$")
    if ylabel:
        ax.set_ylabel(r"$\theta_2$", rotation=0, labelpad=18)
    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.set_aspect("auto")


def plot_contours(ax: plt.Axes, theta1: np.ndarray, theta2: np.ndarray, values: np.ndarray) -> None:
    levels = np.linspace(-150.0, 55.0, 46)
    ax.contourf(theta1, theta2, values, levels=levels, cmap="Oranges_r", extend="both")
    ax.contour(theta1, theta2, values, levels=levels[::4], colors="white", linewidths=0.25, alpha=0.35)


def make_figure(
    adam_run: OptimRun,
    pcgrad_run: OptimRun | None,
    *,
    output: Path,
    grid_size: int,
    dpi: int,
) -> tuple[Path, dict[str, float]]:
    theta1 = np.linspace(-4.0, 2.0, grid_size)
    theta2 = np.linspace(-4.0, 7.0, grid_size)
    x_grid, y_grid = np.meshgrid(theta1, theta2)
    loss1, loss2 = task_losses_xy(x_grid, y_grid)
    total_loss = loss1 + loss2

    wide_theta1 = np.linspace(-6.0, 6.0, max(160, grid_size // 3))
    wide_theta2 = np.linspace(-6.0, 6.0, max(160, grid_size // 3))
    wide_x, wide_y = np.meshgrid(wide_theta1, wide_theta2)
    wide_loss1, wide_loss2 = task_losses_xy(wide_x, wide_y)
    wide_total = np.clip(wide_loss1 + wide_loss2, -155.0, 60.0)

    ncols = 5 if pcgrad_run is not None else 4
    fig = plt.figure(figsize=(3.55 * ncols, 4.25), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.04, hspace=0.02, wspace=0.02)

    surface_ax = fig.add_subplot(1, ncols, 1, projection="3d")
    surface_ax.plot_surface(
        wide_x,
        wide_y,
        wide_total,
        cmap="YlOrBr_r",
        edgecolor=(0.15, 0.15, 0.15, 0.18),
        linewidth=0.25,
        antialiased=True,
        rstride=3,
        cstride=3,
    )
    surface_ax.set_title("Multi-Task Objective", pad=12)
    surface_ax.set_xlabel(r"$\theta_1$", labelpad=-2)
    surface_ax.set_ylabel(r"$\theta_2$", labelpad=-2)
    surface_ax.set_zlabel("")
    surface_ax.view_init(elev=31, azim=-58)
    surface_ax.set_xlim(-6.0, 6.0)
    surface_ax.set_ylim(-6.0, 6.0)
    surface_ax.set_zlim(-155.0, 60.0)
    surface_ax.tick_params(axis="both", labelsize=7, pad=-2)
    surface_ax.zaxis.set_tick_params(labelsize=7, pad=-1)

    task1_ax = fig.add_subplot(1, ncols, 2)
    plot_contours(task1_ax, x_grid, y_grid, loss1)
    configure_contour_axis(task1_ax, ylabel=True)
    task1_ax.set_title("Task 1 Objective")

    task2_ax = fig.add_subplot(1, ncols, 3)
    plot_contours(task2_ax, x_grid, y_grid, loss2)
    configure_contour_axis(task2_ax)
    task2_ax.set_title("Task 2 Objective")

    adam_ax = fig.add_subplot(1, ncols, 4)
    theta1_adam = np.linspace(-4.0, 5.0, grid_size)
    theta2_adam = np.linspace(-4.0, 7.0, grid_size)
    adam_x, adam_y = np.meshgrid(theta1_adam, theta2_adam)
    adam_loss1, adam_loss2 = task_losses_xy(adam_x, adam_y)
    plot_contours(adam_ax, adam_x, adam_y, adam_loss1 + adam_loss2)
    add_trajectory(adam_ax, adam_run.path, "Gradient Conflict")
    annotation_idx, annotation_imbalance, annotation_cosine = choose_annotation_point(adam_run.path)
    arrow_imbalance, arrow_cosine = add_task_descent_arrows(adam_ax, adam_run.path[annotation_idx])
    adam_ax.set_xlim(-4.0, 5.0)
    adam_ax.set_ylim(-4.0, 7.0)
    adam_ax.set_xlabel(r"$\theta_1$")
    adam_ax.tick_params(axis="both", labelsize=8, length=3)

    if pcgrad_run is not None:
        pcgrad_ax = fig.add_subplot(1, ncols, 5)
        plot_contours(pcgrad_ax, adam_x, adam_y, adam_loss1 + adam_loss2)
        add_trajectory(pcgrad_ax, pcgrad_run.path, "Adam + PCGrad")
        pcgrad_ax.set_xlim(-4.0, 5.0)
        pcgrad_ax.set_ylim(-4.0, 7.0)
        pcgrad_ax.set_xlabel(r"$\theta_1$")
        pcgrad_ax.tick_params(axis="both", labelsize=8, length=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "annotation_step_index": float(annotation_idx),
        "annotation_theta1": float(adam_run.path[annotation_idx, 0]),
        "annotation_theta2": float(adam_run.path[annotation_idx, 1]),
        "annotation_imbalance": float(arrow_imbalance),
        "annotation_cosine": float(arrow_cosine),
        "selection_imbalance": float(annotation_imbalance),
        "selection_cosine": float(annotation_cosine),
    }
    return output, metadata


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "pcgrad_toy_landscape.png",
        help="Where to save the generated figure.",
    )
    parser.add_argument("--steps", type=int, default=500_000, help="Number of Adam updates.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--grid-size", type=int, default=480, help="Grid resolution for contour plots.")
    parser.add_argument("--keep-points", type=int, default=2_000, help="Number of trajectory points to keep.")
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI.")
    parser.add_argument(
        "--with-pcgrad",
        action="store_true",
        help="Add a fifth panel comparing Adam with a deterministic two-task PCGrad update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = np.array([0.5, -3.0], dtype=np.float64)

    adam_result = run_adam(
        start=start,
        steps=args.steps,
        learning_rate=args.lr,
        keep_points=args.keep_points,
        use_pcgrad=False,
    )
    pcgrad_result = None
    if args.with_pcgrad:
        pcgrad_result = run_adam(
            start=start,
            steps=args.steps,
            learning_rate=args.lr,
            keep_points=args.keep_points,
            use_pcgrad=True,
        )

    output, metadata = make_figure(
        adam_result,
        pcgrad_result,
        output=args.output,
        grid_size=args.grid_size,
        dpi=args.dpi,
    )

    print(f"Saved figure: {output}")
    print(
        "Adam final theta: "
        f"[{adam_result.final_theta[0]:.6f}, {adam_result.final_theta[1]:.6f}]"
    )
    print(
        "Adam final losses: "
        f"L1={adam_result.final_losses[0]:.6f}, "
        f"L2={adam_result.final_losses[1]:.6f}, "
        f"sum={adam_result.final_losses.sum():.6f}"
    )
    print(
        "Annotated conflict point: "
        f"theta=[{metadata['annotation_theta1']:.6f}, {metadata['annotation_theta2']:.6f}], "
        f"cos(g1,g2)={metadata['annotation_cosine']:.6f}, "
        f"|g|max/|g|min={metadata['annotation_imbalance']:.2f}"
    )


if __name__ == "__main__":
    main()
