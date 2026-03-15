import argparse
from collections.abc import Sequence

from ..runtime.execution import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SKC score between two or more VLMs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Two or more model IDs. 2 models -> pairwise SKC. 3+ -> full matrix + pipeline.",
    )
    parser.add_argument("--dataset", default="textvqa", help="Dataset alias or HF dataset id")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--config", default=None, help="Optional HF dataset config")
    parser.add_argument("--n", type=int, default=200, help="Number of probe samples")
    parser.add_argument(
        "--n_teachers",
        type=int,
        default=2,
        help="Number of teachers to select (matrix mode only, >=3 models)",
    )
    parser.add_argument(
        "--redundancy_threshold",
        type=float,
        default=0.90,
        help="CKA above this is flagged as near-identical",
    )
    parser.add_argument(
        "--vision-layer-index",
        "--layer_index",
        dest="layer_index",
        type=int,
        default=-1,
        help="Vision encoder layer to extract (-1 = last)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))
