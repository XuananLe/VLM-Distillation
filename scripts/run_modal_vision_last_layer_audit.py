import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the vision last-layer audit on Modal."
    )
    parser.add_argument(
        "audit_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/audit_vision_last_layer.py",
    )
    return parser


def main():
    args = build_parser().parse_args()
    audit_args = list(args.audit_args)
    if audit_args[:1] == ["--"]:
        audit_args = audit_args[1:]

    forwarded = " ".join(shlex.quote(arg) for arg in audit_args)
    audit_cmd = "python scripts/audit_vision_last_layer.py"
    if forwarded:
        audit_cmd = f"{audit_cmd} {forwarded}"

    remote_cmd = f"cd /root/VLM-Distillation && {audit_cmd}"
    command = [
        "modal",
        "run",
        "-d",
        "modal_app.py::exec_cmd",
        "--cmd",
        remote_cmd,
    ]

    print("Launching Modal audit:")
    print(" ".join(shlex.quote(part) for part in command))
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
