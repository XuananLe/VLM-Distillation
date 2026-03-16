import argparse
import os
import shlex
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the vision last-layer audit on Modal."
    )
    parser.add_argument(
        "audit_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tests/test_vision_layer_audit.py",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    audit_args = list(args.audit_args)
    if audit_args[:1] == ["--"]:
        audit_args = audit_args[1:]

    forwarded = " ".join(shlex.quote(arg) for arg in audit_args)
    audit_cmd = "python tests/test_vision_layer_audit.py"
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
    return subprocess.call(command, cwd=ROOT)


class ModalVisionLayerAuditTest(unittest.TestCase):
    def test_modal_vision_layer_audit(self):
        if os.environ.get("RUN_MODAL_VISION_LAYER_AUDIT") != "1":
            self.skipTest(
                "Set RUN_MODAL_VISION_LAYER_AUDIT=1 to run the Modal vision-layer audit test."
            )

        audit_args = shlex.split(os.environ.get("MODAL_VISION_LAYER_AUDIT_ARGS", ""))
        self.assertEqual(main(audit_args), 0)


if __name__ == "__main__":
    raise SystemExit(main())
