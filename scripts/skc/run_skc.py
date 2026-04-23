from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main() -> int:
    raise RuntimeError(
        "The legacy SKC entrypoint was removed. Use the CKA utilities in "
        "`src/components/cka.py` and the plotting entrypoint in "
        "`scripts/plot_cka/plot_vlm_layer_cka.py`."
    )


if __name__ == "__main__":
    raise SystemExit(main())
