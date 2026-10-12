"""Manual local Vera-model prompt; advisory text only, never tool actions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_model import LocalModelHold, advise, read_key, probe_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the local Vera GLM24B model")
    parser.add_argument("prompt", help="A bounded question or advisory task")
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    key_file = Path.home() / "AppData/Local/VeraModel/glm24b/api-key.txt"
    try:
        key = read_key(key_file)
        probe_model(key)
        result = advise(args.prompt, key, max_tokens=args.max_tokens)
    except LocalModelHold as exc:
        print("HOLD: " + str(exc), file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
