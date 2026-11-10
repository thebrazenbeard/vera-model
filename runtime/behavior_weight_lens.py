"""Compile bounded Boolean behavior macros to explicit affine weights.

A tiny transparent classifier/lens, NOT a language model, GGUF, tokenizer,
semantic recognizer, or shortcut for changing a pretrained model\u0027s opaque weights.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

FEATURES = (
    # Host-adjudicated evidence flags, NEVER guessed from arbitrary chat text.
    "third_person_self_reference_confirmed",
    "mutable_claim_unverified",
    "user_correction_admitted",
    "correction_conflicts_with_authority",
    "provenance_unresolved",
)
RULES = {
    # Outputs are advisory flags only; none are permission to take an action.
    "REVIEW_SELF_REFERENCE": {"on": ("third_person_self_reference_confirmed",), "off": ()},
    "CHECK_MUTABLE_CLAIM": {"on": ("mutable_claim_unverified",), "off": ()},
    "REVIEW_USER_CORRECTION": {"on": ("user_correction_admitted",), "off": ()},
    "ESCALATE_AUTHORITY_CONFLICT": {
        "on": ("user_correction_admitted", "correction_conflicts_with_authority"), "off": ()
    },
    "SEARCH_PROVENANCE": {"on": ("provenance_unresolved",), "off": ()},
}


def validate_features(features: tuple[int, ...]) -> None:
    if len(features) != len(FEATURES):
        raise ValueError("feature count mismatch")
    if any(type(x) is not int or x not in (0, 1) for x in features):
        raise TypeError("binary features must be exact integers 0 or 1")


def compile_macros(rules: dict = RULES) -> dict:
    """Compile Boolean conjunctions: +1 for on, -1 for off, bias=.5-count(on).

    These are exact algebraic weights in a specified *five-feature classifier*;
    they are not language-model tensors or a discovered map from prose to LLM weights.
    """
    if type(rules) is not dict or not (1 <= len(rules) <= 16):
        raise ValueError("rules must contain 1..16 named macros")
    rows, biases, names = [], [], []
    for name, spec in sorted(rules.items()):
        if (type(name) is not str or len(name) > 64 or not name.isidentifier()
                or not name.startswith(("REVIEW_", "CHECK_", "ESCALATE_", "SEARCH_"))):
            raise ValueError("macros must have an advisory-only name")
        if type(spec) is not dict or set(spec) != {"on", "off"}:
            raise ValueError("macro must declare exactly on and off")
        if type(spec["on"]) not in (list, tuple) or type(spec["off"]) not in (list, tuple):
            raise TypeError("on/off feature groups must be lists or tuples")
        on, off = tuple(spec["on"]), tuple(spec["off"])
        if not on and not off:
            raise ValueError("empty macro unsupported")
        if len(set(on + off)) != len(on + off):
            raise ValueError("duplicated or contradictory feature")
        if any(type(f) is not str or f not in FEATURES for f in on + off):
            raise ValueError("unknown feature")
        row = [int(f in on) - int(f in off) for f in FEATURES]
        rows.append(row)
        biases.append(0.5 - len(on))
        names.append(name)
    return {
        "schema": "VERA_BOOLEAN_AFFINE_MACRO_V1",
        "features": FEATURES, "macros": names, "weights": rows, "biases": biases,
        "rules": {name: {"on": list(rules[name]["on"]), "off": list(rules[name]["off"])}
                  for name in sorted(rules)},
    }


def predict(features: tuple[int, ...], compiled: dict) -> dict[str, bool]:
    validate_features(features)
    if type(compiled) is not dict or set(compiled) != {
        "schema", "features", "macros", "weights", "biases", "rules"
    } or compiled["schema"] != "VERA_BOOLEAN_AFFINE_MACRO_V1":
        raise ValueError("malformed or unsupported compiled network")
    macros, weights, biases = compiled["macros"], compiled["weights"], compiled["biases"]
    if (tuple(compiled["features"]) != FEATURES or type(macros) is not list
            or type(weights) is not list or type(biases) is not list
            or not len(macros) == len(weights) == len(biases)):
        raise ValueError("compiled network has incompatible dimensions")
    if any(type(row) is not list or len(row) != len(FEATURES)
           or any(type(w) is not int or w not in (-1, 0, 1) for w in row)
           for row in weights):
        raise ValueError("compiled network weights invalid")
    if any(type(bias) not in (float, int) for bias in biases):
        raise ValueError("compiled network biases invalid")
    # A compiled blob is untrusted input. Recompile its embedded source rules and
    # compare every normalized field before returning even advisory macro names.
    # Nonfinite floats reject via allow_nan=False; arbitrary effect names are
    # rejected by compile_macros before any prediction is produced.
    try:
        expected = compile_macros(compiled["rules"])
        canonical = lambda obj: json.dumps(
            obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        if canonical(expected) != canonical(compiled):
            raise ValueError("compiled network diverges from its rules")
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        raise ValueError("forged, nonfinite, or malformed compiled network") from exc
    return {
        macro: sum(w * v for w, v in zip(row, features)) + bias > 0
        for macro, row, bias in zip(macros, weights, biases)
    }


def oracle(features: tuple[int, ...], rules: dict = RULES) -> dict[str, bool]:
    """Independent source-rule interpreter, not the weights implementation."""
    validate_features(features)
    view = dict(zip(FEATURES, features))
    return {
        name: all(view[f] == 1 for f in spec["on"])
        and all(view[f] == 0 for f in spec["off"])
        for name, spec in rules.items()
    }


def verify_all(compiled: dict, rules: dict = RULES) -> dict:
    tests = 0
    for features in itertools.product((0, 1), repeat=len(FEATURES)):
        expected = oracle(features, rules)
        actual = predict(features, compiled)
        if actual != expected:
            raise AssertionError(f"mismatch: {features}: {actual} != {expected}")
        tests += 1
    return {"cases": tests, "passed": tests, "exhaustive": True}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate rules key: {key}")
        result[key] = value
    return result


def load_rules(path: Path) -> dict:
    """Read a bounded, caller-selected public macro specification, not private memory."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 8192:
        raise ValueError("rules JSON missing, symlinked, or too large")
    rules = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    compile_macros(rules)
    return rules


def write_receipt(output: Path, rules: dict = RULES) -> dict:
    """Write two new immutable artifacts. No pretrained model, weights or tools loaded."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError("output must be an existing real directory")
    compiled = compile_macros(rules)
    verdict = verify_all(compiled, rules)
    payload = json.dumps(compiled, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    paths = (output / "compiled_macro_weights.json", output / "receipt.json")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("refusing to overwrite existing weights or receipt")
    receipt = {"schema": "SYNTHETIC_MACRO_WEIGHT_PROOF_V1", "sha256": digest, **verdict,
               "gradient_steps": 0, "pretrained_weights": False, "gguf": False,
               "generative_language_model": False, "semantics_recognition_solved": False,
               "language_model_weight_mutation": False}
    with paths[0].open("xb") as stream:
        stream.write(payload)
    with paths[1].open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rules-json", default=None)
    args = parser.parse_args()
    rules = RULES if args.rules_json is None else load_rules(Path(args.rules_json))
    receipt = write_receipt(Path(args.output_dir), rules)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
