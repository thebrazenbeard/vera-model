"""Synthetic, bounded tests for explicit Boolean-to-affine weight compilation."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from behavior_weight_lens import (
    FEATURES, RULES, compile_macros, load_rules, predict, verify_all,
    write_receipt,
)


def evidence(**flags: int) -> tuple[int, ...]:
    return tuple(flags.get(name, 0) for name in FEATURES)


class BehaviorWeightLensTests(unittest.TestCase):
    def test_all_32_truth_table_cases_against_independent_rule_oracle(self):
        result = verify_all(compile_macros())
        self.assertEqual(result, {"cases": 32, "passed": 32, "exhaustive": True})

    def test_no_automatic_third_person_grammar_inference(self):
        # Innocent phrase "I saw her" is NOT evidence of third-person self-reference.
        result = predict(evidence(), compile_macros())
        self.assertFalse(result["REVIEW_SELF_REFERENCE"])
        flagged = predict(evidence(third_person_self_reference_confirmed=1),
                          compile_macros())
        self.assertTrue(flagged["REVIEW_SELF_REFERENCE"])
        self.assertFalse(any(name.startswith(("APPLY_", "EXECUTE_", "WRITE_"))
                             for name in flagged))

    def test_user_correction_conflict_triggers_review_and_escalation_not_action(self):
        result = predict(evidence(user_correction_admitted=1,
                                  correction_conflicts_with_authority=1),
                         compile_macros())
        self.assertTrue(result["REVIEW_USER_CORRECTION"])
        self.assertTrue(result["ESCALATE_AUTHORITY_CONFLICT"])
        self.assertFalse(result["REVIEW_SELF_REFERENCE"])
        self.assertEqual(set(result), set(RULES))

    def test_unverified_mutable_claim_requests_check(self):
        result = predict(evidence(mutable_claim_unverified=1), compile_macros())
        self.assertTrue(result["CHECK_MUTABLE_CLAIM"])
        self.assertFalse(result["SEARCH_PROVENANCE"])

    def test_unresolved_provenance_flag_requests_search(self):
        result = predict(evidence(provenance_unresolved=1), compile_macros())
        self.assertTrue(result["SEARCH_PROVENANCE"])

    def test_nonbinary_or_wrong_width_rejected(self):
        compiled = compile_macros()
        for bits in [(True,) + (0,) * 4, (2,) + (0,) * 4, ("1",) + (0,) * 4]:
            with self.subTest(bits=bits), self.assertRaises(TypeError):
                predict(bits, compiled)
        with self.assertRaises(ValueError):
            predict((0, 1), compiled)

    def test_malformed_compiled_weights_rejected(self):
        compiled = compile_macros()
        compiled["weights"][0] = [1]
        with self.assertRaisesRegex(ValueError, "weights invalid"):
            predict(evidence(), compiled)

    def test_caller_mutated_effect_name_rejected_at_prediction_boundary(self):
        compiled = compile_macros()
        compiled["macros"][0] = "EXECUTE_DEPLOY"
        with self.assertRaises(ValueError):
            predict(evidence(), compiled)

    def test_nonfinite_bias_rejected_at_prediction_boundary(self):
        compiled = compile_macros()
        compiled["biases"][0] = float("inf")
        with self.assertRaises(ValueError):
            predict(evidence(), compiled)

    def test_mutated_legal_shape_coefficient_rejected_against_embedded_rules(self):
        compiled = compile_macros()
        compiled["weights"][0][0] = 1 - compiled["weights"][0][0]
        with self.assertRaises(ValueError):
            predict(evidence(), compiled)

    def test_serialized_compiled_blob_round_trip_remains_accepted(self):
        compiled = json.loads(json.dumps(compile_macros()))
        self.assertEqual(verify_all(compiled)["passed"], 32)

    def test_mutated_rules_without_recompiling_weights_rejected(self):
        compiled = compile_macros()
        compiled["rules"]["CHECK_MUTABLE_CLAIM"]["off"] = ["provenance_unresolved"]
        with self.assertRaises(ValueError):
            predict(evidence(), compiled)

    def test_macros_are_advisory_only_even_when_caller_authors_rules(self):
        bad = {"APPLY_CORRECTION": {"on": ["user_correction_admitted"], "off": []}}
        with self.assertRaisesRegex(ValueError, "advisory"):
            compile_macros(bad)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            compile_macros({"REVIEW_BAD": {
                "on": ["provenance_unresolved"], "off": ["provenance_unresolved"]}})
        with self.assertRaisesRegex(ValueError, "unknown"):
            compile_macros({"SEARCH_UNKNOWN": {"on": ["secret_command"], "off": []}})
        with self.assertRaisesRegex(ValueError, "empty"):
            compile_macros({"REVIEW_EMPTY": {"on": [], "off": []}})

    def test_custom_rules_are_exhaustively_verified(self):
        rules = {"REVIEW_BOTH": {
            "on": ["user_correction_admitted"],
            "off": ["correction_conflicts_with_authority"]}}
        compiled = compile_macros(rules)
        self.assertEqual(verify_all(compiled, rules)["passed"], 32)
        self.assertFalse(predict(evidence(
            user_correction_admitted=1, correction_conflicts_with_authority=1), compiled)["REVIEW_BOTH"])

    def test_write_receipt_new_directory_only_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            receipt = write_receipt(p)
            self.assertFalse(receipt["language_model_weight_mutation"])
            self.assertFalse(receipt["gguf"])
            self.assertEqual(receipt["cases"], 32)
            blob = (p / "compiled_macro_weights.json").read_bytes()
            self.assertEqual(receipt["sha256"], hashlib.sha256(blob).hexdigest())
            with self.assertRaises(FileExistsError):
                write_receipt(p)

    def test_rules_json_rejects_duplicate_and_new_custom_works(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rules.json"
            path.write_text('{"REVIEW_BAD":{"on":[],"off":[]},"REVIEW_BAD":{"on":[],"off":[]}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_rules(path)
            rules = {"CHECK_VALID": {"on": ["provenance_unresolved"], "off": []}}
            path.write_text(json.dumps(rules), encoding="utf-8")
            self.assertEqual(load_rules(path), rules)


if __name__ == "__main__":
    unittest.main()
