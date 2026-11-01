"""Adversarial response-shape contract for a cognition-only local route."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from local_model import LocalModelHold, advise


class AdvisoryResponseBoundaryTests(unittest.TestCase):
    def advise_payload(self, payload):
        with patch("local_model._request", return_value=payload):
            return advise("Review this source", "z" * 64)

    def test_plain_advisory_text_still_allowed(self):
        self.assertEqual(self.advise_payload({"choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "Review first."}}
        ]}), "Review first.")

    def test_legacy_function_call_cannot_pass_as_text(self):
        payload = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "Review first.",
            "function_call": {"name": "erase", "arguments": "{}"}
        }}]}
        with self.assertRaisesRegex(LocalModelHold, "tool calls"):
            self.advise_payload(payload)

    def test_tool_finish_reason_cannot_pass_as_text(self):
        payload = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "Review first.", "tool_calls": []
        }}]}
        with self.assertRaisesRegex(LocalModelHold, "tool calls"):
            self.advise_payload(payload)

    def test_non_object_choice_is_cleanly_held(self):
        with self.assertRaisesRegex(LocalModelHold, "response choices"):
            self.advise_payload({"choices": [42]})

    def test_malformed_tool_call_field_is_held(self):
        payload = {"choices": [{"message": {"content": "Review first.", "tool_calls": {}}}]}
        with self.assertRaisesRegex(LocalModelHold, "tool calls"):
            self.advise_payload(payload)


if __name__ == "__main__":
    unittest.main()
