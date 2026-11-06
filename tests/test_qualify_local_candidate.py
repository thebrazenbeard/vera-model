"""No weights, network, subprocess inference, or real provider effects in tests."""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "runtime" / "qualify_local_candidate.py"
spec = importlib.util.spec_from_file_location("qualify_local_candidate", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateProbeTests(unittest.TestCase):
    def test_false_and_missing_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-a-model.gguf"
            self.assertEqual(module.verify_artifact(path, 0, "x")["reason"], "MISSING_MODEL")
            path.write_bytes(b"a")
            self.assertEqual(module.verify_artifact(path, 2, "x")["reason"], "SIZE_MISMATCH")
            self.assertEqual(module.verify_artifact(path, 1, "0" * 64)["reason"], "HASH_MISMATCH")
            self.assertEqual(module.verify_artifact(path, 1, hashlib.sha256(b"a").hexdigest())["status"], "VERIFIED_LOCAL_BYTES")

    def test_reject_non_int_bounds_and_unregistered_candidates(self):
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", True)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, gpu_layers=100)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, predict=1000)
        with self.assertRaises(ValueError):
            module.build_command("llama-cli", "fixture", 0, reasoning="trust-me")
        with self.assertRaises(ValueError):
            module.probe("arbitrary", Path("missing"))

    def test_missing_file_never_launches_runner(self):
        with patch.object(module, "_bounded_run") as run:
            result = module.probe("gptoss-pruned4-q4", Path("missing.gguf"), execute=True)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["artifact"]["reason"], "MISSING_MODEL")
        run.assert_not_called()

    def test_pinned_bytes_do_not_auto_run(self):
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES", "sha256": "x"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module, "_runner_snapshot", return_value={"path": "llama-cli", "sha256": "runnerhash"}), \
             patch.object(module, "_bounded_run") as run:
            result = module.probe("gptoss-pruned4-q4", Path("any"), execute=False)
        self.assertEqual(result["status"], "READY_FOR_OPT_IN_LOCAL_PROBE")
        self.assertFalse(result["provider_admitted"])
        self.assertFalse(result["behaviorally_qualified"])
        self.assertFalse(result["memory_consumed"])
        run.assert_not_called()

    def test_success_does_not_promote_identity_or_authority(self):
        done = {"returncode": 0, "stdout": "hello", "stderr": "", "stdout_truncated": False, "stderr_truncated": False}
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES", "sha256": "x"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module, "_runner_snapshot", return_value={"path": "llama-cli", "sha256": "runnerhash"}), \
             patch.object(module, "_bounded_run", return_value=done) as run:
            result = module.probe("qwen35-4b-q4-control", Path("any"), execute=True)
        self.assertEqual(result["status"], "RUNNER_EXIT_ZERO_UNQUALIFIED")
        self.assertFalse(result["provider_admitted"])
        self.assertFalse(result["effect_authority_granted"])
        self.assertFalse(result["behaviorally_qualified"])
        self.assertIn("--reasoning", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertFalse(result["model_bytes_loaded_verified"])
        self.assertEqual(result["model_execution_evidence"], "UNVERIFIED_ON_EXEC")

    def test_timeout_stays_unqualified(self):
        import subprocess
        with patch.object(module, "verify_artifact", return_value={"status": "VERIFIED_LOCAL_BYTES"}), \
             patch.object(module.shutil, "which", return_value="llama-cli"), \
             patch.object(module, "_runner_snapshot", return_value={"path": "llama-cli", "sha256": "runnerhash"}), \
             patch.object(module, "_bounded_run", side_effect=subprocess.TimeoutExpired("llama-cli", 120)):
            result = module.probe("gptoss-pruned4-q4", Path("any"), execute=True)
        self.assertEqual(result["status"], "INFERENCE_TIMEOUT")
        self.assertFalse(result["behaviorally_qualified"])

    def test_model_path_swapped_during_runner_is_evidence_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"ABCD")
            expected = hashlib.sha256(b"ABCD").hexdigest()
            good = {"bytes": 4, "sha256": expected, "lineage": "synthetic", "provenance": "TEST"}
            def mutate(_args, *, timeout):
                path.write_bytes(b"WXYZ")
                return {"returncode": 0, "stdout": "hello", "stderr": "",
                        "stdout_truncated": False, "stderr_truncated": False}
            with patch.dict(module.CANDIDATES, {"test-only": good}), \
                 patch.object(module.shutil, "which", return_value="test-runner"), \
                 patch.object(module, "_runner_snapshot", return_value={"path": "test-runner", "sha256": "fixed"}), \
                 patch.object(module, "_bounded_run", side_effect=mutate):
                result = module.probe("test-only", path, execute=True)
            self.assertEqual(result["status"], "HOLD")
            self.assertEqual(result["reason"], "MODEL_CHANGED_DURING_PROBE")
            self.assertFalse(result["model_bytes_loaded_verified"])
            self.assertEqual(result["artifact"]["sha256"], expected)
            self.assertEqual(result["postflight_artifact"]["reason"], "HASH_MISMATCH")

    def test_runner_path_replaced_during_probe_is_evidence_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            runner = Path(tmp) / "runner.exe"
            model.write_bytes(b"ABCD")
            runner.write_bytes(b"first-runner")
            expected = hashlib.sha256(b"ABCD").hexdigest()
            good = {"bytes": 4, "sha256": expected, "lineage": "synthetic", "provenance": "TEST"}
            def mutate(_args, *, timeout):
                runner.write_bytes(b"other-runner")
                return {"returncode": 0, "stdout": "hello", "stderr": "",
                        "stdout_truncated": False, "stderr_truncated": False}
            with patch.dict(module.CANDIDATES, {"test-only": good}), \
                 patch.object(module.shutil, "which", return_value=str(runner)), \
                 patch.object(module, "_bounded_run", side_effect=mutate):
                result = module.probe("test-only", model, execute=True)
            self.assertEqual(result["status"], "HOLD")
            self.assertEqual(result["reason"], "RUNNER_CHANGED_DURING_PROBE")
            self.assertFalse(result["runner_bytes_executed_verified"])

    def test_artifact_permission_error_is_typed_hold(self):
        with patch.object(Path, "is_file", side_effect=OSError("permission denied")):
            result = module.verify_artifact(Path("restricted.gguf"), 4, "0" * 64)
        self.assertEqual(result, {"status": "HOLD", "reason": "MODEL_READ_ERROR"})

    def test_capture_does_not_keep_unbounded_process_output(self):
        import sys
        command = [sys.executable, "-c",
                   "import sys; sys.stdout.write('x'*120000); sys.stderr.write('z'*150000)"]
        done = module._bounded_run(command, timeout=15)
        self.assertEqual(done["returncode"], 0)
        self.assertLessEqual(len(done["stdout"].encode("utf-8")), 8000)
        self.assertLessEqual(len(done["stderr"].encode("utf-8")), 2000)
        self.assertTrue(done["stdout_truncated"])
        self.assertTrue(done["stderr_truncated"])


if __name__ == "__main__":
    unittest.main()
