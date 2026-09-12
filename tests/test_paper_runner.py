"""Runner protocol tests; no model dependencies or GPU needed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("paper_runner", Path(__file__).resolve().parents[1] / "scripts/run_paper_evaluation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PaperRunnerTests(unittest.TestCase):
    def args(self, directory):
        return runner.parse_args(["--dataset-path", directory, "--lenscraft-checkpoint", directory + "/best=099.ckpt",
            "--lenscraft-config", directory + "/config.yaml", "--clatr-checkpoint", directory + "/clatr.ckpt", "--output-dir", directory])

    def test_full_stage_protocol(self):
        args = self.args("/tmp/paper-runner")
        stages = runner.build_stages(args)
        self.assertEqual(len(stages), 13)
        self.assertEqual(sum(s["kind"] == "evaluation" for s in stages), 8)
        for stage in stages[:8]:
            cmd = stage["command"]
            self.assertIn("+n_boot=500", cmd)
            self.assertIn("data.batch_size=128", cmd)
            self.assertIn("test_fraction=1.0", cmd)
            self.assertIn("+data.dataset.config.allowed_movement_types=[]", cmd)
            self.assertNotIn("+limit_test_batches=1", cmd)
            self.assertTrue(any(arg.startswith("test_movement_types=[") for arg in cmd))
            self.assertIn(runner.override("ref_model.inference.checkpoint_path", args.lenscraft_checkpoint), cmd)
        for stage in stages[8:12]:
            self.assertIn("data.batch_size=16", stage["command"])
        self.assertEqual(len(runner.expected_modes("lenscraft")), 13)
        self.assertEqual(len(runner.expected_modes("ccdm")), 3)

    def test_invalid_metric_and_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            stage = runner.build_stages(args)[2]
            results = Path(directory) / "results"
            results.mkdir()
            path = results / "metrics_ccdm_static.json"
            payload = {"model_type": "ccdm", "set": "static", "clatr_backend": "native", "test_movement_types": ["static"], "test_fraction": 1.0,
                "metrics": {mode: {f"{mode}/{name}": 1.0 for name in runner.LEARNED_METRICS} for mode in runner.expected_modes("ccdm")},
                "bootstrap_std": {mode: {f"{mode}/{name}": [1.0, 0.1] for name in runner.LEARNED_METRICS} for mode in runner.expected_modes("ccdm")},
                "evaluation_provenance": {
                    "semantic_evaluator": {"fingerprint": "sem", "checkpoint": {"path": str(args.semantic_evaluator_checkpoint)}},
                    "clatr_evaluator": {"fingerprint": "clatr", "checkpoint": {"path": str(args.clatr_checkpoint)}}}}
            path.write_text(json.dumps(payload))
            self.assertEqual(len(runner.validate_output(args, stage, {})), 1)
            with self.assertRaisesRegex(ValueError, "Different"):
                runner.validate_output(args, stage, {"semantic_evaluator": "different"})
            payload["metrics"]["prompt_generation"]["prompt_generation/fcd"] = float("nan")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                runner.validate_output(args, stage, {})

    def test_atomic_json_and_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            runner.write_json(path, {"status": "running"})
            before = runner.sha256(path)
            runner.write_json(path, {"status": "complete"})
            self.assertNotEqual(before, runner.sha256(path))
            self.assertEqual(json.loads(path.read_text())["status"], "complete")
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
