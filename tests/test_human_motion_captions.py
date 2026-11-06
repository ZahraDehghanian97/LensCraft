"""Human motion captions/cohorts must preserve motion; no model dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


caption = load_module("human_caption", "src/data/simulation/caption.py")
runner = load_module("human_paper_runner", "scripts/run_paper_evaluation.py")
ablation = load_module("human_ablation_runner", "scripts/run_ablation_study.py")


class HumanMotionCaptionTests(unittest.TestCase):
    MOTIONS = {
        "walk": ("walks forward", "straight path"),
        "run": ("runs forward", "straight path"),
        "wander": ("walks", "curving path"),
        "stopStart": ("walks forward", "pauses", "resumes walking"),
        "sidestep": ("sideways", "same facing direction"),
        "backward": ("backward", "face forward"),
        "turnInPlace": (),
        "idle": ("subtly", "facing direction"),
    }

    def test_motion_survives_static_and_moving_camera_captions(self):
        for motion, descriptions in self.MOTIONS.items():
            for camera in ("static", "follow", "dollyIn", "panLeft", "craneUp", "tiltDown"):
                with self.subTest(motion=motion, camera=camera):
                    result = caption.extract_text_prompt({"movement": {"type": camera}}, motion)
                    self.assertNotIn("character remains stationary", result)
                    self.assertNotIn("Both the camera and character remain stationary", result)
                    for description in descriptions:
                        self.assertIn(description, result)
                    if motion == "turnInPlace":
                        self.assertIn("without translating" if camera == "static" else "turns in place", result)

    def test_unknown_motion_is_generic_motion_instead_of_stationary(self):
        for camera in ("static", "follow", "dollyIn"):
            with self.subTest(camera=camera):
                result = caption.extract_text_prompt({"movement": {"type": camera}}, "futureMotion")
                self.assertIn("the character moves", result)
                self.assertNotIn("character remains stationary", result)

    def test_static_and_absent_motion_keep_existing_wording(self):
        for motion in (None, "static"):
            self.assertEqual(
                caption.extract_text_prompt({"movement": {"type": "static"}}, motion),
                "Both the camera and character remain stationary throughout the sequence.",
            )
            self.assertEqual(
                caption.extract_text_prompt({"movement": {"type": "dollyIn"}}, motion),
                "The camera pushes in while the character remains stationary throughout the entire sequence.",
            )
        self.assertEqual(
            caption.extract_text_prompt({"movement": {"type": "follow"}}, "linear"),
            "The camera follows the subject as the character moves along a straight path.",
        )

    def test_caption_is_deterministic_and_preserves_instruction_data(self):
        instructions = {
            "movement": {"type": "dollyIn", "speed": "smoothStartStop"},
            "initial": {"shotSize": "fullShot", "cameraAngle": "eye", "subjectFraming": "left"},
            "final": {"shotSize": "closeUp", "cameraAngle": "high", "subjectFraming": "center"},
        }
        before = copy.deepcopy(instructions)
        first = caption.extract_text_prompt(instructions, "stopStart")
        for _ in range(10):
            self.assertEqual(caption.extract_text_prompt(instructions, "stopStart"), first)
        self.assertEqual(instructions, before)
        self.assertIn("transitioning from a full shot to a close-up", first)
        self.assertIn("repositioning from left side to center of the frame", first)

    def test_both_evaluation_runners_include_all_human_motion_in_dynamic_cohort(self):
        self.assertEqual(runner.COHORTS, ablation.COHORTS)
        self.assertEqual(runner.COHORTS["static"], ["static"])
        self.assertTrue(set(self.MOTIONS).issubset(runner.COHORTS["dynamic"]))
        self.assertNotIn("static", runner.COHORTS["dynamic"])
        args = runner.parse_args([
            "--dataset-path", "/tmp/human-eval/data", "--lenscraft-checkpoint", "/tmp/human-eval/model.ckpt",
            "--lenscraft-config", "/tmp/human-eval/config.yaml", "--clatr-checkpoint", "/tmp/human-eval/clatr.ckpt",
            "--output-dir", "/tmp/human-eval/output",
        ])
        for stage in runner.build_stages(args):
            if stage["kind"] == "evaluation" and stage["set"] == "dynamic":
                override = next(item for item in stage["command"] if item.startswith("test_movement_types="))
                for motion in self.MOTIONS:
                    self.assertIn(motion, override)

    def test_legacy_dynamic_output_cannot_be_mislabeled_with_new_cohort_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args([
                "--dataset-path", temporary, "--lenscraft-checkpoint", temporary + "/model.ckpt",
                "--lenscraft-config", temporary + "/config.yaml", "--clatr-checkpoint", temporary + "/clatr.ckpt",
                "--output-dir", temporary,
            ])
            stage = next(item for item in runner.build_stages(args)
                         if item["kind"] == "evaluation" and item["set"] == "dynamic" and item["model"] == "lenscraft")
            results = Path(temporary) / "results"
            results.mkdir()
            payload = {
                "model_type": "lens_craft", "set": "dynamic", "clatr_backend": "native",
                "metrics": dict.fromkeys(runner.expected_modes("lenscraft"), {}),
                "test_movement_types": [name for name in runner.COHORTS["dynamic"] if name not in self.MOTIONS],
            }
            (results / "metrics_lens_craft_dynamic.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "movement cohort"):
                runner.validate_output(args, stage, {})


if __name__ == "__main__":
    unittest.main()
