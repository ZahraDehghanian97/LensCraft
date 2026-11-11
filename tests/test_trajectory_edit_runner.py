"""The edit runner loads real tiny checkpoints and emits auditable paired scores."""

import importlib.util
import json
from pathlib import Path
import pickle

import msgpack
from omegaconf import OmegaConf
import pytest
import torch

from data.simulation.caption import enum_descriptions
from data.simulation.constants import NumericFeature, cinematography_struct, simulation_struct
from testing.metrics.semantic_vocabulary import _write_embedding_cache_metadata
from testing.metrics.semantic_evaluator import FrozenSemanticEvaluator
from models.camera_trajectory_model import LensCraft


SPEC = importlib.util.spec_from_file_location(
    "trajectory_edit_runner", Path(__file__).parents[1] / "scripts/evaluate_trajectory_editing.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def annotation(struct):
    result = {}
    for name, kind in struct:
        if isinstance(kind, list):
            result[name] = annotation(kind)
        elif isinstance(kind, NumericFeature):
            result[name] = (kind.minimum + kind.maximum) / 2
        elif kind is bool:
            result[name] = False
        else:
            result[name] = next(iter(kind)).value
    return result


def flatten(value, prefix):
    result = []
    for name, child in value.items():
        if isinstance(child, dict):
            result.extend(flatten(child, f"{prefix}__{name}"))
        else:
            result.append((f"{prefix}__{name}", child))
    return result


@pytest.fixture(params=[False, True], ids=["encoder-memory", "direct-pose-checkpoint"])
def experiment(tmp_path, monkeypatch, request):
    monkeypatch.delenv("SIMULATION_NORMALIZATION_DATA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    export = tmp_path / "data"
    export.mkdir()
    cache = tmp_path / "clip_embeddings_clip-vit-base-patch32_cache.pkl"
    monkeypatch.setenv("CLIP_EMBEDDINGS_CACHE_DIR", str(tmp_path))
    random = torch.Generator().manual_seed(3)
    vocabulary = {name: {label: torch.randn(16, generator=random) for label in values}
                  for name, values in enum_descriptions.items()}
    vocabulary["boolean"] = {label: torch.randn(16, generator=random) for label in (True, False)}
    cache.write_bytes(pickle.dumps(vocabulary))
    _write_embedding_cache_metadata(str(cache), "clip-vit-base-patch32", 16)
    fields = flatten(annotation(cinematography_struct), "cinematography")
    instruction = flatten(annotation(simulation_struct), "simulation")
    dictionary = {"schemaVersion": 2, "datasetId": "edit-test",
                  "keys": [name for name, _ in fields + instruction],
                  "values": [[value] for _, value in fields + instruction]}
    shot_index = dictionary["keys"].index("cinematography__initial__shotSize")
    dictionary["values"][shot_index] = ["longShot", "closeUp", None]
    for clip in range(6):
        prompt = [[index, clip % 3 if index == shot_index else 0] for index in range(len(fields))]
        program = [[len(fields) + index, 0] for index in range(len(instruction))]
        body = [1. + .03 * clip, 2. + .05 * clip, .5 + .01 * clip]
        subject = [[.1 * clip, .05 * frame, -.03 * clip, 0., 0., 0.] for frame in range(5)]
        camera = [[.05 * clip, 1. + .04 * frame, 5. + .1 * clip, 0., 0., 0., 35., 16 / 9]
                  for frame in range(5)]
        payload = [prompt, program, [{"i": "person", "c": "human", "d": body,
                    "m": "idle", "f": subject}], camera]
        (export / f"simulation_edit-test_{clip:03}.msgpack").write_bytes(
            msgpack.packb(payload, use_bin_type=True))
    (export / "parameter_dictionary.msgpack").write_bytes(msgpack.packb(dictionary, use_bin_type=True))
    (export / "manifest.json").write_text(json.dumps({"schemaVersion": 2, "datasetId": "edit-test",
        "config": {"subjectCount": 1, "instructionCount": 1}}))
    module = dict(_target_="models.camera_trajectory_model.LensCraft", nhead=2,
        num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32,
        dropout_rate=0., seq_length=5, latent_dim=16,
        keyframe_pose_conditioning=request.param)
    config = OmegaConf.create({"training": {"model": {"module": module}},
        "clip": {"model_name": "clip-vit-base-patch32", "latent_dim": 16},
        "data": {"val_size": 0., "test_size": 1., "split_seed": 19,
            "dataset": {"config": {"_target_": "data.simulation.dataset.SimulationDataset",
                "data_path": str(export), "target_frame_count": 5,
                "normalization_frame_count": 5, "normalize": True}}}})
    config_path = tmp_path / "training.yaml"
    OmegaConf.save(config, config_path)
    judge_module = dict(_target_="testing.metrics.semantic_evaluator.FrozenSemanticEvaluator",
        nhead=2, num_encoder_layers=1, dim_feedforward=32,
        dropout_rate=0., seq_length=5, latent_dim=16, num_query_tokens=46)
    judge_config_path = tmp_path / "evaluator.yaml"
    OmegaConf.save(OmegaConf.create({"module": judge_module, "clip": config.clip}), judge_config_path)
    checkpoints = []
    for seed, name in ((1, "generator"), (2, "evaluator")):
        torch.manual_seed(seed)
        definition = module if name == "generator" else judge_module
        model_type = LensCraft if name == "generator" else FrozenSemanticEvaluator
        model = model_type(**{key: value for key, value in definition.items() if key != "_target_"})
        path = tmp_path / f"{name}.ckpt"
        torch.save({"state_dict": {"model." + key: value for key, value in model.state_dict().items()}}, path)
        checkpoints.append(path)
    output = tmp_path / "result"
    args = runner.parse_args(["--checkpoint", str(checkpoints[0]), "--train-config", str(config_path),
        "--semantic-evaluator-checkpoint", str(checkpoints[1]),
        "--semantic-evaluator-config", str(judge_config_path), "--output", str(output),
        "--limit", "0", "--batch-size", "2", "--save-trajectories"])
    return args


def test_real_checkpoint_dataset_edit_controls_and_reports(experiment, monkeypatch):
    def reject_direct_poses(*args, **kwargs):
        pytest.fail("Every editing control must bypass direct source-pose queries")

    monkeypatch.setattr(LensCraft, "_known_camera_pose_inputs", reject_direct_poses)
    runner.run(experiment)
    result = json.loads((experiment.output / "metrics.json").read_text())
    protocol = result["protocol"]
    assert protocol["status"] == "complete"
    assert protocol["counts"] == {"scanned": 6, "edited": 2,
                                  "skipped_missing": 2, "skipped_unchanged": 2}
    assert protocol["semantic_evaluator"]["frozen"] is True
    assert protocol["generator"]["sha256"] != protocol["semantic_evaluator"]["checkpoint"]["sha256"]
    assert result["metrics"]["target_edited_top1"]["count"] == 2
    assert result["metrics"]["unedited_edited_top1"]["count"] == 18
    assert result["metrics"]["initial_shot_target_log_occupancy_error"]["count"] in (0, 1, 2)
    records = [json.loads(line) for line in (experiment.output / "samples.jsonl").read_text().splitlines()]
    assert len(records) == len({record["sample_id"] for record in records}) == 6
    assert {record["source_value"] for record in records if record["status"] == "edited"} == {"longShot"}
    saved = torch.load(experiment.output / "trajectories.pt", weights_only=True)
    assert saved["coordinate_space"] == "simulation_world"
    assert len(saved["samples"]) == 2
    assert all(sample["edited"].shape == (1, 5, 6) for sample in saved["samples"])
    # The generator uses category IDs; metric vocabulary loading writes no
    # obsolete normalization sidecars into the project or artifact directory.
    assert not (experiment.output / "embedding_means.pkl").exists()
    assert not (experiment.output.parent / "embedding_means.pkl").exists()


def test_no_eligible_edits_is_unmeasured_and_cannot_look_like_perfect_accuracy(experiment):
    # Select one known already-target case in the deterministic holdout order.
    cfg = OmegaConf.load(experiment.train_config)
    seed = next(seed for seed in range(100)
                if torch.randperm(6, generator=torch.Generator().manual_seed(seed))[0].item() % 3 == 1)
    cfg.data.split_seed = seed
    OmegaConf.save(cfg, experiment.train_config)
    experiment.limit = 1
    runner.run(experiment)
    result = json.loads((experiment.output / "metrics.json").read_text())
    assert result["quality_status"] == "no_eligible_edits"
    assert result["protocol"]["counts"]["edited"] == 0
    assert result["metrics"] == {}


def test_output_cannot_overwrite_existing_artifacts(experiment):
    experiment.output.mkdir()
    marker = experiment.output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.run(experiment)
    assert marker.read_text() == "keep"


def test_separate_evaluation_config_cannot_change_training_normalization_grid(experiment):
    config = OmegaConf.load(experiment.train_config)
    config.data.dataset.config.normalization_frame_count = 90
    experiment.data_config = experiment.output.parent / "evaluation.yaml"
    OmegaConf.save(config, experiment.data_config)
    runner.run(experiment)
    manifest = json.loads((experiment.output / "manifest.json").read_text())
    assert manifest["resolved_data_config"]["dataset"]["config"]["normalization_frame_count"] == 5


def test_saved_normalization_is_authoritative(experiment):
    runner.run(experiment)
    baseline = json.loads((experiment.output / "manifest.json").read_text())
    normalization = baseline["normalization_parameters"]
    # Deliberately differ from the export's moments: saved training statistics
    # must normalize/denormalize inputs without being recomputed on the holdout.
    normalization["camera_position"]["mean"][0] += 1.
    original = torch.load(experiment.checkpoint, weights_only=True)
    weights = {key.removeprefix("model."): value for key, value in original["state_dict"].items()}
    robust = experiment.output.parent / "normalized.pt"
    torch.save({"model": weights, "normalization": normalization}, robust)
    experiment.checkpoint = robust
    experiment.output = experiment.output.parent / "robust_result"
    runner.run(experiment)
    manifest = json.loads((experiment.output / "manifest.json").read_text())
    assert manifest["normalization_source"] == "checkpoint"
    assert manifest["normalization_data_path"] is None
    assert manifest["normalization_parameters"]["camera_position"]["mean"] == pytest.approx(
        normalization["camera_position"]["mean"])
    assert manifest["counts"]["edited"] == 2
