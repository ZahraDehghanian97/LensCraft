"""The public mixed-training config carries native captions into categorical labels."""
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from data.ccdm.dataset import CCDMDataset, collate_fn
from data.simulation.constants import cinematography_struct, simulation_struct
from data.simulation.utils import structured_conditioning_from_batch
from train import _validate_multi_config


CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[1] / "config")


def configuration(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIRECTORY, version_base=None):
        return compose(config_name="config", overrides=["data=multi", "training=multi", *overrides])


def empty_structure(structure):
    return {key: empty_structure(kind) if isinstance(kind, list) else None
            for key, kind in structure}


@pytest.fixture
def native_data(tmp_path, monkeypatch):
    monkeypatch.setattr(CCDMDataset, "_normalization_parameters", None)
    monkeypatch.setenv("CCDM_DATA_DIR", str(tmp_path))
    np.save(tmp_path / "Mean_Std.npy", {"Mean": np.zeros(5), "Std": np.ones(5)})
    np.save(tmp_path / "data.npy", {
        "cam": [np.zeros((4, 5)), np.ones((4, 5))],
        "info": [["The camera pushes in."], ["The camera is static."]],
    })
    rows = []
    for index, (caption, movement) in enumerate([
        ("The camera pushes in.", "dollyIn"), ("The camera is static.", "static"),
    ]):
        prompt = empty_structure(cinematography_struct)
        prompt["movement"]["type"] = movement
        rows.append({
            "custom_id": f"ccdm-{index:06d}", "source_prompt": caption,
            "cinematographyPrompts": [prompt],
            "simulationInstructions": [empty_structure(simulation_struct)],
        })
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(rows))
    return annotations


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_public_mixed_training_stops_without_ccdm_labels(monkeypatch, missing):
    monkeypatch.delenv("CCDM_ANNOTATIONS_PATH", raising=False)
    if missing is not None:
        monkeypatch.setenv("CCDM_ANNOTATIONS_PATH", missing)
    with pytest.raises(ValueError, match="CCDM_ANNOTATIONS_PATH"):
        _validate_multi_config(configuration())


def test_mixed_training_rejects_nonexistent_annotation_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CCDM_ANNOTATIONS_PATH", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="CCDM training annotations not found"):
        _validate_multi_config(configuration())


def test_relative_annotation_path_is_anchored_for_hydra_output_directory(native_data, monkeypatch):
    monkeypatch.chdir(native_data.parent)
    monkeypatch.setenv("CCDM_ANNOTATIONS_PATH", native_data.name)
    cfg = configuration()
    _validate_multi_config(cfg)
    assert cfg.data.dataset.ccdm_config.annotations_path == str(native_data)
    assert instantiate(cfg.data.dataset.ccdm_config)[0]["cinematography_ids"][4] == 3


@pytest.mark.parametrize("source", ["environment", "override"])
def test_public_mixed_config_produces_caption_conditioning(native_data, monkeypatch, source):
    monkeypatch.delenv("CCDM_ANNOTATIONS_PATH", raising=False)
    overrides = []
    if source == "environment":
        monkeypatch.setenv("CCDM_ANNOTATIONS_PATH", str(native_data))
    else:
        overrides.append(f"data.dataset.ccdm_config.annotations_path={native_data}")
    cfg = configuration(overrides)
    _validate_multi_config(cfg)
    dataset = instantiate(cfg.data.dataset.ccdm_config)
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["text_prompts"] == ["The camera pushes in.", "The camera is static."]
    assert batch["item_ids"] == ["ccdm-000000", "ccdm-000001"]
    assert batch["cinematography_ids"][:, 4].tolist() == [3, 0]
    assert batch["prompt_none_mask"].sum(1).tolist() == [1, 1]
    tokens = structured_conditioning_from_batch(batch)
    assert tokens[4, 0, 3] == tokens[4, 1, 0] == 1


def test_mixed_subset_config_still_validates_caption_provenance(native_data, monkeypatch):
    rows = json.loads(native_data.read_text())
    native_data.write_text(json.dumps(rows[:1]))
    indices = native_data.with_name("indices.json")
    indices.write_text("[0]")
    monkeypatch.setenv("CCDM_ANNOTATIONS_PATH", str(native_data))
    cfg = configuration([f"data.dataset.ccdm_config.sample_indices_path={indices}"])
    _validate_multi_config(cfg)
    assert instantiate(cfg.data.dataset.ccdm_config)[0]["cinematography_ids"][4] == 3
    rows[0]["source_prompt"] = "Different caption"
    native_data.write_text(json.dumps(rows[:1]))
    with pytest.raises(InstantiationException, match="caption mismatch"):
        instantiate(cfg.data.dataset.ccdm_config)


def test_annotation_free_ccdm_evaluation_remains_supported(native_data):
    cfg = OmegaConf.create({"data": {"use_multi_dataset": False}})
    _validate_multi_config(cfg)
    dataset = CCDMDataset(str(native_data.with_name("data.npy")), original_seq_len=4)
    assert dataset[0]["text_prompts"] == "The camera pushes in."
    assert not dataset[0]["prompt_none_mask"].any()


def test_ccdm_baseline_embeds_original_caption_words_and_case():
    from models.baselines.ccdm_adapter import CCDMAdapter

    # Exercise the public adapter generation call without loading weights or CLIP.
    adapter = object.__new__(CCDMAdapter)
    adapter.device = torch.device("cpu")
    adapter.config = {"guidance_scale": 2.0}
    adapter.seq_len, adapter.n_feature = 4, 5
    adapter.mean, adapter.std = torch.zeros(5), torch.ones(5)
    prompts = ["The camera pushes in.", "The camera pulls out.",
               "The camera PUSHES IN.", "The lens zooms in."]
    seen = []

    class Embedder:
        def extract_clip_embeddings(self, captions, return_seq):
            seen.append(list(captions))
            assert return_seq is False
            return torch.zeros(len(captions), 512)

    class Sampler:
        def sample(self, n_sample, c, size, device, guide_w):
            assert n_sample == len(prompts) and c.shape == (len(prompts), 512)
            return torch.zeros(n_sample, *size)

    adapter.clip_embedder, adapter.ddpm = Embedder(), Sampler()
    result = adapter.generate_using_text(prompts)
    assert seen == [prompts]
    assert prompts[0] == "The camera pushes in."
    assert result.shape == (len(prompts), 4, 5)
