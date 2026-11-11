"""Native evaluator reconstruction keeps labels, split and original preprocessing."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_ccdm_native_classifier.py"
spec = importlib.util.spec_from_file_location("ccdm_native_training", SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def corpus(tmp_path, count=60):
    phrases = list(training.MOTION_PHRASES)
    rng = np.random.RandomState(11)
    cameras, infos = [], []
    for i in range(count):
        cameras.append(rng.normal(size=(3 + i % 3, 5)))
        infos.append(["The camera " + phrases[i % len(phrases)] + ".", "The character is at the middle center of the screen."])
    path = tmp_path / "native.npy"
    np.save(path, {"cam": cameras, "info": infos})
    return path, cameras, infos


def test_native_caption_mapping_covers_both_boom_directions_and_rejects_unknowns():
    for phrase, name in training.MOTION_PHRASES.items():
        label, text = training.motion_label([f"The camera {phrase}.", "The camera shoots at close shot."])
        assert label == training.LABEL_MAP[name]
        assert text.endswith("close shot.")
    for text in ("The camera is mysterious.", "The camera is static. The camera booms up.", "The camera is static-ish."):
        with pytest.raises(ValueError, match="recognized motion sentences"):
            training.motion_label(text)


def test_preparation_preserves_raw_frames_captions_ids_and_unpadded_all_corpus_statistics(tmp_path):
    path, cameras, infos = corpus(tmp_path)
    output = tmp_path / "prepared"
    schema, mean, std, manifest = training.prepare_corpus(path, output)
    train, test = training.split_indices(len(cameras), 42)
    np.testing.assert_array_equal(train, np.random.RandomState(42).permutation(len(cameras))[:54])
    assert not set(train) & set(test)
    assert sorted(np.concatenate((train, test))) == list(range(60))
    expected = np.concatenate([cameras[i] for i in np.concatenate((train, test))])
    np.testing.assert_array_equal(mean, expected.mean(0))
    np.testing.assert_array_equal(std, expected.std(0))
    assert manifest["reconstructed"] and manifest["normalization"]["test_data_included"]
    assert schema["test_info"] == [infos[i] for i in test]
    restored = np.load(output / "data.npy", allow_pickle=True).item()
    np.testing.assert_array_equal(restored["train_cam"][0], cameras[train[0]])
    labels = [json.loads(line) for line in (output / "source_labels.jsonl").read_text().splitlines()]
    assert [row["source_index"] for row in labels] == list(range(60))
    assert labels[0]["raw_caption"] == infos[0]
    dataset = training.NativeDataset(restored["train_cam"], restored["train_label"], mean, std)
    camera, label = dataset[0]
    normalized = ((cameras[train[0]] - mean) / (std + 1e-8)).astype(np.float32)
    np.testing.assert_array_equal(camera[:len(normalized)].numpy(), normalized)
    np.testing.assert_array_equal(camera[-1].numpy(), normalized[-1])
    assert camera.shape == (300, 5) and camera.dtype == torch.float32
    assert label == restored["train_label"][0]


def test_unknown_sample_fails_with_source_index_and_no_silent_drop(tmp_path):
    path, cameras, infos = corpus(tmp_path)
    infos[7] = ["The camera teleports."]
    np.save(path, {"cam": cameras, "info": infos})
    with pytest.raises(ValueError, match="source index 7"):
        training.prepare_corpus(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


class TinyClassifier(torch.nn.Module):
    """Exercise resumable optimizer/RNG machinery without a 4-layer Transformer."""
    def __init__(self, frames):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.pred = torch.nn.Linear(5, 6)

    def forward(self, x):
        return self.pred(self.dropout(x.mean(1)))


def test_explicit_resume_matches_uninterrupted_epoch_rng_and_never_selects_on_test(tmp_path, monkeypatch):
    path, _, _ = corpus(tmp_path)
    monkeypatch.setattr(training, "CCDMNativeClassifier", TinyClassifier)
    def args(name):
        return training.parser().parse_args(["--data-path", str(path), "--output-dir", str(tmp_path / name),
            "--device", "cpu", "--epochs", "2", "--batch-size", "9", "--threads", "1", "--num-workers", "0"])
    uninterrupted = args("whole")
    training.run(uninterrupted)
    interrupted = args("resumed")
    real_save = training.atomic_torch_save
    def stop_after_first_checkpoint(path, value):
        real_save(path, value)
        if Path(path).name == "last.pt" and value["epochs_completed"] == 1:
            raise KeyboardInterrupt("simulated process interruption")
    monkeypatch.setattr(training, "atomic_torch_save", stop_after_first_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        training.run(interrupted)
    monkeypatch.setattr(training, "atomic_torch_save", real_save)
    with pytest.raises(ValueError, match="not empty"):
        training.run(interrupted)
    interrupted.resume = True
    completion = training.run(interrupted)
    a = torch.load(tmp_path / "whole/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    assert a["updates"] == b["updates"] == 12
    assert completion["reconstructed"] and not completion["original_paper_evaluator"]
    assert completion["checkpoint_selection"].startswith("fixed last")
    assert b["last_metrics"]["test_used_for_checkpoint_selection"] is False
    evaluator = json.loads((tmp_path / "resumed/result/evaluator_manifest.json").read_text())
    assert evaluator["classifier_origin"] == "retrained"
    assert evaluator["classifier_checkpoint"]["sha256"] == training.digest(tmp_path / "resumed/result/latest.pth")
    assert evaluator["classifier_data"]["sha256"] == training.digest(tmp_path / "resumed/data.npy")
    assert evaluator["normalization"]["mean"] == json.loads((tmp_path / "resumed/data_manifest.json").read_text())["normalization"]["mean"]
    interrupted.epochs = 3
    with pytest.raises(ValueError, match="configuration differs"):
        training.run(interrupted)


def test_prepare_only_refuses_existing_output_and_requires_explicit_resume_to_train(tmp_path, monkeypatch):
    path, _, _ = corpus(tmp_path)
    output = tmp_path / "prepare"
    args = training.parser().parse_args(["--data-path", str(path), "--output-dir", str(output),
        "--device", "cpu", "--prepare-only", "--threads", "1", "--epochs", "1", "--num-workers", "0"])
    assert training.run(args)["state"] == "prepared"
    assert not (output / "last.pt").exists()
    with pytest.raises(ValueError, match="not empty"):
        training.run(args)
    monkeypatch.setattr(training, "CCDMNativeClassifier", TinyClassifier)
    args.resume, args.prepare_only = True, False
    assert training.run(args)["state"] == "complete"


def test_training_matches_released_shared_rng_shuffle_adam_and_linear_decay(tmp_path, monkeypatch):
    path, _, _ = corpus(tmp_path)
    monkeypatch.setattr(training, "CCDMNativeClassifier", TinyClassifier)
    args = training.parser().parse_args(["--data-path", str(path), "--output-dir", str(tmp_path / "ours"),
        "--device", "cpu", "--epochs", "2", "--batch-size", "9", "--threads", "1", "--num-workers", "0"])
    training.run(args)
    ours = torch.load(tmp_path / "ours/last.pt", weights_only=False)
    schema = np.load(tmp_path / "ours/data.npy", allow_pickle=True).item()
    stats = np.load(tmp_path / "ours/Mean_Std.npy", allow_pickle=True).item()
    training.seed_runtime(42, torch.device("cpu"))
    reference = TinyClassifier(frames=300)
    optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train = torch.utils.data.DataLoader(training.NativeDataset(schema["train_cam"], schema["train_label"],
        stats["Mean"], stats["Std"]), batch_size=9, shuffle=True, num_workers=0)
    test = torch.utils.data.DataLoader(training.NativeDataset(schema["test_cam"], schema["test_label"],
        stats["Mean"], stats["Std"]), batch_size=9, shuffle=False, num_workers=0)
    # Independently follow classify.py's loop, including its shared global RNG.
    for ep in range(2):
        optimizer.param_groups[0]["lr"] = 1e-4 * (1 - ep / 2)
        reference.train()
        for camera, labels in train:
            predictions = reference(camera)
            optimizer.zero_grad()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
        reference.eval()
        for camera, labels in test:
            reference(camera)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, ours["model"][key], atol=0, rtol=0)
    defaults = training.parser().parse_args(["--data-path", str(path), "--output-dir", str(tmp_path / "defaults")])
    assert (defaults.epochs, defaults.batch_size, defaults.learning_rate, defaults.num_workers) == (1000, 128, 1e-4, 5)
    assert defaults.threads is None and defaults.precision == "fp32"
    assert ours["config"]["shuffle"].endswith("no epoch reseeding")
    with pytest.raises(SystemExit):
        training.parser().parse_args(["--data-path", str(path), "--output-dir", str(tmp_path / "bf16"), "--precision", "bf16"])
