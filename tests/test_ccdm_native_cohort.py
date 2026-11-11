"""Pair models using verified classifier-test IDs without inventing generator holdout."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_ccdm_native_cohort.py"
    spec = importlib.util.spec_from_file_location("prepare_ccdm_native_cohort", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cohort_uses_only_frozen_test_ids_and_rejects_changed_split(tmp_path):
    module = runner()
    data = tmp_path / "data.npy"
    np.save(data, {"train_source_indices": np.array([0, 1, 2]),
                  "test_source_indices": np.array([3, 4, 5]), "test_label": np.array([0, 2, 5])})
    manifest = {"source_sha256": "native-corpus-sha",
                "artifacts": {"data.npy": module.digest(data)}}
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest))
    result = module.prepare(tmp_path, tmp_path / "cohort.json", count=24)
    again = module.prepare(tmp_path, tmp_path / "again.json", count=24)
    assert result == again
    assert set(result["source_indices"]) <= {3, 4, 5}
    assert result["generator_heldout_verified"] is False
    assert all(label == {3: 0, 4: 2, 5: 5}[index]
               for index, label in zip(result["source_indices"], result["labels"]))
    with pytest.raises(ValueError, match="cannot exceed"):
        module.prepare(tmp_path, tmp_path / "too_many.json", count=4, unique=True)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        module.prepare(tmp_path, tmp_path / "changed.json", count=1)
