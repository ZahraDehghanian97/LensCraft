# ET evaluation with fresh human-prompt translation

LensCraft can evaluate ET through `src/test.py data/dataset=et`. Explicitly set
`data.dataset.config.split=test data.val_size=0 data.test_size=1`; the default
ET dataset configuration loads the native training split and randomly holds out
part of it.

The September 16 experiment uses the selected intersection of ET's `pure_test`
and `mixed_train` inventories: 2,135 of 2,886 pure-test clips. All selected IDs
also occur in mixed train. This is a cross-dataset evaluation of a
simulation-trained LensCraft checkpoint, not a held-out test for a DIRECTOR
checkpoint trained on mixed ET. The selection is recorded exactly in a manifest.

Prepare the cohort from native ET split lists and captions, without reading
previous structured annotations:

```bash
python scripts/prepare_et_gateway_evaluation.py \
  --dataset-root "$ET_DATA_DIR" --output-dir /path/to/et-evaluation
```

Generate fresh translations using [AI Gateway](gateway_translation.md). The CLI
loads `AI_GATEWAY_API_KEY` from the project `.env`. Use a new cache directory for
a new independent translation run; interrupted runs can reuse their own valid
responses. Identical native captions share one Gateway request.

```bash
python src/translate_prompt.py \
  --input /path/to/et-evaluation/prompts.jsonl \
  --output /path/to/et-evaluation/fresh_annotations.json \
  --metadata-output /path/to/et-evaluation/fresh_annotations.metadata.json \
  --cache-dir /path/to/et-evaluation/gateway_cache \
  --model zai/glm-5.3-flash

python scripts/prepare_et_gateway_evaluation.py \
  --dataset-root "$ET_DATA_DIR" --output-dir /path/to/et-evaluation \
  --annotations /path/to/et-evaluation/fresh_annotations.json \
  --translation-metadata /path/to/et-evaluation/fresh_annotations.metadata.json
```

Finalization checks complete cohort coverage, the current conditioning schema,
native caption identity, requested and returned model, and translation settings.
It writes the `custom_id` annotation list required by `ETDataset`.

Run the existing evaluator, supplying the generator and evaluator checkpoint
paths from the intended experiment:

```bash
python src/test.py data/dataset=et \
  data.dataset.config.dataset_dir=/path/to/et-evaluation/et-data \
  data.dataset.config.et_cin_lang_path=/path/to/et-evaluation/annotations.json \
  data.dataset.config.set_name=pure data.dataset.config.split=test \
  data.val_size=0 data.test_size=1 data.batch_size=32 data.num_workers=0 \
  training.model.inference.checkpoint_path="$TEST_CHECKPOINT_PATH" \
  training.model.inference.config="$TEST_CONFIG_PATH" \
  ref_model.inference.checkpoint_path="$SEMANTIC_EVALUATOR_CHECKPOINT_PATH" \
  ref_model.inference.config="$SEMANTIC_EVALUATOR_CONFIG_PATH" \
  clatr_backend=native clatr_native_checkpoint_path="$CLATR_NATIVE_CHECKPOINT_PATH" \
  trajectory_cache=false +n_boot=0 +eval_set=et_pure_fresh \
  output_dir=/path/to/et-evaluation/results
```

Keep `SIMULATION_DATA_PATH` pointed at the exact simulation dataset used for
training, so ET trajectories use the checkpoint's normalization statistics.
Keep ET normalization enabled. ET's variable-length trajectories are resampled
to the model's 30-frame grid. Subject orientation and bounding-box dimensions
are estimated by the current converter. ET pixel intrinsics are not consumed
by simulation framing metrics; leave shared FOV/aspect overrides unset and
report position/rotation metrics instead of inventing projection parameters.

Native CLaTr scores here use a simulation-trained evaluator. They, and the
30-frame resampling protocol, differ from the published ET benchmark. Report
the checkpoint identities and cohort along with the scores. The command above
reports point estimates without bootstrap confidence intervals. For a smoke
test, prepare a separate directory with `--limit 32` and use `keyframes.counts=[4]`.

The server needs working outbound HTTPS for fresh translation. During the
September 16 run its configured proxy timed out, so requests used a temporary
SSH reverse tunnel to the workstation's working proxy. This is session-local;
no proxy settings or credentials are stored in annotation files.

The completed September 16 run evaluated all 2,135 selected clips in 13 modes,
using 940 newly translated distinct captions from `zai/glm-5.3-flash` with low
reasoning and Pydantic v2 validation. Each mode covered 64,050 poses and reported
zero invalid generated poses. Selected results are:

| Mode | Position error (ET world units) | Rotation error (degrees) |
|---|---:|---:|
| Reconstruction | 1.0366 | 13.0322 |
| Prompt generation | 1.8878 | 136.0489 |
| Hybrid generation | 1.0687 | 19.8110 |
| Four keyframes | 0.9852 | 11.9485 |

Prompt generation has poor agreement with the reference orientation in this
experiment. A caption can describe multiple valid trajectories, so this alone
does not identify whether translation or generation causes the discrepancy.
Pydantic validates structure, types, and supported values; it does not establish
semantic translation accuracy. This checkpoint uses soft keyframe conditioning
(`keyframe_pose_conditioning=false`), so supplied keyframes are not guaranteed
to be reproduced exactly.

A targeted semantic audit found invented numeric `importance` values in 16
distinct translations covering 36 clips, despite valid schemas. The reported
scores retain those actual model outputs. Other fields were not comprehensively
audited for semantic correctness.

Full metrics, annotations, response provenance, and execution logs are stored on
the server under
`/media/external20/morteza_abolghasemi/lens-craft/runs/et_review_20260916/`.
The final annotation files use the `fresh_annotations_pydantic_low` prefix in
`pure_fresh/`; earlier smoke and experimental files are separate diagnostics.

Follow-up diagnostics reproduced the original scores on all 2,135 clips and
ran paired controls on 2,000 simulation holdout clips with the same checkpoint.
Removing simulation conditioning fields according to the fresh ET missingness
patterns raised rotation error from 10.44 to 95.55 degrees. This demonstrates
the impact of missing information under reference-pose scoring; it does not
separate inherent caption ambiguity from model robustness.

A broader translation audit found explicit reversed motion in 237 clips. A
separate diagnostic correcting only their cinematography movement type changed
the full ET rotation error from 136.05 to 135.83 degrees. First-frame rotation
alignment after generation reduced error to 6.46 degrees, but the all-zero
prompt control also reached 6.49 degrees after alignment. These oracle-aligned
scores do not establish successful text-conditioned generation. Full evidence,
per-clip metrics, and plots are in `diagnosis/DIAGNOSIS.md` under the server run
directory above. Original annotations and metrics were preserved.
