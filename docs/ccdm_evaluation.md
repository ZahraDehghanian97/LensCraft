# LensCraft on native CCDM

Current commands require a categorical generator checkpoint. The dated scores
below describe historical frozen executions and are not new categorical-model
measurements. Metric CLIP assets belong to the frozen judges only.

CCDM can be evaluated with the same generator and frozen evaluators as ET.
The released `data.npy` contains 27,810 trajectories and has no official split
IDs. The September 18 review uses 2,781 source indices selected with NumPy
`default_rng(42).choice(..., replace=False)` and sorted before evaluation.
These contain 2,699 distinct complete native captions. This is cross-dataset
evaluation of the original simulation-trained LensCraft model; it is not the
published CCDM benchmark and performs no training on CCDM.

Prepare and record a cohort before generating or scoring any trajectories:

```bash
python scripts/prepare_ccdm_evaluation.py \
  --data-path "$CCDM_DATA_DIR/data.npy" \
  --output-dir /path/to/ccdm/cohort10pct --sample-count 2781 --seed 42

python src/translate_prompt.py \
  --input /path/to/ccdm/cohort10pct/prompts.json \
  --output /path/to/ccdm/cohort10pct/fresh_annotations.json \
  --metadata-output /path/to/ccdm/cohort10pct/translation_metadata.json \
  --cache-dir /path/to/ccdm/gateway_cache \
  --model meta/muse-ember --reasoning-effort low

python scripts/prepare_ccdm_evaluation.py \
  --data-path "$CCDM_DATA_DIR/data.npy" \
  --output-dir /path/to/ccdm/cohort10pct --sample-count 2781 --seed 42 \
  --annotations /path/to/ccdm/cohort10pct/fresh_annotations.json \
  --translation-metadata /path/to/ccdm/cohort10pct/translation_metadata.json
```

The translator, schema, system prompt and low-reasoning configuration match ET.
Every native caption is joined in its stored order; only identical complete
captions share a request. The cohort helper records the original source indices,
data/statistics hashes and caption hashes. Finalization validates complete
coverage, caption identity, schema and translation provenance. Schema validation
checks structure and types, not the semantic correctness of translations.

Set `SIMULATION_DATA_PATH` to the exact training dataset and reuse the ET
generator, semantic evaluator, CLaTr checkpoint and CLIP vocabulary cache:

```bash
python src/test.py data/dataset=ccdm \
  data.dataset.config.sample_indices_path=/path/to/ccdm/cohort10pct/indices.json \
  data.dataset.config.annotations_path=/path/to/ccdm/cohort10pct/annotations.json \
  data.val_size=0 data.test_size=1 data.batch_size=32 data.num_workers=0 \
  training.model.inference.checkpoint_path="$TEST_CHECKPOINT_PATH" \
  training.model.inference.config="$TEST_CONFIG_PATH" \
  ref_model.inference.checkpoint_path="$SEMANTIC_EVALUATOR_CHECKPOINT_PATH" \
  ref_model.inference.config="$SEMANTIC_EVALUATOR_CONFIG_PATH" \
  clatr_backend=native clatr_native_checkpoint_path="$CLATR_NATIVE_CHECKPOINT_PATH" \
  trajectory_cache=false +n_boot=0 +seed=42 +eval_set=ccdm_cohort10pct \
  output_dir=/path/to/ccdm/cohort10pct/evaluation
```

This runs reconstruction, prompt, hybrid and both keyframe modes at K=1,2,4,8,26.
Without structured annotations, CCDM keeps its original reconstruction and
keyframe-only evaluation behavior. Annotation loading rejects missing IDs,
duplicate IDs, mismatched source captions and invalid conditioning schemas.

Native CCDM stores `[relative_x, relative_y, relative_z, screen_x, screen_y]`.
The existing converter reconstructs upright camera rotations using an assumed
vertical FOV of 45 degrees and aspect ratio 16:9, with an implicit stationary
subject at the origin and estimated dimensions `[0.5, 1.7, 0.3]`. Original CCDM
view labels place front at +Z and left at +X, so the synthesized subject has a
180-degree Y rotation to match simulation's zero-heading front=-Z, left=-X.
Camera world poses are unchanged. Native CCDM
normalization is undone before conversion; model inputs use the training
simulation normalization. Valid frames are resampled to the model's 30-frame
grid; last-frame padding is excluded.

Position errors use native scene units. Rotation error is against reconstructed
poses, not measured native orientation; it depends on the declared FOV and
upright-camera assumptions. Do not multiply positions by the Unity demo's
consumer-side scale. Leave shared framing calibration unset: no calibrated
bounding-box/framing scores are reported. Learned metrics use the fixed
simulation-trained evaluators and should not be presented as official CCDM
FID or as directly comparable to published CCDM scores. The experiment reports
point estimates without bootstrap intervals, matching the original ET run.

Server artifacts and the isolated execution source are stored under
`/media/external20/morteza_abolghasemi/lens-craft/runs/ccdm_review_20260918/`.

The reverse benchmark, CCDM on simulation data, uses the same subject basis.
Normalized predictions rotate both position and orientation by
`R_subject @ diag(-1, 1, -1)` and add the subject translation on each frame.
Inference uses the same placement helper. Feature conversion to CCDM applies
the inverse map. Raw/unplaced ablation outputs retain native coordinates.
Coordinate protocol version 3 invalidates old CCDM-related trajectory caches.
The publisher's Unity consumer records screen Y downward, using
`Screen.height - WorldToScreenPoint(...).y`. Version 3 negates stored screen Y
when reconstructing the native Y-up camera ray and reverses that sign when
encoding. It also interprets the declared 45-degree FOV as vertical, consistent
with Unity's camera API. Aspect 16:9 is an evaluation assumption; the demo's
60-degree camera does not calibrate the training data. The exact upright ray
solver is not identical to the consumer's approximate pitch/world-yaw replay.
Previous version-2 CCDM rotation/FCD results are historical and must be rerun.
The archived 2,000-sample simulation comparison is rescored from the same raw
CCDM draws in `ccdm_on_simulation/`, with unchanged evaluator fingerprints and
a raw-output control that must reproduce the old scores.
