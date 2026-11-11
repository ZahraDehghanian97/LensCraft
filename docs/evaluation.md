# Fixed evaluation and sparse keyframes

The categorical LensCraft generator, semantic judge and CLaTr evaluator are
separate models. The generator consumes ten categorical tokens. The semantic
judge is an isolated **encoder-only** 512D model with 46 structured outputs;
it cannot supply generated initial poses or generate trajectories. Its fixed
checkpoint and CLIP vocabulary preserve the existing metric definition.
CLaTr independently supplies trajectory/text alignment, FCD and PRDC.

## Configuration and execution

Set paths in `.env` or the shell:

```bash
export TEST_CHECKPOINT_PATH=/path/to/categorical-generator.pt
export TEST_CONFIG_PATH=/path/to/categorical-training.yaml
export SEMANTIC_EVALUATOR_CHECKPOINT_PATH=/path/to/frozen-semantic.ckpt
export SEMANTIC_EVALUATOR_CONFIG_PATH=/path/to/frozen-semantic-training.yaml
export CLATR_NATIVE_CHECKPOINT_PATH=/path/to/frozen-clatr.ckpt
export CLIP_EMBEDDINGS_CACHE_DIR=/path/to/metric-clip-cache
```

`TEST_*` selects the generator only. There is no fallback from the semantic
judge to the generator checkpoint. The judge's saved configuration is optional
when it matches `config/ref_model/default.yaml`. Its encoder settings and metric
CLIP dimension are independent of the categorical generator. Old 512D generator
checkpoints are rejected by the generator loader; their encoder weights can
still serve the separately configured frozen judge.

```bash
python src/test.py 'keyframes.counts=[1,2,4,8,26]'
python src/aggregate_results.py --results-dir test_results
python src/inference.py num_keyframes=4
```

Select `training/model=et`, `ccdm` or `gendop` only to evaluate those independent
third-party baselines. Baselines retain native and shared-normalized generation
modes. The former judge-generated-initialization mode was removed because the
semantic judge has no decoder. Native baseline text encoders remain unchanged.

Generator memory is always a normalized categorical tensor. There is no camera
vocabulary norm calibration, merged-memory mode, or CLIP cache override for the
generator. Saved categorical configs are migrated without initializing those
removed factories. Current categorical checkpoint keys are retained and loaded
strictly; legacy generator formats do not receive partial compatibility loads.

For source-trajectory interventions, see [trajectory editing](trajectory_editing.md).

## Training and validation policies

Default training samples prompt generation, reconstruction, keyframing and
keyframes plus prompt equally. Keyframe batches choose K from `[1,2,4,8,26]`,
capped by valid sequence length. Observed frames remain clean; hidden camera
inputs are masked. Caption-memory ratios are respectively 1, 0, 0 and 0.5.
Explicit conditioning families disable trajectory teacher forcing and memory
dropout. Noise/mask/mixing schedules remain available when
`training.conditioning.enabled=false`.

Categorical CE and detached-target cycle KL supervise known fields only, with
weight 1 each. Geometry losses include known-pose terms when configured. The
`known_camera_mask` denotes actual constraints; padding never becomes a known
pose. Generated categorical checkpoints pin explicitly supplied keyframes.

Validation computes prompt loss over the configured loader and conditioning
pose/motion diagnostics on a deterministic prefix. `checkpoint_monitor=auto`
selects the enabled validation policy's score consistently for checkpointing,
early stopping and the sweep return. These diagnostics do not use native test
cohorts or CLaTr to select weights. Use the same batch/rank layout and prefix
when comparing validation scores.

Start a new optimizer/schedule from categorical weights with
`initialize_from_checkpoint`, or resume all optimizer/epoch state with
`resume_checkpoint`. They cannot be combined. `training=keyframe_pose` enables
explicit pose/time/presence decoder queries for the same categorical model;
retain its saved configuration for inference. The default is single-step
full-trajectory decoding, with autoregressive decoding still supported.

## Semantic and distribution metrics

Default `src/test.py` **preserves CS46**: mean cosine similarity between the
frozen judge's generated-trajectory embeddings and all known target fields in
its 46-field structured schema. These extra metric fields do not enter the
categorical generator. CS is not CLIP image/text accuracy. Optional caption
classification also uses the frozen judge, never the generator's categorical
encoder as its own evaluator.

Some historical experiment-specific drivers report **CS10**, restricted to the
ten cinematography fields. CS10 and CS46 must be labelled and must not be
compared directly. Display-only `CS × 100` is scaled cosine, not percent accuracy.

FCD and all four PRDC scores use frozen CLaTr trajectory features. PRDC uses
Euclidean distance, `k=3`, five consecutive ordered chunks and corrected recall
version 2; the final value is the mean across chunks. Density can exceed one
because it counts overlapping real neighborhoods divided by k. FCD compares
whole-cohort feature means and covariances. Better PRDC values can coexist with
worse FCD, and neither metric establishes correct caption-to-trajectory pairing.

Keep the cohort membership/order, feature checkpoints, metric vocabulary,
normalization and PRDC protocol fixed. Metrics record checkpoint/content hashes
and evaluator fingerprints under `evaluation_provenance`. Bootstrap uncertainty
is available only when an actual run records it; an empty `bootstrap_std` is not
a confidence interval. Sparse cohorts give less stable distribution estimates.

## Geometry and framing

Geometry is separate from learned semantic metrics. Position error is mean
Euclidean distance in dataset world units, not necessarily meters. Rotation
error is SO(3) geodesic angle in degrees. Motion diagnostics include position
steps, relative rotation, rotation increments and angular path length. Known
and hidden frame errors use their actual masks; padding is excluded.

Framing metrics compare subject-box size/center, screen occupancy, out-of-frame
rate, behind-camera rate and near-plane violations. Their `_count` denominators
show eligibility. Missing measurements are omitted with count zero, rather than
reported as perfect accuracy.

Simulation camera poses use camera-to-world Euler XYZ, radians, local -Z forward
and +Y up. Framing uses stored focal length and aspect ratio with the simulator's
35mm film gauge and 0.1 world-unit near plane. The same reference lens metadata
is applied to generated and reference poses; these scores do not assess optical
zoom prediction or occlusion. A box crossing the near plane contributes to the
violation rate but is excluded from size/center errors.

For a dataset without calibrated projection, framing is unavailable unless an
explicit shared assumption is supplied:

```bash
python src/test.py geometry_metrics.vertical_fov_deg=50 geometry_metrics.aspect_ratio=1.77777778
```

Stored calibration takes precedence. CCDM rotations use the documented upright
camera reconstruction assumptions, rather than measured native orientations.
See [CCDM coordinates](ccdm_native_evaluation.md) before interpreting them.

## Reports and caches

Reports retain mode, K, sample counts, evaluator provenance and all actual scalar
metrics. A cache must match the generator checkpoint/configuration, source data,
normalization, conditioning protocol and keyframe masks. Use a fresh output
when any identity changes. Older generator caches are historical artifacts;
they are not replayed through the categorical model as equivalent outputs.
