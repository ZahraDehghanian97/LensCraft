# Fixed semantic evaluation and sparse keyframes

The generator, structured semantic evaluator, and CLaTr evaluator are separate
models. Keep both evaluator checkpoints fixed across all baselines and ablations.
The structured evaluator is frozen and loaded as a separate object, even when
its weights happen to match the full-model generator. It never generates the
LensCraft trajectories being scored. The separately labelled baseline
`prompt_generation_norm_lenscraft_init` mode still uses this fixed LensCraft
reference to supply an initial position.

## Configuration and execution

Set these in `.env` or your shell:

```bash
export TEST_CHECKPOINT_PATH=/path/to/generator.ckpt
export TEST_CONFIG_PATH=/path/to/generator-training.yaml
export SEMANTIC_EVALUATOR_CHECKPOINT_PATH=/path/to/fixed-full-model.ckpt
export SEMANTIC_EVALUATOR_CONFIG_PATH=/path/to/fixed-full-model-training.yaml
export CLATR_NATIVE_CHECKPOINT_PATH=/path/to/fixed-clatr.ckpt
```

`TEST_*` selects only the generator. There is no implicit fallback from the
semantic evaluator to `TEST_CHECKPOINT_PATH`. An evaluator config is optional
only when its architecture matches `config/ref_model/default.yaml`. If supplied,
its own `training.model.module` is used and its interpolations resolve inside
that saved configuration. Evaluator and dataset targets must use the same CLIP
model and embedding dimension. A mismatch fails before scoring.

Run the default sweep and create reports:

```bash
python src/test.py 'keyframes.counts=[1,2,4,8,26]'
python src/aggregate_results.py --results-dir test_results
```

Use `keyframes.counts=[4]` for one sparse condition, `keyframes.seed=123` for
another selection, or `keyframes.counts=[]` to skip keyframe modes. This seed
affects keyframe selection only, not dataset split membership. The pilot runner
also accepts `--semantic-evaluator-checkpoint` and
`--semantic-evaluator-config`, defaulting to the corresponding environment
variables. Baselines and ablations receive the same fixed evaluator.

For standalone inference:

```bash
python src/inference.py num_keyframes=4
```

### Camera-memory calibration

The default unmerged LensCraft model calibrates encoder token magnitudes to the
fixed conditioning vocabulary before mixing with caption memory and adding slot
identity. Categorical targets are mean vocabulary vector norms; numeric Fourier
targets are `sqrt(embedding_dim / 2)`. No per-sample caption values or presence
mask are used to determine these scales. Encoder outputs used for semantic and
cycle alignment remain raw, and caption-only generation is unchanged.

This is enabled for default training and for inference/test, including loading
an older saved training config. The scales are derived configuration, so legacy
checkpoint weights still load strictly. Keep `CLIP_EMBEDDINGS_CACHE_DIR` and
`clip.model_name` consistent with the training vocabulary. The separate merged
`training=multi` configuration retains its original behavior; applying this
calibration to merged or denormalized memory fails explicitly.

To reproduce the original uncalibrated generation, use:

```bash
python src/test.py training.model.inference.camera_memory_normalization=none output_dir=test_results_raw
```

To evaluate calibrated generation in a separate output directory:

```bash
python src/test.py training.model.inference.camera_memory_normalization=vocabulary output_dir=test_results_calibrated
```

For an explicitly uncalibrated new training run, set
`training.model.module.camera_memory_norms=null`; inference has its own explicit
override above. Calibration itself does not enforce exact keyframe poses.

### Training and validation of conditioning modes

Default simulation training now samples one of four families per batch:
`prompt_generation`, `reconstruction`, `key_framing`, and `key_framing+prompt`.
Their default weights are equal and stay active throughout training. Keyframe
batches sample K uniformly from `[1,2,4,8,26]`, capped by valid sequence length.
Observed frames are clean; unobserved camera values are zeroed and masked.
Caption-memory ratios are respectively 1, 0, 0, and 0.5. Explicit mode training
uses no trajectory teacher forcing or memory dropout. The former noise/mask/text
ratio schedules apply only when `training.conditioning.enabled=false` (or when
loading a legacy trainer config without this policy).

The loss receives the actual `known_mask`. `keyframe_position` is the mean of
`||Δposition||²/3` over known valid frames; `keyframe_rotation` is the mean of
`||R−R_target||²_F/9` over the same frames, multiplied by `rotation_weight`.
Their default coefficients are 4 and 2 and
are configurable. An empty known set contributes a differentiable zero; hidden
and padded frames do not enter these terms. Existing whole-trajectory losses
continue to supervise motion between supplied poses. The new coefficients and
equal mode probabilities are explicit starting settings, not tuned optima.

Validation still computes prompt `val_loss` over the configured validation
loader. In addition, a deterministic prefix (default first four batches per
rank) evaluates prompt, reconstruction and both keyframe modes at every
configured K. All components of `val_conditioning_score`, including prompt,
use this same prefix; the full-loader prompt `val_loss` remains separate.
Dedicated sampling seeds keep these masks fixed across epochs without changing
training RNG. Set `training.validation_conditioning.max_batches` to control
cost. Changing batch size/rank layout changes that prefix; use the same setup
for model comparisons. Distributed totals aggregate sums/counts before means.

Metrics under `val_conditioning/<mode>/` report normalized position error,
SO(3) angle in degrees, known/hidden frame errors, adjacent position-step error,
and step error across known/hidden boundaries, with eligibility counts. These
position values use model-normalized coordinates, unlike the denormalized world
units in final paper evaluation. A step error is displacement residual per
sampled frame, not physical speed. Nonfinite valid poses fail validation.

`checkpoint_monitor=auto` selects `val_conditioning_score` whenever this extra
validation is enabled; otherwise it selects legacy `val_loss`. Checkpointing,
early stopping and the returned sweep objective use the same selected metric.
The score averages four equally weighted families. Each case contributes
`mean_position_error_normalized + mean_rotation_error_deg/180`; keyframe cases
also add `known_pose_weight * (known_position_error_normalized +
known_rotation_error_deg/180)` (default weight 1). K cases are averaged within
their family first, so adding K values does not multiply that family's weight.
The score is a declared selection criterion, not CLaTr or a physical-unit sum.
The independent CLaTr evaluator remains fixed for paper evaluation.

The merged `training=multi` configuration retains its legacy training/validation
policy. To reproduce the previous default trainer policy explicitly, disable
both `training.conditioning.enabled` and `training.validation_conditioning.enabled`;
`checkpoint_monitor=auto` then chooses `val_loss`.

### Fine-tuning and optional direct pose inputs

Use `initialize_from_checkpoint` to start a new optimizer/schedule/epoch history
from existing model weights. `resume_checkpoint` restores an interrupted run's
optimizer and epoch state instead; the two arguments cannot be combined.
For a short, separately named fine-tuning experiment after setting the dataset
and vocabulary environment variables:

```bash
python src/train.py 'initialize_from_checkpoint="/path/to/generator.ckpt"' trainer.max_epochs=20 training.optimizer.lr=0.00001 'training.lr_scheduler.base_lr=[0.00003]' hydra.run.dir=outputs/keyframe_finetune
```

An optional architecture ablation passes visible poses, frame positions and
presence markers directly into single-step decoder queries:

```bash
python src/train.py training=keyframe_pose 'initialize_from_checkpoint="/path/to/generator.ckpt"' trainer.max_epochs=20 training.optimizer.lr=0.00001 'training.lr_scheduler.base_lr=[0.00003]' hydra.run.dir=outputs/keyframe_pose_finetune
```

Keep both layers of quotation around the checkpoint override when its filename
contains `=` (as in `best-val-model-epoch=099-val_loss=19.684.ckpt`).

`training=keyframe_pose` enables `training.model.module.keyframe_pose_conditioning`; the
default is false. It reuses the existing pose-input projection and adds a fixed
presence code, so strict legacy weight loading remains possible. Masked/padded
source values and target trajectories do not enter this path. Caption-only
generation disables pose inputs. Predictions are not hard-copied from supplied
poses; evaluate both supplied-frame error and boundary smoothness. This option
currently supports only `single_step` and requires training/fine-tuning before
judging quality. When evaluating its checkpoint, point `TEST_CONFIG_PATH` at
that run's saved config so the pose-conditioning flag is restored.

Neither command above is started automatically by editing the configuration.

The default is **four visible valid frames**, independent of clip length.
Evaluation emits separate modes such as `key_framing_k1` and
`key_framing+prompt_k1`, through K=26. Given the same per-sample seed and valid
frames, smaller sets are subsets of larger sets; both conditioning modes use
the same constraints. Sampling is independent of batch partitioning. A short
sequence uses `min(K, valid_frame_count)` frames. Padding is never exposed as a
constraint, and does not suppress valid decoder outputs.

## Independent semantic metrics

`clip_score` is the mean cosine similarity of **generated-trajectory embeddings
from the fixed reference** to structured target tokens. It is a custom
structured semantic score, not a direct CLIP image-text score. If enabled,
caption top-1 also uses the fixed evaluator's encoding of generated trajectories;
it no longer measures the ablated generator's input encoder.

Native CLaTr independently supplies CLaTr score, FCD, and PRDC. Metrics JSON
records content SHA-256 identities of semantic and CLaTr checkpoints, the
resolved semantic architecture and CLIP model, and evaluator fingerprints under
`evaluation_provenance`. Changing an ablated generator must not change those
fingerprints. Comparative reports reject incompatible evaluator fingerprints
and mixed legacy/new runs. Legacy-only reports are labelled as unverified.

## Geometry and framing

Each mode also reports frame-weighted means and an eligibility count for each
quantity. Missing measurements are omitted, with count zero; absence is never
reported as perfect accuracy. These geometry means currently have no bootstrap
confidence intervals. `bootstrap_std` applies to the existing learned metrics.

| Metric suffix | Meaning |
| --- | --- |
| `position_error` | Mean Euclidean position error against the reference, in dataset world units. |
| `rotation_error_deg` | Mean SO(3) geodesic angle against the reference, in degrees. |
| `keyframe_position_error`, `keyframe_rotation_error_deg` | The same errors on actual constrained frames only. |
| `hidden_position_error`, `hidden_rotation_error_deg` | The same errors on valid unconstrained frames only. |
| `bbox_size_error`, `bbox_center_error` | L2 error between projected subject-box size/center and the reference-camera projection. |
| `subject_bbox_width`, `subject_bbox_height` | Projected box dimensions as fractions of image dimensions. |
| `subject_bbox_center_x`, `subject_bbox_center_y` | Projected center in image fractions, origin at bottom-left. |
| `out_of_frame_rate` | Fraction with any box part outside the image, or crossing/behind the near plane. |
| `behind_camera_rate`, `near_plane_violation_rate` | Fractions with any subject-box corner behind the camera or failing the near-plane constraint. |
| `invalid_generated_pose_rate` | Fraction of otherwise valid frames with non-finite generated poses. |

Every metric has a `<metric>_count` denominator. Keyframe and hidden metrics
are only defined for keyframe modes. They measure adherence to the supplied
reference, not a guarantee of exact constraint satisfaction by the network.
Intentionally cropped close-ups may have a high `out_of_frame_rate`; interpret
it together with the requested shot and reference framing.

Simulation pose conventions are camera-to-world Euler XYZ in radians, looking
along local -Z with +Y up. The subject volume is a rotated, centered VBox.
Geometry is measured in denormalized coordinates. LensCraft constraints are
evaluated on the generator's native grid; semantic features use the fixed
evaluator's grid. Baseline geometry uses the common reference grid.

Simulation samples retain actual focal length in millimeters and aspect ratio
as `camera_intrinsics`, sampled on the same timeline as camera/subject poses.
Projection follows the simulator's 35 mm film gauge and a 0.1 world-unit near
plane. The **same reference calibration** is applied to generated and reference
poses: these metrics assess extrinsic camera paths, not predicted optical zoom
or scene occlusion. Boxes crossing the near plane contribute to violation rates
but are excluded from size/center errors; the associated counts reveal this.

When a dataset has no calibration, framing metrics are unavailable unless an
explicit shared projection is provided:

```bash
python src/test.py geometry_metrics.vertical_fov_deg=50 geometry_metrics.aspect_ratio=1.77777778
```

Actual stored calibration takes precedence over this fallback. Within mixed
batches, missing calibration remains excluded with its own count. Use
`geometry_metrics.enabled=false` to disable geometric measurements entirely.

## Reports and cache

Alongside the existing tables, aggregation writes:

- `tables/table6_keyframes.md`: K, semantic scores, constrained/hidden errors
  and counts, including ablation variants.
- `tables/table7_geometry.md`: geometric measurements for models and modes.
- `tables/all_metrics.csv`: full-precision geometric values/counts, K, seed,
  and evaluator fingerprints alongside the learned metrics.

The trajectory cache stores the actual source mask and padding mask per mode.
Cache version 6 invalidates results from the former 26/30 default and old
generator/reference coupling. Replay verifies masks against requested K,
padding, and deterministic sampling, then recomputes semantic and geometric
metrics. It never reuses ablated-encoder semantic features.

LensCraft additionally versions its generation protocol for camera-memory
calibration. Cache identity includes the normalization setting, active
vocabulary/metadata files, and relevant saved configuration inputs. Calibration
changes therefore cannot replay the former raw-memory trajectories. Metrics JSON
records the actual applied per-slot scales under `generation_provenance`;
semantic/CLaTr evaluator identities remain separate. Use separate result/run
directories when comparing raw and calibrated generation, and do not resume a
paper run created from an older source/configuration manifest.

Historical tables must be regenerated with the fixed evaluator and the chosen
keyframe protocol before being compared to these results.
