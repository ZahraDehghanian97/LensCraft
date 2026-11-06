# ET adaptation and sparse conditioning

Scope note, **November 6, 2026**: this document preserves the dated adaptation
experiments and the earlier October scratch recipe. For the current component
map and the separate, still-running November robust scratch experiment, see
[Training and evaluation map](training_and_evaluation.md). The historical
`high_level_scratch` configuration below is not the November run's launcher;
final CCDM/ET results for that run are pending.

The September 16 review found that missing structured fields and the simulation
to ET domain change both matter. `training=et_robust` enables simulation training
with grouped field removal, explicit unknown-field tokens, deterministic sparse
validation, and motion-aware checkpoint scoring. Full captions still supervise
the encoder; removed and natively absent fields do not supervise a caption-driven
cycle. Camera-only cycles retain their complete camera-derived targets.

The model option `caption_presence_conditioning=true` replaces missing field
values with a fixed embedding-space marker before blending camera and caption
memory. Slot identity remains available, and a known zero/false value remains
different from an unknown one. No new checkpoint weights are introduced. Old
model configs default to `false`, preserving their generation behavior. Enable
this option during fine-tuning and retain the emitted inference config; toggling
it on for an old checkpoint is not a validated inference improvement. Structured
generation then requires a boolean `[batch, fields]` `caption_presence_mask`
(the dataset's `prompt_none_mask`); the shared generation/evaluation/UI paths
forward it automatically. Pooled free-text tokens do not need this mask.

Use a fresh output directory and the simulation checkpoint's original dataset
and CLIP cache. The paired runner starts both variants from the same checkpoint,
uses equal budgets, and selects on the same simulation validation objective:

```bash
python scripts/run_et_robustness_experiment.py \
  --repo "$PWD" --checkpoint "$INITIAL_CHECKPOINT" \
  --output "$RUN_ROOT/simulation_pilot"
```

Its `dense` variant has no field removal; `sparse` removes fields from 75% of
caption-conditioned examples. Semantic aliases are removed together, including
movement type/direction and endpoint setup fields. Endpoint aliases are grouped
conservatively because `setup.kind` can swap which endpoint an alias describes.
The masks are synthetic, not sampled from ET test annotations. This recipe uses
a constant `1e-5` learning rate. Explicit
`initialize_from_checkpoint` starts a new optimizer; it is not a resumed run.

For supervised ET adaptation, prepare explicit training and validation splits.
The suggested replay comparison uses equal ET/SIM loss weights, not a claim
that 50/50 is optimal:

```bash
python scripts/prepare_et_adaptation.py \
  --dataset-root "$ET_NATIVE_DATA" --annotations "$ET_TRAIN_ANNOTATIONS" \
  --excluded-cohort "$ET_REVIEW/pure_fresh/manifest.json" \
  --output "$RUN_ROOT/et_adaptation_data"

python scripts/train_et_adaptation.py \
  --base-config "$INITIAL_TRAINING_CONFIG" --checkpoint "$INITIAL_CHECKPOINT" \
  --data-dir "$RUN_ROOT/et_adaptation_data" --output "$RUN_ROOT/et_adaptation" \
  --epochs 10 --train-batches 100 --batch-size 64 \
  --conditioning-val-batches 0 \
  --simulation-replay-ratio 0.5 --simulation-val-batches 8
```

Preparation intersects native `pure_train` with available annotations, excludes
all native `pure`, `mixed`, `full`, and `clatr` test IDs and explicitly supplied
cohorts, and separates source videos. Train/validation grouping also uses source
videos. The manifest records IDs, hashes, overlap checks, and repeated caption
texts; it does not claim caption-disjoint evaluation. Legacy lists with several
prompt entries retain the loader's index-0 interpretation. Malformed content is
explicitly audited and excluded before splitting. No annotations are translated
or semantically corrected by this helper.

The training runner consumes the prepared ET train and validation lists directly.
It uses ET native prompts, mixed conditioning, the original simulation
normalization, and a fresh optimizer. Both `--val-batches` and
`--conditioning-val-batches` now default to zero, covering the whole ET validation
split. Positive values explicitly bound a smoke test or pilot and are recorded.

`--simulation-replay-ratio` defaults to zero for an ET-only training control.
A positive value computes one ET batch loss and one original-SIM-train batch loss
per update, combining them as `(1-ratio)*ET + ratio*SIM`. SIM replay uses grouped
caption removal by default; `--no-replay-caption-missingness` disables it. ET
keeps its native sparse annotations. The original simulation dataset inventory,
split seed, train/validation sizes and indices are saved in `simulation_splits.json`;
SIM test samples are never replayed. Changing the adaptation seed does not change
those original memberships.

Even with replay disabled, a fixed prefix of the original SIM validation split
is evaluated independently. Before optimization, both domains are evaluated to
establish the checkpoint's SIM score under the same input/metric settings.
Selection minimizes `val_adaptation_score = ET_score + weight*max(SIM_score -
initial_SIM_score, 0)`, with `--simulation-validation-weight=1` by default.
The SIM reference is saved with the checkpoint. A weight of zero selects on ET
only while still logging SIM validation. The sample counts and initial scores
are saved in the run manifest. This is a measured penalty, not a guarantee that
every general-domain behavior is preserved.

The ET runner enables explicit presence conditioning by default. For an input
representation ablation use `--no-caption-presence-conditioning`; the choice is
saved in both training and inference configs.

`validation_conditioning.motion_score_weight` augments the usual position and
absolute-rotation score with positional step error, relative-to-first SO(3)
rotation error, adjacent SO(3) increment error, and per-clip angular-path error.
It is one in the ET runner and `et_robust` recipe, zero in legacy/default configs.
The two opt-in diagnostic switches `rotation_baselines` and `caption_shuffle`
log fixed-SIM-zero/static-first-reference rotation baselines and a deterministic
paired caption-shuffle gap. The oracle baseline is reference-assisted. Shuffle
diagnostics do not themselves select checkpoints or establish semantic fidelity.

`training_config.yaml` records the actual ET training settings.
`inference_config.yaml` preserves the original model and simulation
normalization configuration and identifies the adaptation run and selected
checkpoint. The evaluator uses the latter:

```bash
python scripts/evaluate_et_robustness.py \
  --train-config "$RUN_ROOT/et_adaptation/inference_config.yaml" \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --et-config "$ET_REVIEW/pure_fresh/evaluation/hydra/.hydra/config.yaml" \
  --output "$RUN_ROOT/et_evaluation" --keyframes 1 4 --save-trajectories
```

The evaluator records absolute pose errors, relative rotations, temporal motion,
subject look angles, per-clip results, provenance, and source-camera invariance.
It also includes constant SIM-zero poses, generated positions with fixed SIM-zero
rotation, frozen first generated poses, and an explicitly named oracle frozen
first reference pose. These baselines are built in denormalized coordinates;
SIM Euler zero means standard/OpenCV `diag(1,-1,-1)`, not CV identity.
Shuffled captions move presence masks with their tokens. Relative orientation
uses the reference only in scoring and must not be reported as absolute
unconditioned generation quality. Keyframe modes supply reference poses and are
soft conditions for the original checkpoint architecture.

ET adaptation changes the protocol from simulation-only transfer to supervised
ET learning. Always report that distinction and preserve the original evaluator
checkpoints when comparing learned metrics. The reviewed 2,135-clip cohort is a
selected subset, not the official complete ET benchmark.

The earlier September 16 ET run selected only on ET absolute pose scores and
used legacy mean-filled captions. Its results are historical and do not measure
the fixes above. New weights must be trained and evaluated before claiming an
ET quality improvement. Annotation semantics, subject-facing estimation, and
new human-motion simulation data remain separate experiments.

## Simulation-only transfer robustness for human-motion data

`training=transfer_robust` extends `et_robust` with synthetic views that expose
the subject information available in ET and CCDM. It trains and selects
checkpoints using only simulation train/validation examples. It does not use
native ET or CCDM test cameras, annotations, or test-derived caption masks.
This protocol is separate from the supervised `train_et_adaptation.py` runner
described above.

Use the checkpoint's architecture, CLIP cache, split settings, and original
normalization source when applying the new recipe. For example, add these
overrides to the usual training invocation:

```bash
python src/train.py training=transfer_robust \
  initialize_from_checkpoint="$HUMAN_MOTION_CHECKPOINT" \
  checkpoint_monitor=auto \
  hydra.run.dir="$RUN_ROOT/transfer_robust"
```

`initialize_from_checkpoint` initializes weights with a fresh optimizer and
epoch counter. The recipe inherits `et_robust`'s constant `1e-5` learning rate,
explicit caption-presence conditioning, grouped caption removal, and four
conditioning families. Keep the desired training budget explicit in the run's
Hydra overrides; the recipe itself does not set an epoch limit.

Each training example independently receives one of three views. The default
probabilities are 50% original simulation, 25% ET-information, and 25%
CCDM-subject-relative. These are sampling probabilities, not an exact count in
each batch. The original fraction must remain positive. The transformation
preserves the input batch and padded values, and never copies target camera
information into subject conditions.

| View | Geometry and available subject information | Caption fields removed |
| --- | --- | --- |
| Original | Original camera, subject poses, and dimensions | Only the existing optional decoder caption dropout |
| ET-information | Camera targets and subject XYZ stay unchanged; subject orientation becomes identity in standard coordinates, and dimensions become `[0.5, 1.7, 0.3]` | Camera-angle/view, shot-size, and framing endpoint aliases; subject-aware interpolation, full-box visibility, and fixed camera/subject rotation claims |
| CCDM-subject-relative | Both camera and subject are expressed in the moving subject frame; subject position becomes zero, orientation becomes the native CCDM yaw basis `diag(-1, 1, -1)`, and dimensions become `[0.5, 1.7, 0.3]` | All motion, speed, interpolation, and constraint labels, plus shot-size and framing endpoints; only subject-local camera-angle/view endpoint aliases and `setup_kind` survive |

The simulator defines camera angle and subject view in subject-local
coordinates, so they survive the coherent CCDM frame change. Its framing uses
projected bounding boxes, including dimension-dependent outer positions;
preserving the projected subject origin is insufficient after dimensions are
replaced. Motion labels cannot survive an arbitrary frame change at each time
step: a camera following a translating or turning subject can become static in
the relative representation.

Domain-invalidated fields are removed from both decoder conditions and encoder
supervision using absent-field masks and neutral token fills. Their annotation
metadata also becomes missing. This differs from ordinary caption dropout,
which keeps valid clean encoder supervision. Metadata-based contrastive loss
is unsupported by `TransferRobustTrainer` and rejected if enabled.

With `checkpoint_monitor=auto`, checkpointing and early stopping minimize
`val_transfer_score`. Alongside clean/sparse simulation validation, the trainer
evaluates deterministic ET and CCDM synthetic views of the simulation validation
split. Defaults use four validation batches, seed `27182`, and four supplied
keyframes for each keyframe mode. Each proxy scores prompt generation,
reconstruction, keyframing, and keyframing with a prompt, with equal family
weights. Sparse-caption removal also runs on the caption-conditioned proxy
cases. Scores include absolute pose, temporal motion, and known-keyframe errors
according to the inherited validation settings.

For `validation_domain_augmentation.score_weight = w`, selection is:

```text
val_transfer_score = (val_robustness_score + w * (ET_proxy + CCDM_proxy)) / (1 + 2*w)
```

`w` defaults to one. Per-domain scores, per-mode metrics, sample counts, and
training view fractions are logged separately. Proxy validation is deliberately
bounded; it is not native ET/CCDM validation or a complete transfer benchmark.

Geometry transforms run after denormalization and are normalized again using
the checkpoint's original simulation statistics. Keep
`SIMULATION_NORMALIZATION_DATA_PATH` pointed at that reference when training on
a different simulation inventory; `SIMULATION_DATA_PATH` can identify the new
human-motion inventory. Preserve these references in the saved training and
inference configurations. For comparison, keep the old model's inference
settings and the learned FCD/CLaTr evaluator checkpoints and normalization
fixed. Native ET/CCDM conversion and test annotations are unchanged.

The CCDM proxy preserves the transformed full camera orientation, including
roll. Native CCDM's five-dimensional representation has a constrained,
roll-free orientation reconstruction; this augmentation does not project the
simulation cameras through that representation. It therefore addresses the
subject-frame/input mismatch without making simulation motion, camera style,
or orientation distributions identical to native CCDM. ET still has its native
annotation and acquisition differences. Report native Simulation, ET, and CCDM
test results before claiming improvement, and treat a small test subset as a
pilot, especially for distribution metrics such as FCD.

## Optional initial-camera training coordinates

`training=transfer_canonical` extends `transfer_robust` with
`domain_augmentation.et_coordinate_frame=first_camera`. The default
`transfer_robust` recipe retains its existing `world` coordinates. Launch the
new recipe with full annotation metadata, because movement-specific masks need
the original simulation labels:

```bash
python src/train.py training=transfer_canonical data.compact_batches=false \
  initialize_from_checkpoint="$HUMAN_MOTION_CHECKPOINT" \
  checkpoint_monitor=auto \
  hydra.run.dir="$RUN_ROOT/transfer_canonical"
```

For each selected ET-style simulation view, let `R0` be the first valid
camera's OpenGL rotation and `s0` the subject position at that frame. In
denormalized coordinates, one constant transform applies to the entire clip:

```text
Q = transpose(R0)
camera_position'(t) = Q * (camera_position(t) - s0)
subject_position'(t) = Q * (subject_position(t) - s0)
camera_rotation'(t) = Q * camera_rotation(t)
```

The first valid camera rotation becomes OpenGL identity, and the first subject
position becomes zero. Camera-local subject rays, distances, and relative
camera rotations are preserved. This does not align each frame independently
or place the first camera at the origin. Subject rotations are then replaced
with identity in standard coordinates and dimensions with `[0.5, 1.7, 0.3]`, matching the ET adapter's
unknown-heading and default-size inputs. Identity is a placeholder, not an
estimate of the person's actual facing direction. Padding is preserved, and
the original simulation normalization remains in use.

The existing ET endpoint/subject-geometry masks still apply. Static, dolly,
truck, pan, tilt, and Dutch motion labels survive the common rotation. Pedestal,
arc, and crane labels depend on the simulator's world-up axis; their
`movement_type`, `dynamic_direction`, and `dynamic_movementMode` fields are
removed. Follow/track labels and those aliases are also removed conservatively
because their rear/side distinction uses the subject heading that is hidden.
All directional movement/rotation lock fields are removed; rotation locks in
the generator use XYZ Euler components, which are not invariant to this frame
change. Speed, acceleration, easing, and constant-distance labels remain.
Removed labels receive absent masks, neutral tokens, and missing annotation
metadata for both conditioning and supervision.

Only simulation training examples provide these synthetic training views;
checkpoint selection uses the original simulation validation split and its
synthetic views. Simulation test examples and native ET/CCDM evaluation data
must not enter optimization or selection. The transform uses the simulation
target camera to choose a common training coordinate frame. Native converters
and inference remain unchanged: generation receives no reference camera for
canonicalization, and no oracle alignment is applied to its output.

This is a transfer hypothesis, not an exact reproduction of ET preprocessing.
ET's published [dataset-centering step](https://github.com/robincourant/the-exceptional-trajectories/blob/main/scripts/misc/dataset_shifting.py)
subtracts the first subject position and leaves camera rotations unchanged;
it does not explicitly rotate every clip into its first camera frame. Native
ET starting orientations need not be exactly identity. Train and evaluate new
weights with frozen evaluator settings before claiming any quality gain, and
report this recipe separately from supervised ET adaptation.

### September 30 simulation-only diagnosis and paired pilot

The frozen mixed-ET cohort in the six-panel plot contains 6,423 clips. It is
different from the later 2,350-clip pure-ET report. The controlled missing-field
test held the same 64 simulation cameras, subjects, and dimensions fixed:
removing the endpoint fields normally absent in ET increased the original
checkpoint's rotation error from 10.74 to 89.66 degrees. Replacing absent tokens
with zero still gave 87.17 degrees. This isolates a conditioning failure without
changing any coordinate conversion. On ET, perturbing the decoder's raw rotation
matrices by relative magnitude `1e-3` changed projected rotations by only 0.053
degrees on average; numerical SVD instability does not explain the roughly
87-degree task error. Native ET does not provide subject heading, so the
adapter's identity subject rotation must not be interpreted as measured facing.

Two 192-update continuations started from the same simulation-only robust
checkpoint, seed, learning rate, and train split. Both selected checkpoints on
the same simulation validation objective, including a synthetic first-camera
view. The first-camera arm changes both coordinates and the validity masks of
affected captions; this is not a geometry-only ablation. Neither native ET
training nor native ET checkpoint selection was used. The common validation
score was 1.4670 for world coordinates and 1.2629 for first-camera coordinates.

The subsequent CPU evaluation fixed all four checkpoints before opening native
test data. It used all 6,423 mixed-ET clips, a deterministic 512-clip subset of
the original simulation test split, the original normalization, and the same
CLaTr epoch-050 checkpoint. Lower is better in the following table:

| Model | ET rotation error (degrees) | ET position error | ET FCD | SIM-512 FCD |
| --- | ---: | ---: | ---: | ---: |
| Original human-motion epoch 096 | 86.758 | 1.886 | 926.059 | 33.654 |
| Existing simulation-only robust epoch 029 | 51.235 | 2.170 | 838.945 | 31.713 |
| Paired world continuation | 52.605 | 2.230 | 840.627 | 33.944 |
| Paired first-camera continuation | 41.890 | 2.469 | 870.633 | 35.571 |

These are fresh CPU measurements. The original published CUDA FCD was 926.379;
the 0.320 difference is numerical, while the regenerated geometric scores match
to within 0.0001 degrees and `1e-7` position units. Source-camera independence
checks had exactly zero output difference for every model in both domains.

The pilot improves absolute orientation but does **not** solve the distribution
shift. For the original model, the FCD centroid/covariance contributions are
669.106/256.953; for the first-camera arm they are 512.279/358.354. Total generated
feature variance changes from 0.617 to 1.417 times reference variance. The new
position path length is 2.640 versus reference 0.741, while its angular path is
10.856 versus reference 34.629 degrees. A fixed OpenGL-zero rotation has only
16.059 degrees of absolute error on this ET cohort. Correct captions also do
not improve the first-128 rotation score over the paired caption shuffle.
Consequently, reduced angle error alone is insufficient evidence of motion
fidelity or successful text-conditioned transfer.

The implementation remains opt-in and the pilot is not a validated replacement
for the published model. Its expected first-camera prompt-only exposure is
small (approximately 384 examples); this experiment does not establish the
best achievable result of longer simulation-only training. The deterministic
decoder and squared pose losses can also average multiple valid camera setups
when setup fields are absent. That mechanism is consistent with the measured
rotation-matrix shrinkage, but is not uniquely established against general
out-of-distribution prediction. Further work needs simulation-only selection
that measures sparse-input distribution and motion as well as pose error;
native ET results must remain evaluation evidence.

Server artifacts are under
`runs/et_coordinate_repair_simonly_20260930/{world,first_camera,evaluation_native_full}`.
Local receipts, full metrics, hashes, and the feature decomposition are under
`../performance_reports/cross_dataset_20260928/coordinate_repair_receipts/`.
The new coordinate/masking behavior and existing transfer behavior passed 68
targeted tests on the server. The frozen published evaluation was preserved.

## Optional high-level-only caption conditions

`training=high_level_sparse` extends the simulation-only `transfer_robust`
recipe in world coordinates. It adds
`caption_missingness.profile=high_level_only`; existing recipes retain
`alias_grouped`, including their previous random draws and alias-mask behavior.
Use the checkpoint's original architecture, simulation normalization, CLIP
cache and train/validation split, and a fresh output directory:

```bash
python src/train.py training=high_level_sparse \
  initialize_from_checkpoint="$HUMAN_MOTION_CHECKPOINT" \
  checkpoint_monitor=auto \
  hydra.run.dir="$RUN_ROOT/high_level_sparse"
```

The inherited probability selects 75% of caption-conditioned training examples
for removal. Selected decoder conditions omit every low-level simulation
instruction field: setup, dynamic program, and constraints. They retain an
observed high-level `movement_type` and `movement_speed`; unobserved fields stay
absent. Endpoint descriptions are sampled independently using the configured
keep ratio and endpoint-drop probability. They need not retain a matching
low-level alias. The remaining 25% retain their incoming caption fields, subject
to any earlier domain-validity masks. Forced sparse validation applies the
profile to every example it evaluates. These percentages are sampling
probabilities, not exact batch counts.

This operation only removes supplied information. It never adds a movement or
speed that the input presence mask marks absent, and it preserves existing
missing-field fills. An optional training-only mask bank remains an upper bound:
its low-level fields are discarded, while absent high-level fields are not
restored. `preserve_movement_type=false` allows the high-level type to be sampled
instead of always retained; the profile still retains observed speed unless
the supplied bank excludes it.

Valid clean captions continue to supervise the camera encoder. Decoder presence
and prompt-conditioned cycle masks receive the same reduced mask, so the cycle
does not enforce an omitted simulator program. Earlier domain invalidations
still remove unsupported fields from encoder supervision as well. Camera-only
conditioning retains its existing behavior.

Training and checkpoint selection use only simulation training/validation data
and their synthetic views. This recipe inherits the existing transfer score;
it does not introduce native ET selection, tune a direction or scale correction,
or report retrained performance. No quality improvement is established until
new weights are trained and evaluated with frozen metrics. The profile addresses
incomplete descriptions; it does not create compound camera motions or change
the meaning of motion classes. Adding simultaneous translation and rotation,
separate movement extent and duration, and varied optics requires additional
simulation data and, where necessary, a richer input/output contract.

### October 1 motion-contract review

The training review separates three issues: missing input information, a
translation-contract defect, and differences between motion definitions. The
Gateway previously encoded dolly/arc/crane as interpolation, while ten original
training samples per motion confirmed that generator v5 uses simple movements
with explicit mode and direction. The Gateway prompt and contradiction checks
now follow that training contract; see [gateway_translation.md](gateway_translation.md).

Native ET's upstream segmentation computes signed camera-local translations
and leaves angular velocity out of its 27-class segmentation input. Thus an ET
translation class alone does not establish locked camera rotation. This is
separate from whether a free-form caption explicitly mentions panning. SIM truck
locks rotation; native ET trucking examples can include substantial rotation.
ET boom is camera-local Y versus SIM pedestal's world Y, and ET push/pull is
camera-local Z versus SIM dolly's subject-relative distance behavior. A global
axis swap cannot reconcile these definitions. Primary sources and existing
cohort statistics are recorded in
`../performance_reports/et_training_review_20261001/native_motion_semantics.json`.

A separate, fixed-checkpoint input experiment repaired only the three dynamic
alias fields for six existing dolly/arc/crane labels in memory. It held their
high-level labels and all other inputs fixed; it did not reinterpret source
captions or train a model. This changed 2,394 of the 6,423 mixed-ET inputs. The
original normalization, CLaTr checkpoint, CPU device, cohort order and metric
protocol matched the September 30 evaluation:

| Fixed model | ET FCD before / aligned aliases | ET rotation degrees before / after | ET position error before / after |
| --- | ---: | ---: | ---: |
| Original human-motion epoch 096 | 926.059 / 931.008 | 86.758 / 89.710 | 1.886 / 1.920 |
| Existing SIM-only robust epoch 029 | 838.945 / 934.906 | 51.235 / 53.559 | 2.170 / 2.335 |

Alias consistency alone did not improve either checkpoint. This input variant
is not a replacement benchmark result. The separate SIM-512 control was exactly
unchanged; unchanged ET rows passed exact token/presence checks, and source-camera
independence remained exact. Historical annotations and results were preserved.
Artifacts are under server `runs/et_training_review_20261001/alias_native_full`
and local `../performance_reports/et_training_review_20261001/remote_receipts/`.

At the end of this review, `high_level_sparse` was an untrained opt-in proposal,
not a measured quality gain. A useful simulation-only comparison would hold the initial
checkpoint, train/validation memberships and update budget fixed, evaluate
prompt-only quality separately for each motion class, and check motion extent,
rotation and distribution along with dense-SIM retention. Existing temporal
and SO(3) losses are already active. Under default selection weights, prompt-only
cases receive 37.5% of the total weight, and the ET-style prompt case alone gets
1/12; improvements in reference-conditioned tasks can therefore dominate.

The paired pilot runner now includes its validated initializer as a selection
candidate and gives it ties. Previously the world arm selected a trained score
of 1.467006 despite an initial score of 1.466112; historical receipts still show
that original result. Future runs record both candidates and select a trained
checkpoint only for strict finite SIM-validation improvement. This guard is in
the paired runner, not automatically added to the general `src/train.py` recipe.

Validation: 86 targeted tests passed on the isolated server source, including
the historical mask/RNG regression, new presence and encoder/cycle gradient
checks, movement aliases and geometry. Five pure checkpoint-selection tests
also passed, and Hydra composition confirmed the new opt-in configuration.

### October 1 training from random initialization

The requested full run uses `training=high_level_scratch` in the isolated server
directory `runs/simonly_scratch_20261001`. It initializes the 41.8M-parameter camera
model randomly; it does not continue either previous checkpoint. CLIP vocabulary
embeddings and the original independent simulation normalization remain fixed.
The original seed-42 manifests define 60,000 training, 20,000 validation and
20,000 held-out test clips. Native ET/CCDM data enters neither optimization nor
checkpoint selection.

The budget is 100 epochs, batch size 128, BF16 mixed precision and early stopping
after 20 validation checks without improvement. The original scratch scheduler
is restored, with ten optimizer warmup steps and peak learning rate 3e-4.
Conditioning draws pure-prompt batches with probability 50%; caption omission
uses the `high_level_only` profile described above. Existing synthetic domain
augmentation remains in the world frame. This run does not add new compound
motion simulations or redefine native ET motion classes.

Checkpoint selection uses `val_transfer_prompt_score`: equal weights for clean,
sparse, ET-shaped and CCDM-shaped **simulation** prompt diagnostics. Each uses
the fixed first 2,048 validation examples; ordinary validation loss covers all
20,000. The random initializer is eligible and wins ties. Only a finite, strictly
better trained candidate replaces it. Epoch-boundary recovery restores this
run's optimizer, scheduler and loader state; bit-exact global RNG recovery is
not claimed.

The detached supervisor automatically evaluates the selected model on the
complete original SIM20,000, mixed ET6,423 and CCDM27,810 cohorts, using all
13 conditioning modes. Original ET-pure2,350 is evaluated separately. CLaTr
epoch50, semantic evaluator, captions, cohort order and numerical metric
definitions stay fixed. Geometry summaries, cached baseline comparisons,
batch-one efficiency at 30 and 300 frames, CSV and Persian/English reports are
required before the pipeline can mark completion. The 300-frame efficiency
case is a length extrapolation, not an evaluated quality claim.

The four-update GPU integration check and 135 server tests passed before
launch. A supervisor interpreter-path error occurred before training began;
its receipts were preserved in `launch_attempt_01_environment_failure`, and
the virtualenv path handling was corrected and regression-tested. Live status
is `runs/simonly_scratch_20261001/status.json`; a launched or running state is
not a completed evaluation or evidence of improved ET transfer.
