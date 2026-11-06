# Training and evaluation map

This guide separates the reusable repository code, earlier experiment recipes,
and the full robust training run started on **November 6, 2026**. That run is
**in progress**; its final SIM, CCDM and ET results are pending. The changes here
prepare the implementation for review and do not establish a quality gain.

## Repository recipes

Hydra selects these configurations with `training=<name>`. They retain distinct
experiments rather than describing successive replacements for the default.
Use the original normalization and CLIP vocabulary when continuing a checkpoint.

| Configuration | Parent | Purpose |
| --- | --- | --- |
| `et_robust` | `default` | SIM-only continuation with grouped caption removal, explicit missing-field tokens and sparse validation. Despite its name, it does not train on native ET. |
| `transfer_robust` | `et_robust` | Adds ET-shaped and CCDM-shaped SIM views and corresponding SIM validation diagnostics. |
| `transfer_canonical` | `transfer_robust` | Re-expresses the ET-shaped training view in the first-camera coordinate frame. |
| `high_level_sparse` | `transfer_robust` | Retains available high-level facts while omitting simulator instructions. |
| `high_level_scratch` | `high_level_sparse` | The **October 1** random-initialization recipe: original scratch scheduler, more prompt-generation batches, and four SIM prompt validation views. |
| `high_level_ccdm` | `high_level_scratch` | SIM-only CCDM-focused continuation with more CCDM-shaped views and compatible motion-caption retention. |

`high_level_scratch.yaml` is the earlier October recipe. It does **not** configure
the November robust run described below. The comments in each configuration
specify required launch overrides; for example, the scratch recipe requests
`checkpoint_monitor=val_transfer_prompt_score`, whereas the CCDM-focused recipe
requests `val_ccdm_prompt_score`. First-camera and metadata-dependent motion
augmentation require `data.compact_batches=false`.

| Trainer | Responsibility |
| --- | --- |
| `LightningLensCraft` | Standard training, conditioning-family validation, sparse-caption validation and diagnostic scores. |
| `TransferRobustTrainer` | SIM-only domain augmentation and clean/sparse/domain validation objectives. |
| `CCDMFocusTrainer` | Uses one fixed validation augmentation when comparing training recipes that retain different CCDM motion fields. Keep this comparison helper even though ordinary Hydra recipes do not select it directly. |
| `ETAdaptationTrainer` | Supervised ET adaptation with optional original-SIM replay and an explicit SIM-regression penalty for checkpoint selection. |

The shared pose metric reducer preserves additive sums/counts across batches
and ranks. Motion-aware scoring includes position steps, relative rotations,
rotation increments and angular path error; supplied keyframe poses are tracked
separately from hidden poses.

## Scripts and supporting changes

| Entry point | Use |
| --- | --- |
| `src/train.py` | General Hydra training entry point for the repository recipes. |
| `scripts/run_et_robustness_experiment.py` | Paired dense/sparse SIM-only continuation from the same checkpoint. |
| `scripts/prepare_et_adaptation.py` | Creates explicit ET train/validation manifests with source-video grouping and evaluation-cohort exclusions. |
| `scripts/train_et_adaptation.py` | Runs supervised ET adaptation or its explicit ET-from-scratch variant; writes the inference configuration used by evaluation. |
| `scripts/evaluate_et_robustness.py` | Evaluates a fixed checkpoint on an ET cohort and optional SIM holdout with pose/motion diagnostics. |
| `scripts/prepare_ccdm_evaluation.py` | Pins CCDM sample IDs and complete source captions, then validates separately generated structured annotation provenance. |

The accompanying source changes cover CCDM projection and subject-basis
conventions, structured caption availability, simulation observation policies,
normalization-source selection, nested device transfer, and optional geometry
losses. Their regression tests live under `tests/`. Native caption text is
translated or parsed into structured CLIP fields; the generator still consumes
structured tokens, not an added full-text encoder.

See [CCDM evaluation](ccdm_evaluation.md), [metric definitions](evaluation.md),
[Gateway translation](gateway_translation.md), and
[third-party patches](third_party_patches.md) for the corresponding contracts.
The dated experiments and commands remain in
[ET adaptation history](et_adaptation.md).

## Opt-in robust scratch components

`src/training/robust_scratch/` brings three components from the November
experiment into the repository:

| Component | Responsibility |
| --- | --- |
| `data_mix.py` | Deterministic natural/primitive batch scheduling and resume identities. Natural examples remain in the epoch; primitive examples are additional exposures. |
| `objective.py` | `RobustForward` and `RobustObjective`: generate first, then supervise a separate clean-target encoder branch used only by the loss. Cycle encoding sees all valid generated frames and uses detached targets under the effective caption-availability mask. |
| `native_motion_tokens.py` | Fixed `nativePan`/`nativeOrbit` features in `movement_type`, with the frozen text-only parser in `caption_bridge.py` and pinned JSON assets. Unambiguous native Pan/Orbit captions change that one slot; the other 45 fields remain intact. |

These are **opt-in building blocks**. Importing them does not switch
`BaseTrainer`, register a new default trainer, or change any Hydra recipe. The
parser and token assets accompany the port; historical paths inside artifact
provenance are records, not executable imports. The focused tests are named
`tests/test_robust_scratch_*.py`.

There is not yet a portable full-training launcher for this recipe in Git. In
particular, selecting `training=high_level_scratch` does not activate these
components or reproduce the November run.

## November full run and its artifacts

The active run, `robust_scratch_20261106`, uses an isolated, frozen source and
helper package on `me_DML4_proxy`. Its source snapshot already differed from the
working repository in **10 files before this cleanup**. The cleanup and ports
must not be described as the exact code currently training. Its pinned manifests
remain the authority for reproducing that execution; the running snapshot is
left unchanged.

The [prospective experiment protocol](experiments/robust_scratch_20261106.protocol.json)
is copied byte for byte into this repository for review. It records the data,
loss, selection and evaluation contract; it is a protocol record rather than a
Hydra launcher configuration. Execution receipts and the complete frozen
dependency chain remain with the external run.

The declared training budget is one random initialization, seed 42, **100 full
epochs and 60,900 optimizer updates**, with no early stopping. Every epoch
retains all 68,192 natural examples: the original 60,000 training clips plus
4,096 narrow and 4,096 broad simulations. Each batch adds 16 samples from a
separate seven-family primitive pool; the final partial natural batch is kept.
Subject sizes and general/mixed paths are retained. Native ET/CCDM examples do
not enter optimization or checkpoint selection.

Selection uses the predeclared SIM mean-plus-worst-group objective, considering
the initializer and each complete epoch. Its 17 score groups include an
identical paired CCDM view, so there are 16 distinct conditions; that cohort has
the recorded double weight. This is part of the frozen protocol. Previously
examined native cohorts are transfer diagnostics, not newly blinded tests.

After selection, the suite evaluates **SIM 20,000, CCDM 27,810 and ET 6,423** on
all **13 primary conditioning modes**. A fourth stage evaluates the same model
on CCDM with the original annotations in prompt-only mode: **40 dataset/mode
rows** in total. Prompt-only generation is reported separately from
reference-assisted reconstruction and keyframing. Native metrics do not select
the checkpoint or the primary caption interface. A separate selected-model
motion audit and final integrity checks precede the report.

The workspace mirror is `performance_reports/robust_scratch_20261106/`, beside
the `LensCraft` Git checkout. Its live observer writes
`local_tools/compact_status.json`; final outputs are `RESULTS.md`,
`report/report.json`, and `completion_audit.json`. A live status or a completed
smoke test does not mean the full run or its evaluations are complete. Keep
generated reports, datasets, checkpoints, logs and machine-specific launchers
outside the source commit.

The full frozen dependency chain also remains outside Git: the suite/training/
evaluation/report drivers; their `protocol.json`, `prepare.json` and freeze
receipts; the source snapshot; prior robust-data/model, geometry-loss,
semantic/motion augmentation and generation/audit helpers; raw simulation and
native evaluation data; cohort manifests and annotations; normalization and
CLIP caches; and the frozen semantic evaluator checkpoint. The preparation
manifest pins these dependencies. The three reusable ports alone do not replace
that chain or authorize editing it while training is active.
