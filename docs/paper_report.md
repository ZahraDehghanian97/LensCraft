# Reproducible paper report

`scripts/build_paper_report.py` generates a complete Persian report in the experiment order of the supplied September 11, 2026 LensCraft manuscript. It uses the Python standard library and reads actual measurements; manuscript numerical claims are never imported. Missing experiments are explicitly pending. The template covers the whole paper, while the completion matrix states which experiments have actually finished.

From the repository root:

```bash
python3 LensCraft/scripts/build_paper_report.py \
  --base-run-dir performance_reports/retrain_eval10pct_20260912_191325_96a9cd7 \
  --output-dir performance_reports/paper_report_20260914 \
  --paper-pdf '/Users/reza/Downloads/LensCraft__Your_Professional_Virtual_Cinematographer (2).pdf'
```

Refresh with new measurements using repeatable `--extra-results-dir /path/to/supplement` and optional `--ablation-dir /path/to/ablation_suite`. Nonexistent optional directories remain pending. Source roots are searched recursively for `metrics_*.json`, `efficiency*.json`, and manifests. Keep output in a dedicated directory. Existing report files are regenerated; no training or external communications are performed.

Outputs:

- `REPORT.html`: standalone styled Persian RTL document, with print CSS for landscape A4. Optional figure assets are copied under `assets/`.
- `REPORT.md`: the same report with all main and supplementary tables.
- `tables/table1_*.tex` through `table5_*.tex`: English LaTeX fragments with escaped identifiers, and CSV equivalents for every table.
- `summary.json`: unrounded measurements, tables, completion matrix, human-study status and metric conventions.
- `all_metrics.csv`: every original scalar/count, original estimate, bootstrap mean, bootstrap SD, native CS and optional display-only CS×100, cohort and source path.
- `report_manifest.json`: freshly computed SHA256 hashes of inputs, generator and outputs, evaluator fingerprints and completeness.
- `human_study_template.csv` and `human_study_protocol.json`: human-study input schema and progress.

## Scientific protocol

Table 1 uses only `prompt_generation`, meaning normalized baselines without LensCraft initialization, for static and dynamic **subject** cohorts. E.T. is labeled by actual architecture (`ca`, `adaln`, `incontext`); the manuscript does not map its A/B/C labels.

Table 2 requires `key_framing_random_1_10` and `key_framing+prompt_random_1_10` in addition to prompt generation, reconstruction and hybrid generation. The supplement's `variant=paper_random_k` is a protocol tag and maps to the main model. Fixed K modes cannot substitute. `keyframe_protocol` must contain `version=paper-random-k-v1`, `same_masks_across_modes=true`, and `sampling_manifest_sha256`.

Table 3 requires `variant` values `full_scheduled`, `without_clip`, `without_first_frame`, `without_relative`, `without_speed`, `without_cycle`, `without_teacher`, `without_volume`, and `without_noise`. `full_scheduled` is the independently trained matched schedule-enabled control, distinct from the main mixed-conditioning checkpoint. Dynamic prompt-only is an explicit inference from the manuscript's matching control values. Retain actual suite and resolved training manifests: current objectives/weights and training-time rotation treatment differ from the manuscript. Legacy teacher/noise ablations are ineffective when those schedules are disabled.

Table 4 accepts only benchmark records containing `protocol.same_length_same_batch=true`, positive `seq_length` and `batch_size` (top-level or under `protocol`), and matching length/batch across all accepted models. Measurements use `inference_time_batch_s`, `inference_time_batch_std_s`, and `gflops_per_traj`. Timing is per batch; FLOPs are per trajectory. Record actual device, precision, warmup/repetition counts, text preprocessing scope, unsupported FLOP operators and actual output length. Old native-length measurements appear only in the supplement.

`status=unsupported` efficiency records with null measurements appear as explicit unsupported rows. `flop_count_is_lower_bound=true` is displayed next to counts and detailed limitations are included. Nested completed ablation stage hashes and supplement stage hashes are verified; their manifests and status snapshots are hashed as inputs.

FCD is labeled honestly as Fréchet distance in CLaTr features. CS retains its native stored scale. Point estimates remain separate from bootstrap means and SDs; SD is not a 95% confidence interval. No absent value is replaced by zero.

## Validation

Every metric file needs both evaluator fingerprints under `evaluation_provenance.semantic_evaluator.fingerprint` and `evaluation_provenance.clatr_evaluator.fingerprint`. They must match across the report. Positive sample counts, identical cohort sizes, equal test fractions, coherent holdout totals, static/dynamic partition totals and finite scalar measurements are checked. Conflicting duplicate model/variant/cohort/mode values abort the build; identical duplicate measurements are coalesced with both source paths. Available `status.json` output hashes are checked against the current files. The report states that legacy metric files do not themselves prove identical per-sample IDs; the base run's split manifest/hash provides the available provenance.

```bash
python3 -m unittest discover -s LensCraft/tests -p test_paper_report.py
python3 -m unittest discover -s LensCraft/tests -p test_paper_qualitative.py
```

## Human study

Pass `--human-study-csv ratings.csv` to report **real** ratings. Required columns:

```text
scene_id,evaluator_id,model,prompt_alignment,naturalness,geometric_stability,movement_correctness,framing_correctness,rank
```

Models are `gendop`, `ccdm`, `et_ca`, `lens_craft`. Scores must be finite in [1,10], ranks integers in [1,4], and ranks cannot tie within an evaluator/scene. Duplicate scene/evaluator/model ratings are rejected. The paper design requires 50 distinct scenes × 30 distinct evaluators × 4 models = 6000 rows. Partial valid ratings are shown explicitly as partial. Structural completeness cannot prove random scene selection, genuine independent raters or their backgrounds; keep that evidence separately. The paper's “Winning Rate” is mean ranking position, lower is better.

## Qualitative assets

An optional `qualitative_manifest.json` in an input root can supply deterministic figures generated from actual model outputs:

```json
{
  "figures": [
    {
      "id": "figure6",
      "path": "figures/conflicting_source.png",
      "caption": "Actual generation using the recorded source and conflicting prompt",
      "protocol": {
        "conflicting_prompt": true,
        "sample_ids": [123, 456],
        "checkpoint_sha256": "actual-checkpoint-hash"
      }
    }
  ]
}
```

Accepted IDs are `figure4` (prompt comparison), `figure5` (keyframe constraints) and `figure6` (conflicting reference/text). Paths are relative to the manifest. Figure 6 requires an explicit conflicting-prompt flag. Missing figures remain pending. The report does not fabricate visually plausible trajectories or human responses.

Generate the actual figures on the server using the frozen source and original environment:

```bash
python scripts/export_paper_qualitative.py \
  --project-dir /path/to/frozen/LenseCraft \
  --base-run-dir /path/to/original/run \
  --output-dir /path/to/new/run/qualitative \
  --device cuda:0
```

The exporter replays cached normalized predictions for the first two static and first two dynamic measured holdout samples, generates actual three-keyframe constraints, and generates controlled time-reversed-source conflicts using the same checkpoint. Source and desired trajectories share the original subject and prompt; this controlled test is not evidence of arbitrary scene transfer. It exports SVG/PDF/PNG and portable bundles with errors and provenance, and prepares 50 random heldout sample IDs and private blinded method orders without creating human ratings. Pass its output directory to the report as another `--extra-results-dir`.

The exporter requires the actual GPU/model environment for generation; its selection/cache-integrity tests require only the standard library. It checks frozen source hashes for the visualization, model loader and simulation conversion; dataset/split/cohort correspondence; cache configuration and sample counts; and actual generation success. Conflict examples are selected by input motion words plus nonzero endpoint displacement, never by generated quality. Padding stays in place when reversing valid source frames.

Synchronize the **entire** remote qualitative export directory locally (including `figures/`, `bundles/`, and `qualitative_manifest.json`) before refreshing a local report. The report copies the displayed SVG into local `assets/`, also copies PDFs, PNGs, metadata and portable trajectory bundles, and uses relative image/download URLs. Its HTML and Markdown therefore remain viewable when the report directory is moved, without access to the server's filesystem. Declared attachment/bundle hashes are verified; an incomplete transfer fails with a specific message. Move the whole report directory, not just the HTML file. The private blinded-method key is not copied into the public report.

All additional fixed-K sweeps, geometry, normalization, native-length timing, training summaries, per-mode metrics and bootstrap distributions follow the main paper report at the end.

## Fetching the active server experiment

From the workspace root, synchronize the small report inputs (weights and prediction caches are excluded), then rebuild:

```bash
python3 LensCraft/scripts/sync_paper_results.py \
  --remote-run-dir /media/external20/morteza_abolghasemi/lens-craft/runs/paper_complete_20260914_96a9cd7 \
  --local-run-dir performance_reports/paper_complete_20260914_96a9cd7

python3 LensCraft/scripts/build_paper_report.py \
  --base-run-dir performance_reports/retrain_eval10pct_20260912_191325_96a9cd7 \
  --extra-results-dir performance_reports/paper_complete_20260914_96a9cd7/supplement \
  --extra-results-dir performance_reports/paper_complete_20260914_96a9cd7/qualitative \
  --ablation-dir performance_reports/paper_complete_20260914_96a9cd7/ablation \
  --output-dir performance_reports/paper_report_20260914 \
  --paper-pdf '/Users/reza/Downloads/LensCraft__Your_Professional_Virtual_Cinematographer (2).pdf'
```

The default SSH alias is `me_DML4_proxy`. Server jobs source `/media/external20/morteza_abolghasemi/LensCraftVenv/bin/activate`. The active study uses frozen base revision `96a9cd7` plus the recorded experiment extensions, preserving comparability with the completed main run. Main-run weights and fixed evaluators are retained.
