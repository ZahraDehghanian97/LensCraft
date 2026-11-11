# Training and evaluation

`python src/train.py` uses the sole categorical LensCraft architecture and
per-field CE plus detached cycle KL. See [the representation contract](categorical_conditioning.md)
for its ten tokens, 93 valid probabilities, masks and checkpoint compatibility.
There is no generator representation selector or CLIP objective.

## Training configurations

All recipes inherit the same categorical model and objective. They vary data
views, conditioning policies, optimizer settings or validation diagnostics.

| Configuration | Purpose |
| --- | --- |
| `default` | SIM training with four conditioning families and sparse-keyframe validation. |
| `multi` | SIM plus CCDM training, with the full categorical and geometry objective on each dataset. |
| `et_robust` | SIM-only continuation with grouped removal of categorical facts and sparse validation. |
| `transfer_robust` | Adds ET-shaped and CCDM-shaped SIM observation views. |
| `transfer_canonical` | Expresses the ET-shaped SIM view in the first-camera frame. |
| `high_level_sparse` | Uses sparse high-level facts with domain augmentation. |
| `high_level_scratch` | Scratch optimizer/scheduler and a larger prompt-generation share. |
| `high_level_ccdm` | SIM continuation with more CCDM-shaped views and compatible motion labels. |
| `keyframe_pose` | Historical alias; explicit keyframe masks supply pose/time/presence queries in every recipe. |

```bash
python src/train.py
python src/train.py data/dataset=et
python src/train.py data/dataset=ccdm
python src/train.py data=multi training=multi
python src/train.py training=transfer_robust data.compact_batches=false
```

Mixed training requires `CCDM_ANNOTATIONS_PATH` or an explicit
`data.dataset.ccdm_config.annotations_path=/path/to/annotations.json`.
Annotations must cover all selected CCDM source IDs, match their native captions
exactly and satisfy the structured schema. For subset annotations, also set
`data.dataset.ccdm_config.sample_indices_path=/path/to/indices.json`.
See [the caption contract](categorical_conditioning.md#training-entry).
Annotation-free native CCDM loading remains available to evaluation and frozen
text-conditioned baselines.

Use `initialize_from_checkpoint` for a new optimizer/schedule initialized from
current categorical weights, or `resume_checkpoint` for an interrupted run.
Checkpoints predating geometric context require `initialize_from_checkpoint`
with `resume_checkpoint=null`, because their optimizer state lacks the new
parameters. Their context weights start at zero and need continued training.
Legacy 512D generator checkpoints are unsupported. The retired CLIP adaptation
and ablation launchers are not training entry points for this architecture.

Sparse facts use zero semantic fields with availability false, rather than a
mean vocabulary vector. SIM view augmentation is a training policy; it does
not add native ET/CCDM test examples to optimization. Metadata-dependent view
policies require `data.compact_batches=false`.

Training normalization is fitted after splitting, using only the training
inventory, and shared with validation/test. Mixed SIM and CCDM splits use
independent seeded generators and retain their membership across setup calls.
New checkpoints save their normalization moments; evaluation restores them
before reading samples. Historical checkpoints without moments require their
original reference for comparable results; fallback preprocessing is reported
as unverifiable. An explicit independent `SIMULATION_NORMALIZATION_DATA_PATH`
remains available when fitting new runs.

The default geometry coefficients are first-frame 2, relative 16, speed 80,
absolute rotation 2, raw rotation 1, keyframe position 4 and keyframe rotation 2.
CE and cycle KL both have weight 1; KL targets are always detached. Empty labels,
padded categories and padded frames are excluded. Total losses from historical
cosine objectives are not comparable to this objective.

## Evaluation

```bash
python src/test.py
python src/inference.py
python src/inference.py training/model=et
python src/inference.py training/model=ccdm
python src/inference.py training/model=gendop
```

`training/model=et|ccdm|gendop` selects independent third-party evaluation
baselines. It does not switch the LensCraft conditioning representation.
The semantic evaluator is a frozen encoder-only judge; CLaTr is a separately
frozen trajectory/text evaluator. Their CLIP assets are evaluation dependencies
and never enter LensCraft training or generator memory.

Each frozen judge uses its own training normalization. Metric adapters convert
generator coordinates through physical space before applying those scales;
raw baseline cameras are identified separately from normalized subject inputs.
New native CLaTr checkpoints save their scales. Older judges require a concrete
original training source in their saved config or an explicit
`SIMULATION_NORMALIZATION_DATA_PATH` reference. Unavailable judge scales stop
scoring instead of substituting the generator's transform. Judge normalization
and its identity are recorded in evaluation provenance.

Keep checkpoint, cohort ordering, normalization, evaluator identities and PRDC
protocol fixed when comparing systems. Read FCD and all four PRDC measures
together with caption alignment and geometry. The baseline modes use their
native coordinate view and the shared normalized view; the former
LensCraft-judge-generated initialization mode has been removed.

See [evaluation definitions](evaluation.md), [CCDM evaluation](ccdm_evaluation.md),
[trajectory editing](trajectory_editing.md), and [visualization](visualization.md).
Historical experiment protocols and receipts remain authoritative for their
frozen executions; this cleaned default does not retrospectively change them.
