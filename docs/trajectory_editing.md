# Partial-prompt trajectory editing

The maintained editing runner evaluates the sole categorical generator. It
changes one requested field of a source trajectory while retaining the other
encoder distributions, and compares that intervention with reconstruction and
a no-op substitution. It runs inference only.

```bash
python scripts/evaluate_trajectory_editing.py \
  --checkpoint /path/to/categorical-generator.pt \
  --train-config /path/to/categorical-training.yaml \
  --semantic-evaluator-checkpoint /path/to/frozen-semantic.ckpt \
  --semantic-evaluator-config /path/to/frozen-semantic-training.yaml \
  --output /path/to/new-edit-evaluation \
  --field initial_shotSize --value closeUp \
  --limit 128 --batch-size 32 --device cuda:0
```

Generator paths accept `TEST_CHECKPOINT_PATH` and `TEST_CONFIG_PATH`; judge paths
accept `SEMANTIC_EVALUATOR_CHECKPOINT_PATH` and `SEMANTIC_EVALUATOR_CONFIG_PATH`.
The judge is an independently loaded frozen encoder-only model. It does not
construct the generator's memory or produce the edited trajectory.

Only current categorical standalone `model` and Lightning `state_dict`
checkpoints are supported. Their saved configs can retain redundant settings
from the former categorical opt-in entry; the loader migrates those settings
after validating categorical weights. Legacy 512D, ID-adapter and robust-wrapper
generator checkpoints fail explicitly. There is no memory norm calibration or
merged-memory option.

## Intervention

The source encoder produces `[10,batch,21]` categorical probabilities. An edit
replaces the selected field with the requested one-hot distribution. Every
other field stays source-derived; no absent simulator program or numerical
constraint is added to memory. The decoder retains ordinary subject context.

| Output | Semantic memory |
| --- | --- |
| Reconstruction | All source-encoder probability fields |
| No-op | Selected field replaced with its original annotated one-hot value |
| Requested edit | Selected field replaced with the requested one-hot value |

The no-op control exposes the effect of replacing a soft distribution with a
one-hot label. A sample is eligible only if its source value is known and differs
from the requested value. No keyframe is pinned, source pose copied, or final
endpoint overwritten. Source annotations select eligibility and the no-op;
they do not fill the remaining edited memory.

The runner uses the simulation held-out split. The limit is applied before
eligibility filtering; `--limit 0` scans the full split. Keep the same field,
value, split, evaluator and normalization when comparing checkpoints. Use the
training simulation export via `--normalization-data-path` if necessary; stored
checkpoint moments remain authoritative and mismatches fail explicitly.

## Scores and artifacts

The frozen judge reports requested-value top-1/cosine change and agreement on
unedited high-level fields. Geometry diagnostics report position, rotation,
motion and subject-relative drift. They measure intervention costs against the
source/reconstruction, not correctness against an unavailable edited ground
truth. Where actual simulator geometry is present, shot-size edits can also be
checked using ROI occupancy. Missing geometry stays unavailable.

Use success, preservation, geometry drift and the no-op control together.
Scores depend on judge quality and are not independent human labels.

The output must be a new directory. Keep `metrics.json`, `manifest.json`,
`samples.jsonl` and optional `trajectories.pt` together so checkpoint, cohort,
evaluator and field intervention identities remain reviewable.
