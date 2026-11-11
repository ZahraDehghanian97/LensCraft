# Categorical conditioning

The categorical architecture encodes a camera trajectory as ten bounded
categorical distributions. The shared semantic tensor has shape
`[10, batch, 21]`: 93 entries are valid category probabilities and the other
117 entries are fixed zeros. Each nonempty field sums to one. The ten sum
constraints leave 83 continuous degrees of freedom across the 93 valid
coordinates. Dataset annotations supply one-hot targets; the encoder predicts
soft probabilities.

| Field | Categories |
| --- | ---: |
| Initial camera angle | 5 |
| Initial shot size | 8 |
| Initial subject view | 8 |
| Initial subject framing | 13 |
| Movement type | 21 |
| Movement speed | 4 |
| Final camera angle | 5 |
| Final shot size | 8 |
| Final subject view | 8 |
| Final subject framing | 13 |

Movement categories include the original 19 values followed by `nativePan`
and `nativeOrbit` at local indices 19 and 20. Simulator instructions and
numerical constraint tokens do not enter this semantic representation.

The transformer hidden width is 256, with eight attention heads, feedforward
width 1024, four encoder layers and four decoder layers. Ten fixed query
codes identify the encoder fields. Each field's logits are masked to its
valid vocabulary before softmax, keeping padded coordinates and their
gradients zero.

The decoder attends directly to the 21-dimensional semantic tokens through
`MultiheadAttention(embed_dim=256, kdim=21, vdim=21)`. Fixed sinusoidal field
codes are added to separate attention keys and values, with category padding
kept zero. The public semantic tensor stays unchanged. Attention's internal
key/value projections are the only learned mapping from these 21D tokens to
the attention hidden space.

Missing fields mean DONTCARE: their semantic token is zero and excluded from
attention. If every field is missing, one zero token without a field code is
unmasked to keep attention finite. There is no learned null token or field
embedding table.

Projected subject trajectory, volume and measured optics remain geometric
context for both the encoder and decoder. `camera_intrinsics` supplies focal
length in millimetres and width/height aspect ratio as `[batch, frames, 2]`
(the model API also accepts constant `[batch, 2]`). NaN marks unobserved optics;
converter projection defaults are not measurements. Separate
`subject_orientation_available` and `subject_dimensions_available` boolean
masks distinguish unknown values, which are masked before projection, from
measured zero rotation or a measured body size. Optical context has its own
projection and does not add categorical tokens or widen the semantic tensor.
Explicit camera keyframes remain pose constraints. Camera information from
the encoder reaches generation through the categorical tensor; encoder
hidden states do not form an additional decoder memory. The camera output
retains the position3 plus SVD9D rotation representation and its existing
rotation projection and autoregressive feedback handling.

## Training entry

This is the only LensCraft generator. From `LensCraft/`:

```bash
python src/train.py
python src/train.py data/dataset=et
python src/train.py data/dataset=ccdm
python src/train.py data=multi training=multi
```

The default model, dataset and loss configs need no representation selector or
CLIP cache. SIM, ET and CCDM produce local categorical IDs and availability;
`data.categorical_conditioning` constructs their one-hot targets. Mixed training
uses the same CE, cycle KL and geometry objective on both labeled datasets.

For `data=multi training=multi`, set `CCDM_ANNOTATIONS_PATH` or override
`data.dataset.ccdm_config.annotations_path=/path/to/annotations.json`.
Each row must contain `custom_id`, the exact native `source_prompt`,
`cinematographyPrompts` and `simulationInstructions`. CCDMDataset validates the
schema and requires coverage of every selected source ID. Use
`data.dataset.ccdm_config.sample_indices_path=/path/to/indices.json` when the
translations cover a subset. Missing annotation configuration stops mixed
training before model construction: native caption strings are not categorical
generator inputs. Annotation-free loading remains supported for evaluation and
independent text baselines. CCDM baseline captions are passed unchanged to its
native text encoder, preserving the distinction between camera travel and zoom.

Each field's known examples supervise cross-entropy from its valid logits.
Cycle consistency uses `KL(original || regenerated-cycle)` and always detaches
the original encoder distribution. Both terms average known examples within
each field, then fields that have labels. Missing labels and category padding
are excluded. The default weights are CE=1 and KL=1; camera geometry losses
retain their configured coefficients in `config/training/default.yaml`.

The complete default model has **6,909,473 parameters**, including 1,536
parameters for geometric context. Parameter count does
not establish runtime speed or generation quality. The current architecture's
results and historical comparisons are recorded separately under
`performance_reports/` in the surrounding research workspace.

## Checkpoints and frozen evaluators

Existing categorical parameter keys retain their shapes. The loader adds the
complete geometric-context group with zero weights only when restoring a
checkpoint that predates it, emitting a warning that further training is needed.
A partially missing group still fails strict loading. Full optimizer resume
requires the current parameter schema; use `initialize_from_checkpoint` and
`resume_checkpoint=null` to continue from older categorical weights with a new
optimizer. Saved training normalization is restored with the generator.
The loader can migrate saved configs from the earlier categorical entry, removing
obsolete representation and CLIP-memory settings after verifying the checkpoint
is categorical. The old opt-in config files and all alternate generator paths
have been removed. A legacy 512D/merged generator checkpoint fails explicitly;
it is not silently reshaped or partially loaded.

The semantic judge is an isolated frozen encoder in
`testing.metrics.semantic_evaluator.FrozenSemanticEvaluator`. Its 512D CLIP
outputs preserve the evaluation contract but cannot generate trajectories.
CLaTr and third-party baseline models retain their own independent text
features. Evaluation constructs those judge inputs separately from the
categorical generator's ten-field memory.
