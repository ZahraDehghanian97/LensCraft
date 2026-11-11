# CCDM native motion-classifier evaluation

The local scripts train and use the authors' released six-motion classifier.
They do not use CLaTr. No remote deployment or experiment was authorized for
this implementation; the files are prepared locally only.

## Exact training reference

The implementation currently reproduces
`third_parties/Camera-control/[2024][EG]Text+keyframe/classify.py`:

| Setting | Released implementation |
| --- | --- |
| Inputs | Raw subject-relative XYZ and screen XY; 5 values per frame |
| Sequence length | 300; normalize, then repeat the final frame |
| Transformer | 4 layers, width 256, 4 heads, feed-forward width 1024 |
| Activation / dropout | GELU / 0.1 |
| Output | Per-frame Linear(256,1), ReLU, flatten, Linear(300,6) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam with its default betas, epsilon and zero weight decay |
| Learning rate | `1e-4 * (1 - epoch_index / 1000)` |
| Training | 1000 epochs, batch 128, FP32, shuffled loader, 5 workers |
| Normalization | Population mean/std over all unpadded train **and test** frames; denominator `std + 1e-8` |
| Checkpoint | Final scheduled epoch; test accuracy is diagnostic, with no early stopping |

The shared-RNG training loop is tested against the released training operations
with identical initialization and data order; final weights match bit for bit.
Runtime versions, source hashes, data IDs and the checkpoint are recorded.

The [paper](https://arxiv.org/html/2402.16143v1), Appendix B, describes **six**
Transformer layers, and its benchmark uses **60** frames. The
[released implementation](https://github.com/jianghd1996/Camera-control/tree/master/%5B2024%5D%5BEG%5DText%2Bkeyframe)
uses **four** layers and **300** frames. This source disagreement is unresolved;
running the released code is not an exact reconstruction of the paper's hidden
60-frame evaluator. The exact-reference choice must be settled before a
scientific training run. The original numeric label mapping, split IDs and
classifier weights have not been recovered.

## Prepare and train

From the `LensCraft` directory, with the released trusted `cam/info` data.npy:

```bash
python scripts/train_ccdm_native_classifier.py \
  --data-path /path/to/ccdm/data.npy \
  --output-dir /path/to/native/classifier \
  --device cuda:0 --prepare-only
```

Preparation preserves every source caption and creates a seeded random 9:1
split, six explicit native-motion labels, the original classifier data schema,
normalization statistics and source manifests. It drops no unknown labels;
unknown or ambiguous motion sentences cause an error. Both boom directions
share one class. To train after reviewing this prepared run, repeat the same
command with `--resume` instead of `--prepare-only`.

The default recipe is the complete released recipe. Non-default epoch counts,
batch size, learning rate, worker count, or capped batches are marked as a
modified/smoke recipe. They cannot be reported as a complete reproduction.

## Freeze shared evaluation inputs

```bash
python scripts/prepare_ccdm_native_cohort.py \
  --classifier-dir /path/to/native/classifier \
  --output /path/to/native/cohort.json --count 12000
```

This selects classifier-test IDs with replacement using seed 42. It checks the
prepared split hash and source disjointness. Use the same JSON for every model.
These IDs are held out from the new classifier, but training overlap of the
released CCDM generator is unknown.

The paired exporter is
`../../performance_reports/ccdm_native_evaluation_20261107/generate_native.py`
relative to this document's directory. Its `--help` describes model-specific
assets. It emits numeric `generated.npz` files and per-batch hash receipts.
The CCDM model uses native, denormalized, smoothed 300-frame outputs. It does
not need the geometric codec to enter the classifier.

The LensCraft exporter currently supports the pinned legacy
`motion_balanced/4721` checkpoint, not an arbitrary or newly trained checkpoint.
Its actual 300-frame generation requires `--allow-length-extrapolation`, because
that checkpoint was trained with 30 frames. This limitation is recorded in its
output. Conversion of LensCraft poses to native screen XY additionally assumes
VFOV 45 degrees and aspect 16:9. It never reads a reference-camera trajectory.

## Score native generations

```bash
python scripts/evaluate_ccdm_native.py \
  --classifier-checkpoint /path/to/native/classifier/result/latest.pth \
  --classifier-data /path/to/native/classifier/data.npy \
  --classifier-origin retrained \
  --training-manifest /path/to/native/classifier/result/evaluator_manifest.json \
  --generated /path/to/native/generation/generated.npz \
  --cohort /path/to/native/cohort.json \
  --generation-manifest /path/to/native/generation/complete.json \
  --output /path/to/native/scores --device cuda:0
```

The manifests bind the checkpoint, split, normalization, corpus and ordered
generation IDs/labels. A mismatch or reference-camera use fails. Original
unpaired artifacts can be scored without a generation manifest, but remain
explicitly unverified as a matched cohort. The scorer extracts the released per-frame post-ReLU features and
normalizes them with training-feature mean/std. Primary `fid_train` follows the
released metric's training reference; test-reference FID is a separately named
diagnostic. Diversity uses 10,000 seeded random pairs with replacement. Motion
classification accuracy is the released interpretation of R Prec, not text
retrieval recall. Missing generated labels leave accuracy unavailable.

Multimodality requires actual repeated generations under identical conditions.
To prepare distinct prompts, use cohort `--unique --count 50`, then exporter
`--repeats 100`. The scorer reports within-condition pair distances separately.
The released class-conditioned diversity script shuffles sample pairs across
classes; that operation is not silently presented as same-prompt multimodality.
With one generation per condition, MM remains unavailable. Deterministic
LensCraft generation can produce zero MM.

All outputs require fresh directories, except an explicitly requested validated
resume. No historical paper or CLaTr result tables are overwritten.
