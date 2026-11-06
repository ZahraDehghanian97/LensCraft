# Simulation dataset (LensCraft) — data contract

Everything a consumer (human or AI agent) needs to use the synthetic
simulation data **without reading the studio's TypeScript generator or
`src/data/simulation/`**.

- Source of truth: the studio's export code (`src/service/dataset/generate.ts`,
  `simulationFormatter.ts`, `parameterDictionary.ts`) for the files, and
  `src/data/simulation/metadata.py` / `loader.py` / `dataset.py` for
  consumption. If this file and the code disagree, the code wins — then
  update this file.
- Data is generated locally (studio → Generator panel → "Generate Dataset" →
  `<datasetId>.zip`); there is nothing to download.

---

## 1. At a glance

| Property | Value |
|---|---|
| Sample id | dataset-scoped file name `simulation_{datasetId}_{i}.msgpack` |
| Sample | 100–500-frame camera 6-DoF + one box subject (matching 6-DoF + volume) + structured prompt & instruction |
| Sequence length | source length is preserved as `original_frame_count`; 100–500-frame clips are reduced to the configured model length (30 by default) by selecting shared, exact camera/subject frame pairs over the full normalized timeline; generic upsampling interpolates the camera in subject-local space |
| Frame rate | v4 manifest records FPS (default 30) and `(frameCount - 1) / FPS` duration; older archives have no physical time metadata |
| Units / world | three.js scene units (~meters), right-handed, **y up**, ground at y = 0 |
| Camera pose | position + Euler **XYZ** radians of an **OpenGL-style** camera (x right, y up, looks down −z) |
| Subject | box: per-frame center position + Euler XYZ, plus one (w, h, d) volume; legacy ground paths rest at y = 0, human root bob keeps the box above it |
| Quantization | exporter stores `round(x·1000)` ints; Python divides by manifest `fixedPointScale` (default 1000, ~±5e-4 noise) |
| Labels | cinematography prompt + simulation instruction with string, boolean, and finite-number values (§3.4), plus a templated English caption rendered at load time |
| Manifest | schema/generator version, seed, export config, and encoding metadata in `manifest.json` |
| Python expectation | trusted schema-v2 studio export with exactly 1 subject/instruction and 100–500 source frames; random export excludes Dutch/roll |
| Normalization | dataset-level per-axis z-score of positions and volume only; rotations stay raw radians |
| Splits | none stored; `CameraTrajectoryDataModule` random-splits by fraction at runtime |

### Optional ET-compatible observation profile

The studio's ET training profile keeps the four-section schema-v2 payload and
full physical subject/camera poses. Its `manifest.modelObservation` declaration
is consumed by `SimulationDataset`:

```json
{
  "profile": "et_compatible_v1",
  "subjectOrientation": "unavailable",
  "subjectDimensions": "default",
  "defaultSubjectDimensions": [0.5, 1.7, 0.3],
  "simulationInstructions": "unavailable",
  "captionPolicy": "motion_only",
  "movementSpeed": "unavailable"
}
```

Model conditions retain subject XYZ and the primary motion type. The optional
`movementSpeed: "unavailable"` declaration also hides the diagnostic speed
token; earlier declarations without it retain supplied speed. Subject
Euler angles become zero placeholders, dimensions become the ET adapter's
default, and all 36 technical plus eight endpoint tokens become absent.
`subject_orientation_available` and `subject_dimensions_available` are false;
the current core model does not interpret these flags as a learned cue.
The generator's optional `observedMotionText` supplies the sparse text caption.

`geometry_reference` preserves physical camera/subject poses and dimensions
in unnormalized scene units for geometry metrics, including the higher
resolution reference path. It is separate from model conditioning. Normal
exports retain their previous item/collate behavior. Keep the manifest with
the payload; removing it removes the observation contract.

ET-profile source lengths may be 30–300 frames, sampled from ET training
duration statistics with an explicit 30-frame floor. The profile uses training
aggregate distributions, not native ET camera/subject trajectories or ET test
statistics. Calibration controls use the ET adapter's common 30-frame timeline
(linear positions and camera SLERP); the generator fits its actual 30 paired
nearest-frame SIM samples to those controls. Reconstructed subject-center drift
is not proof of semantic
walking; matched center rays/FOV proxies are not measured human shot sizes.

## 2. Directory layout & file formats

```
<extraction directory>/
└── <datasetId>/                     # unique archive-local dataset root
    ├── manifest.json                     # schema/generator version, seed, config, encoding
    ├── parameter_dictionary.msgpack      # archive identity + primitive-value table
    ├── simulation_<datasetId>_0000.msgpack ... # one file per simulation
    ├── movement_types.txt                # on-demand subject-movement index: "<file>|<movementType>"
    ├── movement_types.meta.json          # cache version + lightweight source fingerprint
    ├── normalization_parameters.json     # mean/std cache for positions and dimensions
    └── normalization_parameters.meta.json # cache fingerprint
```

Each `simulation_*.msgpack` is a 4-element msgpack list. Trajectory, dimension,
focal-length, and aspect-ratio values are fixed-point integers; dictionary
reference indices are ordinary integers:

```
[cinematography_refs,   # [[key_idx, value_idx], ...] into parameter_dictionary
 simulation_refs,       # same, for the simulation instruction
 subjects,              # [{i: id, c: class, d: [w, h, d],
                        #   f: [[px, py, pz, rx, ry, rz] × sourceFrames], m: movementType,
                        #   a?: [ax, ay, az, aw, ah, ad]}]   # optional attention box
 camera_frames]         # [[px, py, pz, rx, ry, rz, focalLength, aspectRatio] × sourceFrames]
```

Dictionary paths are `__`-joined key chains prefixed `cinematography` /
`simulation` (e.g. `cinematography__movement__type`). Keys and each key's
distinct primitive values are append-only in **first-encounter order** across
the export; they are not sorted. Indices are therefore archive-local and
seed/order-dependent — never treat them as enum ordinals, and never reorder
the dictionary without remapping every reference.

### 2.1 Manifest and reproducibility

New exports include `manifest.json`. Selected fields from the v4 contract
are shown below; the full file also contains normalized effective class/motion
probabilities, realized class-by-motion counts, camera-quality policy, and
rejection counts. These excerpted fields are not a complete manifest:

```json
{
  "schemaVersion": 2,
  "generatorVersion": 4,
  "datasetId": "<archive-unique id>",
  "generatedAt": "<ISO-8601 timestamp>",
  "seed": "<string>",
  "contract": {
    "subjectsPerSample": 1,
    "instructionsPerSample": 1,
    "movementSampling": "largest-remainder-marginal",
    "subjectClassSampling": "largest-remainder-per-compatible-movement",
    "humanMotion": "procedural-root-trajectories-not-articulated-motion-capture"
  },
  "config": {
    "simulationCount": 1000,
    "subjectMotionPreset": "human",
    "minFrameCount": 100,
    "maxFrameCount": 500,
    "framesPerSecond": 30
  },
  "time": {
    "framesPerSecond": 30,
    "timestampSeconds": "frameIndex / framesPerSecond",
    "durationSeconds": "(storedFrameCount - 1) / framesPerSecond"
  },
  "encoding": {
    "archiveRootDirectory": "<datasetId>",
    "simulationFilePattern": "simulation_<datasetId>_*.msgpack",
    "parameterDictionaryFile": "parameter_dictionary.msgpack",
    "parameterDictionarySchemaVersion": 2,
    "fixedPointScale": 1000
  }
}
```

The exporter converts a supplied numeric or string seed to a string; if none
is supplied, it creates one and records it here. Reusing the seed with the
same config and generator version reproduces sample content and dictionary
encounter order (`generatedAt` and the resulting archive bytes can still
differ).

The Python loader treats archives as trusted output from this generator, not
as untrusted datasets to audit. It uses `encoding.fixedPointScale` when that
manifest field is present and otherwise uses the legacy default of 1000,
including for manifest-less archives. The remaining manifest fields record
provenance for humans and future tooling; they are not an archive-wide
validation gate. Training expects one subject and one instruction. The loader
preserves source length as metadata, then resamples matching camera/subject
paths to the configured target length instead of truncating them.

## 3. Conventions — read before touching poses

1. **World frame.** three.js: right-handed, y up, ground plane at y = 0. The
   subject `position` is the box **center**, so a grounded subject sits at
   y = height/2 (apart from intentional vertical motion). The scene is
   **not** subject-centered (unlike E.T.) — use `recenter_rescale_sim` before
   any E.T.-style normalization.
2. **Camera rotations** are Euler XYZ, radians, for an OpenGL/three.js camera
   (local −z is the viewing direction, +y up). `SIMConvertor` bridges to the
   repo's OpenCV-style standard 4×4 by negating rotation columns 1–2
   (`_GL2CV = (1, −1, −1)`); positions pass through unchanged.
3. **Euler order** is dropped at serialization; it is always "XYZ".
4. **Primitive labels survive encoding.** Parameter dictionary values may be
   strings, booleans, or finite numbers. Booleans and nested lock constraints
   are CLIP-conditioned like other categorical tokens. `importance` is mapped
   linearly from [1, 10]; `maxSpeed` and `maxAccelerate` use `log1p` over
   [0, 20] and clip outliers before deterministic Fourier encoding. Missing or
   invalid values receive a zero/mean neutral token and a false presence mask.
   `frameCount` and `subjectIndex` are intentionally omitted: the former is
   observable from the trajectory and every export contains exactly one subject.
5. `focalLength` (default 37.52) and `aspectRatio` (16/9) are stored per
   camera frame but ignored by the loader (features are 6-DoF only).
6. **One instruction, one subject.** The exporter enforces this invariant so
   archive-local paths never lose subject/instruction association.
7. **Dutch/roll is unsupported.** The random dataset exporter excludes
   `dutchLeft` and `dutchRight`; do not pass manually constructed Dutch samples
   to the Python consumer.

## 4. Feature encoding (exact)

### 4.1 Trajectories → tensors (`SimulationDataset`)

```python
indices = round(linspace(0, source_frames - 1, 30))
camera  = source_camera[indices]                  # exact shared frame pairs
subject = source_subject[indices]
volume  = [[w, h, d]]                            # (1, 3)

camera[..., :3]  = (camera[..., :3]  - cam_pos_mean)  / cam_pos_std   # per axis
subject[..., :3] = (subject[..., :3] - subj_pos_mean) / subj_pos_std
volume           = (volume - dim_mean) / dim_std
# rotations (indices 3:6) are NOT normalized
padding_mask = zeros(30, bool)                   # nothing is padded
```

When `normalize=True`, stats are computed once over the same 30 paired samples
used from every simulation file (so long clips are not overweighted) and cached in
`normalization_parameters.json` (historical PyTorch sample std; zero or
singleton stds → 1). In memory they are keyed by resolved dataset path plus
the lightweight dataset fingerprint, and each
`SimulationDataset` retains its own parameters, so loading another directory
does not change an existing instance's normalization. The legacy static
`normalize_item` API uses the most recently requested parameters (fallback
lookup: `SIMULATION_DATA_PATH` env var before any are loaded).
`normalization_parameters.meta.json` records the cache format version and a
lightweight dataset fingerprint. For both manifest-backed and legacy exports,
the fingerprint includes the manifest (when present) plus the
parameter-dictionary and simulation-file name/size/mtime inventory. Payload
contents are not hashed. A missing or nonmatching sidecar causes the stats to be
regenerated.

### 4.2 Prompt / instruction → CLIP token grids

Reconstructed dicts are flattened against two fixed structs; each present
enum value becomes the CLIP text embedding of its English description (cached
in `clip_embeddings_cache.pkl`; D = the configured CLIP text width, 512 in
the shipped configs). Missing tensor slots use a zero neutral embedding, or the
per-enum-type mean when `fill_none_with_mean=True`; their parameter index and
presence metadata remain −1/false (`embedding_means.pkl` /
`embedding_stds.pkl` are written to the working directory).

- `cinematography_prompt` — 10 tokens: initial {cameraAngle, shotSize,
  subjectView, subjectFraming} · movement {type, speed} · final {same 4}.
- `simulation_instruction` — 36 tokens: setup (6) · dynamic (12, including
  type/randomness) · constraints (18, including all locks and three numeric
  limits/weights).
- `prompt_none_mask` — (46,) bool, prompt tokens first, True = value present.

Generator exports preserve boolean constraint values (including explicit
`false`) and the finite numeric speed/acceleration/importance limits.
`dutchAngleScale` remains optional. `MovementMode` mirrors the simulator's
transition, rotation, arc, crane, and roll vocabulary. The
`clip_embeddings_cache.pkl` file has no dataset-provenance sidecar and is
separate from the two dataset-derived caches in §2.

Schema-v2 manifests store normalized effective movement/class probabilities,
the dynamic Car/Bicycle conditional probabilities, realized finite-sample
counts, and the complete effective noise configuration. Noise amplitudes and
frequency are validated rather than silently clamped; exported rotational
noise is capped at 0.15 rad and is applied after vehicle heading recovery, so
the serialized value actually affects the trajectory. Static paths never
receive noise.

### 4.3 Caption

`text_prompt` is rendered at load time from the prompt plus the subject's
movement type with closed-vocabulary templates, e.g. *"As the character moves
in a circular path, the camera pushes in, creating a dynamic relationship."*
Nothing textual is stored in the files.

### 4.4 Camera movement vocabulary

The random dataset exporter uses these 17 camera movements:

| family | values |
|---|---|
| stationary / subject-relative | `static`, `follow`, `track` |
| dolly | `dollyIn`, `dollyOut` |
| pan / tilt | `panLeft`, `panRight`, `tiltUp`, `tiltDown` |
| truck / pedestal | `truckLeft`, `truckRight`, `pedestalUp`, `pedestalDown` |
| arc / crane | `arcLeft`, `arcRight`, `craneUp`, `craneDown` |

The former dolly-zoom combinations (`dollyOutZoomIn`,
`dollyInZoomOut`) are not active movement types. Dutch/roll is also excluded
and unsupported by the Python consumer.

Random endpoint fields are not serialized. The exporter first executes the
camera instruction, then classifies the actual first and last camera/subject
poses back into the cinematography vocabulary. Serialized `initial` is the
observed first setup; `final` contains only observed categories that changed.
The low-level instruction's setup (and interpolation complement) is aligned
to those same observed endpoints while its movement, easing, constraints, and
numeric limits remain untouched. Candidates whose endpoints cannot be
projected are deterministically rejection-sampled. Thus neither a floor clamp
nor an incompatible random endpoint can leave prompt/instruction tokens that
contradict the stored frames.

## 5. Item schema — `SimulationDataset[i]`

| Key | Shape / type | Notes |
|---|---|---|
| `camera_trajectory` | (30, 6) float32 | normalized positions + raw Euler XYZ |
| `subject_trajectory` | (30, 6) float32 | first subject only |
| `subject_volume` | (1, 3) float32 | normalized (w, h, d) |
| `padding_mask` | (30,) bool | all False |
| `cinematography_prompt` | (10, D) float32 | CLIP tokens, neutral zero/mean rows = unknown |
| `simulation_instruction` | (36, D) float32 | categorical CLIP and numeric Fourier tokens |
| `prompt_none_mask` | (46,) bool | True = valid slot |
| `original_frame_count` | integer | source clip length before resampling |
| `*_parameters` | list of tuples | (name, value, enum index, embedding) per slot |
| `raw_prompt` / `raw_instruction` | dict | reconstructed native primitive values (§3.4) |
| `text_prompt` | str | templated caption (§4.3) |

`collate_fn` stacks everything, transposes both token grids to
(tokens, batch, D), and exposes the captions as `text_prompts`.

## 6. Subject motion vocabulary (generator v4)

The exporter supports the original `static`, `circular`, `linear`, `zigzag`,
`spiral`, `figureEight`, and `wave` plus human `walk`, `run`, `wander`,
`stopStart`, `sidestep`, `backward`, `turnInPlace`, and `idle`. Human subjects
have varied bounded body dimensions and procedural root trajectories. Body
heading is independent of travel for sideways/backward motion; static is an
exact freeze, while idle and turning are non-static for cohort selection.
The eight human motion types require a Human subject; the six original moving
families accept Human/Car/Bicycle, and stationary props remain available.

Generation defaults to the human-focused preset, with mixed and legacy presets
available. Sampling remains movement-first: largest-remainder allocation fixes
finite movement counts, then compatible classes are allocated within each
movement. Class weights are conditional, not global proportions. Invalid
class/movement combinations fail before generation. Manifest `config` records
effective distributions and `realized` records movement, class, class-by-movement,
and camera counts. The camera quality and geometry-label checks still run on
the quantized output and the 30 paired training poses.

Generator version 4 retains schema version 2 and the same four-element
MessagePack format. Its manifest records `framesPerSecond`, timestamps
`frameIndex / framesPerSecond`, and duration `(frameCount - 1) / framesPerSecond`.
The current model consumes the same 30 resampled poses and body dimensions;
time is recorded for auditing but is not a model input. Human gait is only a
rigid bounding-box-centre trajectory, not a skeleton or motion capture. There
is no hidden attention ROI. Older manifest-less and version-2/3 exports remain
readable; do not reuse their sample hashes or cached cohort results as v4 results.

The caption vocabulary and paper/ablation dynamic cohort lists cover all eight
new motion names. `idle` and `turnInPlace` belong to the non-static cohort even
without substantial translation. Re-running old exact-cohort validation with
new dynamic names requires a fresh evaluation or the original runner version.
See [generator documentation](../../../../camera_movement_simulation/README.md)
for presets, physical conventions, controls, and verification commands.

`movement_types.txt` is an on-demand per-file index used by
`allowed_movement_types`. Its metadata sidecar uses the same dataset
fingerprint policy as the normalization cache.

## 7. Gotchas (recap)

1. Parameter references are archive-local. Keep every sample with its own
   manifest/dictionary and never merge files across dataset roots.
2. Generate one subject, one instruction, and 100–500 frames. Python records
   the source length and resamples the complete motion to 30 model steps.
3. The movement cache is created only for `allowed_movement_types`; the
   normalization cache is created only for `normalize=True`. Once created,
   each is reused when its `.meta.json` sidecar has the current version and
   lightweight dataset fingerprint. Manifest exports key this from the
   manifest plus file names; manifest-less directories use
   parameter-dictionary and simulation-file size/mtime inventory. Delete a
   sidecar to force regeneration after an out-of-band payload edit that did
   not update that metadata.
4. Dataset instances use normalization stats cached by resolved path and
   dataset fingerprint. Only the backward-compatible static `normalize_item`
   helper relies on the process's most recently selected stats; prefer the
   instance path when multiple simulation datasets are loaded.
5. ×1000 quantization: ~5e-4 absolute noise on positions and radians; don't
   chase sub-millimeter round-trip errors.
6. Train/val/test membership is a runtime `random_split` — it changes with
   the runtime RNG seed (not the manifest's export seed) and split fractions;
   persist your own split file if you need stable membership.
7. Need FOV? Read camera row index 6 (÷`fixedPointScale` already applied) before
   featurization — the 6-DoF features drop it.
8. Padding-mask polarity is **True = pad** throughout this repo (all-False
   here); DIRECTOR's convention is the opposite.
9. CCDM cannot represent camera roll: its five values encode relative
   position plus projected target coordinates, and reconstruction chooses an
   upright orientation. The random exporter excludes Dutch before Python sees
   the data, so do not manually route Dutch samples through CCDM or expect a
   Dutch → CCDM → 6-DoF round trip to preserve roll.

## 8. Appendix — provenance (how the data was made)

The studio generates the subject path, a logically consistent cinematography
prompt/instruction pair, and the rule-based camera path in-browser. It then
quantizes geometric arrays, stores primitive structured parameters in the
archive-local dictionary, writes a schema-v2 manifest, and packages everything
inside a unique dataset root. The manifest and simulator source are the
authoritative provenance for exact generation distributions; LensCraft is
responsible only for strict decoding, validation, normalization, and temporal
resampling.

Schema v2 expands the model from 29 to 46 structured query tokens. Existing
checkpoints therefore have incompatible query-token and merger tensor shapes
and must be retrained (or migrated explicitly); silently loading them is not
supported.
