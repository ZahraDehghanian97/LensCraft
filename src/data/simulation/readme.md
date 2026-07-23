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
  `cinematography_dataset.zip`); there is nothing to download.

---

## 1. At a glance

| Property | Value |
|---|---|
| Sample id | file name `simulation_{i}`, zero-padded to `floor(log10(count)) + 1` digits |
| Sample | 30-frame camera 6-DoF + one box subject (30-frame 6-DoF + volume) + structured prompt & instruction |
| Sequence length | exactly 30 frames; **no padding** (`padding_mask` all-False, hard-coded) |
| Frame rate | not encoded; a clip is one instruction of 30 frames (studio previews at 30 fps) |
| Units / world | three.js scene units (~meters), right-handed, **y up**, ground at y = 0 |
| Camera pose | position + Euler **XYZ** radians of an **OpenGL-style** camera (x right, y up, looks down −z) |
| Subject | box: per-frame center position + Euler XYZ, plus one (w, h, d) volume; grounded baseline y = height/2 + offset [0, 1] |
| Quantization | exporter stores `round(x·1000)` ints; Python divides by manifest `fixedPointScale` (default 1000, ~±5e-4 noise) |
| Labels | cinematography prompt + simulation instruction — **enum strings only** (§3.4) — plus a templated English caption rendered at load time |
| Manifest | schema/generator version, seed, export config, and encoding metadata in `manifest.json` |
| Python expectation | trusted studio export with exactly 1 subject, 1 instruction, and 30 frames; random export excludes Dutch/roll |
| Normalization | dataset-level per-axis z-score of positions and volume only; rotations stay raw radians |
| Splits | none stored; `CameraTrajectoryDataModule` random-splits by fraction at runtime |

## 2. Directory layout & file formats

```
<dataset root>/                       # unzipped cinematography_dataset.zip
├── manifest.json                     # schema/generator version, seed, config, encoding
├── parameter_dictionary.msgpack      # {"keys": [path, ...], "values": [[str, ...], ...]}
│                                     #   shared string table for prompts/instructions
├── simulation_0000.msgpack ...       # one file per simulation (sorted glob simulation_*.msgpack)
├── movement_types.txt                # on-demand subject-movement index: "<file>|<movementType>"
├── movement_types.meta.json          # cache version + lightweight source fingerprint
├── normalization_parameters.json     # first normalize=True load: mean/std for camera_position,
│                                     #   subject_position, subject_dimensions
└── normalization_parameters.meta.json # cache version + lightweight source fingerprint
```

Each `simulation_*.msgpack` is a 4-element msgpack list. Trajectory, dimension,
focal-length, and aspect-ratio values are fixed-point integers; dictionary
reference indices are ordinary integers:

```
[cinematography_refs,   # [[key_idx, value_idx], ...] into parameter_dictionary
 simulation_refs,       # same, for the simulation instruction
 subjects,              # [{i: id, c: class, d: [w, h, d],
                        #   f: [[px, py, pz, rx, ry, rz] × 30], m: movementType,
                        #   a?: [ax, ay, az, aw, ah, ad]}]   # optional attention box
 camera_frames]         # [[px, py, pz, rx, ry, rz, focalLength, aspectRatio] × 30]
```

Dictionary paths are `__`-joined key chains prefixed `cinematography` /
`simulation` (e.g. `cinematography__movement__type`). Keys and each key's
distinct string values are append-only in **first-encounter order** across
the export; they are not sorted. Indices are therefore archive-local and
seed/order-dependent — never treat them as enum ordinals, and never reorder
the dictionary without remapping every reference.

### 2.1 Manifest and reproducibility

New exports include `manifest.json`:

```json
{
  "schemaVersion": 1,
  "generatorVersion": 1,
  "generatedAt": "<ISO-8601 timestamp>",
  "seed": "<string>",
  "config": {
    "simulationCount": 1000,
    "subjectCount": 1,
    "instructionCount": 1,
    "minFrameCount": 30,
    "maxFrameCount": 30,
    "subjectClassProbabilities": null,
    "movementDistribution": null,
    "noiseConfig": null
  },
  "encoding": {
    "simulationFilePattern": "simulation_*.msgpack",
    "parameterDictionaryFile": "parameter_dictionary.msgpack",
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
manifest field is present and otherwise uses the schema-v1 default of 1000,
including for manifest-less archives. The remaining manifest fields record
provenance for humans and future tooling; they are not an archive-wide
validation gate. Training tensors expect the generator profile
`subjectCount = 1`, `instructionCount = 1`, and
`minFrameCount = maxFrameCount = 30`, so generate with that profile rather
than relying on the loader to repair another layout.

## 3. Conventions — read before touching poses

1. **World frame.** three.js: right-handed, y up, ground plane at y = 0. The
   subject `position` is the box **center**, so a grounded subject sits at
   y ≈ height/2 plus the generator's random y offset in [0, 1]. The scene is
   **not** subject-centered (unlike E.T.) — use `recenter_rescale_sim` before
   any E.T.-style normalization.
2. **Camera rotations** are Euler XYZ, radians, for an OpenGL/three.js camera
   (local −z is the viewing direction, +y up). `SIMConvertor` bridges to the
   repo's OpenCV-style standard 4×4 by negating rotation columns 1–2
   (`_GL2CV = (1, −1, −1)`); positions pass through unchanged.
3. **Euler order** is dropped at serialization; it is always "XYZ".
4. **Only strings survive the label encoding.** The parameter dictionary
   indexes string values only, so every numeric or boolean field of the
   prompt / instruction — `frameCount`, `subjectIndex`, `importance`,
   `maxAccelerate`, `maxSpeed`, `subjectAwareInterpolation`,
   `allFramesVisibility`, `staticDistance`, `staticCameraSubjectRotation` —
   is **not in the files**. Reconstructed dicts contain enum strings only,
   and the loader marks the missing slots "none" (index −1).
5. `focalLength` (default 37.52) and `aspectRatio` (16/9) are stored per
   camera frame but ignored by the loader (features are 6-DoF only).
6. **One instruction, one subject.** Schema v1 dictionary paths contain no
   array index, so multiple prompts/instructions lose their grouping. The
   loader reconstructs one flattened prompt/instruction and reads
   `subjectsInfo[0]`; generate with `instructionCount = 1`,
   `subjectCount = 1`, and a frame range fixed at 30–30.
7. **Dutch/roll is unsupported.** The random dataset exporter excludes
   `dutchLeft` and `dutchRight`; do not pass manually constructed Dutch samples
   to the Python consumer.

## 4. Feature encoding (exact)

### 4.1 Trajectories → tensors (`SimulationDataset`)

```python
camera  = [[px, py, pz, rx, ry, rz]] * 30        # float32 (÷fixedPointScale applied)
subject = same layout, subjectsInfo[0]           # (30, 6)
volume  = [[w, h, d]]                            # (1, 3)

camera[..., :3]  = (camera[..., :3]  - cam_pos_mean)  / cam_pos_std   # per axis
subject[..., :3] = (subject[..., :3] - subj_pos_mean) / subj_pos_std
volume           = (volume - dim_mean) / dim_std
# rotations (indices 3:6) are NOT normalized
padding_mask = zeros(30, bool)                   # nothing is padded
```

When `normalize=True`, stats are computed once over every frame of every
simulation file in the archive and cached in
`normalization_parameters.json` (historical PyTorch sample std; zero or
singleton stds → 1). In memory they are keyed by resolved dataset path plus
the lightweight dataset fingerprint, and each
`SimulationDataset` retains its own parameters, so loading another directory
does not change an existing instance's normalization. The legacy static
`normalize_item` API uses the most recently requested parameters (fallback
lookup: `SIMULATION_DATA_PATH` env var before any are loaded).
`normalization_parameters.meta.json` records the cache format version and a
lightweight dataset fingerprint. For manifest-backed exports the fingerprint
uses the manifest plus sorted simulation file names; legacy directories use
parameter-dictionary and simulation-file size/mtime inventory. Simulation
payloads and cache contents are not hashed. A missing or nonmatching sidecar
causes the stats to be regenerated.

### 4.2 Prompt / instruction → CLIP token grids

Reconstructed dicts are flattened against two fixed structs; each present
enum value becomes the CLIP text embedding of its English description (cached
in `clip_embeddings_cache.pkl`; D = the configured CLIP text width, 512 in
the shipped configs). Missing slots stay −1-filled, or get the per-enum-type
mean embedding when `fill_none_with_mean=True` (`embedding_means.pkl` /
`embedding_stds.pkl` are written to the working directory).

- `cinematography_prompt` — 10 tokens: initial {cameraAngle, shotSize,
  subjectView, subjectFraming} · movement {type, speed} · final {same 4}.
- `simulation_instruction` — 19 tokens: setup.config {cameraAngle, shotSize,
  subjectView, framing.position, framing.dutchAngleScale} · setup.kind ·
  dynamic {easing, complementSetup ×5, subjectAwareInterpolation, scale,
  direction, movementMode} · constraints {allFramesVisibility,
  staticDistance, staticCameraSubjectRotation}.
- `prompt_none_mask` — (29,) bool, prompt tokens first, True = value present.

Per §3.4 the four boolean tokens are always "none" in generator exports;
`dutchAngleScale` is too (the studio never emits it). Random exports contain
no Dutch/roll samples (§3.7), and Python's `MovementMode` enum intentionally
has no `"roll"` value. `clip_embeddings_cache.pkl` has no dataset-provenance
sidecar and is separate from the two dataset-derived caches in §2.

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

When a prompt has a meaningful `initial` setup, prompt normalization removes
every `final` field for simple pan, tilt, truck, pedestal, and arc movements;
serialized prompts therefore omit `final` for those cases. `static` likewise
permits no final setup. This prevents an end setup from silently turning a
simple one-axis move into a setup transition.

## 5. Item schema — `SimulationDataset[i]`

| Key | Shape / type | Notes |
|---|---|---|
| `camera_trajectory` | (30, 6) float32 | normalized positions + raw Euler XYZ |
| `subject_trajectory` | (30, 6) float32 | first subject only |
| `subject_volume` | (1, 3) float32 | normalized (w, h, d) |
| `padding_mask` | (30,) bool | all False |
| `cinematography_prompt` | (10, D) float32 | CLIP tokens, −1 rows = none |
| `simulation_instruction` | (19, D) float32 | idem |
| `prompt_none_mask` | (29,) bool | True = valid slot |
| `*_parameters` | list of tuples | (name, value, enum index, embedding) per slot |
| `raw_prompt` / `raw_instruction` | dict | reconstructed, strings only (§3.4) |
| `text_prompt` | str | templated caption (§4.3) |

`collate_fn` stacks everything, transposes both token grids to
(tokens, batch, D), and exposes the captions as `text_prompts`.

## 6. Subject motion vocabulary (10 types)

For the expected one-instruction schema-v1 profile, subject paths are
resampled to exactly 30 frames. Their ground baseline is y = height/2 +
offset (a path such as `bounce` can add vertical displacement); yaw roughly
tracks the path heading. Per-sample randomization: position offset
x/z ∈ [−2, 2], y ∈ [0, 1], radius ×[0.7, 1.3], yaw offset [0, 2π), speed
×[0.7, 1.3], 30 % direction reversal, random phase.

| type | base path (scene units) |
|---|---|
| static | fixed point on an r = 5 circle (30 % chance of a tiny wobble when randomized) |
| circular | full circle, r = 5 |
| linear | straight line through the origin from an r = 8 circle (cyclic back-and-forth when randomized) |
| zigzag | 8 along z, x = 4·sin(6πt) — 3 periods |
| spiral | radius 0 → 6 over 2 turns |
| figureEight | x = 6·sin θ, z = 3·sin 2θ |
| wave | 10 along z, x = 3·sin(6πt) |
| pendulum | ±60° swing on an 8-unit arm, 2 cycles |
| orbital | r = 8 around (4, 0, 0), plane tilted 30° |
| bounce | 8 along x, 3 damped bounces of height 3 |

Optional "movement noise" adds per-axis sinusoids over the frame index
(defaults: amplitude 0.1 pos / 0.02 rad rot, frequency 0.5; **off** by
default). `movement_types.txt` caches the per-file type; the
"static"/"dynamic" eval sets are just `allowed_movement_types` filters over
it. The movement cache is created only when `allowed_movement_types` is
requested. `movement_types.meta.json` uses the same lightweight fingerprint
policy as the normalization cache; a missing or nonmatching sidecar
regenerates the index without hashing all simulation payloads.

## 7. Gotchas (recap)

1. Booleans/numbers of prompts & instructions are not in the files (§3.4);
   the always-none tokens are a format property, not a loader bug.
2. Generate one subject, one instruction, and 30–30 frames. Python tensors
   hard-code 30 frames and zero padding; the trusted-input loader does not
   validate or reshape another generator profile.
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

Generated in-browser by the studio's export loop, per simulation: sample
subjects (class-weighted chair / table / laptop / book / tree / car /
bicycle, per-class Gaussian dimensions with a 0.05 floor); assign each a
weighted-random movement type and synthesize 30 frames with §6's
randomization (+ optional sinusoidal noise). Sample a cinematography prompt —
`initial` fully random over the enums, `movement.type` uniform over the
17-move random-export subset in §4.4, and each permitted `final` field kept
with p = 0.5 subject to per-movement validity rules (e.g. dollyIn may not end
on a wider shot; simple moves omit `final`) — and translate it to a
simulation instruction (movement →
locked-axis constraint set, speed → easing, auto-generated complement setup
such as the next-closer shot size for dollyIn or the next view around the
arc). The camera is then solved rule-based: start/end poses from
Gaussian-jittered vertical-angle / subject-view geometry (std 0.1 rad) with
distance fitted by projecting a region of interest (attention box = a
half-size octant of the subject by default, blended toward the full box for
wide shots, scaled per shot size), and framing fixed by rotating the
projected box into the requested frame section; in-between frames by
lerp/slerp — subject-aware (interpolating subject-relative distance and
bearing) for follow/track/arc/dolly — with per-frame constraint blending
(full strength mid-clip, fading to zero at the endpoints) and a final
30°/frame rotation-rate clamp (translation speed/acceleration clamps exist
but the generator leaves them unset). Everything is ×1000-quantized,
msgpack-packed per §2, enum strings deduplicated into
`parameter_dictionary.msgpack` in encounter order, accompanied by the
schema-v1 manifest, and zipped uncompressed (STORE) as
`cinematography_dataset.zip`.
