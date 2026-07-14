# Simulation dataset (LensCraft) — data contract

Everything a consumer (human or AI agent) needs to use the synthetic
simulation data **without reading the studio's TypeScript generator or
`src/data/simulation/`**.

- Source of truth: the studio's export code (`src/service/dataset/generate.ts`,
  `simulationFormatter.ts`, `parameterDictionary.ts`) for the files, and
  `src/data/simulation/loader.py` / `dataset.py` for consumption. If this file
  and the code disagree, the code wins — then update this file.
- Last verified against: studio @ `<commit>`, loader @ `<commit>`.
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
| Subject | box: per-frame center position + Euler XYZ, plus one (w, h, d) volume; center y ≈ height/2 |
| Quantization | every float stored as `round(x·1000)` int (msgpack); decode ÷1000 (~±5e-4 noise) |
| Labels | cinematography prompt + simulation instruction — **enum strings only** (§3.4) — plus a templated English caption rendered at load time |
| Normalization | dataset-level per-axis z-score of positions and volume only; rotations stay raw radians |
| Splits | none stored; `CameraTrajectoryDataModule` random-splits by fraction at runtime |

## 2. Directory layout & file formats

```
<dataset root>/                       # unzipped cinematography_dataset.zip
├── parameter_dictionary.msgpack      # {"keys": [path, ...], "values": [[str, ...], ...]}
│                                     #   shared string table for prompts/instructions
├── simulation_0000.msgpack ...       # one file per simulation (sorted glob simulation_*.msgpack)
├── movement_types.txt                # generated on first load: "<file>|<movementType>" lines
└── normalization_parameters.json     # generated on first load: mean/std for camera_position,
                                      #   subject_position, subject_dimensions
```

Each `simulation_*.msgpack` is a 4-element msgpack list (all numbers are
×1000 ints):

```
[cinematography_refs,   # [[key_idx, value_idx], ...] into parameter_dictionary
 simulation_refs,       # same, for the simulation instruction
 subjects,              # [{i: id, c: class, d: [w, h, d],
                        #   f: [[px, py, pz, rx, ry, rz] × 30], m: movementType,
                        #   a?: [ax, ay, az, aw, ah, ad]}]   # optional attention box
 camera_frames]         # [[px, py, pz, rx, ry, rz, focalLength, aspectRatio] × 30]
```

Dictionary paths are `__`-joined key chains prefixed `cinematography` /
`simulation` (e.g. `cinematography__movement__type`); values are the sorted
distinct strings seen for that path across the whole export.

## 3. Conventions — read before touching poses

1. **World frame.** three.js: right-handed, y up, ground plane at y = 0. The
   subject `position` is the box **center**, so a grounded subject sits at
   y ≈ height/2 (± the generator's random y offset). The scene is **not**
   subject-centered (unlike E.T.) — use `recenter_rescale_sim` before any
   E.T.-style normalization.
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
6. **One instruction, one subject.** References from multiple prompts
   collapse into a single object at reconstruction (last write wins) and the
   loader reads `subjectsInfo[0]` only; the supported export is
   `instructionCount = 1`, `subjectCount = 1`, frame range fixed at 30–30.

## 4. Feature encoding (exact)

### 4.1 Trajectories → tensors (`SimulationDataset`)

```python
camera  = [[px, py, pz, rx, ry, rz]] * 30        # float32 (÷1000 already applied)
subject = same layout, subjectsInfo[0]           # (30, 6)
volume  = [[w, h, d]]                            # (1, 3)

camera[..., :3]  = (camera[..., :3]  - cam_pos_mean)  / cam_pos_std   # per axis
subject[..., :3] = (subject[..., :3] - subj_pos_mean) / subj_pos_std
volume           = (volume - dim_mean) / dim_std
# rotations (indices 3:6) are NOT normalized
padding_mask = zeros(30, bool)                   # nothing is padded
```

Stats are computed once over every frame of every file in the directory and
cached in `normalization_parameters.json` (zero stds → 1). They live in a
**process-wide singleton**: the first simulation dataset touched fixes them
for the whole process (fallback lookup: `SIMULATION_DATA_PATH` env var).

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
`dutchAngleScale` is too (the studio never emits it).

### 4.3 Caption

`text_prompt` is rendered at load time from the prompt plus the subject's
movement type with closed-vocabulary templates, e.g. *"As the character moves
in a circular path, the camera pushes in, creating a dynamic relationship."*
Nothing textual is stored in the files.

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

All generators emit exactly 30 frames at y = height/2 (+ offset); yaw roughly
tracks the path heading. Per-sample randomization: position offset
x/z ∈ [−2, 2], y ∈ [−0.5, 1], radius ×[0.7, 1.3], yaw offset [0, 2π), speed
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
it.

## 7. Gotchas (recap)

1. Booleans/numbers of prompts & instructions are not in the files (§3.4);
   the always-none tokens are a format property, not a loader bug.
2. The loader hard-codes 30 frames and zero padding; export with
   `instructionCount = 1` and the frame-range slider at 30–30 (the studio UI
   defaults to 30–300 — longer files break batch collation).
3. `movement_types.txt` and `normalization_parameters.json` are **reused if
   present**: regenerate into a fresh directory or delete both, or you will
   train with stale stats and stale movement filters.
4. Normalization stats are per-directory and a process-wide singleton —
   loading two simulation datasets in one process silently normalizes the
   second with the first's stats.
5. A corrupt file makes `parse_simulation_file_to_dict` return `None` and
   `__getitem__` crash with a `TypeError` — validate after unzipping.
6. ×1000 quantization: ~5e-4 absolute noise on positions and radians; don't
   chase sub-millimeter round-trip errors.
7. Train/val/test membership is a runtime `random_split` — it changes with
   the seed and the split fractions; persist your own split file if you need
   stable membership.
8. Need FOV? Read camera row index 6 (÷1000 already applied) before
   featurization — the 6-DoF features drop it.
9. Padding-mask polarity is **True = pad** throughout this repo (all-False
   here); DIRECTOR's convention is the opposite.

## 8. Appendix — provenance (how the data was made)

Generated in-browser by the studio's export loop, per simulation: sample
subjects (class-weighted chair / table / laptop / book / tree / car /
bicycle, per-class Gaussian dimensions with a 0.05 floor); assign each a
weighted-random movement type and synthesize 30 frames with §6's
randomization (+ optional sinusoidal noise). Sample a cinematography prompt —
`initial` fully random over the enums, `movement.type` uniform over all 21
camera moves (incl. dutch and dolly-zoom combos), each `final` field kept
with p = 0.5 subject to per-movement validity rules (e.g. dollyIn may not end
on a wider shot) — and translate it to a simulation instruction (movement →
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
`parameter_dictionary.msgpack`, and zipped uncompressed (STORE) as
`cinematography_dataset.zip`.
