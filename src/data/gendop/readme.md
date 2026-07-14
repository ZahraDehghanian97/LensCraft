# DataDoP dataset (GenDoP) — data contract

Everything a consumer (human or AI agent) needs to use DataDoP data
**without reading `third_parties/GenDoP` or the dataset construction code**.

- Source of truth: upstream repo `3DTopia/GenDoP` (ICCV 2025) —
  `core/provider.py` (loading/tokenization) and
  `dataset/scripts/Dataset_DataDoP.py` (file creation). If this file and that
  code disagree, the code wins — then update this file.
- Last verified against: `third_parties/GenDoP` @ `26099d0`.
- Data download: [huggingface.co/datasets/Dubhe-zmc/DataDoP](https://huggingface.co/datasets/Dubhe-zmc/DataDoP).

---

## 1. At a glance

| Property | Value |
|---|---|
| Sample id | `<VideoID>/<ShotID>`, e.g. `1_0000/shot_0070` (per-sample files, one folder per VideoID) |
| Size | ~29K shots from artistic videos (`0_*` = MovieNet, `1_*` = YouTube) |
| Trajectory | `*_transforms_cleaning.json`: exactly **120** c2w 4×4 poses (cleaned, smoothed, resampled) |
| Camera pose | c2w (translation = camera position), **OpenGL/NeRF axes**: x right, y up, **z backward** |
| Units | none — arbitrary per-shot MonST3R reconstruction scale (monocular; no metric anchor) |
| Frame rate | not encoded; 120 poses uniformly resampled over the shot; tagging assumes 30 fps |
| Subject | none — camera-only dataset (scene context comes from first-frame RGBD instead) |
| Scene condition | first frame `*_rgb.png` + MonST3R-aligned `*_depth.npy` (same arbitrary scale as traj) |
| Captions | `*_caption.json`: `Movement` (motion), `Detailed Interaction`, `Concise Interaction` (directorial) |
| GenDoP feature | 30 poses (every 4th frame) → 10 discrete tokens each (quat 4 + trans 3 + intr 2 + scale 1) = 300 tokens |
| Normalization | per-sample: poses made relative to frame 0, translations scaled to max-norm 1, log-scale kept as a token |
| Splits | `train_valid.txt` (valid ids); train/test = fixed shuffle (seed 42), last `testset_size` held out |

## 2. Directory layout & file formats

```
DataDoP/
├── train_valid.txt                        # newline-separated "<VideoID>/<ShotID>" ids
├── <VideoID>/                             # e.g. 1_0000/
│   ├── <ShotID>_caption.json              # {"Movement": ..., "Detailed Interaction": ...,
│   │                                      #  "Concise Interaction": ...} (plain strings)
│   ├── <ShotID>_rgb.png                   # first (non-black) frame, video crop resolution
│   ├── <ShotID>_depth.npy                 # float (H, W) MonST3R depth of that frame
│   ├── <ShotID>_intrinsics.txt            # one line per frame: 9 floats = row-major 3×3 K
│   │                                      #   [fx 0 cx / 0 fy cy / 0 0 1], pixels
│   ├── <ShotID>_traj.txt                  # raw MonST3R poses, one line per frame:
│   │                                      #   timestamp tx ty tz qw qx qy qz  (⚠ w-FIRST)
│   ├── <ShotID>_transforms_cleaning.json  # THE training trajectory (see below)
│   └── <ShotID>_traj_cleaning.png         # visualization of the cleaned trajectory
```

`*_transforms_cleaning.json` schema (nerfstudio-like):

```json
{
  "w": 640, "h": 360,                  // inferred as 2*int(cx), 2*int(cy)
  "fl_x": ..., "fl_y": ...,            // pixels, from MonST3R frame-0 intrinsics
  "cx": ..., "cy": ...,
  "frames": [                          // exactly 120 entries
    {"transform_matrix": [[...4x4...]],  // c2w, OpenGL convention
     "monst3r_im_id": 1}                 // just index+1, NOT an original frame index
  ]
}
```

Metadata (`metadata.csv` in the upstream repo): `ClipID, YouTubeID, CropSize
(ffmpeg w:h:x:y), StartTime, EndTime` — enough to re-download and re-crop the
source clip; raw video is not distributed.

## 3. Conventions — read before touching poses

1. **Pose direction.** Every stored matrix is camera-to-world (translation =
   camera position). The cleaning code names its argument `w2c_poses` — ignore
   the name (same trap as E.T./DIRECTOR); invert if you need world→camera.
2. **Axes.** `*_traj.txt` is in MonST3R/OpenCV convention (x right, y down,
   z forward). Building `*_transforms_cleaning.json` negates rotation columns
   1–2 (`matrix[:3, 1:3] *= -1`), so the JSON poses are **OpenGL/NeRF**:
   x right, y up, z backward. The motion tags confirm it: pattern +z →
   "move backward", +y → "move up", +x → "move right".
3. **`*_traj.txt` quaternion order is `qw qx qy qz`** (scalar first). This is
   *not* the standard TUM `qx qy qz qw` despite the TUM-like layout — a loader
   that assumes standard TUM silently produces garbage rotations.
4. **No metric scale.** Each shot lives in its own MonST3R scale; positions
   across shots are not comparable. GenDoP handles this by per-sample
   normalization (§4.2) and encodes `log10(scale)` as a token so magnitude
   information survives.
5. **Frame alignment.** Up to 15 leading fully-black frames are dropped when
   building a sample; `_rgb.png`, `_depth.npy`, and line 0 of
   `_intrinsics.txt` / `_traj.txt` all refer to the same (first kept) frame.
6. **Intrinsics are per-frame but treated as constant** — only line 0 is used
   for the transforms JSON; `w`/`h` are back-derived as `2*int(cx)`,
   `2*int(cy)` (assumes a centered principal point; can be off by a pixel vs
   the actual crop size).

## 4. Feature encoding (exact, GenDoP `ShotTrajDataset`)

### 4.1 120 frames → 30 poses → 300 discrete tokens

```python
frames = transforms_cleaning["frames"]              # 120 c2w poses
idx = arange(120)[::4][:30]                         # frames 0, 4, ..., 116
c2ws = poses[idx]                                   # (30, 3, 4)
# intrinsics rescaled to the 512×512 model resolution:
intr = [fx*512/w, fy*512/h, cx*512/w, cy*512/h, 512, 512]

# per-sample normalization (normalized_cameras=True):
c2ws  = inv(c2ws[0]) @ c2ws                         # frame 0 → identity
scale = max(norm(c2ws[:, :3, 3])) + 1e-5
c2ws[:, :3, 3] /= scale                             # translations in [-1, 1]

token = [quat(R) (w first) | t (3) | fx/512, fy/512]   # 9 floats per pose
bins  = 256
q_t   = ((token[:7] + 1) / 2   * bins).clip(0, bins)   # quat+trans: [-1,1] → bin
q_i   = ( token[7:9] / 10      * bins).clip(0, bins)   # intrinsics: [0,10] → bin
q_s   = ((log10(scale) + 2) / 4 * bins).clip(0, bins)  # scale: [1e-2,1e2] → bin
coords = concat per frame [q_t | q_i | q_s]             # 10 ints × 30 = 300
coords += 3                                             # 0=PAD, 1=BOS, 2=EOS
```

Decode (`token_to_camera` + eval): un-bin, quaternion → R (Gram-Schmidt-free,
direct), reassemble c2w, multiply translations back by `10**(4*s/bins - 2)`;
intrinsics decoded with `cx = W/2`, `cy = H/2`.

### 4.2 Conditions

- **Text**: one caption string; `Movement` key for the *motion* checkpoints,
  `Concise Interaction` for the *directorial* ones (`text_key` option).
  Encoded with the frozen SD 2.1 CLIP text encoder, 77 condition tokens.
- **RGBD** (`depth+image+text` mode): `_rgb.png` and `_depth.npy` center-
  cropped / zero-padded to 512×512 (RGB float in [0,1]; depth `(1, H, W)`),
  encoded with frozen CLIP ViT-H-14 (laion2B).

### 4.3 Item schema — `ShotTrajDataset[i]`

| Key | Shape / type | Notes |
|---|---|---|
| `cameras` | `(30, 18)` float32 | flattened 3×4 c2w (12) + `[fx, fy, cx, cy, 512, 512]` |
| `coords` | `(300,)` long | discrete tokens, already offset by +3 |
| `text` | str | one caption (random choice, but the list has length 1) |
| `rgb` | `(3, 512, 512)` float32 | crop/pad, RGB in [0,1] |
| `depth` | `(1, 512, 512)` float32 | crop/pad |
| `path` | str | `<root>/<VideoID>/<ShotID>` file prefix |
| `len` | int | 300 |

Collate pads with PAD and wraps with BOS/EOS; labels mask condition+BOS with
−100; sequences longer than `max_seq_length` are truncated *without* EOS.

## 5. Motion tags and captions

Tags are computed on the 120-frame cleaned trajectory. Per step, relative
motion `inv(P_t) @ P_{t+1}` (camera-frame), translation ×30 (assumed fps):

- **Translation**: sign pattern `(sx, sy, sz) ∈ {0, +1, −1}³` with static
  threshold 0.02 and relative axis-dominance threshold 0.4
  (`(|a|−|b|)/max(|a|,|b|)` suppresses the weaker axis); class id = index in
  `itertools.product([0, 1, -1], repeat=3)` — the same 27-pattern scheme as
  E.T., but in **OpenGL axes** (+y = up, +z = backward) with plain-English
  names ("move right and up", not "truck_right+boom_top").
- **Rotation**: rotation-vector of the relative rotation; the dominant axis
  (threshold 0.005 rad) picks one of 7 classes: `static`, `pitch up/down`,
  `yaw left/right`, `roll left/right`.
- Labels are mode-smoothed (window 18) and chunks < 10 frames merged; each
  chunk becomes an outline line like
  `Between frames 0 and 20: move_right - yaw_left`.

Captions (all OpenAI-generated, so vocabulary is *not* closed-set):

1. `Movement` — GPT-4o (some batches 4o-mini) writes one short factual
   sentence from the chunked outline.
2. `Detailed Interaction` / `Concise Interaction` — GPT-4o-mini sees a 4×4
   grid of 16 frames plus the Movement caption and describes camera–scene
   interaction, one detailed and one concise variant.

## 6. Gotchas (recap)

1. `_traj.txt` quaternions are **wxyz**, not TUM's xyzw (§3.3).
2. The cleaning keeps only the **first** surviving chunk after outlier removal
   (velocity ≥ 18× the 95th percentile dropped, runs ≤ 5 frames dropped), so
   the 120 poses may cover only part of the shot's real time span.
3. Resampling evaluates `t = i/120` for `i = 0..119` — the final original pose
   is never exactly reached.
4. Only camera **positions** are Kalman-smoothed (constant-velocity, process
   std 0.5, measurement std 1); rotations keep raw MonST3R noise.
5. Per-shot arbitrary scale: never mix translations across samples without the
   per-sample normalization + scale token (§4.1); `scale` is clipped to
   `[1e-2, 1e2]` by the log-token encoding.
6. `monst3r_im_id` is `index+1` into the resampled 120, not a source frame id.
7. Train/test is not a published split file: sorted ids are shuffled with seed
   42 and the last `testset_size` (default 1!) held out — reproduce exactly or
   define your own split.
8. Tagging assumed 30 fps and the 120-frame resampling changes per-step
   velocity magnitudes; the 0.02 static threshold is calibrated to the
   *resampled* trajectories, not to real seconds.
9. A sample is valid only if all six files exist; upstream's loader silently
   substitutes a random other sample on any per-item error — don't inherit
   that behavior into evaluation code.

## 7. Appendix — provenance (how the data was made)

Shots come from MovieNet (`0_*`) and curated artistic YouTube videos (`1_*`):
download → black-border crop (ffmpeg) → shot-boundary detection and splitting
→ filtering (quality/length) → per-shot frame extraction. MonST3R (a
DUSt3R-based dynamic-scene reconstructor) estimates per-frame camera poses
(`pred_traj.txt`), per-frame intrinsics (`pred_intrinsics.txt`), and aligned
depth maps. Leading all-black frames are trimmed; the first kept frame's RGB
and depth become the sample's RGBD. Poses are converted OpenCV→OpenGL, cleaned
(velocity outliers > 18× the 95th percentile removed, chunks ≤ 5 frames
dropped, first chunk kept), positions Kalman-smoothed, and slerp/lerp-resampled
to exactly 120 poses → `*_transforms_cleaning.json`. Motion tags (27
translation × 7 rotation patterns, §5) are chunked into an outline; GPT-4o(-
mini) turns the outline into the `Movement` caption and, with a 16-frame
contact sheet, into the `Detailed`/`Concise Interaction` captions. GenDoP
itself (the consumer) is an autoregressive transformer (OPT-style, 24×1024,
trained with bf16) over the 300-token sequences with CLIP text / ViT-H image+
depth conditions.
