# E.T. dataset — data contract

Everything a consumer (human or AI agent) needs to use E.T. data
**without reading `third_parties/DIRECTOR` or the E.T. creation code**.

- Source of truth for constants: DIRECTOR
  `configs/dataset/standardization/0300.yaml`. If this file and that config
  disagree, the config wins — then update this file.
- Last verified against: `third_parties/DIRECTOR` @ `<commit>`.

---

## 1. At a glance

| Property | Value |
|---|---|
| Sample id | `{year}_{videoid}_{shot:05d}_{chunk:05d}`, e.g. `2011_KAeAqaA0Llg_00005_00001` |
| Frame rate | 25 fps |
| Sequence length | ≤ 300 frames (12 s), zero-padded to 300 (`num_cams = 300`) |
| Units | meters (positions), meters/frame (velocities) |
| World frame | origin = main character's position at frame 0; y points **down** (SLAHMR convention) |
| Camera pose | 4×4, used everywhere as **camera pose in world** (translation = camera position) |
| Camera feature | 9 = 6 (rot6d) + 3 (translation: frame 0 absolute, frames 1+ per-frame deltas), z-scored |
| Character feature | 3 (center xyz), same velocity encoding, z-scored with a single stat pair |
| Caption features | CLIP ViT-B/32: per-token `(≤77, 512)` and pooled `(512,)` |
| Motion labels | 27 classes for camera and character (+ pad value **27**), one label per velocity step (length N−1) |
| Splits | `{set_name}_{train|test}_split.txt` in dataset root; default `set_name = "mixed"` |

## 2. Directory layout & file formats

```
<dataset root>/
├── traj/{id}.txt            # KITTI: one line per frame = 12 floats ("%.6e"),
│                            #   row-major 3×4 [R|t] of the camera pose (shifted world)
├── traj_raw/{id}.txt        # same, before the origin shift (not used for training)
├── intrinsics/{id}.npy      # float16 (4,) = fx, fy, cx, cy in pixels
├── char/{id}.npy            # (N, 3) character center per frame (shifted world)
├── char_raw/{id}.npy        # (N, 3) pre-shift centers; frame 0 = the shift offset
│                            #   (used only to re-center the vertices below)
├── vert_decimated/{id}.npy  # dict {vertices: (N, V, 3), faces: (F, 3)} — viz only
├── caption/{id}.txt         # one-sentence caption (camera + character motion)
├── caption_cam/{id}.txt     # camera-only caption variant
├── caption_clip/seq/{id}.npy    # (L≤77, 512) per-token CLIP features
├── caption_clip/token/{id}.npy  # (512,) pooled CLIP feature
├── cam_segments/{id}.npy    # (N−1,) int camera motion class ids (see §6)
├── char_segments/{id}.npy   # (N−1,) int character motion class ids
└── mixed_{train,test}_split.txt # newline-separated sample ids
```

## 3. Conventions — read before touching poses

1. **Pose direction.** Every consumer (DIRECTOR rendering, the shift
   statistics) treats each stored 4×4 as the camera's pose **in** the
   world (c2w-like: translation = camera position). The E.T. *creation* code
   confusingly names the variable `w2c_poses` — ignore the name; invert the
   matrix if you need world→camera. Sanity check: the mean frame-0 translation
   equals `shift_mean ≈ [0.00, −0.27, −1.24]`, i.e. the camera starts ~1.24 m
   behind and ~0.27 m above (y-down) the character at the origin.
2. **Camera-frame axes** (OpenCV-style, used for the motion labels):
   x = right (trucking), y = down (`boom_bottom` = +y), z = forward
   (`push_in` = +z).
3. **rot6d convention.** The 6 stored numbers are the **first two columns** of
   R (DIRECTOR: `R[:, :, :2]`). Beware: pytorch3d's `matrix_to_rotation_6d`
   takes the first two **rows** — bridge with a transpose if you use it.
   rot6d values are in [−1, 1] and are *not* standardized.
4. **Padding-mask polarity.** DIRECTOR items use `1.0 = valid frame`,
   `0.0 = pad`.
5. **Origin shift.** `char[0] ≈ (0,0,0)` for every sample by construction (the
   whole scene is translated so the character starts at the origin). Camera
   frame-0 statistics (`shift_*`) are therefore camera-relative-to-character.

## 4. Feature encoding (exact)

### 4.1 Camera trajectory → `(9, 300)` feature

```python
t = poses[:, :3, 3]                                # (N, 3) absolute positions
v = concat([t[0:1], t[1:] - t[:-1]])               # frame 0 absolute, rest deltas
v[0]  = (v[0]  - shift_mean) / shift_std           # z-score the origin
v[1:] = (v[1:] - norm_mean)  / norm_std            # z-score the velocities
r6 = first_two_columns(R).reshape(N, 6)            # NOT standardized
feat = concat([r6, v], dim=-1)                     # (N, 9); layout [r6 | txyz]
# zero-pad to 300 frames; stored transposed as (9, 300); padding_mask marks valid frames
```

Decode (`get_matrix`): un-z-score → `cumsum` the velocity rows → Gram–Schmidt
(`rotation_6d_to_matrix`) → transpose → assemble 4×4.

### 4.2 Character center → `(3, 300)` feature

Same velocity trick (frame 0 absolute, rest deltas), but **all** frames are
z-scored with the single pair `norm_mean_h / norm_std_h` (the config has 3
values, so the "all-in-one" branch runs). This only works because frame 0 ≈ 0
(§3.5); `norm_std_h ≈ 0.01` is a *per-frame velocity* scale. **If you feed a
subject whose frame-0 position is not near the origin, re-center it first or
the normalized value explodes.**

### 4.3 Standardization constants (DIRECTOR `configs/dataset/standardization/0300.yaml`)

| Constant | Value | Units / applies to |
|---|---|---|
| `shift_mean` | `[0.00201079, −0.27488501, −1.23616805]` | m, camera frame-0 position |
| `shift_std` | `[1.13433516, 1.19061042, 1.58744263]` | m |
| `norm_mean` | `[7.9399e−05, −9.9862e−05, 4.1294e−04]` | m/frame, camera velocities |
| `norm_std` | `[0.027841, 0.01819818, 0.03138536]` | m/frame (≈ 0.45–0.8 m/s at 25 fps) |
| `norm_mean_h` | `[6.676e−05, −5.084e−05, −7.782e−04]` | m/frame, character (all frames) |
| `norm_std_h` | `[0.0105, 0.006958, 0.01145]` | m/frame |
| `velocity` | `True` | translation stored as deltas |

Training-only detail (not part of the stored data or of `normalize_item`):
DIRECTOR additionally multiplies features by `sigma_data = 0.5` inside its
EDM training/sampling loop.

## 5. Item schema — DIRECTOR `MultimodalDataset[i]`

| Key | Shape / type | Notes |
|---|---|---|
| `traj_filename` | str `"{id}.txt"` | |
| `traj_feat` | `(9, 300)` float32 | §4.1, normalized |
| `padding_mask` | `(300,)` float32 | **1 = valid**, 0 = pad |
| `intrinsics` | ndarray `(4,)` **float16** | fx, fy, cx, cy |
| `char_filename` | str `"{id}.npy"` | |
| `char_feat` | `(3, 300)` float32 | §4.2, normalized |
| `char_raw.char_raw_feat` / `.char_centers` | `(3, 300)` float32 | raw (un-normalized) centers, padded |
| `char_raw.char_vertices` / `.char_faces` | `(300, V, 3)` / `(300, F, 3)` | only if `load_vertices=True` |
| `caption_filename` | str `"{id}"` | |
| `caption_feat` | `(512, 77)` float32 | per-token CLIP, zero-padded, transposed |
| `caption_raw.caption` | str | the text |
| `caption_raw.segments` | `(300,)` int64 | camera classes, **padded with 27** |
| `caption_raw.clip_seq_caption` | `(77, 512)` float32 | untransposed copy |
| `caption_raw.clip_seq_mask` | `(77,)` float32 | 1 = valid token |
| `char_padding_mask`, `caption_padding_mask` | `(300,)` | copies of `padding_mask` |

⚠ DIRECTOR's `TrajectoryDataset.get_feature/get_matrix` **bundle** the
z-scoring with the geometric encode/decode — the features they take and
return are the normalized ones.

## 6. Motion labels (27 classes) and captions

Per velocity step, take the sign pattern `(sx, sy, sz) ∈ {0, +1, −1}³` of the
thresholded velocity — **camera**: relative motion `inv(P_t) @ P_{t+1}` in the
camera frame, ×25 → m/s, static threshold 0.02 m/s, axis-dominance threshold
0.4; **character**: world-frame center velocity, thresholds 0.22 / 0.32. The
class id is the pattern's index in `itertools.product([0, 1, -1], repeat=3)`.
Labels are then mode-filtered (window 56 frames) and chunks < 25 frames merged.

| id | camera | character | | id | camera | character |
|---|---|---|---|---|---|---|
| 0 | static | static | | 14 | truck_right+boom_bottom+pull_out | right+down+backward |
| 1 | push_in | move_forward | | 15 | truck_right+boom_top | right+up |
| 2 | pull_out | move_backward | | 16 | truck_right+boom_top+push_in | right+up+forward |
| 3 | boom_bottom | move_down | | 17 | truck_right+boom_top+pull_out | right+up+backward |
| 4 | boom_bottom+push_in | down+forward | | 18 | trucking_left | move_left |
| 5 | boom_bottom+pull_out | down+backward | | 19 | truck_left+push_in | left+forward |
| 6 | boom_top | move_up | | 20 | truck_left+pull_out | left+backward |
| 7 | boom_top+push_in | up+forward | | 21 | truck_left+boom_bottom | left+down |
| 8 | boom_top+pull_out | up+backward | | 22 | truck_left+boom_bottom+push_in | left+down+forward |
| 9 | trucking_right | move_right | | 23 | truck_left+boom_bottom+pull_out | left+down+backward |
| 10 | truck_right+push_in | right+forward | | 24 | truck_left+boom_top | left+up |
| 11 | truck_right+pull_out | right+backward | | 25 | truck_left+boom_top+push_in | left+up+forward |
| 12 | truck_right+boom_bottom | right+down | | 26 | truck_left+boom_top+pull_out | left+up+backward |
| 13 | truck_right+boom_bottom+push_in | right+down+forward | | 27 | *(padding value only)* | *(padding value only)* |

Captions were generated by prompting Mistral-7B / Mixtral-8x7B with the chunked
segment outline ("Between frames 0 and 154: boom top; …") and asking for one
short factual sentence; the vocabulary is therefore limited to the patterns
above. CLIP ViT-B/32 encodings of these captions are precomputed (§2).

## 7. Gotchas (recap)

1. `intrinsics` is stored float16 — cast before math.
2. Segment files have length N−1 (per velocity step); loaders pad with class
   27 to 300; renderers duplicate the first entry to reach N.
3. rot6d uses the first two **columns** of R; pytorch3d's rot6d uses the first
   two **rows** — transpose when bridging (§3.3).
4. `norm_std_h` is a velocity scale; absolute subject positions far from the
   origin explode after normalization (§4.2).
5. The ×0.5 `sigma_data` scaling is a DIRECTOR training-loop detail, not part
   of the data.
6. All statistics assume 25 fps and ≤ 300 frames; resampling a clip to another
   length changes per-frame velocity statistics and pushes features off the
   training distribution.
7. Upstream storage is half precision (traj text at `%.6e`, verts/poses
   float16) — expect ~1e-3 round-trip noise.

## 8. Appendix — provenance (how the data was made)

Movie shots from CondensedMovies (≥ 5 s at 25 fps; shots whose camera path is
< 1 m are dropped). DROID-SLAM estimates camera poses; PHALP + SLAHMR jointly
reconstruct SMPL humans and the camera in 100-frame windows with 10-frame
overlap; windows are stitched by least-squares scale+translation on the shared
cameras; bodies are tracked with a SORT-style Kalman tracker and the main
character is the track maximizing screen-coverage × track-length. Velocity
outliers (> 3× the 95th-percentile speed) are removed, remaining chunks are
smoothed with a constant-velocity Kalman smoother, cropped to ≤ 300 frames
(the "0300" set), and the whole scene is translated so the character's first
position is the world origin. Motion labels and LLM captions are produced as
in §6; all standardization statistics in §4.3 are computed over this final
"0300 / mixed" set.
