# CCDM dataset — data contract

Everything a consumer (human or AI agent) needs to use CCDM data (Cinematographic
Camera Diffusion Model, EG 2024) **without reading
`third_parties/Camera-control` or the original Unity tooling**.

- Source of truth: upstream repo `jianghd1996/Camera-control`, folder
  `[2024][EG]Text+keyframe`. If this file and the upstream code disagree, the
  code wins — then update this file.
- Last verified against:
  `third_parties/Camera-control/[2024][EG]Text+keyframe` @ `<commit>`.

---

## 1. At a glance

| Property | Value |
|---|---|
| Data file | one `data.npy` (numpy dict, `allow_pickle`) — no per-sample files |
| Sample | camera-only trajectory `[T, 5]` + a list of caption sentences |
| Sequence length | variable `T`; padded to 300 by **repeating the last frame**, truncated if longer |
| Frame rate | not encoded; the Unity demo replays one line per `Update()` |
| Feature layout | 5 = `[r_x, r_y, r_z, p_x, p_y]`: camera position **relative to the subject** + subject's on-screen position |
| Units | scene units for `r` (the Unity demo rescales by ×2 at replay, §3.4); `p` is frustum-normalized, ±1 at the frame edge |
| Subject | implicit: the data is subject-relative; no subject trajectory, rotation, or volume is stored |
| Rotation | **not stored** — orientation must be reconstructed at decode time (§3.3) |
| Normalization | global per-dim z-score; `Mean_Std.npy` = 5-vector mean/std over all frames of all sequences |
| Captions | list of template-like sentences per sample; upstream encodes them with frozen CLIP ViT-B/32 projected `encode_text` features (§6) |
| Splits | none in the file; upstream used a random 9:1 split |

## 2. Files & formats

```
<dataset root>/
├── data.npy       # {"cam":  list of float arrays [T_i, 5],
│                  #  "info": list of list-of-sentences (one list per sample)}
└── Mean_Std.npy   # {"Mean": (5,), "Std": (5,)} over np.concatenate(cam, 0)
```

- `data.npy` is available via gdown id `1VxmGy9szWShOKzWvIxrmgaNEkeqGPLJU`.
- `Mean_Std.npy` can be recomputed from `data.npy` (§4.2).
- Upstream-only variants: the classifier split (`train_cam`/`test_cam` +
  6 integer motion labels) and `feature.npy` (classifier features for
  FID/diversity).

## 3. Conventions — read before touching poses

1. **Subject-relative and subject-free.** The data contains no subject
   trajectory. Per frame, `r = camera_position − subject_position`; the world
   frame is effectively the subject's local frame.
2. **Screen position `p`.** With `q` = subject position in the native
   right-handed camera frame (+X left, +Y up, +Z forward):
   `p_x = (q_x/q_z)/tan(hfov/2)` and `p_y = (q_y/q_z)/tan(vfov/2)`, so
   `|p| ≤ 1` ⇔ subject inside the frustum. The FOV itself is **not part of the
   data** — the original Unity consumer applies its own camera FOV at replay
   time, so any pose reconstruction must pick an FOV as a modeling choice.
   The shared standard pose uses OpenCV camera axes (+X right, +Y down,
   +Z forward). The converter bridges these camera bases with
   `R_cv = R_native @ diag(-1, -1, 1)` and reverses that bridge before
   projecting to native screen coordinates. This does not reflect or rescale
   world positions. Omitting the bridge gives an upright camera a 180° roll
   when decoded to simulation; native `[0, 0, 5, 0, 0]` must decode to a
   Three.js/OpenGL camera at `[0, 0, 5]` with zero Euler rotation.
3. **Rotation is not representable beyond look-at.** Only the subject-relative
   position and the subject's screen position exist, so camera **roll** and
   off-subject aim cannot be encoded; the only poses uniquely recoverable from
   the 5-D features are zero-roll, subject-in-frustum look-at poses.
4. **Unity replay quirks (consumer side only).** The demo `camcontrol.cs`
   reads one line per frame (`x y z sx sy`), negates `x` and `sx` (handedness
   bridge) and scales the position by `height = 2` in the character's local
   right/up/forward frame.

## 4. Feature encoding (exact)

### 4.1 Trajectory → `(300, 5)` feature

```python
traj = cam_i                                                  # [T, 5] raw
if T < 300: traj = concat([traj, traj[-1:].repeat(300 - T)])  # last-frame padding
if T > 300: traj = traj[:300]
traj = (traj - Mean) / (Std + 1e-8)                           # z-score, all dims, all frames
```

Decode: `traj * Std + Mean`. Absolute values every frame — no velocity
encoding and no special frame-0 statistics (unlike E.T.).

### 4.2 Normalization constants

`Mean_Std.npy` is computed once over every frame of every sequence
(`d = np.concatenate(data["cam"], 0); Mean, Std = d.mean(0), d.std(0)`).

## 5. Gotchas (recap)

1. Padding repeats the last frame, so padded regions look like a long static
   hold — track valid lengths and mask before any statistics.
2. Roll and orientation are not in the data; only zero-roll, subject-in-frustum
   look-at poses can be reconstructed from the 5-D features (§3.3). Do not
   read a rotation error on out-of-domain poses as a bug.
3. The Unity sign/scale quirks (§3.4) belong to the demo consumer, not to the
   stored data.
4. Upstream training sampled a **random subset** of each sample's caption
   sentences per epoch (joined by `" "`) — the model never saw a fixed
   canonical caption per sample.
5. With the released pretrained weights, prompt "zooms in"/"zooms out" instead
   of "pushes in"/"pulls out" — that is the training vocabulary (upstream
   README).
6. Upstream's exported/generated sequences are denormalized **and
   box-smoothed** (window ±10, 4 passes) — smoothing damps high-frequency
   motion relative to raw diffusion samples.

## 6. Appendix — provenance (how the data was made)

CCDM = "Cinematographic Camera Diffusion Model" (Jiang, Wang, Christie, Liu,
Chen — Eurographics 2024); upstream repo `jianghd1996/Camera-control`, folder
`[2024][EG]Text+keyframe`. Sequences are ≤300-frame, 5-D subject-relative
camera trajectories paired with template-like text descriptions ("The camera
pans to the character. The camera switches from right front view to right back
view. The character is at the middle center of the screen. The camera shoots
at close shot."). Model: DDPM (β 1e-4 → 0.02, T = 1000) over the whole 300×5
sequence with a transformer encoder (4 layers, latent 256, 4 heads, ff 1024);
the text condition is the frozen CLIP ViT-B/32 `encode_text` embedding
(including CLIP's learned text projection, without L2 normalization) added to the
timestep token; classifier-free guidance (cond-mask prob 0.1 at train,
`guide_w = 2.0` at sampling); per epoch, a random subset of each sample's
caption sentences was joined by `" "`. Generated sequences are denormalized
and box-smoothed before export. Upstream evaluation trained a separate
sequence classifier (6 motion classes) and computed FID/diversity on its
features. The Unity 2018.2.13f1 demo replays generated `.txt` files (5 floats
per line) via `camcontrol.cs` with the sign/scale quirks of §3.4.

The pretrained baseline adapter uses OpenAI CLIP directly for this contract.
`CLIPTextModel.pooler_output` omits the learned text projection and cannot
replace `encode_text` when conditioning the released CCDM checkpoint. The
LensCraft dataset's caption features are separate from this pretrained
baseline condition.
