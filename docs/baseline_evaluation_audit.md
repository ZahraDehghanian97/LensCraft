# Baseline and dataset audit — 2026-09-11

This audit follows the LensCraft raw-matrix rotation loss correction. That
failure does not transfer directly to the released CCDM, DIRECTOR/E.T., or
GenDoP checkpoints: their native representations and objectives differ.

## Dataset and coordinate contract

The audited archive is `lenscraft_sim_4fcf7d88-6796-4e19-a8e6-be063e6f99c0`,
with 100,000 samples and generator version 3. A seed-42 sample of 256 clips
covered 76,491 source frames, 512 endpoint labels, and 11,379 required-visibility
checks. The generator's existing geometry and label validators reported no
failures. All 32 existing generator tests also passed.

For 7,680 selected frames, Python conversion and Three.js camera matrices
agreed within `1.28e-7`; visible subject projections agreed within `2.76e-7`
in normalized device coordinates. Euler XYZ radians, the Y-up world,
camera-local negative Z, the OpenGL/OpenCV axis conversion, and fixed-point
decoding by 1000 agree. These are sampled checks, not an exhaustive validation
of every stored clip.

The sampled archive uses focal length 37.52 mm and aspect ratio 1.778 after
quantization, approximately 50.01 degrees horizontal FOV. Constant optics make
their omission from six-DoF trajectory features consistent for this archive.
Datasets with varying optics would need a broader model/evaluation contract.

## Baseline findings

| Model | Native rotation representation | Verified behavior | Remaining limitation |
| --- | --- | --- | --- |
| CCDM | Relative XYZ and screen XY; orientation reconstructed from five scalars | 100,000 valid native projection cases reprojected within `1.54e-13`, with proper decoded rotations; smoothing matches upstream within `2.1e-7` | Five scalars cannot represent general roll or uniquely resolve every extreme orientation; native human target/optics differ from the simulation subjects |
| E.T. | Six-dimensional rotation, converted with Gram–Schmidt | On 960 poses, standardized features exactly matched DIRECTOR; native reconstruction differed by less than `1.8e-7`; world conversion, subject velocities, and EDM2 scaling agree | The checkpoint is trained on its native data distribution, not this simulation archive |
| GenDoP | Quantized quaternion and translation | On 960 poses, conversion and decoded rotation errors were below `2.4e-7`; base-10 scale inversion matches training | Native training uses poses relative to the first camera, without scene/subject conditioning |

E.T. had a separate sampling defect: every batch restarted seeds at zero.
Evaluation now assigns seeds using the configured seed plus absolute sample
offset. Its normalized and unnormalized variants share those same seeds.
Direct adapter calls reserve successive seeds; callers can supply explicit
seeds for reproducible individual requests. E.T. cache keys include a sampling
protocol version so older repeated-seed trajectories are not silently reused.
Other baselines' cache keys are unaffected by this protocol tag.

## Interpretation of the comparison

GenDoP's first-camera-relative trajectories do not specify a world-space
initial pose. The `norm_lenscraft_init` variant translates a generated path
without rotating it, so it cannot resolve this orientation mismatch. In the
existing 128-sample pilot, translating E.T./CCDM paths also reduced the fraction
of frames with the subject in front of the camera. An aligned position alone
therefore does not establish better framing. Supplying the ground-truth
initial orientation would define a different, explicitly conditioned benchmark.

CCDM's adapter declares unused tangent-of-FOV values while its converter uses
45-degree horizontal FOV. The native Unity camera/replay source was not present
in the local or inspected remote provider files, so the correct native
calibration has not been established by this audit. The simulator's 50-degree
FOV alone is not evidence that CCDM's native decoder should be changed.

The archive also has unequal movement frequencies: `track` has 1,131 samples,
`follow` 1,147, and most other camera actions about 5,700–7,700. Static subjects
make up 50.4%. Per-movement scores would help distinguish difficult/rare actions
from an overall model failure. These recorded frequencies are not corrupt labels.

All three baseline pilot caches contained finite trajectories, and the pilot
reported zero GenDoP fallback warnings. At audit time, the full 20,000-sample
run was still processing CCDM; full E.T./GenDoP results were not yet available.
This audit does not establish their final full-test ranking or eliminate the
need to evaluate the corrected LensCraft checkpoint after training.

## Verification and deployment

The full project suite passed all 42 tests in the server's existing Python
environment, using an isolated copy at `/tmp/lenscraft-baseline-audit-vlwfs97s`.
The 11 new E.T. regressions cover distinct sample seeds, batch partitioning,
explicit replay, paired normalization variants, seed validation, and cache
invalidation limited to E.T. The generator's 32 tests passed separately.

Changes are in the local working tree. This audit did not update the live
server checkout, interrupt the running evaluation, or retrain any checkpoint.
E.T. needs a new evaluation with this code to measure the effect of the seed
correction; the defect alone does not establish how much its scores will change.
