from __future__ import annotations

import argparse
import math

import torch

from utils.pytorch3d_transform import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
)
from data.ccdm.convertor import CCDMConvertor
from data.et.convertor import ETConvertor
from data.simulation.convertor import SIMConvertor
from data.gendop.convertor import GenDoPConvertor

# Below this, rotation differences are float32 noise from the geodesic metric,
# not real disagreement.
ROT_NOISE_DEG = 0.05


# --------------------------------------------------------------------------- #
# data generators
# --------------------------------------------------------------------------- #
def random_rotations(B, T, max_angle=1.0, device="cpu", dtype=torch.float32):
    angles = (torch.rand(B, T, 3, device=device, dtype=dtype) * 2 - 1) * max_angle
    return euler_angles_to_matrix(angles, "XYZ")


def make_native(name, B, T, pos_scale=3.0, device="cpu", dtype=torch.float32):
    """A valid sample in each convertor's NATIVE format."""
    if name in ("simulation", "lens_craft"):
        pos = (torch.rand(B, T, 3, device=device, dtype=dtype) * 2 - 1) * pos_scale
        euler = (torch.rand(B, T, 3, device=device, dtype=dtype) * 2 - 1) * 1.0
        return torch.cat([pos, euler], dim=-1)
    if name == "et":
        rot6d = matrix_to_rotation_6d(random_rotations(B, T, device=device, dtype=dtype))
        trans = torch.randn(B, T, 3, device=device, dtype=dtype) * 0.1
        return torch.cat([rot6d, trans], dim=-1)
    if name == "ccdm":
        d = torch.randn(B, T, 3, device=device, dtype=dtype)
        d = d / d.norm(dim=-1, keepdim=True)
        rel = d * (2 + torch.rand(B, T, 1, device=device, dtype=dtype) * 3)
        px = (torch.rand(B, T, 1, device=device, dtype=dtype) * 2 - 1) * 0.5
        py = (torch.rand(B, T, 1, device=device, dtype=dtype) * 2 - 1) * 0.5
        return torch.cat([rel, px, py], dim=-1)
    if name == "gendop":
        R = random_rotations(B, T, device=device, dtype=dtype)
        t = (torch.rand(B, T, 3, device=device, dtype=dtype) * 2 - 1) * pos_scale
        Tm = torch.eye(4, device=device, dtype=dtype).expand(B, T, 4, 4).clone()
        Tm[..., :3, :3] = R
        Tm[..., :3, 3] = t
        return Tm
    raise ValueError(name)


def make_sim_lookat(B, T, device="cpu", dtype=torch.float32):
    """simulation 6dof for zero-roll cameras that LOOK AT the subject at origin
    -- this is the set CCDM can actually represent."""
    d = torch.randn(B, T, 3, device=device, dtype=dtype)
    d[..., 1] *= 0.3                      # bias toward the equator (keep z away from up)
    d = d / d.norm(dim=-1, keepdim=True)
    cam = d * (2 + torch.rand(B, T, 1, device=device, dtype=dtype) * 3)

    z = -cam / cam.norm(dim=-1, keepdim=True)            # camera +z looks at subject
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(z)
    x = torch.cross(up, z, dim=-1)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)                   # columns x,y,z
    euler = matrix_to_euler_angles(R, "XYZ")
    return torch.cat([cam, euler], dim=-1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def sim_to_RT(sim):
    return euler_angles_to_matrix(sim[..., 3:6], "XYZ"), sim[..., :3]


def pos_err(t0, t1):
    return (t0 - t1).norm(dim=-1)


def rot_err_deg(R0, R1):
    """Stable rotation error: chordal distance converted to an angle, computed
    so it does NOT amplify float noise at identity the way acos does."""
    diff = (R0 - R1).reshape(*R0.shape[:-2], 9).norm(dim=-1)   # Frobenius, in [0, 2*sqrt2]
    # ||R0 - R1||_F = 2*sqrt(2)*sin(theta/2)  ->  theta = 2*asin(clamp(d/(2*sqrt2)))
    return torch.rad2deg(2.0 * torch.asin((diff / (2.0 * math.sqrt(2.0))).clamp(0.0, 1.0)))


def verdict(ok):
    return "PASS" if ok else "FAIL  <-- look here"


# --------------------------------------------------------------------------- #
# 1. native round-trip
# --------------------------------------------------------------------------- #
def test_native_roundtrip(B, T, device, dtype):
    print("\n" + "=" * 78)
    print("1. NATIVE ROUND-TRIP  (geometry only, no normalization)")
    print(f"   rot err below ~{ROT_NOISE_DEG} deg is float32 noise, not disagreement")
    print("=" * 78)

    convertors = {
        "simulation": SIMConvertor(),
        "et": ETConvertor(),
        "ccdm": CCDMConvertor(),
        "gendop": GenDoPConvertor(),
    }
    all_ok = True
    print(f"{'convertor':<12}{'native max|Δ|':>16}{'pos err':>14}{'rot err (deg)':>16}   result")
    for name, conv in convertors.items():
        try:
            native = make_native(name, B, T, device=device, dtype=dtype)
            T1 = conv.to_standard(native, None, None)[0]
            native2 = conv.from_standard(T1, None, None)[0]
            T2 = conv.to_standard(native2, None, None)[0]
            nd = (native - native2).abs().max().item()
            pe = pos_err(T1[..., :3, 3], T2[..., :3, 3]).mean().item()
            re = rot_err_deg(T1[..., :3, :3], T2[..., :3, :3]).mean().item()
            # CCDM's p_x/p_y recovery is slightly loose in native units, but the
            # reconstructed *pose* is consistent -- judge on pose, not raw native.
            ok = (pe < 1e-3) and (re < ROT_NOISE_DEG)
            all_ok &= ok
            note = "  (native Δ is in p_x/p_y; pose round-trips)" if name == "ccdm" and nd > 1e-3 else ""
            print(f"{name:<12}{nd:>16.3e}{pe:>14.3e}{re:>16.3e}   {verdict(ok)}{note}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"{name:<12}{'ERROR':>16}   {type(exc).__name__}: {exc}")
    return all_ok


# --------------------------------------------------------------------------- #
# 2. cross round-trip
# --------------------------------------------------------------------------- #
def test_cross_roundtrip(B, T, device, dtype):
    print("\n" + "=" * 78)
    print("2. CROSS ROUND-TRIP  (convert_to_target, geometry + resampling, no norm)")
    print("   simulation -> X -> simulation")
    print("=" * 78)

    from data.convertor.convertor import convert_to_target

    sim_general = make_native("simulation", B, T, device=device, dtype=dtype)
    sim_lookat = make_sim_lookat(B, T, device=device, dtype=dtype)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    # (label, source sim trajectory, target, rot tol deg, note)
    cases = [
        ("et",             sim_general, "et",   ROT_NOISE_DEG, ""),
        ("gendop",         sim_general, "gendop", ROT_NOISE_DEG, ""),
        ("ccdm look-at",   sim_lookat,  "ccdm", 5.0,  "  (CCDM's valid domain)"),
        ("ccdm rnd-rot",   sim_general, "ccdm", 5.0,  "  (out of CCDM domain: expected large)"),
    ]

    all_ok = True
    print(f"{'via':<14}{'pos err':>14}{'rot err (deg)':>16}   result")
    for label, sim, tgt, rtol, note in cases:
        try:
            fwd = convert_to_target("simulation", tgt, sim, None, None, mask,
                                    target_len=T, need_denormal=False, need_normal=False)
            back = convert_to_target(tgt, "simulation", fwd[0], fwd[1], fwd[2], fwd[3],
                                     target_len=T, need_denormal=False, need_normal=False)[0]
            R0, t0 = sim_to_RT(sim)
            R1, t1 = sim_to_RT(back)
            pe = pos_err(t0, t1).mean().item()
            re = rot_err_deg(R0, R1).mean().item()
            ok = (pe < 5e-3) and (re < rtol)
            # rnd-rot ccdm failing is expected and informative, not a code bug
            if label == "ccdm rnd-rot":
                print(f"{label:<14}{pe:>14.3e}{re:>16.3e}   {'(expected large)':<18}{note}")
            else:
                all_ok &= ok
                print(f"{label:<14}{pe:>14.3e}{re:>16.3e}   {verdict(ok)}{note}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"{label:<14}{'ERROR':>14}   {type(exc).__name__}: {exc}")
    return all_ok


# --------------------------------------------------------------------------- #
# 3. ET subject-scale diagnostic
# --------------------------------------------------------------------------- #
def test_et_subject_scale(B, T, device, dtype):
    print("\n" + "=" * 78)
    print("3. ET SUBJECT-SCALE DIAGNOSTIC  (normalization layer, REALISTIC motion)")
    print("=" * 78)

    from data.et.dataset import ETDataset
    from data.et.config import STANDARDIZATION_CONFIG_TORCH as CFG

    et = ETConvertor()
    norm_std = CFG["norm_std"].to(device=device, dtype=dtype)       # camera velocity std (~0.02)
    shift_std = CFG["shift_std"].to(device=device, dtype=dtype)     # camera first-frame std (~1.2)
    norm_std_h = CFG["norm_std_h"].to(device=device, dtype=dtype)   # subject "velocity" std (~0.01)

    # Realistic camera: smooth motion -> per-frame velocity at the training scale.
    vel = torch.randn(B, T, 3, device=device, dtype=dtype) * norm_std
    pos0 = torch.randn(B, 1, 3, device=device, dtype=dtype) * shift_std
    cam_pos = pos0 + torch.cumsum(vel, dim=1)
    R = random_rotations(B, T, max_angle=0.3, device=device, dtype=dtype)
    transforms = torch.eye(4, device=device, dtype=dtype).expand(B, T, 4, 4).clone()
    transforms[..., :3, :3] = R
    transforms[..., :3, 3] = cam_pos
    cam_et = et.from_standard(transforms, None, None)[0]            # velocity-encoded [B,T,9]

    # Realistic subject: smooth motion, ABSOLUTE positions ~ O(1), fed through
    # ETConvertor.from_standard so the test exercises the real pipeline (which
    # velocity-encodes the subject: frame 0 absolute, rest per-frame deltas).
    subj0 = torch.randn(B, 1, 3, device=device, dtype=dtype) * 1.0
    subj_pos = subj0 + torch.cumsum(
        torch.randn(B, T, 3, device=device, dtype=dtype) * norm_std_h, dim=1
    )
    subj_tf = torch.eye(4, device=device, dtype=dtype).expand(B, T, 4, 4).clone()
    subj_tf[..., :3, 3] = subj_pos
    subj_feat = et.from_standard(transforms, subj_tf, None)[1]

    cam_n, subj_n, _ = ETDataset.normalize_item(cam_et.clone(), subj_feat.clone(), None, True)
    cam_max = cam_n[:, 1:].abs().max().item()
    subj_max = subj_n[:, 1:].abs().max().item()
    ratio = subj_max / max(cam_max, 1e-8)

    print(f"   camera  positions  ~ O({cam_pos.abs().mean().item():.2f}),  per-frame velocity ~ O({vel.abs().mean().item():.3f})")
    print(f"   subject positions  ~ O({subj_pos.abs().mean().item():.2f})  (absolute, smooth)")
    print(f"   norm_std_h (subject divisor)   : {norm_std_h.tolist()}")
    print(f"   camera  AFTER norm, abs max     : {cam_max:10.3f}   (frames 1+, expected O(1-10))")
    print(f"   subject AFTER norm, abs max     : {subj_max:10.3f}   (frames 1+, expected O(1-10))")
    print(f"   subject / camera ratio          : {ratio:10.2f}")

    bug = (subj_max > 20.0) and (ratio > 8.0)
    if bug:
        print(f"\n   >>> Subject delta frames normalize ~{ratio:.0f}x larger than the camera.")
        print(f"   >>> ETConvertor.from_standard should hand normalize_item a velocity-")
        print(f"   >>> encoded subject (frame 0 absolute, rest deltas) so norm_std_h")
        print(f"   >>> (~0.01, a per-frame VELOCITY std) applies to deltas -- check that")
        print(f"   >>> encoding has not regressed.")
    else:
        print("\n   subject and camera are on comparable scales (delta frames) -- OK.")
    return not bug


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), torch.float32
    B, T = args.batch, args.seq
    print(f"device={device}  batch={B}  seq_len={T}  seed={args.seed}")

    r1 = test_native_roundtrip(B, T, device, dtype)
    r2 = test_cross_roundtrip(B, T, device, dtype)
    r3 = test_et_subject_scale(B, T, device, dtype)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  1. native geometry round-trip : {verdict(r1)}")
    print(f"  2. cross convert_to_target     : {verdict(r2)}")
    print(f"  3. ET subject scale            : {verdict(r3)}")
    print("=" * 78)
    if r1 and r2 and not r3:
        print("Geometry is correct (incl. CCDM on its valid domain); the failure is the")
        print("ET subject NORMALIZATION (test 3) -- that is what to fix.")


if __name__ == "__main__":
    main()
