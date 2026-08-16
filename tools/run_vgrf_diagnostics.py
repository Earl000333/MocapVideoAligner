# -*- coding: utf-8 -*-
"""Batch-run flat-vGRF diagnostics over paired mocap + reconstructed pressure data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from utils.bvh_parser import load_bvh_motion_preserve_frames
from utils.pressure_alignment import (
    list_reconstructed_segments,
    load_reconstructed_pressure_sensors,
)
from utils.pressure_dynamics_alignment import (
    SEGMENT_TABLE,
    body_com,
    estimate_length_scale_to_meters,
    extract_joint_xyz,
    predict_vgrf_bw,
    scale_joint_xyz,
)
from utils.vgrf_diagnostics import diagnose, estimate_delta_t2_events


def _trial_from_folder(name: str) -> str | None:
    # S13011 -> S1301_1 if last digit is rep, but keep folder name as primary id.
    return name


def _find_bvh(trial_dir: Path) -> Path | None:
    # Prefer Skeleton0/1 if present, else first bvh.
    cands = sorted(trial_dir.glob("*.bvh"))
    if not cands:
        return None
    for preferred in ("Skeleton0", "Skeleton1"):
        for p in cands:
            if preferred.lower() in p.name.lower():
                return p
    return cands[0]


def _session_suffixes(trial_id: str) -> list[str]:
    candidates = [trial_id]
    m = re.fullmatch(r"S(\d+)(\d)", trial_id, flags=re.IGNORECASE)
    if m:
        candidates.append(f"S{m.group(1)}_{m.group(2)}")
        candidates.append(f"S{m.group(1)}{m.group(2)}")
    m2 = re.fullmatch(r"S(\d+)_(\d+)", trial_id, flags=re.IGNORECASE)
    if m2:
        candidates.append(f"S{m2.group(1)}_{m2.group(2)}")
        candidates.append(f"S{m2.group(1)}{m2.group(2)}")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _build_recon_index(recon_root: Path) -> dict[str, Path]:
    """Map session suffix (lower) -> reconstructed rec directory.

    Index once to avoid expensive per-trial recursive searches.
    """
    index: dict[str, Path] = {}
    for manifest in recon_root.rglob("reconstruction_manifest.csv"):
        rec_dir = manifest.parent
        name = rec_dir.name
        m = re.search(r"(S\d+(?:_\d+)?)$", name, flags=re.IGNORECASE)
        if not m:
            continue
        suffix = m.group(1)
        key = suffix.lower()
        index.setdefault(key, rec_dir)
        compact = suffix.replace("_", "").lower()
        index.setdefault(compact, rec_dir)
    return index


def _pair_trials(mocap_root: Path, recon_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    recon_index = _build_recon_index(recon_root)
    print(f"recon index size: {len(recon_index)}")
    for trial_dir in sorted([p for p in mocap_root.iterdir() if p.is_dir()]):
        trial_id = trial_dir.name  # e.g. S13011
        bvh = _find_bvh(trial_dir)
        if bvh is None:
            continue
        rec = None
        for sid in _session_suffixes(trial_id):
            rec = recon_index.get(sid.lower())
            if rec is not None:
                break
        if rec is None:
            continue
        pairs.append((trial_id, bvh, rec))
    return pairs


def _summarize_pred(joint_xyz_m: dict[str, np.ndarray], fs: float, vertical: int) -> dict:
    try:
        com = body_com(joint_xyz_m)
        pred = predict_vgrf_bw(com, fs, vertical=vertical, fc=8.0)
        n = len(pred)
        lo, hi = int(0.1 * n), max(int(0.1 * n) + 1, int(0.9 * n))
        return {
            "pred_mean": float(np.mean(pred)),
            "pred_std_full": float(np.std(pred)),
            "pred_ptp_full": float(np.ptp(pred)),
            "pred_std_mid80": float(np.std(pred[lo:hi])),
            "pred_ptp_mid80": float(np.ptp(pred[lo:hi])),
            "com_z_std_mid80": float(np.std(com[lo:hi, vertical])),
            "com_z_ptp_mid80": float(np.ptp(com[lo:hi, vertical])),
        }
    except Exception as exc:
        return {
            "pred_mean": np.nan,
            "pred_std_full": np.nan,
            "pred_ptp_full": np.nan,
            "pred_std_mid80": np.nan,
            "pred_ptp_mid80": np.nan,
            "com_z_std_mid80": np.nan,
            "com_z_ptp_mid80": np.nan,
            "pred_error": str(exc),
        }


def run_one(trial_id: str, bvh_path: Path, rec_dir: Path, axis_preset: str = "zup") -> dict:
    motion = load_bvh_motion_preserve_frames(bvh_path)
    fs = float(motion.raw_fps if motion.raw_fps > 0 else 120.0)

    joint_xyz = extract_joint_xyz(motion, axis_preset=axis_preset)
    scale = estimate_length_scale_to_meters(joint_xyz)
    joint_xyz_m = scale_joint_xyz(joint_xyz, scale)

    left, right = load_reconstructed_pressure_sensors(rec_dir)
    # Use raw sensor totals (before any curve min-max), on each foot timebase
    # resampled onto a shared grid for diagnostics.
    n = min(len(left.time_s), len(right.time_s))
    if n < 3:
        raise ValueError(f"pressure too short: {rec_dir}")
    # Align by reindexing to common length using left time as reference if close,
    # otherwise interpolate right onto left time.
    left_t = left.time_s
    right_t = right.time_s
    left_sum = left.sensor_totals
    if len(right_t) == len(left_t) and np.allclose(right_t, left_t, atol=1e-6):
        right_sum = right.sensor_totals
        p_time = left_t
    else:
        p_time = left_t
        right_sum = np.interp(left_t, right_t, right.sensor_totals) if len(right_t) else np.zeros_like(left_t)
    # Diagnostics do not require matching fps between mocap and pressure.
    # For event estimator we will use pressure fs independently, but API takes one fs.
    # Here we report diagnostics on mocap fs with pressure as-is (event estimator uses same fs assumption).
    # Better: resample pressure onto its own inferred fps for event estimator later.

    rep = diagnose(
        joint_xyz_m,
        fs,
        left_sum,
        right_sum,
        left_foot_key="LeftFoot",
        right_foot_key="RightFoot",
        segment_table=SEGMENT_TABLE,
    )
    vertical = int(rep["vertical_axis_guess"])
    pred_stats = _summarize_pred(joint_xyz_m, fs, vertical)

    # Event estimator: use pressure native fps approx.
    p_fs = float(left.sample_fps if left.sample_fps > 0 else 40.0)
    # Build joint xyz already in metres; event estimator uses foot trajectories.
    # Resample joints to pressure-ish fps only for event matching? Keep mocap fs and
    # resample pressure to mocap fs for a consistent event API.
    t_m = np.arange(len(next(iter(joint_xyz_m.values())))) / fs
    t_p = p_time
    # interpolate pressure onto mocap timeline for estimator convenience
    left_on_m = np.interp(t_m, t_p, left_sum)
    right_on_m = np.interp(t_m, t_p, right_sum)
    try:
        event = estimate_delta_t2_events(
            joint_xyz_m,
            fs,
            left_on_m,
            right_on_m,
            vertical=vertical,
            t_coarse=0.0,
        )
    except Exception as exc:
        event = {"error": str(exc)}

    row = {
        "trial_id": trial_id,
        "bvh_path": str(bvh_path),
        "rec_dir": str(rec_dir),
        "axis_preset": axis_preset,
        "mocap_fps": fs,
        "pressure_fps": p_fs,
        "length_scale_to_m": scale,
        "segments": ",".join(list_reconstructed_segments(rec_dir)),
        "joint_count": rep.get("joint_count"),
        "segments_matched": rep.get("segments_matched"),
        "segments_total": rep.get("segments_total"),
        "segments_missing": ";".join([f"{a}-{b}" for a, b in rep.get("segments_missing", [])]) if isinstance(rep.get("segments_missing"), list) else "",
        "root_key": rep.get("root_key"),
        "left_foot_key": rep.get("left_foot_key"),
        "vertical_axis_guess": rep.get("vertical_axis_guess"),
        "vertical_axis_scores": ",".join(str(x) for x in rep.get("vertical_axis_scores", [])),
        "root_range_x": rep.get("root_range_per_axis", [np.nan]*3)[0],
        "root_range_y": rep.get("root_range_per_axis", [np.nan]*3)[1],
        "root_range_z": rep.get("root_range_per_axis", [np.nan]*3)[2],
        "foot_range_x": rep.get("foot_range_per_axis", [np.nan]*3)[0],
        "foot_range_y": rep.get("foot_range_per_axis", [np.nan]*3)[1],
        "foot_range_z": rep.get("foot_range_per_axis", [np.nan]*3)[2],
        "unit_guess": rep.get("unit_guess"),
        "root_vert_accel_rms_g": rep.get("root_vert_accel_rms_g"),
        "root_vert_range": rep.get("root_vert_range"),
        "com_channel_usable": rep.get("com_channel_usable"),
        "pressure_total_cv": rep.get("pressure_total_cv"),
        "pressure_left_duty": rep.get("pressure_left_duty"),
        "pressure_right_duty": rep.get("pressure_right_duty"),
        "verdict": " | ".join(rep.get("verdict", [])),
    }
    row.update(pred_stats)
    for k, v in event.items():
        row[f"event_{k}"] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap-root", type=Path, default=Path(r"E:/S13/mocap_ori_bvh"))
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=Path(r"E:/S13/reconstruction_20260816_222245"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sync/output/vgrf_diagnostics_report.csv"),
    )
    parser.add_argument("--axis-preset", default="zup")
    parser.add_argument("--limit", type=int, default=0, help="0 means all pairs")
    args = parser.parse_args()

    if not args.recon_root.exists():
        # fallback to newest reconstruction under E:/S13
        candidates = sorted(Path(r"E:/S13").glob("reconstruction_*"))
        if not candidates:
            raise FileNotFoundError(f"recon root not found: {args.recon_root}")
        args.recon_root = candidates[-1]

    pairs = _pair_trials(args.mocap_root, args.recon_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"pairs: {len(pairs)}")
    print(f"mocap_root: {args.mocap_root}")
    print(f"recon_root: {args.recon_root}")

    rows: list[dict] = []
    for idx, (trial_id, bvh, rec) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {trial_id} ...", flush=True)
        try:
            row = run_one(trial_id, bvh, rec, axis_preset=args.axis_preset)
            row["status"] = "ok"
        except Exception as exc:
            row = {
                "trial_id": trial_id,
                "bvh_path": str(bvh),
                "rec_dir": str(rec),
                "status": "error",
                "error": str(exc),
            }
            print("  ERROR", exc)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # stable header union
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}  rows={len(rows)}")


if __name__ == "__main__":
    main()