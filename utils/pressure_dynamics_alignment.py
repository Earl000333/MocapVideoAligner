# -*- coding: utf-8 -*-
"""Strike-based mocap <-> plantar-pressure temporal alignment (new mechanism).

The legacy curve correlates foot height against pressure, but foot height
saturates into a plateau during stance so the correlation peak is blunt.
Take only the moment of touchdown instead, on both sides, and take the
median of the paired differences. No centre of mass, no dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np
from scipy.signal import butter, filtfilt

from models import BVHMotion
from utils.bvh_pose import compute_joint_positions, transform_display_positions
from utils.pressure_alignment import (
    PressureAlignmentResult,
    PressureCurveSet,
    PressureMetaInfo,
    _guess_session_id,
)

AXIS_PRESET = {"zup": 2, "yup": 1, "xup": 0}

# Keep a lightweight curve container so the UI can still plot two totals.
@dataclass(frozen=True)
class DynamicsVgrfCurveSet:
    """Display curves for the new (strike) mechanism.

    These are NOT used for the offset estimator. They only give the UI two
    comparable total traces: inverted foot-height contact proxy vs total
    pressure / mean.
    """

    time_s: np.ndarray
    mocap_vgrf_bw: np.ndarray
    tactile_vgrf_bw: np.ndarray
    sample_fps: float
    filter_cutoff_hz: float
    length_scale_to_m: float = 1.0
    tactile_valid: np.ndarray | None = None
    # Extra diagnostics from the strike estimator.
    event_n: int = 0
    event_scatter: float = float("nan")
    event_ok: bool = False
    event_note: str = ""
    # Absolute event times used by the estimator (seconds on each stream's own clock).
    # UI shifts mocap strikes by delta_t2 when drawing on the shared preview axis.
    mocap_strike_times_s: np.ndarray | None = None
    pressure_strike_times_s: np.ndarray | None = None


JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "Hips": ("Hips", "Pelvis", "Hip", "Root"),
    "LeftLeg": ("LeftLeg", "LeftShin", "L_Leg", "LeftCalf"),
    "RightLeg": ("RightLeg", "RightShin", "R_Leg", "RightCalf"),
    "LeftFoot": ("LeftFoot", "L_Foot"),
    "RightFoot": ("RightFoot", "R_Foot"),
    "LeftToeBase": ("LeftToeBase", "LeftToe", "LeftFootEnd", "LeftToeBaseEnd"),
    "RightToeBase": ("RightToeBase", "RightToe", "RightFootEnd", "RightToeBaseEnd"),
    "LeftUpLeg": ("LeftUpLeg", "LeftThigh", "L_UpLeg", "LeftHip"),
    "RightUpLeg": ("RightUpLeg", "RightThigh", "R_UpLeg", "RightHip"),
}


def _lowpass(x, fs, fc=10.0):
    x = np.asarray(x, dtype=float)
    if fc >= fs / 2.0:
        return x
    b, a = butter(4, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def _unit_scale(joints: Mapping[str, np.ndarray]) -> float:
    """BVH units to metres, from a rigid bone length.

    Guessing the scale from motion extent flattened eight trials by a factor
    of ten; bone length is constant for the whole trial.
    """
    if "LeftLeg" in joints and "LeftFoot" in joints:
        d = float(
            np.median(
                np.linalg.norm(
                    np.asarray(joints["LeftLeg"], dtype=float)
                    - np.asarray(joints["LeftFoot"], dtype=float),
                    axis=1,
                )
            )
        )
        return min([1.0, 0.01, 0.001], key=lambda s: abs(d * s - 0.40))
    # Fallbacks if left shank missing.
    for a, b, ref in (
        ("RightLeg", "RightFoot", 0.40),
        ("LeftUpLeg", "LeftLeg", 0.43),
        ("RightUpLeg", "RightLeg", 0.43),
    ):
        if a in joints and b in joints:
            d = float(
                np.median(
                    np.linalg.norm(
                        np.asarray(joints[a], dtype=float)
                        - np.asarray(joints[b], dtype=float),
                        axis=1,
                    )
                )
            )
            return min([1.0, 0.01, 0.001], key=lambda s: abs(d * s - ref))
    return 0.01


def estimate_length_scale_to_meters(joint_xyz: Mapping[str, np.ndarray]) -> float:
    """Public wrapper used by other modules / diagnostics."""
    try:
        return float(_unit_scale(joint_xyz))
    except Exception:
        return 0.01


def scale_joint_xyz(joint_xyz: Mapping[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    if abs(scale - 1.0) < 1e-15:
        return {name: np.asarray(values, dtype=np.float64) for name, values in joint_xyz.items()}
    return {name: np.asarray(values, dtype=np.float64) * scale for name, values in joint_xyz.items()}


def _cross(y, i, thr, fs):
    """Threshold crossing between i-1 and i, linearly interpolated."""
    y0, y1 = y[i - 1], y[i]
    return (i - 1 + (thr - y0) / (y1 - y0 + 1e-12)) / fs


def _foot_strikes(heel, toe, fs, vertical, min_gap=0.35):
    """Touchdown times from the foot's own vertical trajectory."""
    z = _lowpass(np.minimum(heel[:, vertical], toe[:, vertical]), fs)
    lo, hi = np.percentile(z, [2, 98])
    gate = lo + 0.20 * (hi - lo)  # relative, so the unit cannot bite
    v = np.gradient(z) * fs
    out, last = [], -9.0
    for i in range(1, len(z)):
        if z[i] < gate <= z[i - 1] and v[i - 1] < 0:
            t = _cross(z, i, gate, fs)
            if t - last >= min_gap:
                out.append(t)
                last = t
    return np.asarray(out, dtype=float)


def _pressure_strikes(p, fs, min_gap=0.35):
    """Loading onset times, same construction as the mocap side."""
    y = _lowpass(p, fs, min(15.0, fs / 2.5))
    base, top = np.percentile(y, [5, 95])
    thr = base + 0.25 * (top - base)
    out, last = [], -9.0
    for i in range(1, len(y)):
        if y[i] > thr >= y[i - 1]:
            t = _cross(y, i, thr, fs)
            if t - last >= min_gap:
                out.append(t)
                last = t
    return np.asarray(out, dtype=float)


def align(
    joints,
    fs_mocap,
    left_sum,
    right_sum,
    fs_pressure,
    t_coarse=0.0,
    vertical=2,
    window=0.20,
):
    """joints maps a BVH joint name to an (T, 3) world position array.

    Returns delta_t2 in seconds, the number of matched events, and the
    median absolute deviation of those matches. Flag the result in the UI
    when scatter is large or n is small.
    """
    s = _unit_scale(joints)
    j = {k: np.asarray(v, dtype=float) * s for k, v in joints.items()}

    required = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
    missing = [name for name in required if name not in j]
    if missing:
        return dict(
            delta_t2=float(t_coarse),
            n=0,
            scatter=float("nan"),
            ok=False,
            note=f"missing joints: {','.join(missing)}",
            length_scale=float(s),
            mocap_events=0,
            pressure_events=0,
            mocap_strike_times_s=np.zeros(0, dtype=float),
            pressure_strike_times_s=np.zeros(0, dtype=float),
        )

    mocap = np.sort(
        np.concatenate(
            [
                _foot_strikes(j["LeftFoot"], j["LeftToeBase"], fs_mocap, vertical),
                _foot_strikes(j["RightFoot"], j["RightToeBase"], fs_mocap, vertical),
            ]
        )
    )
    press = np.sort(
        np.concatenate(
            [
                _pressure_strikes(left_sum, fs_pressure),
                _pressure_strikes(right_sum, fs_pressure),
            ]
        )
    )

    if mocap.size < 3 or press.size < 3:
        return dict(
            delta_t2=float(t_coarse),
            n=0,
            scatter=float("nan"),
            ok=False,
            note="not enough steps, inherit the offset",
            length_scale=float(s),
            mocap_events=int(mocap.size),
            pressure_events=int(press.size),
            mocap_strike_times_s=mocap,
            pressure_strike_times_s=press,
        )

    d = [
        press[k] - t
        for t in mocap
        for k in [int(np.argmin(np.abs(press - (t + t_coarse))))]
        if abs(press[k] - t - t_coarse) <= window
    ]
    if len(d) < 3:
        return dict(
            delta_t2=float(t_coarse),
            n=len(d),
            scatter=float("nan"),
            ok=False,
            note="no match inside the window, check t_coarse",
            length_scale=float(s),
            mocap_events=int(mocap.size),
            pressure_events=int(press.size),
            mocap_strike_times_s=mocap,
            pressure_strike_times_s=press,
        )

    d = np.asarray(d, dtype=float)
    delta = float(np.median(d))
    return dict(
        delta_t2=delta,
        n=len(d),
        scatter=float(np.median(np.abs(d - delta))),
        ok=True,
        note="",
        length_scale=float(s),
        mocap_events=int(mocap.size),
        pressure_events=int(press.size),
        mocap_strike_times_s=mocap,
        pressure_strike_times_s=press,
    )


def _lookup_joint_index(name_to_index: dict[str, int], target: str) -> int | None:
    lower_map = {name.lower(): index for name, index in name_to_index.items()}
    for alias in JOINT_ALIASES.get(target, (target,)):
        if alias in name_to_index:
            return name_to_index[alias]
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def extract_joint_xyz(
    motion: BVHMotion,
    *,
    axis_preset: str = "zup",
    target_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Extract world trajectories for foot/leg joints used by strike alignment."""
    if len(motion.raw_frames) == 0:
        raise ValueError(f"BVH has no frames: {motion.path}")

    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    needed = {
        "LeftLeg",
        "RightLeg",
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
        "LeftUpLeg",
        "RightUpLeg",
        "Hips",
    }
    if target_names is not None:
        needed.update(target_names)

    selected: dict[str, int] = {}
    for name in needed:
        index = _lookup_joint_index(name_to_index, name)
        if index is not None:
            selected[name] = index

    if "LeftFoot" not in selected or "RightFoot" not in selected:
        raise ValueError(f"Unable to map foot joints in {motion.path.name}")

    frame_count = len(motion.raw_frames)
    trajectories = {name: np.zeros((frame_count, 3), dtype=np.float64) for name in selected}
    for frame_index in range(frame_count):
        positions = transform_display_positions(
            compute_joint_positions(motion, frame_index), axis_preset
        )
        for name, joint_index in selected.items():
            trajectories[name][frame_index] = positions[joint_index]
    return trajectories


def _resample_signal_to_fps(
    time_s: np.ndarray, values: np.ndarray, reference_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(time_s) == 0 or len(values) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if reference_fps <= 0:
        return time_s, values
    duration = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
    count = max(2, int(round(duration * reference_fps)) + 1)
    grid = np.arange(count, dtype=np.float64) / reference_fps + float(time_s[0])
    if len(time_s) == 1:
        return grid, np.repeat(values[:1], count).astype(np.float64)
    return grid, np.interp(grid, time_s, values).astype(np.float64)


def _build_display_curves(
    joint_xyz_m: Mapping[str, np.ndarray],
    mocap_time: np.ndarray,
    left_sum: np.ndarray,
    right_sum: np.ndarray,
    pressure_time: np.ndarray,
    *,
    vertical: int,
    fs: float,
    length_scale: float,
) -> DynamicsVgrfCurveSet:
    """Two total curves for UI: inverted foot height vs total pressure / mean."""
    lf = joint_xyz_m.get("LeftFoot")
    rf = joint_xyz_m.get("RightFoot")
    lt = joint_xyz_m.get("LeftToeBase", lf)
    rt = joint_xyz_m.get("RightToeBase", rf)
    if lf is None or rf is None:
        raise ValueError("Missing foot joints for display curves")

    left_h = np.minimum(lf[:, vertical], lt[:, vertical])
    right_h = np.minimum(rf[:, vertical], rt[:, vertical])
    # Contact proxy: larger means closer to the floor.
    contact = -0.5 * (left_h + right_h)
    contact = _lowpass(contact, float(len(mocap_time) / max(mocap_time[-1] - mocap_time[0], 1e-6) if len(mocap_time) > 1 else 120.0), 8.0)
    c_lo, c_hi = float(np.min(contact)), float(np.max(contact))
    if c_hi > c_lo:
        contact = (contact - c_lo) / (c_hi - c_lo)
    else:
        contact = np.zeros_like(contact)

    total = np.asarray(left_sum, dtype=float) + np.asarray(right_sum, dtype=float)
    mean_p = float(np.mean(total)) if len(total) else 1.0
    meas = total / mean_p if mean_p > 0 else total

    grid, mocap_rs = _resample_signal_to_fps(mocap_time, contact, fs)
    _, meas_rs = _resample_signal_to_fps(pressure_time, meas, fs)
    n = min(len(grid), len(mocap_rs), len(meas_rs))
    return DynamicsVgrfCurveSet(
        time_s=grid[:n],
        mocap_vgrf_bw=np.asarray(mocap_rs[:n], dtype=np.float64),
        tactile_vgrf_bw=np.asarray(meas_rs[:n], dtype=np.float64),
        sample_fps=float(fs),
        filter_cutoff_hz=8.0,
        length_scale_to_m=float(length_scale),
    )


def estimate_pressure_alignment_dynamics(
    motion: BVHMotion,
    pressure: PressureCurveSet,
    meta: PressureMetaInfo,
    *,
    axis_preset: str = "zup",
    search_window_ms: int = 200,
    filter_cutoff_hz: float = 8.0,
    reference_fps: float | None = None,
    left_total: np.ndarray | None = None,
    right_total: np.ndarray | None = None,
    pressure_time_s: np.ndarray | None = None,
) -> tuple[PressureAlignmentResult, DynamicsVgrfCurveSet]:
    """New-mechanism adapter: strike-event offset estimation.

    Returns:
      result: delta_t2 from matched touchdown / loading-onset pairs
      curves: simple total traces for UI (not used by the estimator)
    """
    vertical = AXIS_PRESET.get(str(axis_preset).lower(), 2)
    joint_xyz = extract_joint_xyz(motion, axis_preset=axis_preset)
    scale = float(_unit_scale(joint_xyz))
    joint_xyz_m = scale_joint_xyz(joint_xyz, scale)

    fs_mocap = float(motion.raw_fps if motion.raw_fps > 0 else 120.0)
    fs_pressure = float(pressure.sample_fps if pressure.sample_fps > 0 else 40.0)
    if fs_pressure <= 0:
        fs_pressure = 40.0

    if left_total is not None and right_total is not None:
        left_sum = np.asarray(left_total, dtype=float)
        right_sum = np.asarray(right_total, dtype=float)
        p_time = np.asarray(
            pressure_time_s if pressure_time_s is not None else pressure.time_s,
            dtype=float,
        )
    else:
        left_sum = np.asarray(pressure.left_sum, dtype=float)
        right_sum = np.asarray(pressure.right_sum, dtype=float)
        p_time = np.asarray(pressure.time_s, dtype=float)

    t_coarse = float(meta.t_coarse if meta is not None else 0.0)
    window = max(0.05, float(search_window_ms) / 1000.0)
    # If only a weak global prior exists, keep the provided window (caller may widen).
    metrics = align(
        joint_xyz,  # unscaled; align() applies bone unit scale itself
        fs_mocap,
        left_sum,
        right_sum,
        fs_pressure,
        t_coarse=t_coarse,
        vertical=vertical,
        window=window,
    )

    delta = float(metrics["delta_t2"])
    # Use inverse scatter as a soft peak surrogate for UI rows.
    scatter = metrics.get("scatter", float("nan"))
    if metrics.get("ok") and np.isfinite(scatter) and scatter > 0:
        peak = float(1.0 / (1.0 + 10.0 * scatter))
    elif metrics.get("ok"):
        peak = 1.0
    else:
        peak = 0.0

    display_fs = float(reference_fps if reference_fps is not None and reference_fps > 0 else fs_pressure)
    mocap_time = (
        np.arange(len(motion.raw_frames), dtype=np.float64) / fs_mocap
        if fs_mocap > 0
        else np.arange(len(motion.raw_frames), dtype=np.float64)
    )
    curves = _build_display_curves(
        joint_xyz_m,
        mocap_time,
        left_sum,
        right_sum,
        p_time,
        vertical=vertical,
        fs=display_fs,
        length_scale=scale,
    )
    curves = DynamicsVgrfCurveSet(
        time_s=curves.time_s,
        mocap_vgrf_bw=curves.mocap_vgrf_bw,
        tactile_vgrf_bw=curves.tactile_vgrf_bw,
        sample_fps=curves.sample_fps,
        filter_cutoff_hz=curves.filter_cutoff_hz,
        length_scale_to_m=float(metrics.get("length_scale", scale)),
        tactile_valid=None,
        event_n=int(metrics.get("n", 0) or 0),
        event_scatter=float(scatter) if scatter is not None else float("nan"),
        event_ok=bool(metrics.get("ok", False)),
        event_note=str(metrics.get("note", "") or ""),
        mocap_strike_times_s=np.asarray(
            metrics.get("mocap_strike_times_s", np.zeros(0)), dtype=np.float64
        ),
        pressure_strike_times_s=np.asarray(
            metrics.get("pressure_strike_times_s", np.zeros(0)), dtype=np.float64
        ),
    )

    session_id = _guess_session_id(
        motion.path,
        pressure.source_path,
        meta.source_path if meta is not None else None,
    )
    result = PressureAlignmentResult(
        session_id=session_id,
        delta_t2=delta,
        delta_t2_left=delta,
        delta_t2_right=delta,
        peak_left=peak,
        peak_right=peak,
        t_coarse=t_coarse,
        search_window_ms=int(round(window * 1000.0)),
        manual_adjusted=False,
        reference_fps=float(display_fs),
        mocap_source_file=motion.path.name,
        pressure_source_file=pressure.source_path.name,
        axis_preset=axis_preset,
        exported_at=datetime.now().isoformat(timespec="seconds"),
    )
    return result, curves