"""Dynamics-based mocap <-> plantar-pressure temporal alignment (delta_t2).

The legacy foot-height contact proxy saturates during stance and yields a broad
cross-correlation peak. This module converts both sides into estimates of the
same physical quantity (vertical GRF in body weights), then compares them.

Reference design is kept in a separate module so the legacy visual-mocap path
and the original foot-curve pressure alignment remain untouched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from models import BVHMotion
from utils.bvh_pose import compute_joint_positions, transform_display_positions
from dataclasses import dataclass

from utils.pressure_alignment import (
    PressureAlignmentResult,
    PressureCurveSet,
    PressureMetaInfo,
    _guess_session_id,
)


@dataclass(frozen=True)
class DynamicsVgrfCurveSet:
    """The two total curves compared by the dynamics alignment mechanism.

    Per 新机制.md, both sides estimate the same quantity:
      mocap  : pred_total = a_com_z / g + 1
      tactile: meas_total = p_total / mean(p_total)
    after identical zero-phase low-pass filtering.
    """

    time_s: np.ndarray
    mocap_vgrf_bw: np.ndarray
    tactile_vgrf_bw: np.ndarray
    sample_fps: float
    filter_cutoff_hz: float
    length_scale_to_m: float = 1.0


G = 9.80665

# Winter (2009) segment mass fractions and CoM position as a fraction of the
# segment length measured from the proximal joint.  Fractions sum to 1.0.
# (proximal, distal, mass_fraction, com_ratio)
SEGMENT_TABLE = [
    ("Hips", "Neck", 0.497, 0.500),  # trunk (pelvis -> neck)
    ("Neck", "Head", 0.081, 0.500),  # head + neck
    ("LeftArm", "LeftForeArm", 0.028, 0.436),
    ("RightArm", "RightForeArm", 0.028, 0.436),
    ("LeftForeArm", "LeftHand", 0.016, 0.430),
    ("RightForeArm", "RightHand", 0.016, 0.430),
    ("LeftHand", "LeftHandEnd", 0.006, 0.506),
    ("RightHand", "RightHandEnd", 0.006, 0.506),
    ("LeftUpLeg", "LeftLeg", 0.100, 0.433),
    ("RightUpLeg", "RightLeg", 0.100, 0.433),
    ("LeftLeg", "LeftFoot", 0.0465, 0.433),
    ("RightLeg", "RightFoot", 0.0465, 0.433),
    ("LeftFoot", "LeftToeBase", 0.0145, 0.500),
    ("RightFoot", "RightToeBase", 0.0145, 0.500),
]

# Common Vicon / BVH naming variants mapped onto SEGMENT_TABLE names.
JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "Hips": ("Hips", "Pelvis", "Hip", "Root"),
    "Spine": ("Spine", "Spine1", "Chest", "Torso"),
    "Neck": ("Neck", "Neck1", "Spine2"),
    "Head": ("Head", "HeadTop_End", "HeadEnd"),
    "LeftArm": ("LeftArm", "LeftUpperArm", "L_UpperArm"),
    "RightArm": ("RightArm", "RightUpperArm", "R_UpperArm"),
    "LeftForeArm": ("LeftForeArm", "LeftLowerArm", "L_Forearm"),
    "RightForeArm": ("RightForeArm", "RightLowerArm", "R_Forearm"),
    "LeftHand": ("LeftHand", "L_Hand"),
    "RightHand": ("RightHand", "R_Hand"),
    "LeftHandEnd": (
        "LeftHandEnd",
        "LeftHandEndSite",
        "LeftHand_EndSite1",
        "LeftFingerBase",
        "LeftHandEnd_End",
        "LeftHand_End",
    ),
    "RightHandEnd": (
        "RightHandEnd",
        "RightHandEndSite",
        "RightHand_EndSite2",
        "RightFingerBase",
        "RightHandEnd_End",
        "RightHand_End",
    ),
    "LeftUpLeg": ("LeftUpLeg", "LeftThigh", "L_UpLeg", "LeftHip"),
    "RightUpLeg": ("RightUpLeg", "RightThigh", "R_UpLeg", "RightHip"),
    "LeftLeg": ("LeftLeg", "LeftShin", "L_Leg", "LeftCalf"),
    "RightLeg": ("RightLeg", "RightShin", "R_Leg", "RightCalf"),
    "LeftFoot": ("LeftFoot", "L_Foot"),
    "RightFoot": ("RightFoot", "R_Foot"),
    "LeftToeBase": ("LeftToeBase", "LeftToe", "LeftFootEnd", "LeftToeBaseEnd"),
    "RightToeBase": ("RightToeBase", "RightToe", "RightFootEnd", "RightToeBaseEnd"),
}


def body_com(joint_xyz: Mapping[str, np.ndarray], table=SEGMENT_TABLE) -> np.ndarray:
    """Whole-body CoM trajectory, shape (T, 3)."""
    total, acc = 0.0, None
    for prox, dist, mass, ratio in table:
        if prox not in joint_xyz or dist not in joint_xyz:
            continue
        p, d = joint_xyz[prox], joint_xyz[dist]
        seg_com = p + ratio * (d - p)
        acc = seg_com * mass if acc is None else acc + seg_com * mass
        total += mass
    if acc is None or total == 0.0:
        raise ValueError("no usable segment found in joint_xyz")
    return acc / total


def zero_phase_lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low pass, applied along axis 0."""
    if fc >= fs / 2.0:
        return np.asarray(x, dtype=float)
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=0)


def second_derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """Central second difference; filter BEFORE calling this."""
    d = np.gradient(np.gradient(np.asarray(x, dtype=float), axis=0), axis=0)
    return d * (fs ** 2)


def predict_vgrf_bw(com_xyz: np.ndarray, fs: float, vertical: int = 2, fc: float = 8.0) -> np.ndarray:
    """Predicted total vertical GRF in body weights from the CoM trajectory."""
    com_f = zero_phase_lowpass(com_xyz, fs, fc)
    a_v = second_derivative(com_f[:, vertical], fs)
    return a_v / G + 1.0


def measured_vgrf_bw(p_total: np.ndarray, fs: float, fc: float = 8.0) -> np.ndarray:
    """Insole total, band-matched to the mocap side and scaled to body weights."""
    p = zero_phase_lowpass(np.asarray(p_total, dtype=float), fs, fc)
    m = float(np.mean(p))
    return p / m if m > 0 else p


def foot_contact(
    heel_xyz: np.ndarray,
    toe_xyz: np.ndarray,
    fs: float,
    vertical: int = 2,
    speed_thr: float = 0.35,
    height_band: float = 0.03,
) -> np.ndarray:
    """Boolean stance mask from foot kinematics (velocity + height gate)."""
    pos = np.minimum(heel_xyz[:, vertical], toe_xyz[:, vertical])
    pos = zero_phase_lowpass(pos, fs, min(10.0, fs / 2.5))
    vel = np.abs(np.gradient(pos) * fs)
    floor = np.percentile(pos, 2.0)
    return (pos < floor + height_band) & (vel < speed_thr)


def split_vgrf(vgrf: np.ndarray, contact_l: np.ndarray, contact_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distribute the total onto both feet for optional cross-check channels."""
    n = len(vgrf)
    wl = np.zeros(n)
    wl[contact_l & ~contact_r] = 1.0
    wl[~contact_l & contact_r] = 0.0
    both = contact_l & contact_r
    idx = 0
    while idx < n:
        if not both[idx]:
            idx += 1
            continue
        j = idx
        while j < n and both[j]:
            j += 1
        lead_left = bool(idx > 0 and not contact_l[idx - 1])
        s = np.linspace(0.0, 1.0, j - idx)
        ramp = 3 * s ** 2 - 2 * s ** 3
        wl[idx:j] = ramp if lead_left else 1.0 - ramp
        idx = j
    return vgrf * wl, vgrf * (1.0 - wl)


def _resample(t: np.ndarray, y: np.ndarray, fs_out: float):
    grid = np.arange(t[0], t[-1], 1.0 / fs_out)
    if len(grid) < 2:
        grid = np.asarray([t[0], t[-1]], dtype=float)
    return grid, CubicSpline(t, y)(grid)


def xcorr_lag(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    t_center: float,
    half_window_s: float,
    fs_work: float = 500.0,
    use_derivative: bool = True,
) -> tuple[float, float]:
    """Normalised cross-correlation lag of b relative to a, in seconds."""
    if len(a) < 3 or len(b) < 3:
        return float(t_center), 0.0

    t = np.arange(len(a)) / fs
    _, ai = _resample(t, a, fs_work)
    _, bi = _resample(np.arange(len(b)) / fs, b, fs_work)
    if use_derivative:
        ai, bi = np.gradient(ai), np.gradient(bi)
    ai = (ai - ai.mean()) / (ai.std() + 1e-12)
    bi = (bi - bi.mean()) / (bi.std() + 1e-12)

    max_lag = int(round((abs(t_center) + half_window_s) * fs_work)) + 2
    full = np.correlate(ai, bi, mode="full") / min(len(ai), len(bi))
    lags = (np.arange(full.size) - (len(bi) - 1)) / fs_work
    keep = np.abs(lags - t_center) <= half_window_s
    if not keep.any():
        keep = np.abs(lags) <= max_lag / fs_work
    seg, seg_lags = full[keep], lags[keep]

    k = int(np.argmax(seg))
    peak = float(seg[k])
    # np.correlate(a, b) lag convention is opposite to "how much later b is than a".
    # Convert to: positive lag means b is delayed relative to a (matches event refine
    # and the app-wide meaning of delta_t2 when shifting mocap by +delta).
    lag = -float(seg_lags[k])
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            lag -= 0.5 * (y0 - y2) / denom / fs_work
    return lag, peak


def onset_times(sig: np.ndarray, fs: float, rel_thr: float = 0.15, min_gap_s: float = 0.35) -> np.ndarray:
    """Rising threshold crossings at a fixed fraction of the stance peak."""
    s = np.asarray(sig, dtype=float)
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)
    thr = lo + rel_thr * (hi - lo)
    above = s > thr
    cross = np.where(~above[:-1] & above[1:])[0]
    out, last = [], -np.inf
    for i in cross:
        y0, y1 = s[i], s[i + 1]
        t = (i + (thr - y0) / (y1 - y0 + 1e-12)) / fs
        if t - last >= min_gap_s:
            out.append(t)
            last = t
    return np.asarray(out)


def refine_by_events(
    a: np.ndarray,
    b: np.ndarray,
    fs: float,
    coarse_lag: float,
    tol_s: float = 0.12,
):
    """Median offset over matched events, plus scatter and clock drift."""
    ea, eb = onset_times(a, fs), onset_times(b, fs)
    if ea.size == 0 or eb.size == 0:
        return coarse_lag, np.nan, np.nan, 0
    pairs = []
    for t in ea:
        k = int(np.argmin(np.abs(eb - (t + coarse_lag))))
        d = eb[k] - t
        if abs(d - coarse_lag) <= tol_s:
            pairs.append((t, d))
    if len(pairs) < 3:
        return coarse_lag, np.nan, np.nan, len(pairs)
    tt = np.array([p[0] for p in pairs])
    dd = np.array([p[1] for p in pairs])
    delta = float(np.median(dd))
    mad = float(np.median(np.abs(dd - delta)))
    slope = float(np.polyfit(tt, dd, 1)[0]) if tt.ptp() > 5.0 else 0.0
    return delta, mad, slope * 1e6, len(pairs)


def estimate_delta_t2(
    joint_xyz: Mapping[str, np.ndarray],
    fs_mocap: float,
    left_sum: np.ndarray,
    right_sum: np.ndarray,
    fs_pressure: float,
    t_coarse: float = 0.0,
    half_window_s: float = 0.20,
    vertical: int = 2,
    fc: float = 8.0,
) -> dict:
    """Both streams are assumed already resampled to a common reference fps."""
    if abs(fs_mocap - fs_pressure) > 1e-6:
        raise ValueError("resample both streams to reference_fps first")
    fs = fs_mocap

    com = body_com(joint_xyz)
    pred_total = predict_vgrf_bw(com, fs, vertical=vertical, fc=fc)
    meas_total = measured_vgrf_bw(np.asarray(left_sum) + np.asarray(right_sum), fs, fc=fc)
    n = min(len(pred_total), len(meas_total))
    pred_total, meas_total = pred_total[:n], meas_total[:n]

    lag, peak = xcorr_lag(pred_total, meas_total, fs, t_coarse, half_window_s)
    delta, mad, drift_ppm, n_ev = refine_by_events(pred_total, meas_total, fs, lag)

    shifted = CubicSpline(np.arange(n) / fs, meas_total, extrapolate=False)(np.arange(n) / fs - delta)
    ok = np.isfinite(shifted)
    ss_res = float(np.sum((pred_total[ok] - shifted[ok]) ** 2))
    ss_tot = float(np.sum((pred_total[ok] - pred_total[ok].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    se = mad * 1.4826 / np.sqrt(n_ev) if n_ev > 0 else np.nan

    return dict(
        delta_t2=delta,
        delta_t2_xcorr=lag,
        xcorr_peak=peak,
        event_mad_s=mad,
        event_count=n_ev,
        standard_error_s=se,
        clock_drift_ppm=drift_ppm,
        r2_total=r2,
        t_coarse=t_coarse,
        search_window_ms=half_window_s * 1000.0,
        filter_cutoff_hz=fc,
    )



def estimate_delta_t2_from_totals(
    pred_total: np.ndarray,
    meas_total: np.ndarray,
    fs: float,
    *,
    t_coarse: float = 0.0,
    half_window_s: float = 0.20,
    fc: float = 8.0,
) -> dict:
    """Lag estimation on the two total vGRF/BW curves from 新机制.md."""
    pred_total = np.asarray(pred_total, dtype=np.float64)
    meas_total = np.asarray(meas_total, dtype=np.float64)
    n = min(len(pred_total), len(meas_total))
    if n < 3:
        raise ValueError("Not enough samples for dynamics-based pressure alignment")
    pred_total, meas_total = pred_total[:n], meas_total[:n]

    lag, peak = xcorr_lag(pred_total, meas_total, fs, t_coarse, half_window_s)
    delta, mad, drift_ppm, n_ev = refine_by_events(pred_total, meas_total, fs, lag)

    shifted = CubicSpline(np.arange(n) / fs, meas_total, extrapolate=False)(np.arange(n) / fs - delta)
    ok = np.isfinite(shifted)
    ss_res = float(np.sum((pred_total[ok] - shifted[ok]) ** 2))
    ss_tot = float(np.sum((pred_total[ok] - pred_total[ok].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    se = mad * 1.4826 / np.sqrt(n_ev) if n_ev > 0 else np.nan

    return dict(
        delta_t2=delta,
        delta_t2_xcorr=lag,
        xcorr_peak=peak,
        event_mad_s=mad,
        event_count=n_ev,
        standard_error_s=se,
        clock_drift_ppm=drift_ppm,
        r2_total=r2,
        t_coarse=t_coarse,
        search_window_ms=half_window_s * 1000.0,
        filter_cutoff_hz=fc,
    )


def _lookup_joint_index(name_to_index: dict[str, int], target: str) -> int | None:
    if target in name_to_index:
        return name_to_index[target]
    lower_map = {key.lower(): value for key, value in name_to_index.items()}
    for alias in JOINT_ALIASES.get(target, (target,)):
        if alias in name_to_index:
            return name_to_index[alias]
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def estimate_length_scale_to_meters(joint_xyz: Mapping[str, np.ndarray]) -> float:
    """Infer a multiplier that converts BVH coordinates into metres.

    Vicon exports in this project are typically centimetres (standing height
    ~150-190). The dynamics formula a/g expects SI metres; using cm/mm as if
    they were metres makes predicted vGRF either explode or look almost flat
    after later comparisons.
    """
    if not joint_xyz:
        return 1.0
    stacked = np.concatenate([np.asarray(v, dtype=np.float64) for v in joint_xyz.values()], axis=0)
    if stacked.size == 0:
        return 1.0
    extents = stacked.max(axis=0) - stacked.min(axis=0)
    height = float(np.max(extents))
    if not np.isfinite(height) or height <= 1e-9:
        return 1.0
    # Typical adult stature proxies.
    if 0.4 <= height <= 2.8:
        return 1.0  # already metres
    if 40.0 <= height <= 400.0:
        return 0.01  # centimetres -> metres
    if 400.0 < height <= 4000.0:
        return 0.001  # millimetres -> metres
    if height > 4000.0:
        return 0.001
    if height > 10.0:
        return 0.01
    return 1.0


def scale_joint_xyz(joint_xyz: Mapping[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    if abs(scale - 1.0) < 1e-15:
        return {name: np.asarray(values, dtype=np.float64) for name, values in joint_xyz.items()}
    return {name: np.asarray(values, dtype=np.float64) * scale for name, values in joint_xyz.items()}


def extract_joint_xyz(
    motion: BVHMotion,
    *,
    axis_preset: str = "zup",
    target_names: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Extract world trajectories for the joints needed by the segment table."""
    if len(motion.raw_frames) == 0:
        raise ValueError(f"BVH has no frames: {motion.path}")

    name_to_index = {joint.name: index for index, joint in enumerate(motion.joints)}
    needed = set()
    for prox, dist, _mass, _ratio in SEGMENT_TABLE:
        needed.add(prox)
        needed.add(dist)
    if target_names is not None:
        needed.update(target_names)

    selected: dict[str, int] = {}
    for name in needed:
        index = _lookup_joint_index(name_to_index, name)
        if index is not None:
            selected[name] = index

    if len(selected) < 2:
        raise ValueError(f"Unable to map enough joints for CoM estimation in {motion.path.name}")

    frame_count = len(motion.raw_frames)
    trajectories = {name: np.zeros((frame_count, 3), dtype=np.float64) for name in selected}
    for frame_index in range(frame_count):
        positions = transform_display_positions(compute_joint_positions(motion, frame_index), axis_preset)
        for name, joint_index in selected.items():
            trajectories[name][frame_index] = positions[joint_index]
    return trajectories


def _resample_signal_to_fps(time_s: np.ndarray, values: np.ndarray, reference_fps: float) -> tuple[np.ndarray, np.ndarray]:
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


def build_dynamics_vgrf_curves(
    motion: BVHMotion,
    pressure: PressureCurveSet,
    *,
    axis_preset: str = "zup",
    filter_cutoff_hz: float = 8.0,
    reference_fps: float | None = None,
    vertical: int = 2,
    length_scale: float | None = None,
    left_total: np.ndarray | None = None,
    right_total: np.ndarray | None = None,
    pressure_time_s: np.ndarray | None = None,
) -> DynamicsVgrfCurveSet:
    """Build the two total vGRF/BW curves used by the dynamics mechanism.

    Mocap side uses whole-body CoM second derivative in metres:
        pred = a_com_z / g + 1
    Tactile side uses total plantar pressure scaled by its own mean:
        meas = p_total / mean(p_total)
    Both pass through the same zero-phase low-pass before comparison.
    """
    fs = float(reference_fps if reference_fps is not None else 0.0)
    if fs <= 0:
        fs = float(pressure.sample_fps if pressure.sample_fps > 0 else motion.raw_fps)
    if fs <= 0:
        fs = 40.0

    joint_xyz = extract_joint_xyz(motion, axis_preset=axis_preset)
    if length_scale is None:
        length_scale = estimate_length_scale_to_meters(joint_xyz)
    joint_xyz = scale_joint_xyz(joint_xyz, float(length_scale))

    mocap_time = (
        np.arange(len(motion.raw_frames), dtype=np.float64) / motion.raw_fps
        if motion.raw_fps > 0
        else np.arange(len(motion.raw_frames), dtype=np.float64)
    )
    resampled_joints: dict[str, np.ndarray] = {}
    common_time = None
    for name, traj in joint_xyz.items():
        cols = []
        for axis in range(3):
            grid, values = _resample_signal_to_fps(mocap_time, traj[:, axis], fs)
            common_time = grid
            cols.append(values)
        resampled_joints[name] = np.column_stack(cols)

    # Prefer unnormalized total pressure if provided; otherwise fall back to
    # the already-normalized left/right curves (legacy path).
    if left_total is not None and right_total is not None:
        p_time = np.asarray(
            pressure_time_s if pressure_time_s is not None else pressure.time_s,
            dtype=np.float64,
        )
        left_src = np.asarray(left_total, dtype=np.float64)
        right_src = np.asarray(right_total, dtype=np.float64)
        _, left_rs = _resample_signal_to_fps(p_time, left_src, fs)
        _, right_rs = _resample_signal_to_fps(p_time, right_src, fs)
    else:
        _, left_rs = _resample_signal_to_fps(pressure.time_s, pressure.left_sum, fs)
        _, right_rs = _resample_signal_to_fps(pressure.time_s, pressure.right_sum, fs)

    if common_time is None or len(common_time) < 3 or len(left_rs) < 3:
        raise ValueError("Not enough samples for dynamics-based pressure alignment")

    n = min(len(common_time), len(left_rs), len(right_rs))
    for name in list(resampled_joints):
        resampled_joints[name] = resampled_joints[name][:n]
    left_rs = left_rs[:n]
    right_rs = right_rs[:n]
    time_s = np.asarray(common_time[:n], dtype=np.float64)

    com = body_com(resampled_joints)
    pred_total = predict_vgrf_bw(com, fs, vertical=vertical, fc=filter_cutoff_hz)
    meas_total = measured_vgrf_bw(np.asarray(left_rs) + np.asarray(right_rs), fs, fc=filter_cutoff_hz)
    m = min(len(pred_total), len(meas_total), len(time_s))
    return DynamicsVgrfCurveSet(
        time_s=time_s[:m],
        mocap_vgrf_bw=np.asarray(pred_total[:m], dtype=np.float64),
        tactile_vgrf_bw=np.asarray(meas_total[:m], dtype=np.float64),
        sample_fps=float(fs),
        filter_cutoff_hz=float(filter_cutoff_hz),
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
    """Adapter for the dynamics estimator (新机制.md).

    Returns:
      result: delta_t2 estimated by comparing the two total vGRF/BW curves
      curves: those same two total curves (mocap pred_total, tactile meas_total)
    """
    curves = build_dynamics_vgrf_curves(
        motion,
        pressure,
        axis_preset=axis_preset,
        filter_cutoff_hz=filter_cutoff_hz,
        reference_fps=reference_fps,
        vertical=2,
        left_total=left_total,
        right_total=right_total,
        pressure_time_s=pressure_time_s,
    )
    fs = float(curves.sample_fps)
    half_window_s = max(0.0, float(search_window_ms) / 1000.0)
    t_coarse = float(meta.t_coarse if meta is not None else 0.0)
    if meta is not None and meta.t_coarse_source == "fallback_global":
        half_window_s = max(half_window_s, float(curves.time_s[-1] if len(curves.time_s) else 1.0))

    metrics = estimate_delta_t2_from_totals(
        curves.mocap_vgrf_bw,
        curves.tactile_vgrf_bw,
        fs,
        t_coarse=t_coarse,
        half_window_s=half_window_s,
        fc=filter_cutoff_hz,
    )

    delta = float(metrics["delta_t2"])
    peak = float(metrics.get("xcorr_peak", 0.0))
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
        search_window_ms=int(round(float(metrics.get("search_window_ms", search_window_ms)))),
        manual_adjusted=False,
        reference_fps=float(fs),
        mocap_source_file=motion.path.name,
        pressure_source_file=pressure.source_path.name,
        axis_preset=axis_preset,
        exported_at=datetime.now().isoformat(timespec="seconds"),
    )
    return result, curves
