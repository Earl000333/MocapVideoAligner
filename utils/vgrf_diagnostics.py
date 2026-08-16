# -*- coding: utf-8 -*-
"""Diagnostics for the flat mocap vGRF curve, plus an event-based delta_t2
estimator that does not depend on root translation quality.

Two entry points:

    diagnose(...)                -> printable report, run this FIRST
    estimate_delta_t2_events(...) -> the replacement estimator
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

G = 9.80665


def zero_phase_lowpass(x: np.ndarray, fs: float, fc: float,
                       order: int = 4) -> np.ndarray:
    if fc >= fs / 2.0:
        return np.asarray(x, dtype=float)
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=0)


def detect_vertical_axis(foot_xyz: np.ndarray) -> tuple[int, list[float]]:
    """Guess the gravity axis from a foot joint trajectory."""
    scores = []
    for k in range(3):
        s = np.asarray(foot_xyz[:, k], dtype=float)
        lo, hi = np.percentile(s, 1), np.percentile(s, 99)
        if hi - lo < 1e-9:
            scores.append(0.0)
            continue
        u = (s - lo) / (hi - lo)
        low_frac = float(np.mean(u < 0.15))
        skew = float(np.mean(((u - u.mean()) / (u.std() + 1e-12)) ** 3))
        scores.append(low_frac * max(skew, 0.0))
    return int(np.argmax(scores)), scores


def band_rms(x: np.ndarray, fs: float, f_lo: float = 0.5,
             f_hi: float = 6.0) -> float:
    """RMS of x inside a band, used to judge whether motion is present."""
    if f_hi >= fs / 2.0:
        f_hi = fs / 2.0 * 0.9
    if f_lo >= f_hi:
        return float(np.std(x))
    b, a = butter(2, [f_lo / (fs / 2.0), f_hi / (fs / 2.0)], btype="band")
    return float(np.std(filtfilt(b, a, np.asarray(x, dtype=float))))


def diagnose(joint_xyz: dict[str, np.ndarray], fs: float,
             left_sum: np.ndarray, right_sum: np.ndarray,
             left_foot_key: str = "LeftFoot",
             right_foot_key: str = "RightFoot",
             segment_table=None) -> dict:
    """Report what the mocap side actually contains before trusting any curve."""
    rep: dict = {}

    keys = list(joint_xyz.keys())
    rep["joint_count"] = len(keys)
    rep["joint_names_sample"] = keys[:12]

    if segment_table is not None:
        matched = [(p, d) for p, d, _, _ in segment_table
                   if p in joint_xyz and d in joint_xyz]
        rep["segments_matched"] = len(matched)
        rep["segments_total"] = len(segment_table)
        rep["segments_missing"] = [(p, d) for p, d, _, _ in segment_table
                                   if p not in joint_xyz or d not in joint_xyz]

    if left_foot_key not in joint_xyz:
        # Fallback: first available *Foot* key.
        for k in keys:
            if "foot" in k.lower() and "left" in k.lower():
                left_foot_key = k
                break
        else:
            left_foot_key = keys[0]
    foot = np.asarray(joint_xyz[left_foot_key], dtype=float)
    axis, axis_scores = detect_vertical_axis(foot)
    rep["vertical_axis_guess"] = axis
    rep["vertical_axis_scores"] = [round(s, 3) for s in axis_scores]
    rep["left_foot_key"] = left_foot_key

    root_key = next((k for k in ("Hips", "Hip", "Root", "Pelvis")
                     if k in joint_xyz), keys[0])
    root = np.asarray(joint_xyz[root_key], dtype=float)
    rep["root_key"] = root_key
    rep["root_range_per_axis"] = [float(root[:, k].ptp()) for k in range(3)]
    rep["foot_range_per_axis"] = [float(foot[:, k].ptp()) for k in range(3)]

    span = float(foot[:, axis].ptp())
    if span < 0.02:
        rep["unit_guess"] = "root translation looks absent or FK not applied"
    elif span < 0.6:
        rep["unit_guess"] = "metres"
    elif span < 60.0:
        rep["unit_guess"] = "centimetres, divide positions by 100"
    else:
        rep["unit_guess"] = "millimetres, divide positions by 1000"

    v = zero_phase_lowpass(root[:, axis], fs, min(8.0, fs / 2.5))
    a_root = np.gradient(np.gradient(v)) * fs ** 2
    rep["root_vert_accel_rms_g"] = round(band_rms(a_root, fs) / G, 4)
    rep["root_vert_range"] = round(float(root[:, axis].ptp()), 4)
    rep["com_channel_usable"] = bool(rep["root_vert_accel_rms_g"] > 0.03)

    ls = np.asarray(left_sum, dtype=float)
    rs = np.asarray(right_sum, dtype=float)
    tot = ls + rs
    rep["pressure_total_cv"] = round(float(np.std(tot) / (np.mean(tot) + 1e-12)), 3)
    rep["pressure_left_duty"] = round(float(np.mean(ls > 0.1 * np.percentile(ls, 95))), 3)
    rep["pressure_right_duty"] = round(float(np.mean(rs > 0.1 * np.percentile(rs, 95))), 3)

    verdict = []
    if not rep["com_channel_usable"]:
        verdict.append("CoM channel dead, fall back to the event estimator")
    if rep["unit_guess"] != "metres":
        verdict.append("fix position units before any dynamics")
    if axis != 2:
        verdict.append(f"vertical axis is {axis}, not the default 2")
    if rep["pressure_total_cv"] > 0.6:
        verdict.append("pressure total swings too hard for walking, "
                       "check whether flight phases or dropouts are present")
    rep["verdict"] = verdict or ["inputs look sane"]
    return rep


def _break_point(y: np.ndarray, i0: int, fs: float, k_pre: int, k_post: int,
                 pre_is_ramp: bool = True) -> float:
    n = len(y)
    a0, a1 = max(0, i0 - k_pre), max(2, i0)
    b0, b1 = min(n - 2, i0), min(n, i0 + k_post)
    if a1 - a0 < 2 or b1 - b0 < 2:
        return i0 / fs
    xa = np.arange(a0, a1)
    xb = np.arange(b0, b1)
    pa = np.polyfit(xa, y[a0:a1], 1)
    pb = np.polyfit(xb, y[b0:b1], 1)
    if abs(pa[0] - pb[0]) < 1e-12:
        return i0 / fs
    xi = (pb[1] - pa[1]) / (pa[0] - pb[0])
    if not (a0 - 2 <= xi <= b1 + 2):
        return i0 / fs
    return float(xi) / fs


def mocap_contact_onsets(heel_xyz: np.ndarray, toe_xyz: np.ndarray, fs: float,
                         vertical: int = 1, height_band: float = 0.03,
                         min_gap_s: float = 0.35) -> np.ndarray:
    z = np.minimum(np.asarray(heel_xyz, dtype=float)[:, vertical],
                   np.asarray(toe_xyz, dtype=float)[:, vertical])
    z = zero_phase_lowpass(z, fs, min(10.0, fs / 2.5))
    floor = float(np.percentile(z, 2.0))
    v = np.gradient(z) * fs
    low = z < floor + height_band
    onsets, last = [], -np.inf
    k = max(2, int(round(0.06 * fs)))
    for i in range(1, len(z)):
        if low[i] and not low[i - 1] and v[i - 1] < 0:
            t = _break_point(z, i, fs, k_pre=k, k_post=k)
            if t - last >= min_gap_s:
                onsets.append(t)
                last = t
    return np.asarray(onsets)


def pressure_contact_onsets(p: np.ndarray, fs: float,
                            min_gap_s: float = 0.35) -> np.ndarray:
    y = np.asarray(p, dtype=float)
    y = zero_phase_lowpass(y, fs, min(15.0, fs / 2.5))
    hi = float(np.percentile(y, 95))
    lo = float(np.percentile(y, 5))
    thr = lo + 0.25 * (hi - lo)
    above = y > thr
    onsets, last = [], -np.inf
    k = max(2, int(round(0.06 * fs)))
    for i in range(1, len(y)):
        if above[i] and not above[i - 1]:
            t = _break_point(y, i, fs, k_pre=k, k_post=k)
            if t - last >= min_gap_s:
                onsets.append(t)
                last = t
    return np.asarray(onsets)


def robust_offset(t_mocap: np.ndarray, t_press: np.ndarray,
                  coarse: float, tol_s: float = 0.12) -> dict:
    if t_mocap.size == 0 or t_press.size == 0:
        return dict(delta=coarse, mad=np.nan, n=0, drift_ppm=np.nan)
    pairs = []
    for t in t_mocap:
        k = int(np.argmin(np.abs(t_press - (t + coarse))))
        d = float(t_press[k] - t)
        if abs(d - coarse) <= tol_s:
            pairs.append((t, d))
    if len(pairs) < 3:
        return dict(delta=coarse, mad=np.nan, n=len(pairs), drift_ppm=np.nan)
    tt = np.array([p[0] for p in pairs])
    dd = np.array([p[1] for p in pairs])
    delta = float(np.median(dd))
    mad = float(np.median(np.abs(dd - delta)))
    drift = float(np.polyfit(tt, dd, 1)[0]) * 1e6 if tt.ptp() > 5.0 else 0.0
    return dict(delta=delta, mad=mad, n=len(pairs), drift_ppm=drift,
                sigma=mad * 1.4826,
                se=mad * 1.4826 / np.sqrt(len(pairs)))


def estimate_delta_t2_events(joint_xyz: dict[str, np.ndarray], fs: float,
                             left_sum: np.ndarray, right_sum: np.ndarray,
                             vertical: int = 1, t_coarse: float = 0.0,
                             tol_s: float = 0.12,
                             keys=("LeftFoot", "LeftToeBase",
                                   "RightFoot", "RightToeBase")) -> dict:
    """Primary estimator: match contact onset trains, pool both feet."""
    lh, lt, rh, rt = (np.asarray(joint_xyz[k], dtype=float) for k in keys)

    m_left = mocap_contact_onsets(lh, lt, fs, vertical=vertical)
    m_right = mocap_contact_onsets(rh, rt, fs, vertical=vertical)
    p_left = pressure_contact_onsets(left_sum, fs)
    p_right = pressure_contact_onsets(right_sum, fs)

    res_l = robust_offset(m_left, p_left, t_coarse, tol_s)
    res_r = robust_offset(m_right, p_right, t_coarse, tol_s)

    pooled_m = np.concatenate([m_left, m_right])
    pooled_p = np.concatenate([p_left, p_right])
    order_m, order_p = np.argsort(pooled_m), np.argsort(pooled_p)
    res = robust_offset(pooled_m[order_m], pooled_p[order_p], t_coarse, tol_s)

    out = dict(delta_t2=res["delta"], event_count=res["n"],
               event_mad_s=res["mad"], standard_error_s=res.get("se", np.nan),
               clock_drift_ppm=res.get("drift_ppm", np.nan),
               delta_t2_left=res_l["delta"], delta_t2_right=res_r["delta"],
               event_count_left=res_l["n"], event_count_right=res_r["n"],
               mocap_steps_left=int(m_left.size),
               mocap_steps_right=int(m_right.size),
               pressure_steps_left=int(p_left.size),
               pressure_steps_right=int(p_right.size))
    out["lr_disagreement_s"] = abs(res_l["delta"] - res_r["delta"])
    return out