from __future__ import annotations

import numpy as np

from models import BVHSourceData, CameraSignal, SessionData
from utils.energy import normalize_zscore, resample_camera_energy, resample_signal


def quantize_time(value: float, fps: float) -> float:
    if fps <= 0:
        return float(value)
    return float(round(value * fps) / fps)


def estimate_initial_offset(
    camera_energy_mean: np.ndarray,
    visual_fps: float,
    bvh_energy_sig: np.ndarray,
    bvh_fps: float,
) -> tuple[float, float]:
    if len(camera_energy_mean) == 0 or len(bvh_energy_sig) == 0:
        return 0.0, 0.0

    bvh_rs = resample_signal(bvh_energy_sig, bvh_fps, visual_fps)
    min_len = min(len(camera_energy_mean), len(bvh_rs))
    if min_len == 0:
        return 0.0, 0.0

    a = normalize_zscore(camera_energy_mean[:min_len])
    b = normalize_zscore(bvh_rs[:min_len])
    if not np.any(a) or not np.any(b):
        return 0.0, 0.0

    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a), dtype=np.int64)
    best_idx = int(np.argmax(corr))
    delta_t = float(lags[best_idx] / visual_fps)

    max_possible = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    confidence = float(np.clip(corr[best_idx] / max_possible, 0.0, 1.0)) if max_possible > 1e-12 else 0.0
    return delta_t, confidence


def aligned_start_frames(
    session: SessionData,
    delta_t: float,
    bvh_source: BVHSourceData | None = None,
) -> tuple[int, dict[str, int]]:
    target_bvh = bvh_source or session.alignment_bvh
    if delta_t >= 0:
        bvh_start = int(round(delta_t * target_bvh.motion.raw_fps)) if target_bvh is not None else 0
        camera_starts = {camera.label: 0 for camera in session.cameras}
    else:
        bvh_start = 0
        camera_starts = {
            camera.label: camera.frame_index_at_time(abs(delta_t))
            for camera in session.cameras
        }
    return bvh_start, camera_starts


def build_aligned_curve_matrix(session: SessionData, delta_t: float) -> tuple[np.ndarray, list[str]]:
    ref_fps = session.reference_visual_fps
    camera_signals = {
        camera.label: resample_camera_energy(camera, ref_fps)
        for camera in session.cameras
    }
    bvh_signal = session.alignment_bvh.energy_visual if session.alignment_bvh is not None else np.zeros(0, dtype=np.float64)

    if delta_t >= 0:
        bvh_start = int(round(delta_t * ref_fps))
        camera_starts = {label: 0 for label in camera_signals}
    else:
        bvh_start = 0
        camera_starts = {label: int(round(abs(delta_t) * ref_fps)) for label in camera_signals}

    available_lengths = []
    for label, signal in camera_signals.items():
        available_lengths.append(len(signal) - camera_starts[label])
    if len(bvh_signal):
        available_lengths.append(len(bvh_signal) - bvh_start)

    if not available_lengths:
        return np.zeros((0, 1), dtype=np.float64), ["time_s"]

    min_len = max(0, min(available_lengths))
    columns = [np.arange(min_len, dtype=np.float64) / ref_fps]
    headers = ["time_s"]
    for label in sorted(camera_signals):
        start = camera_starts[label]
        columns.append(camera_signals[label][start : start + min_len])
        headers.append(label)
    if len(bvh_signal):
        columns.append(bvh_signal[bvh_start : bvh_start + min_len])
        headers.append("bvh")
    return np.column_stack(columns), headers


def preview_time_to_camera_frame(camera: CameraSignal, preview_time: float) -> int:
    return camera.frame_index_at_time(preview_time)


def preview_time_to_bvh_frame(
    session: SessionData,
    preview_time: float,
    delta_t: float,
    bvh_source: BVHSourceData | None = None,
) -> int:
    target_bvh = bvh_source or session.display_bvh
    if target_bvh is None:
        return 0
    bvh_time = preview_time - delta_t
    return int(round(max(0.0, bvh_time) * target_bvh.motion.raw_fps))
