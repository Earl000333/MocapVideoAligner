from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from config import ARM_LENGTH_CM, SMOOTH_MS
from models import BVHMotion, CameraSignal


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return np.array([1.0], dtype=np.float64)
    return kernel / kernel_sum


def _read_image_bgr(image_path: Path) -> np.ndarray | None:
    image = None
    if hasattr(cv2, "imdecode"):
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if len(data) > 0:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            image = None
    if image is None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return image


def _load_rgb_image(image_path: Path) -> np.ndarray:
    image = _read_image_bgr(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _gray_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _parse_hhmmss_timestamp_seconds(frame_path: Path) -> float | None:
    stem = frame_path.stem
    matches = re.findall(r"(?<!\d)(\d{6})[._-](\d{1,6})(?!\d)", stem)
    if not matches:
        return None
    hhmmss, fraction = matches[-1]
    hour = int(hhmmss[:2])
    minute = int(hhmmss[2:4])
    second = int(hhmmss[4:6])
    fraction_seconds = int(fraction) / (10 ** len(fraction))
    if hour > 23 or minute > 59 or second > 59:
        return None
    return float(hour * 3600 + minute * 60 + second + fraction_seconds)


def infer_frame_times_from_paths(frame_paths: tuple[Path, ...], fallback_fps: float) -> tuple[np.ndarray | None, float]:
    parsed = [_parse_hhmmss_timestamp_seconds(path) for path in frame_paths]
    if any(value is None for value in parsed) or len(parsed) < 2:
        return None, float(fallback_fps)

    raw_times = np.array([float(value) for value in parsed], dtype=np.float64)
    for index in range(1, len(raw_times)):
        if raw_times[index] < raw_times[index - 1]:
            raw_times[index:] += 24.0 * 3600.0

    frame_times = raw_times - raw_times[0]
    total_duration = float(frame_times[-1])
    if total_duration <= 0:
        return None, float(fallback_fps)

    fps = (len(frame_times) - 1) / total_duration
    if not np.isfinite(fps) or fps <= 0:
        return None, float(fallback_fps)
    return frame_times, float(fps)


def image_sequence_energy(frame_paths: tuple[Path, ...], fps: float) -> tuple[np.ndarray, float, int]:
    if not frame_paths:
        raise FileNotFoundError("Image sequence is empty")

    prev_gray = _gray_frame(_load_rgb_image(frame_paths[0]))
    energy = []
    for frame_path in frame_paths[1:]:
        gray = _gray_frame(_load_rgb_image(frame_path))
        energy.append(np.abs(gray - prev_gray).mean())
        prev_gray = gray

    total = len(frame_paths)
    print(f"  [Frames] {frame_paths[0].parent.name}  fps={fps:.2f}  frames={total}  duration={total / fps:.2f}s")
    return np.array(energy, dtype=np.float64), float(fps), total


def video_energy(video_path: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [Video] {video_path.name}  fps={fps:.2f}  frames={total}  duration={total / fps:.2f}s")

    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from: {video_path}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)

    energy = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        energy.append(np.abs(gray - prev_gray).mean())
        prev_gray = gray

    cap.release()
    return np.array(energy, dtype=np.float64), fps, total


def smooth_energy(energy: np.ndarray, fps: float, smooth_ms: float = SMOOTH_MS) -> np.ndarray:
    if len(energy) == 0 or smooth_ms <= 0:
        return energy.copy()

    sigma = fps * smooth_ms / 1000.0
    if sigma < 0.5:
        return energy.copy()

    kernel = _gaussian_kernel(sigma)
    pad = len(kernel) // 2
    padded = np.pad(energy.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_bvh_energy(motion: BVHMotion, smooth_ms: float = SMOOTH_MS) -> tuple[np.ndarray, float]:
    frames = motion.raw_frames.copy()
    fps = motion.raw_fps

    pos_idx = [index for index, (_, name) in enumerate(motion.channel_info) if "position" in name.lower()]
    rot_idx = [index for index, (_, name) in enumerate(motion.channel_info) if "rotation" in name.lower()]

    if rot_idx:
        frames[:, rot_idx] = np.unwrap(np.deg2rad(frames[:, rot_idx]), axis=0)

    vel = np.diff(frames, axis=0)
    dt = 1.0 / fps if fps > 0 else 0.0

    if pos_idx and dt > 0:
        pos_speed = np.sqrt((vel[:, pos_idx] ** 2).sum(axis=1)) / dt
    else:
        pos_speed = np.zeros(len(vel), dtype=np.float64)

    if rot_idx and dt > 0:
        rot_speed = np.sqrt((vel[:, rot_idx] ** 2).mean(axis=1)) / dt
    else:
        rot_speed = np.zeros(len(vel), dtype=np.float64)

    energy = np.sqrt(pos_speed ** 2 + (rot_speed * ARM_LENGTH_CM) ** 2)
    energy = smooth_energy(energy, fps, smooth_ms=smooth_ms)
    print(f"  [BVH energy] fps={fps:.2f}  max={energy.max(initial=0.0):.2f}  mean={energy.mean() if len(energy) else 0.0:.2f}")
    return energy, float(fps)


def norm01(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    lo = float(x.min())
    hi = float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    std = float(x.std())
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def resample_signal(signal: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if len(signal) == 0:
        return signal.copy()
    if abs(src_fps - dst_fps) < 1e-6:
        return signal.copy()
    if len(signal) == 1 or src_fps <= 0 or dst_fps <= 0:
        return np.repeat(signal[:1], 1).astype(np.float64)

    src_times = np.arange(len(signal), dtype=np.float64) / src_fps
    dst_count = max(1, int(round((len(signal) - 1) * dst_fps / src_fps)) + 1)
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_fps
    return np.interp(dst_times, src_times, signal).astype(np.float64)


def resample_signal_at_times(signal: np.ndarray, src_times: np.ndarray, dst_fps: float) -> np.ndarray:
    if len(signal) == 0:
        return signal.copy()
    if len(signal) == 1 or len(src_times) == 0 or dst_fps <= 0:
        return np.repeat(signal[:1], 1).astype(np.float64)

    count = min(len(signal), len(src_times))
    times = np.asarray(src_times[:count], dtype=np.float64)
    values = np.asarray(signal[:count], dtype=np.float64)
    if len(times) == 1:
        return values[:1].copy()

    times = times - times[0]
    duration = float(times[-1])
    if duration <= 0:
        return values.copy()

    dst_count = max(1, int(round(duration * dst_fps)) + 1)
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_fps
    return np.interp(dst_times, times, values).astype(np.float64)


def resample_camera_energy(camera: CameraSignal, reference_fps: float) -> np.ndarray:
    if camera.frame_times is not None and len(camera.frame_times) > 1:
        return resample_signal_at_times(camera.energy, camera.energy_times(), reference_fps)
    return resample_signal(camera.energy, camera.fps, reference_fps)


def build_combined_camera_energy(cameras: list[CameraSignal], reference_fps: float | None = None) -> tuple[np.ndarray, float]:
    if not cameras:
        raise ValueError("At least one camera signal is required")

    ref_fps = reference_fps or cameras[0].fps
    resampled = [normalize_zscore(resample_camera_energy(camera, ref_fps)) for camera in cameras]
    min_len = min(len(signal) for signal in resampled)
    stacked = np.vstack([signal[:min_len] for signal in resampled])
    return stacked.mean(axis=0), float(ref_fps)
