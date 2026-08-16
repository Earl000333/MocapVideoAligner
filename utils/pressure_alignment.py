from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import re

import numpy as np

from config import DEFAULT_OUTPUT_ROOT
from models import BVHMotion
from utils.bvh_pose import compute_joint_positions, transform_display_positions


@dataclass(frozen=True)
class PressureCurveSet:
    time_s: np.ndarray
    left_sum: np.ndarray
    right_sum: np.ndarray
    source_path: Path
    sample_fps: float


@dataclass(frozen=True)
class PressureSensorFrameSet:
    time_s: np.ndarray
    sensor_values: np.ndarray
    source_path: Path
    sample_fps: float

    @property
    def sensor_totals(self) -> np.ndarray:
        if len(self.sensor_values) == 0:
            return np.zeros(0, dtype=np.float64)
        return np.sum(self.sensor_values, axis=1)


@dataclass(frozen=True)
class MocapFootCurveSet:
    time_s: np.ndarray
    left_sum: np.ndarray
    right_sum: np.ndarray
    source_path: Path
    reference_fps: float
    axis_preset: str


@dataclass(frozen=True)
class PressureMetaInfo:
    source_path: Path | None
    started_at_iso: str | None
    epoch_monotonic_us: int | None
    t_coarse: float
    t_coarse_source: str


@dataclass(frozen=True)
class PressureAlignmentResult:
    session_id: str
    delta_t2: float
    delta_t2_left: float
    delta_t2_right: float
    peak_left: float
    peak_right: float
    t_coarse: float
    search_window_ms: int
    manual_adjusted: bool
    reference_fps: float
    mocap_source_file: str
    pressure_source_file: str
    axis_preset: str
    exported_at: str


def _norm01(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64, copy=True)
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _infer_fps(time_s: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=np.float64)
    if len(time_s) < 2:
        return 0.0
    diffs = np.diff(time_s)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return 0.0
    step = float(np.median(diffs))
    return 1.0 / step if step > 0 else 0.0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_pressure_curve_csv(csv_path: Path) -> PressureCurveSet:
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Pressure CSV is empty: {csv_path}")

    fieldnames = list(rows[0].keys())
    time_col = _find_column(fieldnames, ("time_s", "time", "t_s"))
    left_col = _find_column(fieldnames, ("left_sum", "left", "pressure_left"))
    right_col = _find_column(fieldnames, ("right_sum", "right", "pressure_right"))
    if time_col is None or left_col is None or right_col is None:
        raise ValueError(f"Pressure CSV must contain time_s, left_sum and right_sum: {csv_path}")

    time_values: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    for row in rows:
        try:
            time_values.append(float(row.get(time_col, "").strip()))
            left_values.append(float(row.get(left_col, "").strip()))
            right_values.append(float(row.get(right_col, "").strip()))
        except (TypeError, ValueError):
            continue

    if not time_values:
        raise ValueError(f"Pressure CSV has no numeric rows: {csv_path}")

    time_s = np.asarray(time_values, dtype=np.float64)
    left_sum = _norm01(np.asarray(left_values, dtype=np.float64))
    right_sum = _norm01(np.asarray(right_values, dtype=np.float64))
    sample_fps = _infer_fps(time_s)
    return PressureCurveSet(time_s=time_s, left_sum=left_sum, right_sum=right_sum, source_path=csv_path, sample_fps=sample_fps)


def load_pressure_sensor_csv(csv_path: Path) -> PressureSensorFrameSet:
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Pressure CSV is empty: {csv_path}")

    fieldnames = list(rows[0].keys())
    time_col = _find_column(fieldnames, ("t_us", "time_us", "timestamp_us", "time_s", "time"))
    if time_col is None:
        raise ValueError(f"Pressure sensor CSV must contain t_us or time_s: {csv_path}")

    excluded = {
        time_col,
        "frame_idx",
        "frame",
        "idx",
        "valid_mask",
        "source_frame_idx",
        "source_t_us",
        "source_time_us",
        "source_time_s",
    }
    sensor_cols = [name for name in fieldnames if name not in excluded]
    if not sensor_cols:
        raise ValueError(f"Pressure sensor CSV has no sensor columns: {csv_path}")

    time_values: list[float] = []
    sensor_rows: list[list[float]] = []
    for row in rows:
        try:
            raw_time = float(row.get(time_col, "").strip())
        except (TypeError, ValueError):
            continue

        values: list[float] = []
        try:
            for column in sensor_cols:
                values.append(float(row.get(column, "").strip()))
        except (TypeError, ValueError):
            continue

        time_values.append(raw_time / 1_000_000.0 if time_col.lower().endswith("us") else raw_time)
        sensor_rows.append(values)

    if not time_values:
        raise ValueError(f"Pressure sensor CSV has no numeric rows: {csv_path}")

    time_s = np.asarray(time_values, dtype=np.float64)
    time_s = time_s - float(time_s[0])
    sensor_values = np.asarray(sensor_rows, dtype=np.float64)
    if len(sensor_values):
        max_value = float(np.max(sensor_values))
        if max_value > 0:
            sensor_values = np.clip(sensor_values / max_value, 0.0, 1.0)
    sample_fps = _infer_fps(time_s)
    return PressureSensorFrameSet(
        time_s=time_s,
        sensor_values=sensor_values,
        source_path=csv_path,
        sample_fps=sample_fps,
    )


def build_pressure_curve_set(
    left: PressureSensorFrameSet,
    right: PressureSensorFrameSet,
    *,
    reference_fps: float = 0.0,
) -> PressureCurveSet:
    fps = float(reference_fps)
    if fps <= 0:
        fps = max(left.sample_fps, right.sample_fps, 40.0)

    duration = max(
        float(left.time_s[-1]) if len(left.time_s) else 0.0,
        float(right.time_s[-1]) if len(right.time_s) else 0.0,
    )
    sample_count = max(1, int(round(duration * fps)) + 1)
    time_s = np.arange(sample_count, dtype=np.float64) / fps

    left_totals = left.sensor_totals
    right_totals = right.sensor_totals
    left_sum = np.interp(time_s, left.time_s, left_totals) if len(left.time_s) else np.zeros(sample_count, dtype=np.float64)
    right_sum = np.interp(time_s, right.time_s, right_totals) if len(right.time_s) else np.zeros(sample_count, dtype=np.float64)
    source_path = left.source_path.parent / "pressure_left+right.csv"
    return PressureCurveSet(
        time_s=time_s,
        left_sum=_norm01(left_sum),
        right_sum=_norm01(right_sum),
        source_path=source_path,
        sample_fps=fps,
    )


def _side_matches(name: str, side: str) -> bool:
    lower = re.sub(r"\s+", "", name.lower())
    if side == "left":
        return "left" in lower or lower.startswith("l")
    return "right" in lower or lower.startswith("r")


def _is_foot_joint(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("foot", "toe", "heel", "ankle", "ball", "endsite"))


def _joint_vertical_axis(axis_preset: str) -> int:
    # transform_display_positions(zup) maps original Y-up to display (x, -z, y),
    # so the upright/height axis becomes index 2. Raw mode keeps original axes
    # and we treat Z as upright by convention for this project.
    return 2


def _candidate_joint_indices(motion: BVHMotion, side: str) -> list[int]:
    foot_indices = [
        index
        for index, joint in enumerate(motion.joints)
        if _side_matches(joint.name, side) and _is_foot_joint(joint.name)
    ]
    if foot_indices:
        return foot_indices
    fallback = [index for index, joint in enumerate(motion.joints) if _side_matches(joint.name, side)]
    return fallback


def load_mocap_foot_curves(motion: BVHMotion, *, axis_preset: str = "zup") -> MocapFootCurveSet:
    if len(motion.raw_frames) == 0:
        raise ValueError(f"BVH has no frames: {motion.path}")

    vertical_axis = _joint_vertical_axis(axis_preset)
    left_indices = _candidate_joint_indices(motion, "left")
    right_indices = _candidate_joint_indices(motion, "right")
    if not left_indices or not right_indices:
        raise ValueError(f"Unable to locate left/right foot joints in {motion.path.name}")

    left_curve: list[float] = []
    right_curve: list[float] = []
    for frame_index in range(len(motion.raw_frames)):
        positions = compute_joint_positions(motion, frame_index)
        display_positions = transform_display_positions(positions, axis_preset)
        left_curve.append(float(np.min(display_positions[left_indices, vertical_axis])))
        right_curve.append(float(np.min(display_positions[right_indices, vertical_axis])))

    time_s = np.arange(len(motion.raw_frames), dtype=np.float64) / motion.raw_fps if motion.raw_fps > 0 else np.arange(len(motion.raw_frames), dtype=np.float64)
    # h(t) = lowest foot-joint height; contact proxy = -h(t), then min-max to [0,1]
    # => higher values mean closer to ground / more plant-like contact.
    left_sum = _norm01(-np.asarray(left_curve, dtype=np.float64))
    right_sum = _norm01(-np.asarray(right_curve, dtype=np.float64))
    return MocapFootCurveSet(
        time_s=time_s,
        left_sum=left_sum,
        right_sum=right_sum,
        source_path=motion.path,
        reference_fps=float(motion.raw_fps),
        axis_preset=axis_preset,
    )


def load_pressure_meta(meta_path: Path | None, aligned_bvh_path: Path | None = None) -> PressureMetaInfo:
    if meta_path is None or not meta_path.exists():
        return PressureMetaInfo(source_path=meta_path, started_at_iso=None, epoch_monotonic_us=None, t_coarse=0.0, t_coarse_source="fallback_global")

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    started_at_iso = payload.get("started_at_iso")
    epoch_monotonic_us = payload.get("epoch_monotonic_us")
    t_coarse = 0.0
    t_coarse_source = "fallback_global"

    if started_at_iso and aligned_bvh_path is not None and aligned_bvh_path.exists():
        try:
            started_at = datetime.fromisoformat(str(started_at_iso))
            bvh_mtime = datetime.fromtimestamp(aligned_bvh_path.stat().st_mtime)
            candidate = (bvh_mtime - started_at).total_seconds()
            if abs(candidate) <= 300.0:
                t_coarse = float(candidate)
                t_coarse_source = "aligned_bvh_mtime_minus_pressure_start"
        except Exception:
            pass

    try:
        epoch_monotonic_us = int(epoch_monotonic_us) if epoch_monotonic_us is not None else None
    except (TypeError, ValueError):
        epoch_monotonic_us = None

    return PressureMetaInfo(
        source_path=meta_path,
        started_at_iso=str(started_at_iso) if started_at_iso is not None else None,
        epoch_monotonic_us=epoch_monotonic_us,
        t_coarse=t_coarse,
        t_coarse_source=t_coarse_source,
    )


def _interp_signal(signal_time: np.ndarray, signal_values: np.ndarray, sample_time: np.ndarray) -> np.ndarray:
    if len(signal_time) == 0 or len(signal_values) == 0:
        return np.zeros(len(sample_time), dtype=np.float64)
    count = min(len(signal_time), len(signal_values))
    signal_time = np.asarray(signal_time[:count], dtype=np.float64)
    signal_values = np.asarray(signal_values[:count], dtype=np.float64)
    if len(signal_time) == 1:
        return np.repeat(signal_values[:1], len(sample_time)).astype(np.float64)
    return np.interp(sample_time, signal_time, signal_values, left=np.nan, right=np.nan).astype(np.float64)


def _score_shift(
    mocap_time: np.ndarray,
    mocap_values: np.ndarray,
    pressure_time: np.ndarray,
    pressure_values: np.ndarray,
    delta_t2: float,
) -> float:
    shifted = _interp_signal(mocap_time, mocap_values, pressure_time - delta_t2)
    mask = np.isfinite(shifted) & np.isfinite(pressure_values)
    if int(mask.sum()) < 3:
        return float("-inf")
    x = shifted[mask]
    y = pressure_values[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return float("-inf")
    x /= x_std
    y /= y_std
    return float(np.mean(x * y))


def _search_delta(
    mocap_time: np.ndarray,
    mocap_values: np.ndarray,
    pressure_time: np.ndarray,
    pressure_values: np.ndarray,
    *,
    center: float,
    window_ms: int,
    fallback_span_s: float,
    step_s: float,
) -> tuple[float, float]:
    half_window = max(0.0, window_ms / 1000.0)
    if window_ms > 0:
        start = center - half_window
        end = center + half_window
    else:
        start = -fallback_span_s
        end = fallback_span_s

    if end < start:
        start, end = end, start

    if step_s <= 0:
        step_s = 1.0 / max(_infer_fps(pressure_time), 40.0)
    steps = max(3, int(round((end - start) / step_s)) + 1)
    candidates = np.linspace(start, end, steps, dtype=np.float64)

    best_delta = float(center)
    best_peak = float("-inf")
    for candidate in candidates:
        score = _score_shift(mocap_time, mocap_values, pressure_time, pressure_values, float(candidate))
        if score > best_peak:
            best_peak = score
            best_delta = float(candidate)
    return best_delta, best_peak


def estimate_pressure_alignment(
    mocap: MocapFootCurveSet,
    pressure: PressureCurveSet,
    meta: PressureMetaInfo,
    *,
    search_window_ms: int = 200,
) -> PressureAlignmentResult:
    ref_fps = pressure.sample_fps if pressure.sample_fps > 0 else mocap.reference_fps
    if ref_fps <= 0:
        ref_fps = 40.0
    step_s = 1.0 / ref_fps

    fallback_span_s = float(max(pressure.time_s[-1] if len(pressure.time_s) else 0.0, mocap.time_s[-1] if len(mocap.time_s) else 0.0, 1.0))
    center = meta.t_coarse if meta is not None else 0.0
    delta_left, peak_left = _search_delta(
        mocap.time_s,
        mocap.left_sum,
        pressure.time_s,
        pressure.left_sum,
        center=center,
        window_ms=search_window_ms if meta.t_coarse_source != "fallback_global" else 0,
        fallback_span_s=fallback_span_s,
        step_s=step_s,
    )
    delta_right, peak_right = _search_delta(
        mocap.time_s,
        mocap.right_sum,
        pressure.time_s,
        pressure.right_sum,
        center=center,
        window_ms=search_window_ms if meta.t_coarse_source != "fallback_global" else 0,
        fallback_span_s=fallback_span_s,
        step_s=step_s,
    )

    if np.isfinite(delta_left) and np.isfinite(delta_right):
        delta_t2 = float((delta_left + delta_right) / 2.0)
    elif np.isfinite(delta_left):
        delta_t2 = float(delta_left)
    elif np.isfinite(delta_right):
        delta_t2 = float(delta_right)
    else:
        delta_t2 = 0.0

    session_id = _guess_session_id(mocap.source_path, pressure.source_path, meta.source_path)
    exported_at = datetime.now().isoformat(timespec="seconds")
    return PressureAlignmentResult(
        session_id=session_id,
        delta_t2=delta_t2,
        delta_t2_left=float(delta_left),
        delta_t2_right=float(delta_right),
        peak_left=float(peak_left),
        peak_right=float(peak_right),
        t_coarse=float(meta.t_coarse if meta is not None else 0.0),
        search_window_ms=int(search_window_ms),
        manual_adjusted=False,
        reference_fps=float(ref_fps),
        mocap_source_file=mocap.source_path.name,
        pressure_source_file=pressure.source_path.name,
        axis_preset=mocap.axis_preset,
        exported_at=exported_at,
    )


def _guess_session_id(*paths: Path | None) -> str:
    for path in paths:
        if path is None:
            continue
        for candidate in (path.name, path.stem, path.parent.name):
            match = re.search(r"S\d+_\d+", candidate)
            if match:
                return match.group(0)
    return "pressure_session"


def normalize_trial_code(session_id: str | None) -> str | None:
    if not session_id:
        return None
    text = str(session_id).strip()
    if not text:
        return None
    match = re.search(r"S(\d+)_(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"S{match.group(1)}{match.group(2)}"
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    return compact or None


def load_visual_alignment_offset(csv_path: Path) -> float:
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Visual alignment CSV is empty: {csv_path}")

    fieldnames = list(rows[0].keys())
    offset_col = _find_column(
        fieldnames,
        (
            "偏移量(s)",
            "偏移量",
            "delta_t",
            "offset_s",
            "offset",
        ),
    )
    if offset_col is None:
        raise ValueError(f"Visual alignment CSV must contain 偏移量(s): {csv_path}")

    for row in rows:
        raw = str(row.get(offset_col, "")).strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid offset value in {csv_path}: {raw}") from exc
    raise ValueError(f"Visual alignment CSV has no numeric offset: {csv_path}")


def resolve_visual_alignment_csv(
    session_id: str | None,
    review_root: Path | None = None,
) -> Path | None:
    if review_root is None or not review_root.exists():
        return None

    trial_code = normalize_trial_code(session_id)
    if trial_code is None:
        return None

    candidates = [
        review_root / f"{trial_code}.csv",
        review_root / f"{trial_code.lower()}.csv",
        review_root / f"{trial_code.upper()}.csv",
    ]
    if session_id:
        candidates.append(review_root / f"{session_id}.csv")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in sorted(review_root.glob("*.csv")):
        if path.name.lower().startswith("conversion_summary"):
            continue
        stem_code = normalize_trial_code(path.stem)
        if stem_code is not None and stem_code.lower() == trial_code.lower():
            return path
    return None



def _session_suffix_candidates(session_id: str | None) -> list[str]:
    if not session_id:
        return []
    text = str(session_id).strip()
    if not text:
        return []
    candidates = [text]
    compact = normalize_trial_code(text)
    if compact and compact not in candidates:
        candidates.append(compact)
    if text.upper().startswith("S") and "_" in text:
        candidates.append(text.replace("_", ""))

    # Expand compact codes like S14011 -> S1401_1 for folder suffix matching.
    match = re.fullmatch(r"S(\d+)(\d)", text, flags=re.IGNORECASE)
    if match is not None:
        candidates.append(f"S{match.group(1)}_{match.group(2)}")
        candidates.append(f"S{match.group(1)}{match.group(2)}")
    else:
        match = re.fullmatch(r"S(\d+)_(\d+)", text, flags=re.IGNORECASE)
        if match is not None:
            candidates.append(f"S{match.group(1)}_{match.group(2)}")
            candidates.append(f"S{match.group(1)}{match.group(2)}")

    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def resolve_reconstructed_rec_dir(
    session_id: str | None,
    reconstruction_root: Path | None,
) -> Path | None:
    """Locate the reconstructed rec... directory for a trial.

    Expected layout:
      reconstruction_<timestamp>/<date>/<Sx>/rec..._<session_id>/
    """
    if reconstruction_root is None:
        return None
    root = Path(reconstruction_root)
    if not root.exists() or not root.is_dir():
        return None

    suffixes = _session_suffix_candidates(session_id)
    if not suffixes:
        return None

    matches: list[Path] = []
    for path in root.rglob("reconstruction_manifest.csv"):
        rec_dir = path.parent
        name_lower = rec_dir.name.lower()
        for suffix in suffixes:
            suffix_lower = suffix.lower()
            if name_lower.endswith(f"_{suffix_lower}") or name_lower.endswith(suffix_lower):
                matches.append(rec_dir)
                break

    if not matches:
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            name_lower = path.name.lower()
            for suffix in suffixes:
                suffix_lower = suffix.lower()
                if name_lower.endswith(f"_{suffix_lower}") or name_lower == suffix_lower:
                    left_hits = list(path.glob("pressure_left_t*.csv"))
                    right_hits = list(path.glob("pressure_right_t*.csv"))
                    if left_hits and right_hits:
                        matches.append(path)
                        break

    if not matches:
        return None

    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p).lower()), reverse=True)
    return matches[0]


def list_reconstructed_segments(rec_dir: Path) -> list[str]:
    rec_dir = Path(rec_dir)
    manifest_path = rec_dir / "reconstruction_manifest.csv"
    segments: list[str] = []
    if manifest_path.exists():
        rows = _read_csv_rows(manifest_path)
        for row in rows:
            name = str(row.get("segment_name", "")).strip()
            if name:
                segments.append(name)
    if segments:
        return segments

    discovered: set[str] = set()
    for path in rec_dir.glob("pressure_left_t*.csv"):
        stem = path.stem
        if stem.startswith("pressure_left_"):
            discovered.add(stem[len("pressure_left_"):])
    for path in rec_dir.glob("pressure_right_t*.csv"):
        stem = path.stem
        if stem.startswith("pressure_right_"):
            discovered.add(stem[len("pressure_right_"):])

    def _segment_key(name: str) -> tuple[int, str]:
        text = name[1:] if name.lower().startswith("t") else name
        if text.isdigit():
            return (int(text), name)
        return (10**9, name)

    return sorted(discovered, key=_segment_key)


def _segment_csv_path(rec_dir: Path, side: str, segment: str) -> Path | None:
    rec_dir = Path(rec_dir)
    candidates = [
        rec_dir / f"pressure_{side}_{segment}.csv",
    ]
    if not segment.lower().startswith("t"):
        candidates.append(rec_dir / f"pressure_{side}_t{segment}.csv")
    else:
        # segment already t0
        candidates.append(rec_dir / f"pressure_{side}_{segment}.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_pressure_sensor_rows(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load absolute times (seconds) and sensor matrix without rezero/normalization."""
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Pressure CSV is empty: {csv_path}")

    fieldnames = list(rows[0].keys())
    time_col = _find_column(fieldnames, ("t_us", "time_us", "timestamp_us", "time_s", "time"))
    if time_col is None:
        raise ValueError(f"Pressure sensor CSV must contain t_us or time_s: {csv_path}")

    excluded = {
        time_col,
        "frame_idx",
        "frame",
        "idx",
        "valid_mask",
        "source_frame_idx",
        "source_t_us",
        "source_time_us",
        "source_time_s",
    }
    sensor_cols = [name for name in fieldnames if name not in excluded]
    if not sensor_cols:
        raise ValueError(f"Pressure sensor CSV has no sensor columns: {csv_path}")

    time_values: list[float] = []
    sensor_rows: list[list[float]] = []
    for row in rows:
        try:
            raw_time = float(str(row.get(time_col, "")).strip())
        except (TypeError, ValueError):
            continue
        values: list[float] = []
        try:
            for column in sensor_cols:
                values.append(float(str(row.get(column, "")).strip()))
        except (TypeError, ValueError):
            continue
        time_values.append(raw_time / 1_000_000.0 if time_col.lower().endswith("us") else raw_time)
        sensor_rows.append(values)

    if not time_values:
        raise ValueError(f"Pressure sensor CSV has no numeric rows: {csv_path}")

    return (
        np.asarray(time_values, dtype=np.float64),
        np.asarray(sensor_rows, dtype=np.float64),
    )


def _finalize_sensor_frames(
    time_s: np.ndarray,
    sensor_values: np.ndarray,
    source_path: Path,
    *,
    rezero: bool = True,
    normalize: bool = True,
) -> PressureSensorFrameSet:
    time_s = np.asarray(time_s, dtype=np.float64)
    sensor_values = np.asarray(sensor_values, dtype=np.float64)
    if rezero and len(time_s):
        time_s = time_s - float(time_s[0])
    if normalize and len(sensor_values):
        max_value = float(np.max(sensor_values))
        if max_value > 0:
            sensor_values = np.clip(sensor_values / max_value, 0.0, 1.0)
    return PressureSensorFrameSet(
        time_s=time_s,
        sensor_values=sensor_values,
        source_path=source_path,
        sample_fps=_infer_fps(time_s),
    )


def load_reconstructed_pressure_sensors(
    rec_dir: Path,
) -> tuple[PressureSensorFrameSet, PressureSensorFrameSet]:
    """Load reconstructed left/right sensor streams for one rec... directory.

    All segments listed in reconstruction_manifest.csv (or discovered by glob)
    are concatenated in segment order using absolute t_us timelines.
    """
    rec_dir = Path(rec_dir)
    if not rec_dir.exists() or not rec_dir.is_dir():
        raise FileNotFoundError(f"Reconstructed tactile directory does not exist: {rec_dir}")

    segments = list_reconstructed_segments(rec_dir)
    if not segments:
        raise FileNotFoundError(f"No reconstructed pressure segments found in {rec_dir}")

    left_times: list[np.ndarray] = []
    left_values: list[np.ndarray] = []
    right_times: list[np.ndarray] = []
    right_values: list[np.ndarray] = []
    left_source: Path | None = None
    right_source: Path | None = None

    for segment in segments:
        left_path = _segment_csv_path(rec_dir, "left", segment)
        right_path = _segment_csv_path(rec_dir, "right", segment)
        if left_path is not None:
            try:
                t_arr, v_arr = _load_pressure_sensor_rows(left_path)
            except ValueError as exc:
                # Some reconstructed segments are header-only / zero-row.
                if "empty" in str(exc).lower() or "no numeric rows" in str(exc).lower():
                    pass
                else:
                    raise
            else:
                if len(t_arr) > 0:
                    left_times.append(t_arr)
                    left_values.append(v_arr)
                    left_source = left_path
        if right_path is not None:
            try:
                t_arr, v_arr = _load_pressure_sensor_rows(right_path)
            except ValueError as exc:
                if "empty" in str(exc).lower() or "no numeric rows" in str(exc).lower():
                    pass
                else:
                    raise
            else:
                if len(t_arr) > 0:
                    right_times.append(t_arr)
                    right_values.append(v_arr)
                    right_source = right_path

    if not left_times or not right_times:
        raise FileNotFoundError(
            f"Reconstructed tactile directory is missing left/right segment CSVs: {rec_dir}"
        )

    left_time = np.concatenate(left_times)
    left_mat = np.vstack(left_values)
    right_time = np.concatenate(right_times)
    right_mat = np.vstack(right_values)

    # Preserve left/right relative timing by rezeroing both to the earlier foot start.
    origin = float(min(left_time[0], right_time[0]))
    left = _finalize_sensor_frames(
        left_time - origin,
        left_mat,
        left_source if left_source is not None else rec_dir / "pressure_left_reconstructed.csv",
        rezero=False,
        normalize=True,
    )
    right = _finalize_sensor_frames(
        right_time - origin,
        right_mat,
        right_source if right_source is not None else rec_dir / "pressure_right_reconstructed.csv",
        rezero=False,
        normalize=True,
    )
    return left, right


def build_pressure_aligned_curve_matrix(
    mocap: MocapFootCurveSet,
    pressure: PressureCurveSet,
    delta_t2: float,
) -> tuple[np.ndarray, list[str]]:
    time_s = np.asarray(pressure.time_s, dtype=np.float64)
    mocap_left = _interp_signal(mocap.time_s, mocap.left_sum, time_s - delta_t2)
    mocap_right = _interp_signal(mocap.time_s, mocap.right_sum, time_s - delta_t2)
    matrix = np.column_stack((time_s, mocap_left, mocap_right, pressure.left_sum, pressure.right_sum))
    headers = ["time_s", "mocap_left", "mocap_right", "pressure_left", "pressure_right"]
    return matrix, headers


def export_pressure_alignment_bundle(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    result: PressureAlignmentResult,
    mocap: MocapFootCurveSet,
    pressure: PressureCurveSet,
    figure=None,
) -> dict[str, Path]:
    output_dir = output_root / result.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / f"{result.session_id}_pressure_alignment.json"
    csv_path = output_dir / f"{result.session_id}_pressure_aligned_curves.csv"
    image_path = output_dir / f"{result.session_id}_pressure_calibration.png"

    payload = {
        "session_id": result.session_id,
        "delta_t2": result.delta_t2,
        "delta_t2_left": result.delta_t2_left,
        "delta_t2_right": result.delta_t2_right,
        "peak_left": result.peak_left,
        "peak_right": result.peak_right,
        "t_coarse": result.t_coarse,
        "search_window_ms": result.search_window_ms,
        "manual_adjusted": result.manual_adjusted,
        "reference_fps": result.reference_fps,
        "mocap_source_file": result.mocap_source_file,
        "pressure_source_file": result.pressure_source_file,
        "axis_preset": result.axis_preset,
        "exported_at": result.exported_at,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix, headers = build_pressure_aligned_curve_matrix(mocap, pressure, result.delta_t2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in matrix:
            writer.writerow(["" if not np.isfinite(value) else f"{float(value):.6f}" for value in row])

    if figure is not None and hasattr(figure, "savefig"):
        figure.savefig(image_path, dpi=150, bbox_inches="tight")

    return {
        "metadata": metadata_path,
        "curves": csv_path,
        "image": image_path,
    }
