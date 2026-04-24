from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def _blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


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


def _resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame
    height, width = frame.shape[:2]
    target_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


@dataclass
class CameraSignal:
    label: str
    fps: float
    frame_count: int
    energy: np.ndarray
    frame_times: np.ndarray | None = field(default=None, kw_only=True)
    source_kind: str = field(default="unknown", init=False)
    _frame_cache: OrderedDict[tuple[int, int], np.ndarray] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _cache_limit: int = field(default=12, init=False, repr=False)

    @property
    def duration(self) -> float:
        if self.frame_times is not None and len(self.frame_times):
            if len(self.frame_times) > 1:
                diffs = np.diff(self.frame_times.astype(np.float64, copy=False))
                positive_diffs = diffs[diffs > 0]
                step = float(np.median(positive_diffs)) if len(positive_diffs) else (1.0 / self.fps if self.fps > 0 else 0.0)
            else:
                step = 1.0 / self.fps if self.fps > 0 else 0.0
            return float(self.frame_times[-1] + step)
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def frame_index_at_time(self, preview_time: float) -> int:
        if self.frame_times is not None and len(self.frame_times):
            target_time = max(0.0, float(preview_time))
            index = int(np.searchsorted(self.frame_times, target_time, side="left"))
            if index <= 0:
                return 0
            if index >= len(self.frame_times):
                return max(self.frame_count - 1, 0)
            previous_delta = abs(target_time - float(self.frame_times[index - 1]))
            next_delta = abs(float(self.frame_times[index]) - target_time)
            return index if next_delta < previous_delta else index - 1
        return int(round(max(0.0, preview_time) * self.fps))

    def energy_times(self) -> np.ndarray:
        if self.frame_times is not None and len(self.frame_times) > 1 and len(self.energy):
            count = min(len(self.energy), len(self.frame_times) - 1)
            times = (self.frame_times[:count] + self.frame_times[1 : count + 1]) * 0.5
            return (times - times[0]).astype(np.float64, copy=False)
        return np.arange(len(self.energy), dtype=np.float64) / self.fps if self.fps > 0 else np.zeros(len(self.energy), dtype=np.float64)

    def _read_frame(self, frame_index: int) -> np.ndarray:
        raise NotImplementedError

    def input_paths(self) -> tuple[Path, ...]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def get_frame(self, frame_index: int, scale: float = 1.0) -> np.ndarray:
        frame_index = int(np.clip(frame_index, 0, max(self.frame_count - 1, 0)))
        scale_key = max(1, int(round(scale * 1000.0)))
        cache_key = (frame_index, scale_key)
        cached = self._frame_cache.pop(cache_key, None)
        if cached is not None:
            self._frame_cache[cache_key] = cached
            return cached

        frame = self._read_frame(frame_index)
        if frame is None:
            frame = _blank_frame()
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]
        frame = frame.astype(np.uint8, copy=False)
        frame = _resize_frame(frame, scale_key / 1000.0)

        self._frame_cache[cache_key] = frame
        while len(self._frame_cache) > self._cache_limit:
            self._frame_cache.popitem(last=False)
        return frame


@dataclass
class ImageSequenceSource(CameraSignal):
    source_dir: Path
    frame_paths: tuple[Path, ...]
    source_kind: str = field(default="frames", init=False)

    def _read_frame(self, frame_index: int) -> np.ndarray:
        frame_path = self.frame_paths[frame_index]
        image = _read_image_bgr(frame_path)
        if image is None:
            return _blank_frame()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def input_paths(self) -> tuple[Path, ...]:
        return self.frame_paths


@dataclass
class VideoSource(CameraSignal):
    path: Path
    capture: cv2.VideoCapture | None = field(default=None, init=False, repr=False)
    _capture_index: int = field(default=-1, init=False, repr=False)
    source_kind: str = field(default="mp4", init=False)

    def open(self) -> None:
        if self.capture is None:
            self.capture = cv2.VideoCapture(str(self.path))
            self._capture_index = -1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self._capture_index = -1

    def _read_frame(self, frame_index: int) -> np.ndarray:
        self.open()
        if self.capture is None:
            return _blank_frame()

        if frame_index == self._capture_index + 1:
            ok, frame = self.capture.read()
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.capture.read()
        if not ok or frame is None:
            return _blank_frame()

        self._capture_index = frame_index
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def input_paths(self) -> tuple[Path, ...]:
        return (self.path,)


@dataclass(frozen=True)
class RuntimeOptions:
    source_mode: str
    lite_mode: bool
    preview_scale: float
    axis_preset: str
    defer_heavy_refresh: bool = False


@dataclass
class BVHJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]
    channel_indices: list[int]


@dataclass
class BVHMotion:
    path: Path
    text: str
    frame_time_str: str
    raw_fps: float
    raw_frames: np.ndarray
    joints: list[BVHJoint]
    edges: list[tuple[int, int]]
    channel_info: list[tuple[str, str]]

    @property
    def duration(self) -> float:
        return len(self.raw_frames) / self.raw_fps if self.raw_fps > 0 else 0.0

    @property
    def has_joint_channels(self) -> bool:
        return self.raw_frames.shape[1] > 6

    @property
    def joint_motion_max(self) -> float:
        if len(self.raw_frames) < 2 or not self.has_joint_channels:
            return 0.0
        return float(np.max(np.abs(np.diff(self.raw_frames[:, 6:], axis=0))))

    @property
    def joint_motion_mean(self) -> float:
        if len(self.raw_frames) < 2 or not self.has_joint_channels:
            return 0.0
        return float(np.mean(np.abs(np.diff(self.raw_frames[:, 6:], axis=0))))

    @property
    def root_motion_mean(self) -> float:
        if len(self.raw_frames) < 2:
            return 0.0
        limit = min(6, self.raw_frames.shape[1])
        return float(np.mean(np.abs(np.diff(self.raw_frames[:, :limit], axis=0))))

    @property
    def motion_score(self) -> float:
        return self.joint_motion_mean * 100.0 + self.root_motion_mean

    @property
    def has_joint_motion(self) -> bool:
        return self.joint_motion_max > 1e-5


@dataclass
class BVHSourceData:
    role: str
    motion: BVHMotion
    energy_raw: np.ndarray
    energy_visual: np.ndarray

    @property
    def raw_duration(self) -> float:
        return self.motion.duration

    @property
    def display_name(self) -> str:
        return self.motion.path.name

    @property
    def has_joint_motion(self) -> bool:
        return self.motion.has_joint_motion

    @property
    def motion_score(self) -> float:
        return self.motion.motion_score


@dataclass
class SessionData:
    session_id: str
    cam_session: Path | None
    mocap_subject: Path | None
    cameras: list[CameraSignal]
    position_bvh: BVHSourceData | None
    order_bvh: BVHSourceData | None
    combined_camera_energy: np.ndarray
    reference_visual_fps: float
    output_dir: Path
    cache_dir: Path
    source_mode: str
    runtime_options: RuntimeOptions
    extra_bvhs: list[BVHSourceData] = field(default_factory=list)

    @property
    def bvh_sources(self) -> tuple[BVHSourceData, ...]:
        sources: list[BVHSourceData] = []
        seen: set[tuple[str, str]] = set()
        for item in (self.position_bvh, self.order_bvh, *self.extra_bvhs):
            if item is None:
                continue
            key = (item.role, str(item.motion.path))
            if key in seen:
                continue
            seen.add(key)
            sources.append(item)
        return tuple(sources)

    def _rank_bvh(self) -> list[BVHSourceData]:
        ranked = list(self.bvh_sources)
        ranked.sort(
            key=lambda item: (
                1 if item.has_joint_motion else 0,
                item.motion_score,
                1 if item.role == "order" else 0,
            ),
            reverse=True,
        )
        return ranked

    @property
    def display_bvh(self) -> BVHSourceData | None:
        ranked = self._rank_bvh()
        if ranked:
            return ranked[0]
        return None

    @property
    def alignment_bvh(self) -> BVHSourceData | None:
        ranked = self._rank_bvh()
        if ranked:
            return ranked[0]
        return None

    @property
    def has_visual(self) -> bool:
        return bool(self.cameras)

    @property
    def has_bvh(self) -> bool:
        return bool(self.bvh_sources)

    @property
    def camera_max_duration(self) -> float:
        return max((cam.duration for cam in self.cameras), default=0.0)

    @property
    def bvh_raw_duration(self) -> float:
        alignment_bvh = self.alignment_bvh
        return alignment_bvh.raw_duration if alignment_bvh is not None else 0.0

    @property
    def bvh_visual_duration(self) -> float:
        alignment_bvh = self.alignment_bvh
        if alignment_bvh is None or self.reference_visual_fps <= 0:
            return 0.0
        return len(alignment_bvh.energy_visual) / self.reference_visual_fps

    @property
    def preview_duration(self) -> float:
        display_duration = self.display_bvh.raw_duration if self.display_bvh is not None else 0.0
        return max(self.camera_max_duration, display_duration, 0.0)

    @property
    def available_camera_labels(self) -> tuple[str, ...]:
        return tuple(camera.label for camera in sorted(self.cameras, key=lambda item: item.label))


@dataclass
class AlignmentState:
    delta_t: float
    preview_time: float
    auto_delta_t: float
    auto_confidence: float
    status_message: str = ""


@dataclass(frozen=True)
class TrialInfo:
    subject: int
    action: int
    rep: int

    @property
    def mocap_folder_name(self) -> str:
        return f"S{self.subject}{self.action:02d}{self.rep}"

    @property
    def cam_session_suffix(self) -> str:
        return f"S{self.subject}{self.action:02d}_{self.rep}"

    @property
    def display_name(self) -> str:
        return f"对象{self.subject} 动作{self.action:02d} 第{self.rep}次"
