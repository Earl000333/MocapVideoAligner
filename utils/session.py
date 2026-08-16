from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from config import (
    CACHE_VERSION,
    DEFAULT_AXIS_PRESET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CAM_ROOT,
    DEFAULT_MOCAP_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREVIEW_SCALE,
    DEFAULT_SOURCE_MODE,
    FRAME_EXTENSIONS,
    FRAME_SEQUENCE_FPS,
    LITE_PREVIEW_SCALE,
    SMOOTH_MS,
)
from models import BVHSourceData, ImageSequenceSource, RuntimeOptions, SessionData, TrialInfo, VideoSource
from utils.bvh_parser import load_bvh_motion
from utils.energy import (
    build_combined_camera_energy,
    compute_bvh_energy,
    image_sequence_energy,
    infer_frame_times_from_paths,
    resample_signal,
    smooth_energy,
    video_energy,
)


def find_cam_session(session_id: str, cam_root: Path) -> Path:
    if not cam_root.exists():
        raise FileNotFoundError(f"相机会话根目录不存在：{cam_root}")

    matches = [path for path in cam_root.iterdir() if path.is_dir() and path.name.endswith(f"_{session_id}")]
    if not matches:
        for action_folder in cam_root.iterdir():
            if not action_folder.is_dir():
                continue
            matches.extend(
                path
                for path in action_folder.iterdir()
                if path.is_dir() and path.name.endswith(f"_{session_id}")
            )
    if not matches:
        raise FileNotFoundError(
            f"没有在 {cam_root} 或其动作子目录下找到匹配 *_{session_id} 的相机会话目录。"
        )
    matches.sort(key=lambda path: str(path).lower())
    if len(matches) > 1:
        print(f"  [warn] 找到多个相机会话，默认使用：{matches[0].name}")
    return matches[0]


def _parse_trial_folder_name(folder_name: str) -> TrialInfo | None:
    match = re.match(r"^S(\d+)(\d{2})(\d)$", folder_name)
    if not match:
        return None
    return TrialInfo(
        subject=int(match.group(1)),
        action=int(match.group(2)),
        rep=int(match.group(3)),
    )


def _iter_mocap_trial_folders(mocap_root: Path):
    if not mocap_root.exists():
        return
    for folder in mocap_root.iterdir():
        if not folder.is_dir():
            continue
        if _parse_trial_folder_name(folder.name) is not None:
            yield folder
            continue
        for nested in folder.iterdir():
            if nested.is_dir() and _parse_trial_folder_name(nested.name) is not None:
                yield nested


def find_mocap_subject(session_id: str, mocap_root: Path) -> Path:
    if not mocap_root.exists():
        raise FileNotFoundError(f"动捕根目录不存在：{mocap_root}")

    match = re.match(r"S(\d+)_(\d+)", session_id)
    if not match:
        raise ValueError(f"session_id 格式应为 S<编号>_<次数>，当前收到：{session_id}")
    folder_name = f"S{match.group(1)}{match.group(2)}"
    direct = mocap_root / folder_name
    if direct.exists():
        return direct
    for folder in _iter_mocap_trial_folders(mocap_root):
        if folder.name == folder_name:
            return folder
    raise FileNotFoundError(f"动捕目录不存在：{direct}，也没有在下一层分组目录中找到 {folder_name}。")


def get_cam_frame_dirs(cam_session: Path) -> dict[str, Path]:
    frame_dirs = {}
    for index in range(1, 5):
        for candidate_name in (f"cam{index}", str(index)):
            frame_dir = cam_session / candidate_name
            if frame_dir.is_dir():
                frame_dirs[f"cam{index}"] = frame_dir
                break
    return frame_dirs


def get_cam_videos(cam_session: Path) -> dict[str, Path]:
    videos = {}
    for mp4 in sorted(cam_session.glob("*.mp4")):
        match = re.match(r"^(\d)_", mp4.name)
        if match:
            videos[f"cam{match.group(1)}"] = mp4
    return videos


def _frame_sort_key(frame_path: Path) -> tuple[int, tuple[int, ...] | str]:
    matches = re.findall(r"(\d+)", frame_path.stem)
    if matches:
        return (0, tuple(int(value) for value in matches))
    return (1, frame_path.name)


def list_frame_paths(frame_dir: Path) -> tuple[Path, ...]:
    frame_paths = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS]
    frame_paths.sort(key=_frame_sort_key)
    return tuple(frame_paths)


def get_mocap_files(mocap_subject: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in mocap_subject.glob("*.bvh") if path.is_file()))


def enumerate_trials(mocap_root: Path, cam_root: Path | None = None) -> list[TrialInfo]:
    if not mocap_root.exists():
        return []
    trials = []
    seen: set[tuple[int, int, int]] = set()
    for folder in _iter_mocap_trial_folders(mocap_root):
        trial = _parse_trial_folder_name(folder.name)
        if trial is None:
            continue
        key = (trial.subject, trial.action, trial.rep)
        if key in seen:
            continue
        seen.add(key)
        trials.append(trial)
    trials.sort(key=lambda t: (t.subject, t.action, t.rep))
    return trials


def choose_bvh_roles(mocap_files: tuple[Path, ...]) -> tuple[Path | None, Path | None]:
    position_bvh: Path | None = None
    order_bvh: Path | None = None

    for path in mocap_files:
        stem = path.stem.lower()
        if position_bvh is None and ("skeleton0" in stem or "position" in stem):
            position_bvh = path
        if order_bvh is None and ("skeleton1" in stem or "order" in stem or "motion" in stem):
            order_bvh = path

    remaining = [path for path in mocap_files if path not in {position_bvh, order_bvh}]
    if position_bvh is None and remaining:
        position_bvh = remaining.pop(0)
    if order_bvh is None and remaining:
        order_bvh = remaining.pop(0)
    return position_bvh, order_bvh


def choose_bvh_paths(mocap_files: tuple[Path, ...]) -> tuple[Path | None, Path | None, tuple[Path, ...]]:
    position_bvh, order_bvh = choose_bvh_roles(mocap_files)
    selected = {path for path in (position_bvh, order_bvh) if path is not None}
    extra_bvhs = tuple(sorted(path for path in mocap_files if path not in selected))
    return position_bvh, order_bvh, extra_bvhs


def _safe_bvh_role(path: Path, fallback: str) -> str:
    skeleton_match = re.search(r"skeleton(\d+)", path.stem, re.IGNORECASE)
    if skeleton_match:
        return f"skeleton{skeleton_match.group(1)}"
    role = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_").lower()
    return role or fallback


def _extra_bvh_roles(extra_bvh_paths: tuple[Path, ...]) -> tuple[tuple[str, Path], ...]:
    used = {"position", "order"}
    role_paths: list[tuple[str, Path]] = []
    for index, path in enumerate(extra_bvh_paths, start=1):
        role = _safe_bvh_role(path, f"extra{index}")
        if role in used:
            role = f"extra{index}"
        base_role = role
        suffix = 2
        while role in used:
            role = f"{base_role}_{suffix}"
            suffix += 1
        used.add(role)
        role_paths.append((role, path))
    return tuple(role_paths)


def derive_session_id(
    cam_session: Path | None,
    position_bvh_path: Path | None,
    order_bvh_path: Path | None,
    provided_session_id: str | None = None,
    extra_bvh_paths: tuple[Path, ...] = (),
) -> str:
    if provided_session_id:
        return provided_session_id

    candidates = []
    for path in (cam_session, position_bvh_path, order_bvh_path, *extra_bvh_paths):
        if path is None:
            continue
        candidates.extend([path.name, path.stem, path.parent.name])

    for candidate in candidates:
        match = re.search(r"S\d+_\d+", candidate)
        if match:
            return match.group(0)

    merged = "_".join(candidate for candidate in candidates if candidate)
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", merged).strip("_")
    return sanitized or "manual_session"


def resolve_source_mode(
    requested_source_mode: str,
    frame_dirs: dict[str, Path],
    videos: dict[str, Path],
) -> str:
    if requested_source_mode == "frames":
        if not frame_dirs:
            raise FileNotFoundError("frames 模式下至少需要一个逐帧目录（1-4 或 cam1-cam4）。")
        return "frames"
    if requested_source_mode == "mp4":
        if not videos:
            raise FileNotFoundError("mp4 模式下至少需要一个以 1_ 到 4_ 开头的视频文件。")
        return "mp4"
    if frame_dirs:
        return "frames"
    if videos:
        return "mp4"
    return "none"


def build_runtime_options(source_mode: str, lite_mode: bool, axis_preset: str) -> RuntimeOptions:
    preview_scale = LITE_PREVIEW_SCALE if lite_mode else DEFAULT_PREVIEW_SCALE
    return RuntimeOptions(
        source_mode=source_mode,
        lite_mode=lite_mode,
        preview_scale=preview_scale,
        axis_preset=axis_preset,
        # 拖动滑条时默认推迟重绘重负载组件，提升交互流畅度。
        defer_heavy_refresh=True,
    )


def _file_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def build_cache_manifest(
    *,
    session_id: str,
    source_mode: str,
    visual_inputs: dict[str, tuple[Path, ...]],
    position_bvh_path: Path | None,
    order_bvh_path: Path | None,
    extra_bvh_paths: tuple[Path, ...] = (),
) -> dict:
    return {
        "version": CACHE_VERSION,
        "session_id": session_id,
        "source_mode": source_mode,
        "frame_sequence_fps": FRAME_SEQUENCE_FPS,
        "smooth_ms": SMOOTH_MS,
        "visual_inputs": {
            label: [_file_record(path) for path in paths]
            for label, paths in sorted(visual_inputs.items())
        },
        "position_bvh": _file_record(position_bvh_path) if position_bvh_path is not None else None,
        "order_bvh": _file_record(order_bvh_path) if order_bvh_path is not None else None,
        "extra_bvhs": [_file_record(path) for path in extra_bvh_paths],
    }


def _load_cached_precompute(cache_dir: Path, manifest: dict) -> dict[str, np.ndarray] | None:
    manifest_path = cache_dir / "manifest.json"
    precompute_path = cache_dir / "precompute.npz"
    if not manifest_path.exists() or not precompute_path.exists():
        return None

    cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cached_manifest != manifest:
        return None

    with np.load(precompute_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_cached_precompute(cache_dir: Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(cache_dir / "precompute.npz", **arrays)


def _build_frame_sources(
    frame_dirs: dict[str, Path],
    cached: dict[str, np.ndarray] | None,
) -> tuple[list[ImageSequenceSource], float]:
    cameras: list[ImageSequenceSource] = []
    for label in sorted(frame_dirs):
        frame_paths = list_frame_paths(frame_dirs[label])
        if not frame_paths:
            raise FileNotFoundError(f"目录里没有找到可读图片：{frame_dirs[label]}")

        if cached is not None and f"{label}_energy" in cached:
            energy = cached[f"{label}_energy"]
            fps = float(cached[f"{label}_fps"][0])
            frame_count = int(cached[f"{label}_frame_count"][0])
            frame_times = cached[f"{label}_frame_times"] if f"{label}_frame_times" in cached else None
        else:
            frame_times, fps = infer_frame_times_from_paths(frame_paths, FRAME_SEQUENCE_FPS)
            raw_energy, fps, frame_count = image_sequence_energy(frame_paths, fps)
            energy = smooth_energy(raw_energy, fps)

        cameras.append(
            ImageSequenceSource(
                label=label,
                fps=fps,
                frame_count=frame_count,
                energy=energy,
                frame_times=frame_times,
                source_dir=frame_dirs[label],
                frame_paths=frame_paths,
            )
        )
    reference_fps = float(np.median([camera.fps for camera in cameras])) if cameras else FRAME_SEQUENCE_FPS
    return cameras, reference_fps


def _build_video_sources(
    video_paths: dict[str, Path],
    cached: dict[str, np.ndarray] | None,
) -> tuple[list[VideoSource], float]:
    cameras: list[VideoSource] = []
    ref_fps = 0.0
    for label in sorted(video_paths):
        if cached is not None and f"{label}_energy" in cached:
            energy = cached[f"{label}_energy"]
            fps = float(cached[f"{label}_fps"][0])
            frame_count = int(cached[f"{label}_frame_count"][0])
        else:
            raw_energy, fps, frame_count = video_energy(video_paths[label])
            energy = smooth_energy(raw_energy, fps)
        cameras.append(
            VideoSource(
                label=label,
                fps=fps,
                frame_count=frame_count,
                energy=energy,
                path=video_paths[label],
            )
        )
    if cameras:
        ref_fps = float(np.median([camera.fps for camera in cameras]))
    return cameras, ref_fps


def _build_bvh_source(
    role: str,
    motion,
    reference_visual_fps: float,
    cached: dict[str, np.ndarray] | None,
) -> BVHSourceData | None:
    if motion is None:
        return None

    raw_key = f"{role}_bvh_energy_raw"
    visual_key = f"{role}_bvh_energy_visual"
    if cached is not None and raw_key in cached and visual_key in cached:
        energy_raw = cached[raw_key]
        energy_visual = cached[visual_key]
    else:
        energy_raw, _ = compute_bvh_energy(motion)
        energy_visual = resample_signal(energy_raw, motion.raw_fps, reference_visual_fps)
    return BVHSourceData(
        role=role,
        motion=motion,
        energy_raw=energy_raw.astype(np.float64, copy=False),
        energy_visual=energy_visual.astype(np.float64, copy=False),
    )


def _bvh_fps_fallback(*motions) -> float:
    for motion in motions:
        if motion is not None:
            return float(motion.raw_fps)
    return FRAME_SEQUENCE_FPS


def load_session_from_paths(
    *,
    cam_session: Path | None = None,
    bvh_path: Path | None = None,
    position_bvh_path: Path | None = None,
    order_bvh_path: Path | None = None,
    extra_bvh_paths: tuple[Path, ...] = (),
    session_id: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    source_mode: str = DEFAULT_SOURCE_MODE,
    lite_mode: bool = False,
    axis_preset: str = DEFAULT_AXIS_PRESET,
) -> SessionData:
    extra_bvh_paths = tuple(extra_bvh_paths)
    if position_bvh_path is None and bvh_path is not None:
        position_bvh_path = bvh_path

    primary_paths = {path for path in (position_bvh_path, order_bvh_path) if path is not None}
    extra_bvh_paths = tuple(path for path in extra_bvh_paths if path not in primary_paths)

    if cam_session is None and position_bvh_path is None and order_bvh_path is None and not extra_bvh_paths:
        raise ValueError("至少需要提供相机会话目录或一个 BVH 文件。")

    if cam_session is not None and (not cam_session.exists() or not cam_session.is_dir()):
        raise FileNotFoundError(f"相机会话目录不存在：{cam_session}")
    validation_items = [("位置 BVH", position_bvh_path), ("顺序 BVH", order_bvh_path)]
    validation_items.extend((f"额外 BVH {index}", path) for index, path in enumerate(extra_bvh_paths, start=1))
    for label, path in validation_items:
        if path is not None and (not path.exists() or not path.is_file()):
            raise FileNotFoundError(f"{label} 不存在：{path}")

    frame_dirs = get_cam_frame_dirs(cam_session) if cam_session is not None else {}
    video_paths = get_cam_videos(cam_session) if cam_session is not None else {}
    effective_source_mode = resolve_source_mode(source_mode, frame_dirs, video_paths)
    runtime_options = build_runtime_options(effective_source_mode, lite_mode, axis_preset)

    position_motion = load_bvh_motion(position_bvh_path) if position_bvh_path is not None else None
    order_motion = load_bvh_motion(order_bvh_path) if order_bvh_path is not None else None
    extra_motion_pairs = tuple(
        (role, load_bvh_motion(path))
        for role, path in _extra_bvh_roles(extra_bvh_paths)
    )

    resolved_session_id = derive_session_id(
        cam_session,
        position_bvh_path,
        order_bvh_path,
        extra_bvh_paths=extra_bvh_paths,
        provided_session_id=session_id,
    )
    first_bvh_path = next(
        (path for path in (position_bvh_path, order_bvh_path, *extra_bvh_paths) if path is not None),
        None,
    )
    mocap_subject = first_bvh_path.parent if first_bvh_path is not None else None

    print(f"\n{'=' * 60}")
    print(f"  Session: {resolved_session_id}")
    print(f"{'=' * 60}")
    print(f"\n  相机会话: {cam_session if cam_session is not None else '未提供'}")
    print(f"  视觉源模式: {effective_source_mode}")
    print(f"  位置 BVH: {position_bvh_path.name if position_bvh_path is not None else '未提供'}")
    print(f"  顺序 BVH: {order_bvh_path.name if order_bvh_path is not None else '未提供'}")
    if extra_bvh_paths:
        print(f"  额外 BVH: {', '.join(path.name for path in extra_bvh_paths)}")
    else:
        print("  额外 BVH: 未提供")
    print()

    if effective_source_mode == "frames":
        visual_inputs = {label: list_frame_paths(frame_dir) for label, frame_dir in sorted(frame_dirs.items())}
    elif effective_source_mode == "mp4":
        visual_inputs = {label: (video_paths[label],) for label in sorted(video_paths)}
    else:
        visual_inputs = {}

    cache_dir = cache_root / resolved_session_id
    output_dir = output_root / resolved_session_id
    manifest = build_cache_manifest(
        session_id=resolved_session_id,
        source_mode=effective_source_mode,
        visual_inputs=visual_inputs,
        position_bvh_path=position_bvh_path,
        order_bvh_path=order_bvh_path,
        extra_bvh_paths=extra_bvh_paths,
    )
    cached = _load_cached_precompute(cache_dir, manifest)
    if cached is not None:
        print("  [cache] 已加载预计算结果")

    if effective_source_mode == "frames":
        cameras, reference_visual_fps = _build_frame_sources(frame_dirs, cached)
    elif effective_source_mode == "mp4":
        cameras, reference_visual_fps = _build_video_sources(video_paths, cached)
    else:
        cameras = []
        if cached is not None and "reference_visual_fps" in cached:
            reference_visual_fps = float(cached["reference_visual_fps"][0])
        else:
            reference_visual_fps = _bvh_fps_fallback(order_motion, position_motion, *(motion for _, motion in extra_motion_pairs))

    if reference_visual_fps <= 0:
        reference_visual_fps = _bvh_fps_fallback(order_motion, position_motion, *(motion for _, motion in extra_motion_pairs))

    position_bvh = _build_bvh_source("position", position_motion, reference_visual_fps, cached)
    order_bvh = _build_bvh_source("order", order_motion, reference_visual_fps, cached)
    extra_bvhs = [
        bvh_source
        for role, motion in extra_motion_pairs
        if (bvh_source := _build_bvh_source(role, motion, reference_visual_fps, cached)) is not None
    ]

    if cached is None or "combined_camera_energy" not in cached:
        if cameras:
            combined_camera_energy, _ = build_combined_camera_energy(cameras, reference_visual_fps)
        else:
            combined_camera_energy = np.zeros(0, dtype=np.float64)

        cache_arrays: dict[str, np.ndarray] = {
            "reference_visual_fps": np.array([reference_visual_fps], dtype=np.float64),
            "combined_camera_energy": combined_camera_energy.astype(np.float64, copy=False),
        }
        for camera in cameras:
            cache_arrays[f"{camera.label}_energy"] = camera.energy.astype(np.float64, copy=False)
            cache_arrays[f"{camera.label}_fps"] = np.array([camera.fps], dtype=np.float64)
            cache_arrays[f"{camera.label}_frame_count"] = np.array([camera.frame_count], dtype=np.int64)
            if camera.frame_times is not None:
                cache_arrays[f"{camera.label}_frame_times"] = camera.frame_times.astype(np.float64, copy=False)
        for bvh_source in (position_bvh, order_bvh, *extra_bvhs):
            if bvh_source is None:
                continue
            cache_arrays[f"{bvh_source.role}_bvh_energy_raw"] = bvh_source.energy_raw.astype(np.float64, copy=False)
            cache_arrays[f"{bvh_source.role}_bvh_energy_visual"] = bvh_source.energy_visual.astype(np.float64, copy=False)
        _save_cached_precompute(cache_dir, manifest, cache_arrays)
    else:
        combined_camera_energy = cached["combined_camera_energy"]

    return SessionData(
        session_id=resolved_session_id,
        cam_session=cam_session,
        mocap_subject=mocap_subject,
        cameras=cameras,
        position_bvh=position_bvh,
        order_bvh=order_bvh,
        combined_camera_energy=combined_camera_energy.astype(np.float64, copy=False),
        reference_visual_fps=float(reference_visual_fps),
        output_dir=output_dir,
        cache_dir=cache_dir,
        source_mode=effective_source_mode,
        runtime_options=runtime_options,
        extra_bvhs=extra_bvhs,
    )


def load_session_data(
    session_id: str,
    cam_root: Path = DEFAULT_CAM_ROOT,
    mocap_root: Path = DEFAULT_MOCAP_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    source_mode: str = DEFAULT_SOURCE_MODE,
    lite_mode: bool = False,
    axis_preset: str = DEFAULT_AXIS_PRESET,
) -> SessionData:
    cam_session = find_cam_session(session_id, cam_root)
    mocap_subject = find_mocap_subject(session_id, mocap_root)
    mocap_files = get_mocap_files(mocap_subject)
    position_bvh_path, order_bvh_path, extra_bvh_paths = choose_bvh_paths(mocap_files)
    return load_session_from_paths(
        cam_session=cam_session,
        position_bvh_path=position_bvh_path,
        order_bvh_path=order_bvh_path,
        extra_bvh_paths=extra_bvh_paths,
        session_id=session_id,
        output_root=output_root,
        cache_root=cache_root,
        source_mode=source_mode,
        lite_mode=lite_mode,
        axis_preset=axis_preset,
    )


def close_session_data(session: SessionData | None) -> None:
    if session is None:
        return
    for camera in session.cameras:
        camera.close()


def load_trial(
    trial: TrialInfo,
    cam_root: Path = DEFAULT_CAM_ROOT,
    mocap_root: Path = DEFAULT_MOCAP_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    source_mode: str = DEFAULT_SOURCE_MODE,
    lite_mode: bool = False,
    axis_preset: str = DEFAULT_AXIS_PRESET,
) -> SessionData:
    try:
        cam_session = find_cam_session(trial.cam_session_suffix, cam_root)
    except FileNotFoundError:
        cam_session = None

    mocap_folder = find_mocap_subject(trial.cam_session_suffix, mocap_root)

    mocap_files = get_mocap_files(mocap_folder)
    position_bvh_path, order_bvh_path, extra_bvh_paths = choose_bvh_paths(mocap_files)

    return load_session_from_paths(
        cam_session=cam_session,
        position_bvh_path=position_bvh_path,
        order_bvh_path=order_bvh_path,
        extra_bvh_paths=extra_bvh_paths,
        session_id=trial.cam_session_suffix,
        output_root=output_root,
        cache_root=cache_root,
        source_mode=source_mode,
        lite_mode=lite_mode,
        axis_preset=axis_preset,
    )
