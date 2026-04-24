from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models import BVHMotion, BVHSourceData, SessionData
from utils.alignment import aligned_start_frames, build_aligned_curve_matrix


def write_bvh(original_motion: BVHMotion, aligned_frames: np.ndarray, output_path: Path) -> None:
    motion_idx = original_motion.text.find("MOTION")
    if motion_idx == -1:
        raise ValueError("原始 BVH 文件中没有找到 MOTION 段。")
    header = original_motion.text[:motion_idx].rstrip()

    lines = [header, "MOTION", f"Frames: {len(aligned_frames)}", f"Frame Time: {original_motion.frame_time_str}"]
    for frame in aligned_frames:
        lines.append(" ".join(f"{value:.6f}" for value in frame))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_one_bvh(
    session: SessionData,
    bvh_source: BVHSourceData,
    delta_t: float,
) -> tuple[Path, int]:
    bvh_start, _ = aligned_start_frames(session, delta_t, bvh_source=bvh_source)
    if bvh_start >= len(bvh_source.motion.raw_frames):
        raise RuntimeError(
            f"{bvh_source.role} BVH 的起始帧 {bvh_start} 超过总帧数 {len(bvh_source.motion.raw_frames)}。"
        )

    aligned_bvh_frames = bvh_source.motion.raw_frames[bvh_start:]
    output_path = session.output_dir / f"{session.session_id}_{bvh_source.role}_aligned.bvh"
    write_bvh(bvh_source.motion, aligned_bvh_frames, output_path)
    return output_path, bvh_start


def export_alignment_bundle(
    session: SessionData,
    delta_t: float,
    figure=None,
) -> dict[str, Path]:
    if not session.has_bvh:
        raise RuntimeError("当前没有加载任何 BVH，无法导出对齐结果。")

    session.output_dir.mkdir(parents=True, exist_ok=True)
    _, camera_starts = aligned_start_frames(session, delta_t)

    outputs: dict[str, Path] = {}
    bvh_metadata = {}
    for bvh_source in session.bvh_sources:
        role = bvh_source.role
        output_path, start_frame = _export_one_bvh(session, bvh_source, delta_t)
        outputs[f"{role}_bvh"] = output_path
        bvh_metadata[role] = {
            "file": output_path.name,
            "start_frame": start_frame,
            "raw_fps": bvh_source.motion.raw_fps,
            "source_file": bvh_source.motion.path.name,
        }

    metadata_path = session.output_dir / f"{session.session_id}_alignment.json"
    csv_path = session.output_dir / f"{session.session_id}_aligned_curves.csv"
    image_path = session.output_dir / f"{session.session_id}_calibration.png"

    matrix, headers = build_aligned_curve_matrix(session, delta_t)
    if len(matrix):
        np.savetxt(csv_path, matrix, delimiter=",", header=",".join(headers), comments="")
    else:
        csv_path.write_text("time_s\n", encoding="utf-8")

    metadata = {
        "session_id": session.session_id,
        "source_mode": session.source_mode,
        "delta_t": delta_t,
        "camera_start_frames": camera_starts,
        "reference_visual_fps": session.reference_visual_fps,
        "axis_preset": session.runtime_options.axis_preset,
        "bvh": bvh_metadata,
        "visual_cameras": [camera.label for camera in session.cameras],
        "visual_camera_fps": {camera.label: camera.fps for camera in session.cameras},
        "visual_camera_frame_count": {camera.label: camera.frame_count for camera in session.cameras},
        "display_bvh_role": session.display_bvh.role if session.display_bvh is not None else None,
        "alignment_bvh_role": session.alignment_bvh.role if session.alignment_bvh is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if figure is not None and hasattr(figure, "savefig"):
        figure.savefig(image_path, dpi=150, bbox_inches="tight")
        outputs["image"] = image_path

    outputs["metadata"] = metadata_path
    outputs["curves"] = csv_path
    return outputs
