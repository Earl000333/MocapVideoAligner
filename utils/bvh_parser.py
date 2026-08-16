from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from config import BVH_TRIM_LEADING_FRAMES, BVH_TRIM_LEADING_SECONDS
from models import BVHJoint, BVHMotion


def _read_bvh_text(bvh_path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return bvh_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return bvh_path.read_text()


def parse_bvh_hierarchy(text: str) -> tuple[list[BVHJoint], list[tuple[int, int]], list[tuple[str, str]]]:
    motion_pos = text.find("MOTION")
    if motion_pos < 0:
        raise ValueError("BVH 文件缺少 MOTION 段。")
    hierarchy = text[:motion_pos]

    joints: list[BVHJoint] = []
    stack: list[int] = []
    pending_name: str | None = None
    channel_cursor = 0
    end_site_count = 0

    for line in hierarchy.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            pending_name = s.split(None, 1)[1]
        elif s.startswith("End Site"):
            parent_name = joints[stack[-1]].name if stack else "End"
            pending_name = f"{parent_name}_EndSite{end_site_count}"
            end_site_count += 1
        elif s == "{":
            if pending_name is None:
                continue
            parent = stack[-1] if stack else -1
            joints.append(
                BVHJoint(
                    name=pending_name,
                    parent=parent,
                    offset=np.zeros(3, dtype=np.float64),
                    channels=[],
                    channel_indices=[],
                )
            )
            stack.append(len(joints) - 1)
            pending_name = None
        elif s == "}":
            if stack:
                stack.pop()
        elif s.startswith("OFFSET"):
            if stack:
                joints[stack[-1]].offset = np.array(s.split()[1:4], dtype=np.float64)
        elif s.startswith("CHANNELS"):
            if not stack:
                continue
            parts = s.split()
            count = int(parts[1])
            channels = parts[2 : 2 + count]
            joints[stack[-1]].channels = channels
            joints[stack[-1]].channel_indices = list(range(channel_cursor, channel_cursor + count))
            channel_cursor += count

    edges = [(joint.parent, index) for index, joint in enumerate(joints) if joint.parent >= 0]
    channel_info = []
    for joint in joints:
        for channel_name in joint.channels:
            channel_info.append((joint.name, channel_name))
    return joints, edges, channel_info


def load_bvh_motion(bvh_path: Path) -> BVHMotion:
    text = _read_bvh_text(bvh_path)

    ft_match = re.search(r"Frame Time:\s*([\d.eE+\-]+)", text)
    if not ft_match:
        raise ValueError(f"BVH 文件里没有找到 Frame Time：{bvh_path}")
    frame_time_str = ft_match.group(1)
    raw_fps = 1.0 / float(frame_time_str)

    motion_start = text.find("Frame Time:")
    if motion_start < 0:
        raise ValueError(f"BVH 文件缺少 Frame Time 行：{bvh_path}")

    lines = text[motion_start:].strip().split("\n")[1:]
    lines = [line.strip() for line in lines if line.strip()]
    raw_frames = np.array([[float(value) for value in line.split()] for line in lines], dtype=np.float64)

    trim_frames = int(round(BVH_TRIM_LEADING_SECONDS * raw_fps)) + int(BVH_TRIM_LEADING_FRAMES)
    trim_frames = max(0, min(trim_frames, max(len(raw_frames) - 2, 0)))
    if trim_frames > 0:
        raw_frames = raw_frames[trim_frames:]

    joints, edges, channel_info = parse_bvh_hierarchy(text)

    print(
        f"  [BVH]   {bvh_path.name}  fps={raw_fps:.2f}  frames={len(raw_frames)}  "
        f"ch={raw_frames.shape[1]}  duration={len(raw_frames) / raw_fps:.2f}s"
    )
    if trim_frames > 0:
        print(f"  [BVH trim] removed leading frames: {trim_frames}")
    return BVHMotion(
        path=bvh_path,
        text=text,
        frame_time_str=frame_time_str,
        raw_fps=raw_fps,
        raw_frames=raw_frames,
        joints=joints,
        edges=edges,
        channel_info=channel_info,
    )


def load_bvh_motion_preserve_frames(bvh_path: Path) -> BVHMotion:
    text = _read_bvh_text(bvh_path)

    ft_match = re.search(r"Frame Time:\s*([\d.eE+\-]+)", text)
    if not ft_match:
        raise ValueError(f"BVH 文件里没有找到 Frame Time：{bvh_path}")
    frame_time_str = ft_match.group(1)
    raw_fps = 1.0 / float(frame_time_str)

    motion_start = text.find("Frame Time:")
    if motion_start < 0:
        raise ValueError(f"BVH 文件缺少 Frame Time 行：{bvh_path}")

    lines = text[motion_start:].strip().split("\n")[1:]
    lines = [line.strip() for line in lines if line.strip()]
    raw_frames = np.array([[float(value) for value in line.split()] for line in lines], dtype=np.float64)
    joints, edges, channel_info = parse_bvh_hierarchy(text)

    print(
        f"  [BVH aligned] {bvh_path.name}  fps={raw_fps:.2f}  frames={len(raw_frames)}  "
        f"ch={raw_frames.shape[1]}  duration={len(raw_frames) / raw_fps:.2f}s"
    )
    return BVHMotion(
        path=bvh_path,
        text=text,
        frame_time_str=frame_time_str,
        raw_fps=raw_fps,
        raw_frames=raw_frames,
        joints=joints,
        edges=edges,
        channel_info=channel_info,
    )
