from __future__ import annotations

import numpy as np

from models import BVHMotion


def axis_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def compute_joint_positions(motion: BVHMotion, frame_index: int) -> np.ndarray:
    frame_index = int(np.clip(frame_index, 0, max(len(motion.raw_frames) - 1, 0)))
    frame = motion.raw_frames[frame_index]

    positions = np.zeros((len(motion.joints), 3), dtype=np.float64)
    global_rotations: list[np.ndarray] = []

    for joint_index, joint in enumerate(motion.joints):
        translation = joint.offset.astype(np.float64).copy()
        local_rotation = np.eye(3, dtype=np.float64)

        if joint.channel_indices:
            for channel_name, value in zip(joint.channels, frame[joint.channel_indices]):
                lower = channel_name.lower()
                axis = lower[0]
                if lower.endswith("position"):
                    translation[axis_to_index(axis)] += value
                elif lower.endswith("rotation"):
                    local_rotation = local_rotation @ rotation_matrix(axis, np.deg2rad(value))

        if joint.parent < 0:
            global_rot = local_rotation
            global_pos = translation
        else:
            parent_rot = global_rotations[joint.parent]
            parent_pos = positions[joint.parent]
            global_rot = parent_rot @ local_rotation
            global_pos = parent_pos + parent_rot @ translation

        positions[joint_index] = global_pos
        global_rotations.append(global_rot)

    return positions


def transform_display_positions(positions: np.ndarray, axis_preset: str = "zup") -> np.ndarray:
    if axis_preset == "raw":
        return positions
    if axis_preset == "zup":
        return np.column_stack((positions[:, 0], -positions[:, 2], positions[:, 1]))
    raise ValueError(f"Unsupported axis preset: {axis_preset}")
