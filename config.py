from __future__ import annotations

from pathlib import Path


APP_TITLE = "动捕视频对齐工具"

DEFAULT_CAM_ROOT = Path("E:/Data/1_Data/20260422/S2")
DEFAULT_MOCAP_ROOT = Path("E:/Data/1_Data/20260422/VICON")
DEFAULT_OUTPUT_ROOT = Path("sync/output")
DEFAULT_CACHE_ROOT = Path("sync/cache")

DEFAULT_SOURCE_MODE = "auto"
DEFAULT_AXIS_PRESET = "zup"

FRAME_SEQUENCE_FPS = 40.0
TARGET_BVH_FPS = 120.0
SMOOTH_MS = 50.0
ARM_LENGTH_CM = 30.0

# 删除动捕开头 T-pose 过渡段（用于对齐前预处理）
# 实际删除帧数 = round(BVH_TRIM_LEADING_SECONDS * bvh_fps) + BVH_TRIM_LEADING_FRAMES
BVH_TRIM_LEADING_SECONDS = 0.40
BVH_TRIM_LEADING_FRAMES = 0

FRAME_STEP_SMALL = 1
FRAME_STEP_MEDIUM = 5
FRAME_STEP_LARGE = 10

DEFAULT_PREVIEW_SCALE = 1.0
LITE_PREVIEW_SCALE = 0.5

FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg")
SUPPORTED_SOURCE_MODES = ("auto", "frames", "mp4")
SUPPORTED_AXIS_PRESETS = ("raw", "zup")

CACHE_VERSION = 4


def configure_matplotlib_backend(interactive: bool, prefer_qt: bool = False):
    import matplotlib

    if interactive:
        backends = ("Qt5Agg", "QtAgg", "TkAgg") if prefer_qt else ("TkAgg", "Qt5Agg", "QtAgg")
        for backend in backends:
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg")

    matplotlib.rcParams["font.family"] = "Microsoft YaHei"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib
