"""动捕视频对齐工具入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_AXIS_PRESET, DEFAULT_SOURCE_MODE, SUPPORTED_AXIS_PRESETS, SUPPORTED_SOURCE_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="动捕视频对齐工具")
    parser.add_argument(
        "--source",
        choices=SUPPORTED_SOURCE_MODES,
        default=DEFAULT_SOURCE_MODE,
        help="视觉源模式，默认使用 auto。",
    )
    parser.add_argument("--lite", action="store_true", help="以轻量模式启动。")
    parser.add_argument("--auto-load", action="store_true", help="启动后自动加载当前目录下的第一个试次。")
    parser.add_argument(
        "--axis-preset",
        choices=SUPPORTED_AXIS_PRESETS,
        default=DEFAULT_AXIS_PRESET,
        help="骨架显示坐标系。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ui.main_window import LaunchOptions, main as qt_main
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or "未知模块"
        raise SystemExit(
            f"当前环境缺少依赖：{missing_name}。请先安装 PyQt5、opencv-python、matplotlib 后再启动图形界面。"
        ) from exc
    except ImportError as exc:
        raise SystemExit("图形界面依赖加载失败。请确认已安装 PyQt5，并且 matplotlib 的 Qt 后端可用。") from exc

    qt_main(
        LaunchOptions(
            source_mode=args.source,
            axis_preset=args.axis_preset,
            lite_mode=args.lite,
            auto_load=args.auto_load,
        )
    )


if __name__ == "__main__":
    main()
