# MocapVideoAligner

> Lightweight desktop tool for aligning multi-camera visual recordings and plantar-pressure data with BVH motion-capture.
> 用于对齐多相机视觉数据、足底触觉压力数据与 BVH 动捕数据的轻量级桌面工具。

![Python](https://img.shields.io/badge/Python-3.11%2B-355166)
![GUI](https://img.shields.io/badge/GUI-PyQt5-AF5A3B)
![License](https://img.shields.io/badge/License-MIT-667C58)

## Language / 语言

- [中文文档](README.zh-CN.md)
- [English Documentation](README.en.md)
- [触觉对齐机制说明](docs/tactile_alignment.md)

## What It Does / 功能概览

Two alignment tabs in one GUI window / 同一 GUI 内提供两个对齐页签：

1. **Visual-Mocap alignment** / **视觉-动捕对齐** → `delta_t`
2. **Mocap-Tactile alignment** / **动捕-触觉对齐** → `delta_t2`

The original visual-mocap workflow stays independent. Tactile alignment is added as a separate tab and module set.
原有视觉-动捕流程保持独立；触觉对齐以新增页签和模块方式接入。

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The application opens a single GUI window. Select the camera root folder and mocap root folder from the toolbar, then click reload. Use the second tab for mocap-tactile alignment after visual-mocap results are available.

软件会打开单一主窗口。请在右上角选择相机根目录和动捕根目录，然后点击重新加载。完成视觉-动捕对齐后，可切换到第二页签进行动捕-触觉对齐。

## Repository

This repository contains only the lightweight desktop alignment tool. Runtime cache, exported results, videos, and local IDE files are intentionally excluded.

本仓库只包含轻量级桌面对齐工具。运行缓存、导出结果、视频和本地 IDE 文件不会进入仓库。
