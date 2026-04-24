# MocapVideoAligner

> Lightweight desktop tool for aligning multi-camera visual recordings with BVH motion-capture data.
> 用于对齐多相机视觉数据与 BVH 动捕数据的轻量级桌面工具。

![Python](https://img.shields.io/badge/Python-3.11%2B-355166)
![GUI](https://img.shields.io/badge/GUI-PyQt5-AF5A3B)
![License](https://img.shields.io/badge/License-MIT-667C58)

## Language / 语言

- [中文文档](README.zh-CN.md)
- [English Documentation](README.en.md)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The application opens a single GUI window. Select the camera root folder and mocap root folder from the toolbar, then click reload.

软件会打开单一主窗口。请在右上角选择相机根目录和动捕根目录，然后点击重新加载。

## Repository

This repository contains only the lightweight desktop alignment tool. Runtime cache, exported results, videos, and local IDE files are intentionally excluded.

本仓库只包含轻量级桌面对齐工具。运行缓存、导出结果、视频和本地 IDE 文件不会进入仓库。
