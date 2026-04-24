# MocapVideoAligner / 动捕视频对齐工具

> A lightweight PyQt desktop tool for aligning multi-camera visual recordings with BVH motion-capture data.  
> 一个用于对齐四相机视觉数据与 BVH 动捕数据的轻量级 PyQt 桌面工具。

![Python](https://img.shields.io/badge/Python-3.11%2B-355166)
![GUI](https://img.shields.io/badge/GUI-PyQt5-AF5A3B)
![License](https://img.shields.io/badge/License-MIT-667C58)

---

## 中文说明

### 1. 项目简介

`MocapVideoAligner` 用于把视觉系统采集的多相机画面和 VICON / BVH 动捕数据放到同一个时间轴中进行检查、自动对齐、手动微调和导出。

当前版本重点解决以下实验数据问题：

- 四相机视觉数据可能是逐帧图片，也可能是 MP4。
- 视觉系统设定为 40fps，但实际采集帧率可能偏离，例如 37.4-37.8fps。
- 动捕数据通常为 120fps，但同一试次可能存在 1 个、2 个、3 个或 4 个 BVH 文件。
- BVH 原始坐标系可能导致骨架预览中人物“躺下”，需要显示层坐标修正。
- 低配或无独显电脑也需要能够运行。

软件采用单窗口 GUI，默认启动后先进入界面，由用户在右上角选择相机目录和动捕目录，再加载数据。

### 2. 功能特性

- 单窗口 PyQt GUI，不依赖 Web 服务。
- 支持相机目录和动捕目录在界面中手动选择。
- 支持逐帧图片目录：`1/2/3/4` 或 `cam1/cam2/cam3/cam4`。
- 支持四路 MP4：`1_*.mp4`、`2_*.mp4`、`3_*.mp4`、`4_*.mp4`。
- `auto` 模式优先使用逐帧图片，缺失图片时回退 MP4。
- 逐帧图片支持 `png`、`jpg`、`jpeg`。
- 图片文件名包含时间戳时，会自动推断实际帧率和真实时间轴。
- MP4 模式使用 OpenCV 读取到的实际 FPS，不假设固定 40fps。
- 多相机合成时按各相机实际 FPS 重采样到统一参考时间轴。
- 支持 BVH 原始 120fps，不修改原始运动数据。
- 支持多个 BVH 文件，自动选择关节运动最明显的 BVH 用于骨架预览和自动对齐。
- 默认 `zup` 显示坐标修正：`display_xyz = [bvh_x, -bvh_z, bvh_y]`。
- 支持骨架悬浮叠加到相机区域，可拖拽、滚轮缩放、双击重置。
- 支持自动对齐、手动按帧微调、播放 / 暂停、起点 / 终点记录。
- 支持导出裁剪记录 CSV。
- 支持导出对齐后的 BVH、对齐曲线 CSV、元数据 JSON 和曲线截图 PNG。
- 支持轻量模式，降低预览尺寸和拖动时的重绘开销。

### 3. 界面概览

主界面分为四个区域：

- 顶部工具栏：试次导航、目录选择、重新加载、自动对齐、导出。
- 左侧相机预览：四路相机画面，同步显示当前视觉帧。
- 中间骨架预览：显示当前 BVH 骨架，可开启悬浮叠加。
- 底部对齐曲线：显示相机综合能量、单相机能量和动捕能量曲线。
- 右侧状态栏：显示当前试次、运行设置、实际 FPS、偏移、帧位置和操作日志。

### 4. 环境要求

推荐环境：

- Windows 10 / Windows 11
- Python 3.11+
- CPU 即可运行，不要求独立显卡
- 建议内存 8GB+

核心依赖：

- `numpy`
- `matplotlib`
- `opencv-python`
- `PyQt5`

### 5. 安装方式

使用 Conda：

```bash
conda env create -f environment.yml
conda activate mocap-align
python main.py
```

使用 pip：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

如果 PyQt5 安装失败，可以单独安装：

```bash
pip install PyQt5 opencv-python matplotlib numpy
```

### 6. 启动方式

默认启动，只打开 GUI，不自动加载数据：

```bash
python main.py
```

轻量模式：

```bash
python main.py --lite
```

强制使用 MP4：

```bash
python main.py --source mp4
```

强制使用逐帧图片：

```bash
python main.py --source frames
```

启动后自动加载当前目录配置下的第一条试次：

```bash
python main.py --auto-load
```

坐标显示模式：

```bash
python main.py --axis-preset zup
python main.py --axis-preset raw
```

### 7. 推荐数据目录结构

相机数据目录可以是逐帧图片：

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1/
      1_162852.002.jpg
      1_162852.035.jpg
    2/
    3/
    4/
```

也可以是 `cam1` 到 `cam4`：

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    cam1/
    cam2/
    cam3/
    cam4/
```

也支持 MP4：

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1_20260422_163247.mp4
    2_20260422_163247.mp4
    3_20260422_163247.mp4
    4_20260422_163247.mp4
```

动捕目录可以是直接试次目录：

```text
mocap_root/
  S4011/
    S4011_Skeleton0.bvh
    S4011_Skeleton1.bvh
    S4011_Skeleton2.bvh
    S4011_Skeleton3.bvh
```

也支持按被试分组：

```text
mocap_root/
  S4/
    S4011/
      S4011_Skeleton0.bvh
      S4011_Skeleton1.bvh
      S4011_Skeleton2.bvh
      S4011_Skeleton3.bvh
```

试次命名规则：

- 动捕目录名：`S<对象><动作两位><重复次数>`，例如 `S4011`。
- 相机会话名：建议以 `_S401_1` 结尾，软件会用该后缀匹配动捕试次。

### 8. 标准使用流程

1. 运行 `python main.py`。
2. 在右上角点击“相机目录”，选择相机数据根目录。
3. 点击“动捕目录”，选择 VICON / BVH 数据根目录。
4. 使用“上一试次 / 下一试次”选择目标试次。
5. 点击“重新加载”。
6. 检查四路相机预览、骨架预览和对齐曲线。
7. 点击“自动对齐”获得初始偏移。
8. 用底部按钮、滑条或方向键进行人工微调。
9. 需要裁剪时，点击“记录起点”和“记录终点”。
10. 点击“导出裁剪记录”或“导出当前结果”。

### 9. 时间轴与实际 FPS 逻辑

视觉侧不再简单假设固定 40fps：

- MP4 模式读取 OpenCV 返回的实际 FPS。
- 逐帧图片模式如果文件名中有类似 `162852.002` 的时间戳，会推断真实帧时间和实际 FPS。
- 如果图片文件名无法解析时间戳，则回退 `config.py` 中的 `FRAME_SEQUENCE_FPS = 40.0`。
- 多相机参考 FPS 使用当前已加载相机实际 FPS 的中位数。
- 每路相机的取帧都按自己的真实时间轴计算。
- 对齐曲线会把各路相机能量重采样到统一参考时间轴。

这可以减少“设定 40fps，但实际只有 37.xfps”导致的长时间漂移。

### 10. BVH 处理逻辑

软件支持任意数量 BVH 文件：

- `Skeleton0` 优先标记为 `position`。
- `Skeleton1` 优先标记为 `order`。
- `Skeleton2`、`Skeleton3` 等作为额外 BVH 一起加载。
- 软件会计算每个 BVH 的运动评分。
- 骨架预览和自动对齐会自动选择关节运动最明显的 BVH。
- 导出时会导出全部已加载 BVH。

显示坐标修正只影响 GUI 预览，不会改写原始 BVH 数据。

### 11. 导出内容

默认输出目录：

```text
sync/output/<session_id>/
```

导出文件包括：

- `<session_id>_<role>_aligned.bvh`：按当前偏移裁切后的 BVH。
- `<session_id>_alignment.json`：对齐元数据。
- `<session_id>_aligned_curves.csv`：对齐后的曲线数据。
- `<session_id>_calibration.png`：当前对齐曲线截图。
- `sync/output/results_csv/*.csv`：起点 / 终点裁剪记录。

`alignment.json` 会记录：

- `delta_t`
- `reference_visual_fps`
- 每路相机实际 FPS
- 每路相机帧数
- BVH 起始帧
- 显示和对齐使用的 BVH 角色
- 坐标预设

### 12. 轻量模式与性能

无独显机器可以运行本工具。主要耗时来自：

- 图片或视频解码
- Matplotlib 曲线绘制
- 3D 骨架绘制

建议低配机器使用：

```bash
python main.py --lite
```

轻量模式会：

- 降低预览图缩放比例。
- 拖动滑条时减少重负载刷新。
- 优先复用预计算缓存。

缓存目录：

```text
sync/cache/
```

如果数据或代码逻辑更新后需要重新计算，可以删除对应 session 的缓存目录。

### 13. 快捷键

| 快捷键 | 功能 |
|---|---|
| `Space` | 播放 / 暂停 |
| `Left` / `Right` | 对齐偏移 -1 / +1 帧 |
| `Shift + Left/Right` | 对齐偏移 -10 / +10 帧 |
| `Ctrl + Left/Right` | 对齐偏移 -5 / +5 帧 |
| `Up` / `Down` | 视觉时间前进 / 后退 |
| `Home` | 自动对齐 |
| `1` | 记录起点 |
| `2` | 记录终点 |
| `Enter` | 导出当前结果 |
| `PageUp` / `PageDown` | 上一试次 / 下一试次 |

### 14. 常见问题

#### 运行 `main.py` 后没有自动加载数据

这是当前设计。默认只打开界面，避免默认配置路径错误时直接失败。请在 GUI 中选择目录后点击“重新加载”。

如果需要自动加载：

```bash
python main.py --auto-load
```

#### 图像读取失败

检查：

- 路径是否存在。
- 图片是否损坏。
- 目录名是否为 `1/2/3/4` 或 `cam1/cam2/cam3/cam4`。
- 后缀是否为 `png/jpg/jpeg`。

#### 骨架躺下或方向不对

先使用默认 `zup`。如果数据本身已经是正确坐标系，可以切换为 `raw`。

#### 有多个 BVH 文件不知道选哪个

不需要手动判断。软件会全部加载并根据运动评分自动选择用于预览和对齐的 BVH。

#### 对齐后仍然有轻微漂移

优先确认：

- 是否使用了实际帧率。
- 逐帧图片文件名是否包含可靠时间戳。
- MP4 是否为可变帧率视频。若是，建议使用逐帧图片和时间戳。

### 15. 开发与测试

运行测试：

```bash
python -m unittest discover -s tests
```

语法检查：

```bash
python -m py_compile main.py visualize_energy.py models.py ui/main_window.py ui/widgets.py utils/session.py utils/energy.py utils/alignment.py utils/exporter.py
```

项目主要文件：

```text
main.py                 兼容入口
visualize_energy.py     GUI 启动入口
config.py               默认路径、帧率、缓存版本等配置
models.py               数据模型
ui/                     PyQt 界面
utils/session.py        数据发现、加载和缓存
utils/energy.py         能量计算、实际 FPS 和重采样
utils/alignment.py      对齐和帧映射
utils/exporter.py       导出逻辑
tests/                  单元测试和轻量 fixture
```

---

## English Documentation

### 1. Overview

`MocapVideoAligner` is a desktop GUI tool for aligning multi-camera visual recordings with BVH motion-capture data. It is designed for lab workflows where visual recordings and mocap files are collected by separate systems and need to be synchronized after acquisition.

The current version focuses on practical experimental issues:

- Visual recordings may be image sequences or MP4 files.
- The intended visual frame rate may be 40fps, while the real recorded rate can be around 37.xfps.
- Mocap data is commonly 120fps.
- A single trial may contain one, two, three, four, or more BVH files.
- BVH coordinate conventions may make the skeleton appear lying down in a direct 3D preview.
- The tool should run on CPU-only machines without a dedicated GPU.

The application starts as a single main window. By default it does not auto-load data; users select the camera and mocap root folders in the GUI.

### 2. Features

- Single-window PyQt GUI.
- Folder selection directly from the toolbar.
- Image sequence input from `1/2/3/4` or `cam1/cam2/cam3/cam4`.
- MP4 input with `1_*.mp4`, `2_*.mp4`, `3_*.mp4`, `4_*.mp4`.
- `auto` mode prefers image sequences and falls back to MP4.
- Image sequence formats: `png`, `jpg`, `jpeg`.
- Timestamp-aware image sequence timing.
- Actual MP4 FPS from OpenCV.
- Multi-camera energy fusion on a unified reference timeline.
- Native BVH frame rate preserved.
- Automatic BVH selection based on joint motion score.
- Default `zup` display transform: `display_xyz = [bvh_x, -bvh_z, bvh_y]`.
- Draggable and scalable skeleton overlay on top of camera previews.
- Automatic alignment and manual frame-level adjustment.
- Start / end mark recording.
- CSV clip-log export.
- Aligned BVH, metadata JSON, curve CSV, and figure PNG export.
- Lite mode for lower-end computers.

### 3. Requirements

Recommended:

- Windows 10 / Windows 11
- Python 3.11+
- CPU-only is supported
- 8GB RAM or more recommended

Core dependencies:

- `numpy`
- `matplotlib`
- `opencv-python`
- `PyQt5`

### 4. Installation

With Conda:

```bash
conda env create -f environment.yml
conda activate mocap-align
python main.py
```

With pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 5. Usage

Launch GUI:

```bash
python main.py
```

Lite mode:

```bash
python main.py --lite
```

Force MP4:

```bash
python main.py --source mp4
```

Force image sequences:

```bash
python main.py --source frames
```

Auto-load the first available trial:

```bash
python main.py --auto-load
```

Skeleton display axis preset:

```bash
python main.py --axis-preset zup
python main.py --axis-preset raw
```

### 6. Data Layout

Image sequence camera data:

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1/
      1_162852.002.jpg
      1_162852.035.jpg
    2/
    3/
    4/
```

MP4 camera data:

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1_20260422_163247.mp4
    2_20260422_163247.mp4
    3_20260422_163247.mp4
    4_20260422_163247.mp4
```

Mocap data:

```text
mocap_root/
  S4/
    S4011/
      S4011_Skeleton0.bvh
      S4011_Skeleton1.bvh
      S4011_Skeleton2.bvh
      S4011_Skeleton3.bvh
```

### 7. Timing Model

The visual timeline is based on actual timing whenever possible:

- MP4 mode uses the FPS reported by OpenCV.
- Image-sequence mode parses timestamps from filenames such as `1_162852.002.jpg`.
- If timestamps are unavailable, the fallback frame rate is `FRAME_SEQUENCE_FPS = 40.0`.
- Multi-camera fusion uses the median FPS of all loaded cameras as the reference visual FPS.
- Each camera maps preview time to its own frame index.
- Energy curves are resampled to the shared reference timeline.

This reduces alignment drift when the real visual FPS differs from the intended 40fps.

### 8. BVH Handling

The tool supports any number of BVH files:

- `Skeleton0` is treated as `position`.
- `Skeleton1` is treated as `order`.
- `Skeleton2`, `Skeleton3`, and additional files are loaded as extra BVH sources.
- The GUI automatically selects the BVH file with the strongest joint motion for preview and alignment.
- Export includes every loaded BVH file.

The display-axis transform affects only preview rendering. Exported BVH motion data remains in the original coordinate system.

### 9. Export

Default output folder:

```text
sync/output/<session_id>/
```

Exported files:

- `<session_id>_<role>_aligned.bvh`
- `<session_id>_alignment.json`
- `<session_id>_aligned_curves.csv`
- `<session_id>_calibration.png`
- `sync/output/results_csv/*.csv`

The JSON metadata includes:

- current `delta_t`
- reference visual FPS
- per-camera FPS
- per-camera frame count
- BVH start frame
- display and alignment BVH roles
- axis preset

### 10. Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `Left` / `Right` | Alignment offset -1 / +1 frame |
| `Shift + Left/Right` | Alignment offset -10 / +10 frames |
| `Ctrl + Left/Right` | Alignment offset -5 / +5 frames |
| `Up` / `Down` | Move visual timeline |
| `Home` | Auto-align |
| `1` | Mark start |
| `2` | Mark end |
| `Enter` | Export current result |
| `PageUp` / `PageDown` | Previous / next trial |

### 11. Testing

Run tests:

```bash
python -m unittest discover -s tests
```

Compile check:

```bash
python -m py_compile main.py visualize_energy.py models.py ui/main_window.py ui/widgets.py utils/session.py utils/energy.py utils/alignment.py utils/exporter.py
```

### 12. License

This project is released under the MIT License.
