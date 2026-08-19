# MocapVideoAligner 中文文档

> 一个用于对齐四相机视觉数据、足底触觉压力数据与 BVH 动捕数据的轻量级 PyQt 桌面工具。

[English Documentation](README.en.md) | [返回语言选择](README.md) | [触觉对齐机制说明](docs/tactile_alignment.md)

---

## 1. 项目简介

`MocapVideoAligner` 用于把视觉系统采集的多相机画面、足底压力数据，以及 VICON / BVH 动捕数据放到同一个时间轴中进行检查、自动对齐、手动微调和导出。

本工具主要面向实验室数据采集后的离线对齐流程。它不依赖 GPU，不需要 Web 服务，也不要求数据一开始就完整。只要有相机数据、动捕数据或触觉数据中的可用部分，软件都可以尽可能加载并提供对应功能。

当前版本包含两个对齐页签：

1. **视觉-动捕对齐**：求解 `delta_t`，并导出 aligned BVH 等结果。
2. **动捕-触觉对齐**：在已对齐动捕基础上求解 `delta_t2`，检查压力热图 / 曲线，并支持 Fake 片段标记。

当前版本重点解决以下问题：

- 四相机视觉数据可能是逐帧图片，也可能是 MP4。
- 视觉系统设定为 40fps，但实际采集帧率可能偏离，例如 37.4-37.8fps。
- 动捕数据通常为 120fps。
- 同一试次可能存在 1 个、2 个、3 个、4 个或更多 BVH 文件。
- BVH 原始坐标系可能导致骨架预览中人物“躺下”。
- 足底压力可能来自重建触觉连续文件或旧分段文件。
- 部分试次存在触觉缺失，需要在导航中跳过并在列表中置灰。
- 低配或无独显电脑也需要能够运行。

软件默认启动后只打开主界面，不自动加载默认路径数据。用户需要在 GUI 右上角选择相机目录和动捕目录，再点击重新加载。

---

## 2. 核心功能

### 2.1 视觉-动捕对齐

- 单窗口 PyQt GUI。
- 支持在界面中选择相机根目录和动捕根目录。
- 支持试次自动枚举和上一试次 / 下一试次导航。
- 支持逐帧图片目录：`1/2/3/4` 或 `cam1/cam2/cam3/cam4`。
- 支持四路 MP4：`1_*.mp4`、`2_*.mp4`、`3_*.mp4`、`4_*.mp4`。
- `auto` 模式优先读取逐帧图片，找不到图片时回退 MP4。
- 逐帧图片支持 `png`、`jpg`、`jpeg`。
- 图片文件名包含时间戳时，自动推断实际 FPS 和真实时间轴。
- MP4 模式使用 OpenCV 读取到的实际 FPS。
- 多相机能量曲线会按实际 FPS 重采样到统一参考时间轴。
- 支持 BVH 原始 120fps，不改写原始数据。
- 支持任意数量 BVH 文件，并自动选择运动信息最明显的文件用于预览和对齐。
- 默认 `zup` 骨架显示修正，解决人物躺下问题。
- 支持骨架悬浮叠加到相机区域，可拖拽、滚轮缩放、双击重置。
- 支持自动对齐、手动按帧微调、播放 / 暂停。
- 支持记录起点和终点。
- 支持导出裁剪记录 CSV。
- 支持导出裁切后的 BVH、对齐曲线 CSV、元数据 JSON 和曲线截图 PNG。
- 支持轻量模式，降低低配电脑上的刷新压力。

### 2.2 动捕-触觉对齐

- 新增独立页签，不改动原视觉-动捕导出字段。
- 支持旧机制：足底贴地曲线互相关。
- 支持新机制：触地事件对齐。
- 支持重建触觉新连续格式与旧分段格式。
- 支持 `valid_mask` 显示：原始帧实线、重建帧虚线；热图可显示 `Valid`。
- 支持左右脚成对曲线勾选，切换试次时保留勾选状态。
- 支持点击 / 拖动曲线移动预览光标。
- 支持多段 Fake 起点 / 终点标记，并以左脚帧为基准导出。
- 支持质量表 `C/D` 试次跳过与选择列表置灰。
- 支持导出 `*_pressure_alignment.json`、曲线 CSV 与 Fake 帧表。

更细的机制说明见：[docs/tactile_alignment.md](docs/tactile_alignment.md)

---

## 3. 环境要求

推荐环境：

- Windows 10 或 Windows 11。
- Python 3.11 及以上。
- CPU 即可运行，不要求独立显卡。
- 建议内存 8GB 及以上。

核心依赖：

- `numpy`
- `matplotlib`
- `opencv-python`
- `PyQt5`

---

## 4. 安装方式

### Conda

```bash
conda env create -f environment.yml
conda activate mocap-align
python main.py
```

### pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

如果 PyQt5 安装失败，可以单独安装：

```bash
pip install PyQt5 opencv-python matplotlib numpy scipy
```

---

## 5. 启动方式

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

启动后自动加载当前配置目录下的第一条试次：

```bash
python main.py --auto-load
```

坐标显示模式：

```bash
python main.py --axis-preset zup
python main.py --axis-preset raw
```

---

## 6. 推荐数据结构

### 相机逐帧图片

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

也支持：

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    cam1/
    cam2/
    cam3/
    cam4/
```

### 相机 MP4

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1_20260422_163247.mp4
    2_20260422_163247.mp4
    3_20260422_163247.mp4
    4_20260422_163247.mp4
```

### 动捕 BVH

直接试次目录：

```text
mocap_root/
  S4011/
    S4011_Skeleton0.bvh
    S4011_Skeleton1.bvh
    S4011_Skeleton2.bvh
    S4011_Skeleton3.bvh
```

按被试分组：

```text
mocap_root/
  S4/
    S4011/
      S4011_Skeleton0.bvh
      S4011_Skeleton1.bvh
      S4011_Skeleton2.bvh
      S4011_Skeleton3.bvh
```

### 重建触觉数据

新连续格式（优先）：

```text
reconstruction_<timestamp>/
  20260804/S5/rec20260804_192218_xxx_S501_1/
    pressure_left.csv
    pressure_right.csv
    reconstruction_manifest.csv
```

旧分段格式（兼容）：

```text
rec..._S501_1/
  pressure_left_t0.csv
  pressure_right_t0.csv
  reconstruction_manifest.csv
```

可选质量表（放在工程根目录）：

```text
missing_pressure_objects.csv
```

试次命名建议：

- 动捕目录名：`S<对象><动作两位><重复次数>`，例如 `S4011`。
- 相机会话名：建议以 `_S401_1` 结尾，软件会用该后缀匹配动捕试次。

---

## 7. 标准使用流程

### 7.1 视觉-动捕对齐

1. 运行 `python main.py`。
2. 在右上角点击“相机目录”，选择相机数据根目录。
3. 点击“动捕目录”，选择 VICON / BVH 数据根目录。
4. 使用“上一试次 / 下一试次”选择目标试次。
5. 点击“重新加载”。
6. 检查四路相机预览、骨架预览和对齐曲线。
7. 点击“自动对齐”获得初始偏移 `delta_t`。
8. 使用底部按钮、滑条或方向键进行人工微调。
9. 需要裁剪时，点击“记录起点”和“记录终点”。
10. 点击“导出裁剪记录”或“导出当前结果”。

### 7.2 动捕-触觉对齐

1. 建议先完成视觉-动捕对齐，得到 `*_aligned.bvh`。
2. 切换到「动捕-触觉对齐」页签。
3. 如有需要，点击“导入重建触觉数据”，选择 `reconstruction_<timestamp>` 根目录。
4. 可选：导入视频-动捕对齐结果 CSV，辅助初始化。
5. 选择旧机制或新机制，软件会自动估计 `delta_t2`。
6. 点击 / 拖动曲线检查热图、骨架与压力曲线。
7. 如需标注异常压力段，记录多段 Fake 起点 / 终点并导出。
8. 导出压力对齐结果。

---

## 8. 时间轴与实际 FPS

视觉侧不再简单假设固定 40fps：

- MP4 模式读取 OpenCV 返回的实际 FPS。
- 逐帧图片模式如果文件名中有类似 `162852.002` 的时间戳，会推断真实帧时间和实际 FPS。
- 如果图片文件名无法解析时间戳，则回退 `config.py` 中的 `FRAME_SEQUENCE_FPS = 40.0`。
- 多相机参考 FPS 使用当前已加载相机实际 FPS 的中位数。
- 每路相机取帧都按自己的真实时间轴计算。
- 对齐曲线会把各路相机能量重采样到统一参考时间轴。

这可以减少“设定 40fps，但实际只有 37.xfps”导致的长时间漂移。

触觉侧：

- 优先使用重建压力文件中的 `t_us` 时间轴。
- 左右脚可在各自时间轴上存在，导出 Fake 时以左脚离散帧为基准，右脚取最近邻。

---

## 9. BVH 处理逻辑

软件支持任意数量 BVH 文件：

- `Skeleton0` 优先标记为 `position`。
- `Skeleton1` 优先标记为 `order`。
- `Skeleton2`、`Skeleton3` 等作为额外 BVH 一起加载。
- 软件会计算每个 BVH 的运动评分。
- 骨架预览和自动对齐会自动选择关节运动最明显的 BVH。
- 导出时会导出全部已加载 BVH。

显示坐标修正只影响 GUI 预览，不会改写原始 BVH 数据。

动捕-触觉页优先读取视觉对齐后的 `*_aligned.bvh`。

---

## 10. 界面说明

主界面包含两个页签：

### 10.1 视觉-动捕对齐

- 顶部工具栏：试次导航、目录选择、重新加载、自动对齐、导出。
- 相机预览区：四路相机画面，同步显示当前视觉帧。
- 骨架预览区：显示当前 BVH 骨架，并支持悬浮叠加到相机区域。
- 对齐曲线区：显示相机综合能量、单相机能量和动捕能量曲线。
- 右侧状态区：显示当前试次、运行设置、实际 FPS、偏移、帧位置和操作日志。

骨架悬浮叠加：

- 点击“悬浮叠加到相机区”开启。
- 鼠标左键拖拽移动。
- 鼠标滚轮缩放。
- 双击重置位置。

### 10.2 动捕-触觉对齐

- 左右脚压力热图 + 动捕骨架预览。
- 对齐曲线：旧机制四条 L/R 曲线，或新机制两条总曲线。
- 可点击 / 拖动曲线移动预览光标。
- 右侧可切换旧机制 / 新机制，并导入重建触觉 / 视觉对齐先验。
- 支持 Fake 起点、终点、撤销与导出。

试次导航补充规则：

- 质量表 `C/D` 会在上一 / 下一试次中跳过。
- 选择批次列表仍显示 `C/D`，但置灰不可选。

---

## 11. 导出内容

默认输出目录：

```text
output/video_mocap/<session_id>/
```

### 11.1 视觉-动捕导出

- `<session_id>_<role>_aligned.bvh`：按当前偏移裁切后的 BVH。
- `<session_id>_alignment.json`：对齐元数据。
- `<session_id>_aligned_curves.csv`：对齐后的曲线数据。
- `<session_id>_calibration.png`：当前对齐曲线截图。
- `output/video_mocap/results_csv/*.csv`：起点 / 终点裁剪记录。

`alignment.json` 会记录：

- `delta_t`
- `reference_visual_fps`
- 每路相机实际 FPS
- 每路相机帧数
- BVH 起始帧
- 显示和对齐使用的 BVH 角色
- 坐标预设

### 11.2 动捕-触觉导出

- `<session_id>_pressure_alignment.json`
- `<session_id>_pressure_aligned_curves.csv`
- `<session_id>_pressure_calibration.png`
- `output/video_mocap/fake_tactile_csv/*_fake_frames.csv`

Fake 帧表头：

```csv
segment_id,left_frame_idx,left_time_s,right_frame_idx,right_time_s
```

---

## 12. 轻量模式与性能

无独显机器可以运行本工具。主要耗时来自：

- 图片或视频解码。
- Matplotlib 曲线绘制。
- 3D 骨架绘制。
- 重建触觉与 BVH 事件估计。

低配机器建议：

```bash
python main.py --lite
```

轻量模式会：

- 降低预览图缩放比例。
- 拖动滑条时减少重负载刷新。
- 优先复用预计算缓存。

缓存目录：

```text
cache/
```

如果数据或代码逻辑更新后需要重新计算，可以删除对应 session 的缓存目录。

---

## 13. 快捷键

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

动捕-触觉页的 Fake 标记、机制切换与导出以页面按钮为主。

---

## 14. 常见问题

### 运行 `main.py` 后没有自动加载数据

这是当前设计。默认只打开界面，避免默认配置路径错误时直接失败。

如果需要自动加载：

```bash
python main.py --auto-load
```

### 图像读取失败

请检查：

- 路径是否存在。
- 图片是否损坏。
- 目录名是否为 `1/2/3/4` 或 `cam1/cam2/cam3/cam4`。
- 后缀是否为 `png/jpg/jpeg`。

### 骨架躺下或方向不对

先使用默认 `zup`。如果数据本身已经是正确坐标系，可以切换为 `raw`。

### 有多个 BVH 文件不知道选哪个

不需要手动判断。软件会全部加载并根据运动评分自动选择用于预览和对齐的 BVH。

### 对齐后仍然有轻微漂移

优先确认：

- 是否使用了实际帧率。
- 逐帧图片文件名是否包含可靠时间戳。
- MP4 是否为可变帧率视频。若是，建议使用逐帧图片和时间戳。

### 某些试次被跳过或选择列表置灰

检查 `missing_pressure_objects.csv` 中对应 `dataset_id` 是否为 `C/D`，  
或重建目录是否左右脚一侧无有效帧。

### 动捕-触觉页加载不到压力

确认：

- 已导入正确的 `reconstruction_<timestamp>` 根目录。
- 目录中存在 `pressure_left.csv` / `pressure_right.csv`，或旧格式 `pressure_*_t*.csv`。
- 文件不是仅有表头的空侧数据。

---

## 15. 开发与测试

运行测试：

```bash
python -m unittest discover -s tests
```

重点相关测试：

```bash
python -m unittest tests.test_pressure_alignment -v
```

语法检查：

```bash
python -m py_compile main.py visualize_energy.py models.py ui/main_window.py ui/pressure_alignment_page.py ui/widgets.py utils/session.py utils/energy.py utils/alignment.py utils/pressure_alignment.py utils/pressure_dynamics_alignment.py utils/exporter.py
```

主要文件：

```text
main.py                              兼容入口
visualize_energy.py                  GUI 启动入口
config.py                            默认路径、帧率、缓存版本、触觉默认路径
models.py                            数据模型
ui/main_window.py                    主窗口与视觉-动捕页
ui/pressure_alignment_page.py        动捕-触觉对齐页
utils/session.py                     数据发现、加载和缓存
utils/energy.py                      能量计算、实际 FPS 和重采样
utils/alignment.py                   视觉-动捕对齐和帧映射
utils/pressure_alignment.py          压力加载、旧机制、导出、质量跳过
utils/pressure_dynamics_alignment.py 新机制（触地事件）
utils/exporter.py                    视觉页导出逻辑
tests/                               单元测试和轻量 fixture
docs/tactile_alignment.md                   触觉页专项说明
```

---

## 16. 许可协议

本项目使用 MIT License。
