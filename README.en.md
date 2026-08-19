# MocapVideoAligner English Documentation

> A lightweight PyQt desktop tool for aligning multi-camera visual recordings and plantar-pressure data with BVH motion-capture data.

[中文文档](README.zh-CN.md) | [Back to language selector](README.md) | [Tactile Alignment Notes (Chinese)](docs/tactile_alignment.md)

---

## 1. Overview

`MocapVideoAligner` is a desktop GUI tool for aligning:

1. multi-camera visual recordings with BVH motion-capture data (`delta_t`)
2. aligned mocap with plantar-pressure / tactile data (`delta_t2`)

The tool is designed for offline lab workflows where visual data, pressure data, and motion-capture data are collected by separate systems and need to be synchronized after recording. It does not require a GPU, does not run a web server, and can still operate when only part of the expected data is available.

The current version focuses on the following practical issues:

- Visual recordings may be image sequences or MP4 files.
- The intended visual frame rate may be 40fps, while the actual recorded FPS can be around 37.4-37.8fps.
- Mocap data is commonly recorded at 120fps.
- A single trial may contain one, two, three, four, or more BVH files.
- Raw BVH coordinates may make the skeleton appear lying down in a direct 3D preview.
- Plantar pressure may come from reconstructed continuous CSVs or legacy segmented CSVs.
- Some trials have missing tactile data and must be skipped in navigation while remaining visible but disabled in the trial picker.
- The tool should run on lower-end CPU-only computers.

By default, the application opens the GUI only. Users select the camera root folder and mocap root folder from the toolbar, then load the trial.

---

## 2. Key Features

### 2.1 Visual-Mocap Alignment

- Single-window PyQt GUI.
- Camera root and mocap root folder selection from the toolbar.
- Automatic trial enumeration with previous / next trial navigation.
- Image sequence input from `1/2/3/4` or `cam1/cam2/cam3/cam4`.
- MP4 input with `1_*.mp4`, `2_*.mp4`, `3_*.mp4`, `4_*.mp4`.
- `auto` mode prefers image sequences and falls back to MP4.
- Image sequence formats: `png`, `jpg`, `jpeg`.
- Timestamp-aware image sequence timing.
- Actual MP4 FPS from OpenCV.
- Multi-camera energy fusion on a unified reference timeline.
- Native BVH frame rate preserved.
- Automatic BVH selection based on joint motion score.
- Default `zup` display transform to correct lying-down skeleton previews.
- Draggable and scalable skeleton overlay on top of camera previews.
- Automatic alignment and manual frame-level adjustment.
- Play / pause preview.
- Start / end mark recording.
- CSV clip-log export.
- Aligned BVH, metadata JSON, curve CSV, and figure PNG export.
- Lite mode for lower-end computers.

### 2.2 Mocap-Tactile Alignment

- Separate tab that does not rewrite visual-mocap export fields.
- Legacy mechanism: foot-grounding curve cross-correlation.
- New mechanism: touchdown / loading-onset event matching.
- Reconstructed tactile continuous format and legacy segmented format support.
- `valid_mask` visualization: original frames as solid lines, reconstructed frames as dashed lines; heatmap can show `Valid`.
- Paired left / right curve toggles that persist across trial switches.
- Click / drag playhead on the alignment curve.
- Multi-segment Fake start / end marking, exported with left-foot frame indexing.
- Quality-table `C/D` trial skip in navigation and disabled items in the trial picker.
- Export of `*_pressure_alignment.json`, pressure curve CSV, and Fake frame CSV.

Detailed notes (Chinese): [docs/tactile_alignment.md](docs/tactile_alignment.md)

---

## 3. Requirements

Recommended environment:

- Windows 10 or Windows 11.
- Python 3.11 or later.
- CPU-only is supported.
- 8GB RAM or more recommended.

Core dependencies:

- `numpy`
- `matplotlib`
- `opencv-python`
- `PyQt5`
- `scipy` (used by the mocap-tactile new mechanism)

---

## 4. Installation

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

If PyQt5 installation fails, install the core packages manually:

```bash
pip install PyQt5 opencv-python matplotlib numpy scipy
```

---

## 5. Launch Options

Open the GUI without auto-loading data:

```bash
python main.py
```

Lite mode:

```bash
python main.py --lite
```

Force MP4 input:

```bash
python main.py --source mp4
```

Force image sequence input:

```bash
python main.py --source frames
```

Auto-load the first available trial from the configured roots:

```bash
python main.py --auto-load
```

Skeleton display axis preset:

```bash
python main.py --axis-preset zup
python main.py --axis-preset raw
```

---

## 6. Recommended Data Layout

### Image Sequence Camera Data

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

The following layout is also supported:

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    cam1/
    cam2/
    cam3/
    cam4/
```

### MP4 Camera Data

```text
camera_root/
  rec20260422_163247_subject_S401_1/
    1_20260422_163247.mp4
    2_20260422_163247.mp4
    3_20260422_163247.mp4
    4_20260422_163247.mp4
```

### BVH Mocap Data

Direct trial folder:

```text
mocap_root/
  S4011/
    S4011_Skeleton0.bvh
    S4011_Skeleton1.bvh
    S4011_Skeleton2.bvh
    S4011_Skeleton3.bvh
```

Subject-grouped folder:

```text
mocap_root/
  S4/
    S4011/
      S4011_Skeleton0.bvh
      S4011_Skeleton1.bvh
      S4011_Skeleton2.bvh
      S4011_Skeleton3.bvh
```

### Reconstructed Tactile Data

Preferred continuous format:

```text
reconstruction_<timestamp>/
  20260804/S5/rec20260804_192218_xxx_S501_1/
    pressure_left.csv
    pressure_right.csv
    reconstruction_manifest.csv
```

Legacy segmented format:

```text
rec..._S501_1/
  pressure_left_t0.csv
  pressure_right_t0.csv
  reconstruction_manifest.csv
```

Optional quality table in the project root:

```text
missing_pressure_objects.csv
```

Recommended naming:

- Mocap folder name: `S<subject><two-digit-action><repetition>`, for example `S4011`.
- Camera session folder name: preferably ends with `_S401_1`, so the tool can match the camera session to the mocap trial.

---

## 7. Standard Workflow

### 7.1 Visual-Mocap Alignment

1. Run `python main.py`.
2. Click `Camera Folder` in the toolbar and select the camera root folder.
3. Click `Mocap Folder` and select the VICON / BVH root folder.
4. Use previous / next trial navigation to select a trial.
5. Click reload.
6. Check the camera previews, skeleton preview, and alignment curves.
7. Click auto-align to estimate the initial offset `delta_t`.
8. Fine-tune with buttons, sliders, or keyboard shortcuts.
9. Mark start and end when clip boundaries are needed.
10. Export the clip log or the current alignment result.

### 7.2 Mocap-Tactile Alignment

1. Preferably finish visual-mocap alignment first and obtain `*_aligned.bvh`.
2. Switch to the `Mocap-Tactile Alignment` tab.
3. Import a reconstructed tactile root (`reconstruction_<timestamp>`) when needed.
4. Optionally import visual-alignment CSV priors.
5. Choose the legacy or new mechanism; the page auto-estimates `delta_t2`.
6. Click / drag the curve playhead and inspect heatmaps, skeleton, and curves.
7. Mark multi-segment Fake intervals if needed, then export the Fake frame CSV.
8. Export the pressure-alignment result.

---

## 8. Timing Model and Actual FPS

The visual timeline no longer assumes a fixed 40fps:

- MP4 mode uses the FPS reported by OpenCV.
- Image-sequence mode parses timestamps from filenames such as `1_162852.002.jpg`.
- If timestamps are unavailable, the fallback is `FRAME_SEQUENCE_FPS = 40.0` in `config.py`.
- Multi-camera fusion uses the median FPS of all loaded cameras as the reference visual FPS.
- Each camera maps preview time to its own frame index.
- Energy curves are resampled to the shared reference timeline.

This reduces long-term drift when the real visual FPS differs from the intended 40fps.

On the tactile side:

- Pressure timelines prefer absolute `t_us` from reconstructed files.
- Left and right feet may use their own timelines. Fake export uses left-foot discrete frames as the base and nearest-neighbor mapping for the right foot.

---

## 9. BVH Handling

The tool supports any number of BVH files:

- `Skeleton0` is treated as `position`.
- `Skeleton1` is treated as `order`.
- `Skeleton2`, `Skeleton3`, and additional files are loaded as extra BVH sources.
- The tool computes a motion score for each BVH file.
- The skeleton preview and automatic alignment use the BVH with the strongest joint motion.
- Export includes every loaded BVH file.

The display-axis transform affects only GUI rendering. Exported BVH motion data remains in the original coordinate system.

The mocap-tactile page prefers visual-aligned `*_aligned.bvh` files.

---

## 10. GUI Layout

The main window contains two tabs.

### 10.1 Visual-Mocap Alignment

- Toolbar: trial navigation, folder selection, reload, auto-align, export.
- Camera preview area: four synchronized camera views.
- Skeleton preview area: current BVH skeleton and optional skeleton overlay.
- Alignment curve area: combined camera energy, per-camera energy, and mocap energy.
- Status panel: current trial, runtime settings, actual FPS, offset, frame position, and operation log.

Skeleton overlay:

- Click the overlay button to enable it.
- Drag with the left mouse button.
- Use the mouse wheel to scale.
- Double-click to reset the overlay position.

### 10.2 Mocap-Tactile Alignment

- Left / right pressure heatmaps plus skeleton preview.
- Alignment curves: four L/R curves in legacy mode, or two total curves in the new mechanism.
- Click / drag playhead on the curve.
- Sidebar mechanism buttons and import actions for reconstructed tactile data / visual priors.
- Fake start / end / undo / export controls.

Trial navigation extras:

- Quality classes `C/D` are skipped by previous / next navigation.
- The trial picker still shows `C/D` items, but they are grayed out and not selectable.

---

## 11. Exported Files

Default output directory:

```text
output/video_mocap/<session_id>/
```

### 11.1 Visual-Mocap Exports

- `<session_id>_<role>_aligned.bvh`: BVH clipped according to the current offset.
- `<session_id>_alignment.json`: alignment metadata.
- `<session_id>_aligned_curves.csv`: aligned curve data.
- `<session_id>_calibration.png`: current alignment curve figure.
- `output/video_mocap/results_csv/*.csv`: start / end clip records.

`alignment.json` contains:

- `delta_t`
- `reference_visual_fps`
- per-camera actual FPS
- per-camera frame count
- BVH start frame
- display and alignment BVH roles
- axis preset

### 11.2 Mocap-Tactile Exports

- `<session_id>_pressure_alignment.json`
- `<session_id>_pressure_aligned_curves.csv`
- `<session_id>_pressure_calibration.png`
- `output/video_mocap/fake_tactile_csv/*_fake_frames.csv`

Fake frame CSV header:

```csv
segment_id,left_frame_idx,left_time_s,right_frame_idx,right_time_s
```

---

## 12. Lite Mode and Performance

The tool can run without a dedicated GPU. The main performance costs are:

- image or video decoding
- Matplotlib curve rendering
- 3D skeleton rendering
- reconstructed tactile loading and event estimation

For lower-end machines:

```bash
python main.py --lite
```

Lite mode:

- reduces preview image scale
- avoids heavy redraws while dragging sliders
- reuses precomputed cache when possible

Cache directory:

```text
cache/
```

If data or timing logic changes, delete the corresponding session cache directory to force recomputation.

---

## 13. Keyboard Shortcuts

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

Fake marking, mechanism switching, and pressure-specific exports on the tactile tab are primarily button-driven.

---

## 14. Troubleshooting

### `main.py` does not auto-load data

This is intentional. The GUI opens first to avoid failing immediately when default paths are invalid.

Use this if auto-load is required:

```bash
python main.py --auto-load
```

### Images cannot be read

Check:

- whether the path exists
- whether the image files are corrupted
- whether camera folders are named `1/2/3/4` or `cam1/cam2/cam3/cam4`
- whether file extensions are `png`, `jpg`, or `jpeg`

### The skeleton appears lying down

Use the default `zup` preset first. If your BVH data already uses the correct display coordinate system, switch to `raw`.

### There are multiple BVH files

No manual selection is required. The tool loads all BVH files and selects the best one for preview and alignment based on motion score.

### Alignment still drifts slightly

Check:

- whether actual FPS is being used
- whether image sequence filenames contain reliable timestamps
- whether MP4 files are variable-frame-rate videos. If they are, image sequences with timestamps are preferred.

### Some trials are skipped or grayed out

Check whether `missing_pressure_objects.csv` marks the `dataset_id` as `C/D`, or whether one reconstructed pressure side has no numeric frames.

### Pressure data cannot be loaded on the tactile tab

Check:

- whether the correct `reconstruction_<timestamp>` root was imported
- whether `pressure_left.csv` / `pressure_right.csv` or legacy `pressure_*_t*.csv` files exist
- whether either side is header-only empty data

---

## 15. Development and Tests

Run tests:

```bash
python -m unittest discover -s tests
```

Tactile-focused tests:

```bash
python -m unittest tests.test_pressure_alignment -v
```

Compile check:

```bash
python -m py_compile main.py visualize_energy.py models.py ui/main_window.py ui/pressure_alignment_page.py ui/widgets.py utils/session.py utils/energy.py utils/alignment.py utils/pressure_alignment.py utils/pressure_dynamics_alignment.py utils/exporter.py
```

Main files:

```text
main.py                              compatibility entrypoint
visualize_energy.py                  GUI startup entrypoint
config.py                            default paths, frame rate, cache version, tactile defaults
models.py                            data models
ui/main_window.py                    main window and visual-mocap page
ui/pressure_alignment_page.py        mocap-tactile alignment page
utils/session.py                     data discovery, loading, and cache
utils/energy.py                      energy calculation, actual FPS, resampling
utils/alignment.py                   visual-mocap alignment and frame mapping
utils/pressure_alignment.py          pressure loading, legacy mechanism, export, quality skip
utils/pressure_dynamics_alignment.py new mechanism (touchdown events)
utils/exporter.py                    visual-page export logic
tests/                               unit tests and lightweight fixtures
docs/tactile_alignment.md                   tactile-page design notes
```

---

## 16. License

This project is released under the MIT License.
