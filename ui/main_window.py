from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import csv
import datetime
import re
import traceback

import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 15

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from PyQt5 import QtCore, QtGui, QtWidgets

from config import (
    APP_TITLE,
    DEFAULT_AXIS_PRESET,
    DEFAULT_CAM_ROOT,
    DEFAULT_MOCAP_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREVIEW_SCALE,
    DEFAULT_SOURCE_MODE,
    LITE_PREVIEW_SCALE,
)
from models import AlignmentState, BVHSourceData, SessionData, TrialInfo
from ui.style import app_stylesheet, create_palette
from ui.widgets import DraggableOverlayLabel, InfoRow, PreviewTile, TechPanel, build_alignment_logo_pixmap, repolish, rgb_array_to_qpixmap
from utils.alignment import estimate_initial_offset, preview_time_to_bvh_frame, preview_time_to_camera_frame, quantize_time
from utils.bvh_pose import compute_joint_positions, transform_display_positions
from utils.energy import build_combined_camera_energy, norm01
from utils.exporter import export_alignment_bundle
from utils.session import close_session_data, enumerate_trials, load_session_from_paths, load_trial


MIN_WINDOW_WIDTH = 1480
MIN_WINDOW_HEIGHT = 940


def _camera_display_name(label: str) -> str:
    return f"相机 {label.replace('cam', '')}"


def _join_camera_display_names(labels: tuple[str, ...] | list[str]) -> str:
    if not labels:
        return "未勾选"
    return "、".join(_camera_display_name(label) for label in labels)


def _join_camera_indexes(labels: tuple[str, ...] | list[str]) -> str:
    if not labels:
        return "未选"
    return " / ".join(label.replace("cam", "") for label in labels)


@dataclass(frozen=True)
class LaunchOptions:
    camera_session: Path | None = None
    position_bvh: Path | None = None
    order_bvh: Path | None = None
    source_mode: str = DEFAULT_SOURCE_MODE
    axis_preset: str = DEFAULT_AXIS_PRESET
    lite_mode: bool = False
    auto_load: bool = False


class SkeletonCanvas(FigureCanvas):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(4.6, 4.6), facecolor="#F3E6D7")
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax.set_facecolor("#F3E6D7")
        self._current_role_key: tuple[str, str] | None = None
        self._edge_artists = []
        self._scatter = None
        self._position_cache: OrderedDict[tuple[str, int, str], np.ndarray] = OrderedDict()
        self._cache_limit = 48
        self._show_placeholder("请先加载动捕骨架")

    def _show_placeholder(self, text: str) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#F3E6D7")
        self.ax.text2D(0.5, 0.5, text, transform=self.ax.transAxes, ha="center", va="center", color="#241C17")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_zlabel("")
        self.draw_idle()

    def _ensure_artists(self, bvh_source: BVHSourceData) -> None:
        role_key = (bvh_source.role, str(bvh_source.motion.path))
        if self._current_role_key == role_key and self._edge_artists:
            return

        self.ax.clear()
        self.ax.set_facecolor("#F3E6D7")
        self.ax.view_init(elev=18, azim=-62)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_zlabel("")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        self._edge_artists = []
        for _ in bvh_source.motion.edges:
            artist, = self.ax.plot([], [], [], color="#AF5A3B", linewidth=2.4)
            self._edge_artists.append(artist)
        self._scatter = self.ax.scatter([], [], [], color="#355166", s=24)
        self._current_role_key = role_key

    def _get_positions(self, bvh_source: BVHSourceData, frame_index: int, axis_preset: str) -> np.ndarray:
        cache_key = (str(bvh_source.motion.path), frame_index, axis_preset)
        cached = self._position_cache.pop(cache_key, None)
        if cached is not None:
            self._position_cache[cache_key] = cached
            return cached

        positions = compute_joint_positions(bvh_source.motion, frame_index)
        positions = transform_display_positions(positions, axis_preset)
        self._position_cache[cache_key] = positions
        while len(self._position_cache) > self._cache_limit:
            self._position_cache.popitem(last=False)
        return positions

    def clear_position_cache(self) -> None:
        self._position_cache.clear()

    def _resolve_positions(self, session: SessionData | None, preview_time: float, delta_t: float) -> tuple[BVHSourceData, np.ndarray] | None:
        if session is None or session.display_bvh is None:
            return None

        bvh_source = session.display_bvh
        self._ensure_artists(bvh_source)
        frame_index = preview_time_to_bvh_frame(session, preview_time, delta_t, bvh_source=bvh_source)
        positions = self._get_positions(bvh_source, frame_index, session.runtime_options.axis_preset)

        center = positions.mean(axis=0)
        span = float(np.max(np.ptp(positions, axis=0))) if len(positions) > 0 else 1.0
        span = max(span, 1.0)
        radius = span * 0.65
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        return bvh_source, positions

    def build_overlay_pixmap(self, session: SessionData | None, preview_time: float, delta_t: float, side: int = 1400) -> QtGui.QPixmap:
        resolved = self._resolve_positions(session, preview_time, delta_t)
        if resolved is None:
            return QtGui.QPixmap()

        bvh_source, positions = resolved
        proj_x, proj_y, _ = proj3d.proj_transform(positions[:, 0], positions[:, 1], positions[:, 2], self.ax.get_proj())
        projected = np.column_stack((proj_x, proj_y)).astype(np.float64, copy=False)
        valid = np.all(np.isfinite(projected), axis=1)
        if not np.any(valid):
            return QtGui.QPixmap()

        projected = projected[valid]
        min_xy = projected.min(axis=0)
        max_xy = projected.max(axis=0)
        span_xy = np.maximum(max_xy - min_xy, 1e-6)
        margin = max(24.0, side * 0.10)
        scale = min((side - margin * 2.0) / span_xy[0], (side - margin * 2.0) / span_xy[1])
        center_xy = (min_xy + max_xy) / 2.0
        output_points = np.zeros((len(positions), 2), dtype=np.float64)
        output_points[:, 0] = (proj_x - center_xy[0]) * scale + side / 2.0
        output_points[:, 1] = (center_xy[1] - proj_y) * scale + side / 2.0

        pixmap = QtGui.QPixmap(side, side)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        edge_width = max(3.0, side * 0.006)
        point_radius = max(4.0, side * 0.008)
        edge_pen = QtGui.QPen(QtGui.QColor(175, 90, 59, 245), edge_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        point_brush = QtGui.QBrush(QtGui.QColor(53, 81, 102, 245))
        point_pen = QtGui.QPen(QtCore.Qt.NoPen)

        painter.setPen(edge_pen)
        for parent, child in bvh_source.motion.edges:
            parent_point = QtCore.QPointF(float(output_points[parent, 0]), float(output_points[parent, 1]))
            child_point = QtCore.QPointF(float(output_points[child, 0]), float(output_points[child, 1]))
            painter.drawLine(parent_point, child_point)

        painter.setPen(point_pen)
        painter.setBrush(point_brush)
        for x_value, y_value in output_points:
            painter.drawEllipse(QtCore.QPointF(float(x_value), float(y_value)), point_radius, point_radius)
        painter.end()
        return pixmap

    def render(self, session: SessionData | None, preview_time: float, delta_t: float) -> None:
        resolved = self._resolve_positions(session, preview_time, delta_t)
        if resolved is None:
            self._show_placeholder("请先加载动捕骨架")
            return

        bvh_source, positions = resolved

        for artist, (parent, child) in zip(self._edge_artists, bvh_source.motion.edges):
            points = positions[[parent, child]]
            artist.set_data(points[:, 0], points[:, 1])
            artist.set_3d_properties(points[:, 2])
        if self._scatter is not None:
            self._scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        self.draw_idle()


class EnergyCanvas(FigureCanvas):
    clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(10.8, 4.8), facecolor="#F3E6D7")
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.95)
        self.ax_main = self.figure.add_subplot(111)
        self.mpl_connect("button_press_event", self._on_click)
        self._render_empty()

    def _style_axis(self) -> None:
        self.ax_main.set_facecolor("#F7EEE4")
        self.ax_main.tick_params(colors="#5E5045", labelsize=13)
        for spine in self.ax_main.spines.values():
            spine.set_color("#D4BFAE")
        self.ax_main.title.set_color("#241C17")
        self.ax_main.xaxis.label.set_color("#241C17")
        self.ax_main.yaxis.label.set_color("#241C17")
        self.ax_main.grid(alpha=0.28, color="#D2BCA9")

    def _render_empty(self) -> None:
        self.ax_main.clear()
        self._style_axis()
        self.ax_main.text(0.5, 0.5, "请先加载用于对齐的数据", transform=self.ax_main.transAxes, ha="center", va="center")
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_main or event.xdata is None:
            return
        self.clicked.emit(float(event.xdata))

    def render(
        self,
        session: SessionData | None,
        preview_time: float,
        delta_t: float,
        combined_camera_energy: np.ndarray,
        show_combined_energy: bool,
        visible_camera_labels: tuple[str, ...],
        mark_start_time: float | None = None,
        mark_end_time: float | None = None,
    ) -> None:
        if session is None:
            self._render_empty()
            return

        self.ax_main.clear()
        self._style_axis()

        ref_fps = session.reference_visual_fps
        visible_set = set(visible_camera_labels)
        has_content = False
        palette = ["#667C58", "#355166", "#A94F5B", "#D97706"]

        if show_combined_energy and len(combined_camera_energy):
            time_values = np.arange(len(combined_camera_energy), dtype=np.float64) / ref_fps
            self.ax_main.plot(
                time_values,
                norm01(combined_camera_energy),
                color="#241C17",
                linewidth=2.8,
                alpha=0.92,
                label="相机综合",
            )
            has_content = True

        for index, camera in enumerate(sorted(session.cameras, key=lambda item: item.label)):
            if camera.label not in visible_set:
                continue
            time_values = camera.energy_times()
            energy = camera.energy[: len(time_values)]
            self.ax_main.plot(
                time_values,
                norm01(energy),
                linewidth=1.6,
                alpha=0.80,
                color=palette[index % len(palette)],
                label=_camera_display_name(camera.label),
            )
            has_content = True

        if session.alignment_bvh is not None:
            bvh_energy = session.alignment_bvh.energy_visual
            time_bvh = np.arange(len(bvh_energy), dtype=np.float64) / ref_fps + delta_t
            self.ax_main.plot(
                time_bvh,
                norm01(bvh_energy),
                color="#355166",
                linestyle="--",
                linewidth=2.4,
                label="动捕能量",
            )
            has_content = True

        self.ax_main.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.55)
        if mark_start_time is not None:
            self.ax_main.axvline(mark_start_time, color="#16A34A", linewidth=2.2, label="起点")
        if mark_end_time is not None:
            self.ax_main.axvline(mark_end_time, color="#2563EB", linewidth=2.2, linestyle="--", label="终点")

        self.ax_main.set_xlabel("时间（秒）")
        self.ax_main.set_ylabel("归一化能量")

        if has_content:
            self.ax_main.legend(loc="upper right", fontsize=11, ncol=2)
        else:
            self.ax_main.text(0.5, 0.5, "当前没有可显示的相机或动捕曲线", transform=self.ax_main.transAxes, ha="center", va="center")

        max_x = max(session.preview_duration, session.bvh_visual_duration, 1.0)
        min_x = min(0.0, delta_t)
        self.ax_main.set_xlim(min_x, max_x)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, options: LaunchOptions) -> None:
        super().__init__()
        self.options = options
        self.cam_root = DEFAULT_CAM_ROOT
        self.mocap_root = DEFAULT_MOCAP_ROOT
        self.session: SessionData | None = None
        self.state = AlignmentState(delta_t=0.0, preview_time=0.0, auto_delta_t=0.0, auto_confidence=0.0)
        self._dragging_visual = False
        self._dragging_mocap = False
        self._dragging_delta = False
        self._selected_camera_labels: tuple[str, ...] = ()
        self._use_explicit_paths = any(
            path is not None for path in (options.camera_session, options.position_bvh, options.order_bvh)
        )

        self._trial_list: list[TrialInfo] = [] if self._use_explicit_paths else enumerate_trials(self.mocap_root, self.cam_root)
        self._trial_index: int = 0

        self.body_splitter: QtWidgets.QSplitter | None = None
        self.workspace_splitter: QtWidgets.QSplitter | None = None
        self.preview_splitter: QtWidgets.QSplitter | None = None
        self.camera_tiles: dict[str, PreviewTile] = {}
        self.camera_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.combined_check: QtWidgets.QCheckBox | None = None
        self.energy_canvas = EnergyCanvas()
        self.skeleton_canvas = SkeletonCanvas()
        self.camera_stage: QtWidgets.QWidget | None = None
        self.overlay_toggle_button: QtWidgets.QPushButton | None = None
        self.skeleton_overlay: DraggableOverlayLabel | None = None
        self._mark_start: dict | None = None
        self._mark_end: dict | None = None
        self._maximized_camera: str | None = None
        self._pending_initial_layout = True

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._connect_signals()
        self._apply_launch_options()
        self._update_trial_display()
        self._clear_marks(update_view=False)
        self._set_play_button_state(False)

        if self.options.auto_load and self._use_explicit_paths:
            QtCore.QTimer.singleShot(0, self.load_explicit_selection)
        elif self.options.auto_load and self._trial_list:
            QtCore.QTimer.singleShot(0, self.load_current_selection)
        elif self._trial_list:
            self.log("已发现试次列表。请确认目录后点击“重新加载”，或使用上一试次/下一试次选择目标试次。")
        else:
            self.log("请先在右上角选择相机目录和动捕目录，然后点击“重新加载”。")

    def _build_ui(self) -> None:
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        root.addWidget(self._build_header())
        root.addWidget(self._build_body(), 1)
        self.setCentralWidget(central)

    def _build_header(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        logo_label = QtWidgets.QLabel()
        logo_label.setPixmap(build_alignment_logo_pixmap(76))
        logo_label.setFixedSize(76, 76)
        layout.addWidget(logo_label, 0, QtCore.Qt.AlignVCenter)

        title = QtWidgets.QLabel(APP_TITLE)
        title.setProperty("role", "title")
        layout.addWidget(title, 0, QtCore.Qt.AlignVCenter)

        layout.addStretch(1)

        self.prev_button = QtWidgets.QPushButton("上一试次")
        self.prev_button.setMinimumWidth(110)
        self.trial_info_label = QtWidgets.QLabel("未加载")
        self.trial_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trial_info_label.setMinimumWidth(340)
        self.trial_info_label.setProperty("headerValue", True)
        self.next_button = QtWidgets.QPushButton("下一试次")
        self.next_button.setMinimumWidth(110)
        layout.addWidget(self.prev_button, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.trial_info_label, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.next_button, 0, QtCore.Qt.AlignVCenter)

        self.state_badge = QtWidgets.QLabel("未加载")
        self.state_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.state_badge.setProperty("stateBadge", True)
        self.state_badge.setProperty("level", "warning")
        layout.addWidget(self.state_badge, 0, QtCore.Qt.AlignVCenter)

        self.pick_cam_root_button = QtWidgets.QPushButton("相机目录")
        self.pick_cam_root_button.setMinimumWidth(118)
        self.pick_mocap_root_button = QtWidgets.QPushButton("动捕目录")
        self.pick_mocap_root_button.setMinimumWidth(118)
        self.load_button = QtWidgets.QPushButton("重新加载")
        self.load_button.setProperty("accent", True)
        self.auto_button = QtWidgets.QPushButton("自动对齐")
        self.export_button = QtWidgets.QPushButton("导出当前结果")
        self.export_button.setMinimumWidth(180)
        layout.addWidget(self.pick_cam_root_button, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.pick_mocap_root_button, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.load_button, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.auto_button, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.export_button, 0, QtCore.Qt.AlignVCenter)
        return panel

    def _build_body(self) -> QtWidgets.QWidget:
        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(10)
        self.body_splitter.addWidget(self._build_workspace())
        sidebar = self._build_sidebar()
        sidebar.setMinimumWidth(330)
        sidebar.setMaximumWidth(430)
        self.body_splitter.addWidget(sidebar)
        self.body_splitter.setStretchFactor(0, 7)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([1640, 360])
        return self.body_splitter

    def _build_workspace(self) -> QtWidgets.QWidget:
        self.workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(10)
        self.workspace_splitter.addWidget(self._build_preview_area())
        self.workspace_splitter.addWidget(self._build_energy_panel())
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 3)
        self.workspace_splitter.setSizes([760, 470])
        return self.workspace_splitter

    def _build_preview_area(self) -> QtWidgets.QWidget:
        self.preview_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setHandleWidth(10)
        self.preview_splitter.addWidget(self._build_camera_panel())
        self.preview_splitter.addWidget(self._build_skeleton_panel())
        self.preview_splitter.setStretchFactor(0, 5)
        self.preview_splitter.setStretchFactor(1, 3)
        self.preview_splitter.setSizes([1040, 620])
        return self.preview_splitter

    def apply_default_window_layout(self) -> None:
        if self.body_splitter is None or self.workspace_splitter is None or self.preview_splitter is None:
            return

        total_width = max(self.width() - 32, MIN_WINDOW_WIDTH - 32)
        total_height = max(self.height() - 132, MIN_WINDOW_HEIGHT - 132)

        sidebar_width = max(330, int(total_width * 0.17))
        body_width = max(1080, total_width - sidebar_width)

        camera_width = max(880, int(body_width * 0.60))
        skeleton_width = max(560, body_width - camera_width)

        preview_height = max(640, int(total_height * 0.66))
        energy_height = max(340, total_height - preview_height)

        self.body_splitter.setSizes([body_width, sidebar_width])
        self.workspace_splitter.setSizes([preview_height, energy_height])
        self.preview_splitter.setSizes([camera_width, skeleton_width])

    def _build_section_header(self, accent_text: str, title_text: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if title_text:
            accent = QtWidgets.QLabel(accent_text)
            accent.setProperty("panelAccent", True)
            title = QtWidgets.QLabel(title_text)
            title.setProperty("panelTitle", True)
            layout.addWidget(accent)
            layout.addWidget(title)
        else:
            title = QtWidgets.QLabel(accent_text)
            title.setProperty("panelTitle", True)
            layout.addWidget(title)
        return widget

    def _build_control_group(self, widgets: tuple[QtWidgets.QWidget, ...], spacing: int = 8) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setProperty("controlGroup", True)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(spacing)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        for widget in widgets:
            layout.addWidget(widget)
        return frame

    def _build_camera_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._build_section_header("相机预览", ""))

        self.camera_stage = QtWidgets.QWidget()
        self.camera_stage.installEventFilter(self)
        stage_layout = QtWidgets.QVBoxLayout(self.camera_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)

        self._camera_grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self._camera_grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        for index in range(1, 5):
            label = f"cam{index}"
            tile = PreviewTile(_camera_display_name(label))
            self.camera_tiles[label] = tile
            tile.double_clicked.connect(lambda lbl=label: self._toggle_camera_maximize(lbl))
            grid.addWidget(tile, (index - 1) // 2, (index - 1) % 2)

        stage_layout.addWidget(self._camera_grid_widget, 1)
        self.skeleton_overlay = DraggableOverlayLabel(self.camera_stage)
        self.skeleton_overlay.raise_()

        layout.addWidget(self.camera_stage, 1)
        return panel

    def _toggle_camera_maximize(self, label: str) -> None:
        if self._maximized_camera == label:
            for tile in self.camera_tiles.values():
                tile.show()
            self._maximized_camera = None
            return

        for camera_label, tile in self.camera_tiles.items():
            tile.setVisible(camera_label == label)
        self._maximized_camera = label

    def _update_skeleton_overlay_state(self) -> None:
        has_display_bvh = self.session is not None and self.session.display_bvh is not None
        if self.overlay_toggle_button is not None:
            self.overlay_toggle_button.setEnabled(has_display_bvh)
            if not has_display_bvh:
                self.overlay_toggle_button.blockSignals(True)
                self.overlay_toggle_button.setChecked(False)
                self.overlay_toggle_button.blockSignals(False)
        if self.skeleton_overlay is not None and not has_display_bvh:
            self.skeleton_overlay.hide()

    def _toggle_skeleton_overlay(self, checked: bool) -> None:
        if self.overlay_toggle_button is None or self.skeleton_overlay is None:
            return
        if not checked:
            self.skeleton_overlay.hide()
            return
        if self.session is None or self.session.display_bvh is None:
            self.overlay_toggle_button.blockSignals(True)
            self.overlay_toggle_button.setChecked(False)
            self.overlay_toggle_button.blockSignals(False)
            return
        self._refresh_skeleton_overlay(force_reset=True)
        self.log("已开启骨架悬浮叠加，可直接拖到相机画面上对照。")

    def _refresh_skeleton_overlay(self, force_reset: bool = False) -> None:
        if self.skeleton_overlay is None or self.camera_stage is None:
            return
        overlay_active = self.overlay_toggle_button is not None and self.overlay_toggle_button.isChecked()
        if not overlay_active or self.session is None or self.session.display_bvh is None:
            self.skeleton_overlay.hide()
            return
        overlay_side = max(1200, int(max(self.camera_stage.width(), self.camera_stage.height()) * 1.6))
        pixmap = self.skeleton_canvas.build_overlay_pixmap(self.session, self.state.preview_time, self.state.delta_t, side=overlay_side)
        if pixmap.isNull():
            self.skeleton_overlay.hide()
            return
        self.skeleton_overlay.set_overlay_pixmap(pixmap)
        if force_reset or not self.skeleton_overlay.isVisible():
            self.skeleton_overlay.reset_in_parent()
        else:
            self.skeleton_overlay.clamp_to_parent()
        self.skeleton_overlay.show()
        self.skeleton_overlay.raise_()

    def _build_skeleton_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        header_row.addWidget(self._build_section_header("骨架预览", ""), 1)
        self.overlay_toggle_button = QtWidgets.QPushButton("悬浮叠加到相机区")
        self.overlay_toggle_button.setCheckable(True)
        self.overlay_toggle_button.setEnabled(False)
        self.overlay_toggle_button.setProperty("overlayToggle", True)
        self.overlay_toggle_button.setToolTip("把当前骨架预览以半透明悬浮层叠到相机画面上，可拖拽对照。")
        header_row.addWidget(self.overlay_toggle_button, 0, QtCore.Qt.AlignTop)
        layout.addLayout(header_row)

        self.skeleton_info_frame = QtWidgets.QFrame()
        self.skeleton_info_frame.setProperty("infoValueCard", True)
        skeleton_info_layout = QtWidgets.QVBoxLayout(self.skeleton_info_frame)
        skeleton_info_layout.setContentsMargins(14, 10, 14, 10)
        skeleton_info_layout.setSpacing(0)
        self.skeleton_info_label = QtWidgets.QLabel("当前骨架：未加载")
        self.skeleton_info_label.setProperty("metricValue", True)
        self.skeleton_info_label.setWordWrap(True)
        skeleton_info_layout.addWidget(self.skeleton_info_label)
        layout.addWidget(self.skeleton_info_frame, 0)
        layout.addWidget(self.skeleton_canvas, 1)
        return panel

    def _build_energy_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)
        top_row.addWidget(self._build_section_header("对齐曲线", ""), 1)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(10)
        selector_label = QtWidgets.QLabel("显示曲线")
        selector_label.setProperty("metricTitle", True)
        selector_row.addWidget(selector_label)
        self.combined_check = QtWidgets.QCheckBox("综合")
        self.combined_check.setChecked(True)
        self.combined_check.setEnabled(False)
        selector_row.addWidget(self.combined_check)
        for index in range(1, 5):
            label = f"cam{index}"
            checkbox = QtWidgets.QCheckBox(str(index))
            checkbox.setChecked(True)
            checkbox.setEnabled(False)
            self.camera_checks[label] = checkbox
            selector_row.addWidget(checkbox)
        self.camera_hint_label = QtWidgets.QLabel("当前默认显示全部相机。")
        self.camera_hint_label.setProperty("metricValue", True)
        selector_row.addWidget(self.camera_hint_label)
        top_row.addLayout(selector_row, 0)
        layout.addLayout(top_row)

        layout.addWidget(self.energy_canvas, 1)

        timeline_grid = QtWidgets.QGridLayout()
        timeline_grid.setContentsMargins(0, 0, 0, 0)
        timeline_grid.setHorizontalSpacing(14)
        timeline_grid.setVerticalSpacing(12)

        self.visual_label = QtWidgets.QLabel("视觉进度：未加载")
        self.visual_label.setProperty("metricValue", True)
        self.visual_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.visual_slider.setRange(0, 0)
        timeline_grid.addWidget(self.visual_label, 0, 0)
        timeline_grid.addWidget(self.visual_slider, 0, 1)

        self.mocap_label = QtWidgets.QLabel("动捕进度：未加载")
        self.mocap_label.setProperty("metricValue", True)
        self.mocap_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mocap_slider.setRange(0, 0)
        timeline_grid.addWidget(self.mocap_label, 1, 0)
        timeline_grid.addWidget(self.mocap_slider, 1, 1)

        self.delta_label = QtWidgets.QLabel("对齐偏移：+0.000 秒")
        self.delta_label.setProperty("metricValue", True)
        self.delta_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.delta_slider.setRange(0, 0)
        timeline_grid.addWidget(self.delta_label, 2, 0)
        timeline_grid.addWidget(self.delta_slider, 2, 1)
        layout.addLayout(timeline_grid)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(18)
        control_row.setAlignment(QtCore.Qt.AlignCenter)
        self.play_button = QtWidgets.QPushButton("开始播放")
        self.play_button.setCheckable(True)
        self.visual_prev = QtWidgets.QPushButton("上一帧")
        self.visual_next = QtWidgets.QPushButton("下一帧")
        self.delta_minus_10 = QtWidgets.QPushButton("对齐 -10 帧")
        self.delta_minus_1 = QtWidgets.QPushButton("对齐 -1 帧")
        self.delta_plus_1 = QtWidgets.QPushButton("对齐 +1 帧")
        self.delta_plus_10 = QtWidgets.QPushButton("对齐 +10 帧")
        self.mark_start_btn = QtWidgets.QPushButton("📍 记录起点")
        self.mark_end_btn = QtWidgets.QPushButton("🏁 记录终点")
        self.export_log_btn = QtWidgets.QPushButton("导出裁剪记录")
        self.mark_start_label = QtWidgets.QLabel("起点：未记录")
        self.mark_start_label.setProperty("metricValue", True)
        self.mark_start_label.setMinimumWidth(220)
        self.mark_start_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mark_end_label = QtWidgets.QLabel("终点：未记录")
        self.mark_end_label.setProperty("metricValue", True)
        self.mark_end_label.setMinimumWidth(220)
        self.mark_end_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        play_group = self._build_control_group((
            self.play_button,
            self.visual_prev,
            self.visual_next,
            self.delta_minus_10,
            self.delta_minus_1,
            self.delta_plus_1,
            self.delta_plus_10,
        ), spacing=4)

        mark_group = self._build_control_group((
            self.mark_start_btn,
            self.mark_start_label,
            self.mark_end_btn,
            self.mark_end_label,
        ), spacing=2)

        export_group = self._build_control_group((self.export_log_btn,), spacing=0)

        control_row.addWidget(play_group, 0)
        control_row.addWidget(mark_group, 0)
        control_row.addWidget(export_group, 0)
        layout.addLayout(control_row)
        return panel

    def _build_sidebar(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        panel = TechPanel()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(18)

        nav_group = QtWidgets.QGroupBox("当前试次")
        nav_layout = QtWidgets.QVBoxLayout(nav_group)
        nav_layout.setContentsMargins(14, 6, 14, 14)
        nav_layout.setSpacing(6)
        self.nav_trial_label = QtWidgets.QLabel("未加载")
        self.nav_trial_label.setWordWrap(True)
        self.nav_trial_label.setProperty("metricValue", True)
        nav_layout.addWidget(self.nav_trial_label)
        panel_layout.addWidget(nav_group)

        option_group = QtWidgets.QGroupBox("运行设置")
        option_layout = QtWidgets.QGridLayout(option_group)
        option_layout.setContentsMargins(14, 14, 14, 14)
        option_layout.setHorizontalSpacing(14)
        option_layout.setVerticalSpacing(14)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["auto", "frames", "mp4"])
        self.source_combo.setProperty("settingControl", True)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["zup", "raw"])
        self.axis_combo.setProperty("settingControl", True)
        self.lite_checkbox = QtWidgets.QCheckBox("轻量模式（低配机器推荐）")
        option_layout.addWidget(QtWidgets.QLabel("视觉源模式"), 0, 0)
        option_layout.addWidget(self.source_combo, 0, 1)
        option_layout.addWidget(QtWidgets.QLabel("骨架坐标"), 1, 0)
        option_layout.addWidget(self.axis_combo, 1, 1)
        self.lite_checkbox.setProperty("settingToggle", True)
        option_layout.addWidget(self.lite_checkbox, 2, 0, 1, 2)
        panel_layout.addWidget(option_group)

        info_group = QtWidgets.QGroupBox("当前状态")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(14, 14, 14, 14)
        info_layout.setSpacing(14)
        self.visual_row = InfoRow("视觉输入")
        self.fps_row = InfoRow("参考帧率")
        self.delta_row = InfoRow("当前偏移")
        self.frame_row = InfoRow("当前位置")
        for row in (
            self.visual_row,
            self.fps_row,
            self.delta_row,
            self.frame_row,
        ):
            info_layout.addWidget(row)
        panel_layout.addWidget(info_group)

        log_group = QtWidgets.QGroupBox("操作日志")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(14, 14, 14, 14)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMinimumHeight(220)
        self.log_view.setProperty("console", True)
        log_font = QtGui.QFont("Microsoft YaHei UI", 10)
        log_font.setBold(True)
        self.log_view.setFont(log_font)
        log_layout.addWidget(self.log_view)
        panel_layout.addWidget(log_group, 1)

        layout.addWidget(panel)
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _connect_signals(self) -> None:
        self.pick_cam_root_button.clicked.connect(self._choose_cam_root)
        self.pick_mocap_root_button.clicked.connect(self._choose_mocap_root)
        self.load_button.clicked.connect(self._reload_by_context)
        self.auto_button.clicked.connect(self.auto_align)
        self.export_button.clicked.connect(self.export_current_result)

        self.prev_button.clicked.connect(self.navigate_prev)
        self.next_button.clicked.connect(self.navigate_next)

        self.visual_slider.valueChanged.connect(self._on_visual_slider_changed)
        self.visual_slider.sliderPressed.connect(lambda: self._set_drag_state("visual", True))
        self.visual_slider.sliderReleased.connect(lambda: self._set_drag_state("visual", False))

        self.mocap_slider.valueChanged.connect(self._on_mocap_slider_changed)
        self.mocap_slider.sliderPressed.connect(lambda: self._set_drag_state("mocap", True))
        self.mocap_slider.sliderReleased.connect(lambda: self._set_drag_state("mocap", False))

        self.delta_slider.valueChanged.connect(self._on_delta_slider_changed)
        self.delta_slider.sliderPressed.connect(lambda: self._set_drag_state("delta", True))
        self.delta_slider.sliderReleased.connect(lambda: self._set_drag_state("delta", False))

        self.play_button.toggled.connect(self._toggle_playback)
        self.visual_prev.clicked.connect(lambda: self.shift_visual_frames(-1))
        self.visual_next.clicked.connect(lambda: self.shift_visual_frames(1))
        self.delta_minus_10.clicked.connect(lambda: self.shift_delta_frames(-10))
        self.delta_minus_1.clicked.connect(lambda: self.shift_delta_frames(-1))
        self.delta_plus_1.clicked.connect(lambda: self.shift_delta_frames(1))
        self.delta_plus_10.clicked.connect(lambda: self.shift_delta_frames(10))
        self.mark_start_btn.clicked.connect(self._record_mark_start)
        self.mark_end_btn.clicked.connect(self._record_mark_end)
        self.export_log_btn.clicked.connect(self._export_clip_log)

        self.energy_canvas.clicked.connect(self._on_plot_clicked)

        if self.combined_check is not None:
            self.combined_check.toggled.connect(self._on_camera_selection_changed)
        for checkbox in self.camera_checks.values():
            checkbox.toggled.connect(self._on_camera_selection_changed)
        if self.overlay_toggle_button is not None:
            self.overlay_toggle_button.toggled.connect(self._toggle_skeleton_overlay)
        self.axis_combo.currentTextChanged.connect(self._on_axis_preset_changed)
        self.lite_checkbox.toggled.connect(self._on_lite_mode_changed)

    def _apply_launch_options(self) -> None:
        self.source_combo.setCurrentText(self.options.source_mode)
        self.axis_combo.setCurrentText(self.options.axis_preset)
        self.lite_checkbox.setChecked(self.options.lite_mode)
        self._update_root_button_state()
        self._update_status_texts()

    def _on_axis_preset_changed(self, axis_preset: str) -> None:
        if self.session is None:
            return
        self.session.runtime_options.axis_preset = axis_preset
        self.skeleton_canvas.clear_position_cache()
        self._refresh_view(force_heavy=True)
        self.log(f"骨架坐标显示已切换为：{axis_preset}")

    def _on_lite_mode_changed(self, checked: bool) -> None:
        if self.session is None:
            return
        self.session.runtime_options.lite_mode = checked
        self.session.runtime_options.preview_scale = LITE_PREVIEW_SCALE if checked else DEFAULT_PREVIEW_SCALE
        self.session.runtime_options.defer_heavy_refresh = True
        self._refresh_view(force_heavy=True)
        self.log("轻量模式已开启。" if checked else "轻量模式已关闭。")

    def _reload_by_context(self) -> None:
        if self._use_explicit_paths:
            self.load_explicit_selection()
        else:
            self.load_current_selection()

    def _update_root_button_state(self) -> None:
        button_pairs = (
            (self.pick_cam_root_button, "相机根目录", self.cam_root),
            (self.pick_mocap_root_button, "动捕根目录", self.mocap_root),
        )
        manual_tip = "当前为手动路径模式，请重新以试次模式启动后再选择根目录。"
        for button, label, path in button_pairs:
            button.setEnabled(not self._use_explicit_paths)
            if self._use_explicit_paths:
                button.setToolTip(manual_tip)
                continue
            button.setToolTip(f"{label}：{path}")

    def _reset_session_view(self) -> None:
        self.stop_playback()
        self._release_session()
        self._clear_marks(update_view=False)
        self.state.preview_time = 0.0
        self.state.delta_t = 0.0
        self.state.auto_delta_t = 0.0
        self.state.auto_confidence = 0.0
        self.state.status_message = ""
        self._configure_sliders()
        self._reset_camera_selection()
        self._refresh_view(force_heavy=True)

    def _rebuild_trial_list(self, auto_load: bool) -> None:
        if self._use_explicit_paths:
            return

        self._trial_list = enumerate_trials(self.mocap_root, self.cam_root)
        if self._trial_list:
            self._trial_index = min(self._trial_index, len(self._trial_list) - 1)
            self._update_trial_display()
            if auto_load:
                self.load_current_selection()
            return

        self._trial_index = 0
        self._reset_session_view()
        self._update_trial_display()

    def _choose_cam_root(self) -> None:
        if self._use_explicit_paths:
            return

        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择相机根目录",
            str(self.cam_root),
        )
        if not selected:
            return

        next_root = Path(selected)
        if next_root == self.cam_root:
            return

        self.cam_root = next_root
        self.log(f"相机根目录已切换：{self.cam_root}")
        self._update_root_button_state()
        if self._trial_list:
            self.load_current_selection()
        else:
            self._update_trial_display()

    def _choose_mocap_root(self) -> None:
        if self._use_explicit_paths:
            return

        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择动捕根目录",
            str(self.mocap_root),
        )
        if not selected:
            return

        next_root = Path(selected)
        if next_root == self.mocap_root:
            return

        self.mocap_root = next_root
        self.log(f"动捕根目录已切换：{self.mocap_root}")
        self._update_root_button_state()
        self._rebuild_trial_list(auto_load=True)

    def _update_trial_display(self) -> None:
        if self._use_explicit_paths:
            manual_name = self.options.camera_session.name if self.options.camera_session is not None else "手动选择"
            self.trial_info_label.setText(f"手动模式 | {manual_name}")
            self.nav_trial_label.setText(
                "当前使用手动传入的路径。\n"
                "相机会话、位置 BVH 和顺序 BVH 会按你提供的内容分别加载。"
            )
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self._update_root_button_state()
            return

        if not self._trial_list:
            self.trial_info_label.setText("未找到试次数据")
            self.nav_trial_label.setText(
                "未找到任何可加载的试次，请在右上角重新选择目录。\n"
                f"动捕根目录：{self.mocap_root}\n"
                f"相机根目录：{self.cam_root}"
            )
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self._update_root_button_state()
            return

        trial = self._trial_list[self._trial_index]
        total = len(self._trial_list)
        self.trial_info_label.setText(f"[{self._trial_index + 1}/{total}] {trial.display_name}")
        self.nav_trial_label.setText(
            f"试次序号：{self._trial_index + 1} / {total}\n"
            f"实验对象：{trial.subject}\n"
            f"动作编号：{trial.action:02d}\n"
            f"重复次数：第 {trial.rep} 次\n"
            f"动捕文件夹：{trial.mocap_folder_name}\n"
            f"相机会话标识：{trial.cam_session_suffix}"
        )
        self.prev_button.setEnabled(self._trial_index > 0)
        self.next_button.setEnabled(self._trial_index < total - 1)
        self._update_root_button_state()

    def navigate_prev(self) -> None:
        if self._trial_index > 0:
            self._trial_index -= 1
            self._update_trial_display()
            self.load_current_selection()

    def navigate_next(self) -> None:
        if self._trial_index < len(self._trial_list) - 1:
            self._trial_index += 1
            self._update_trial_display()
            self.load_current_selection()

    def log(self, message: str) -> None:
        if not message:
            return
        timestamp = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _set_state_badge(self, text: str, level: str) -> None:
        self.state_badge.setText(text)
        self.state_badge.setProperty("level", level)
        repolish(self.state_badge)

    def _set_play_button_state(self, playing: bool) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.blockSignals(False)
        self.play_button.setText("暂停播放" if playing else "开始播放")
        self.play_button.setProperty("playing", playing)
        repolish(self.play_button)

    def _toggle_playback(self, checked: bool) -> None:
        if checked:
            self.start_playback()
        else:
            self.stop_playback()

    def _playback_fps(self) -> float:
        if self.visual_slider.isEnabled():
            return self._visual_fps()
        if self.mocap_slider.isEnabled():
            return self._mocap_fps()
        return 0.0

    def start_playback(self) -> None:
        if self.session is None:
            self._set_play_button_state(False)
            return
        fps = self._playback_fps()
        if fps <= 0:
            self._set_play_button_state(False)
            return
        interval = max(20, int(round(1000.0 / fps)))
        self.play_timer.start(interval)
        self._set_play_button_state(True)

    def stop_playback(self) -> None:
        self.play_timer.stop()
        self._set_play_button_state(False)

    def _advance_playback(self) -> None:
        if self.visual_slider.isEnabled():
            next_value = self.visual_slider.value() + 1
            if next_value > self.visual_slider.maximum():
                self.stop_playback()
                return
            self.visual_slider.setValue(next_value)
            return

        if self.mocap_slider.isEnabled():
            next_value = self.mocap_slider.value() + 1
            if next_value > self.mocap_slider.maximum():
                self.stop_playback()
                return
            self.mocap_slider.setValue(next_value)
            return

        self.stop_playback()

    def _release_session(self) -> None:
        close_session_data(self.session)
        self.session = None
        self._update_skeleton_overlay_state()

    def _clear_marks(self, update_view: bool = True) -> None:
        self._mark_start = None
        self._mark_end = None
        self.mark_start_label.setText("起点：未记录")
        self.mark_end_label.setText("终点：未记录")
        if update_view:
            self._refresh_view(force_heavy=False)

    def load_explicit_selection(self) -> None:
        self.stop_playback()
        self._set_state_badge("加载中", "warning")
        self.log("正在加载手动指定的数据路径。")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            session = load_session_from_paths(
                cam_session=self.options.camera_session,
                position_bvh_path=self.options.position_bvh,
                order_bvh_path=self.options.order_bvh,
                source_mode=self.source_combo.currentText(),
                lite_mode=self.lite_checkbox.isChecked(),
                axis_preset=self.axis_combo.currentText(),
            )
        except Exception as exc:
            self._set_state_badge("加载失败", "danger")
            self.log(f"手动路径加载失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._release_session()
        self.session = session
        self._update_skeleton_overlay_state()
        self._clear_marks(update_view=False)
        self._reset_camera_selection()
        self._configure_sliders()
        self.state.preview_time = 0.0
        self._report_bvh_decision()
        self.auto_align()
        self.log("手动指定数据已加载完成。")

    def load_current_selection(self) -> None:
        if not self._trial_list:
            QtWidgets.QMessageBox.warning(self, "无可用试次", "未找到任何试次数据，请先在右上角选择动捕目录和相机目录。")
            return

        trial = self._trial_list[self._trial_index]
        self.stop_playback()
        self._set_state_badge("加载中", "warning")
        self.log(f"正在加载试次：{trial.display_name}（{trial.mocap_folder_name}）")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            session = load_trial(
                trial,
                cam_root=self.cam_root,
                mocap_root=self.mocap_root,
                source_mode=self.source_combo.currentText(),
                lite_mode=self.lite_checkbox.isChecked(),
                axis_preset=self.axis_combo.currentText(),
            )
        except Exception as exc:
            self._set_state_badge("加载失败", "danger")
            self.log(f"加载失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._release_session()
        self.session = session
        self._update_skeleton_overlay_state()
        self._clear_marks(update_view=False)
        self._reset_camera_selection()
        self._configure_sliders()
        self.state.preview_time = 0.0
        self._report_bvh_decision()
        self.auto_align()
        self.log("试次数据加载完成。")

    def _report_bvh_decision(self) -> None:
        if self.session is None:
            return

        for index, bvh_source in enumerate(self.session.bvh_sources, start=1):
            self.log(
                f"BVH 文件 {index}：{bvh_source.display_name} | "
                f"角色 {bvh_source.role} | "
                f"关节运动 {'有' if bvh_source.has_joint_motion else '无'} | "
                f"运动评分 {bvh_source.motion_score:.4f}"
            )
        if self.session.display_bvh is not None:
            self.log(f"骨架预览实际使用：{self.session.display_bvh.display_name}")
        if self.session.alignment_bvh is not None:
            self.log(f"自动对齐实际使用：{self.session.alignment_bvh.display_name}")

    def _reset_camera_selection(self) -> None:
        available = set(self.session.available_camera_labels) if self.session is not None else set()
        self._selected_camera_labels = tuple(sorted(available))
        if self.combined_check is not None:
            self.combined_check.blockSignals(True)
            self.combined_check.setEnabled(self.session is not None and self.session.has_visual)
            self.combined_check.setChecked(self.session is not None and self.session.has_visual)
            self.combined_check.blockSignals(False)
        for label, checkbox in self.camera_checks.items():
            checkbox.blockSignals(True)
            checkbox.setEnabled(label in available)
            checkbox.setChecked(label in available)
            checkbox.blockSignals(False)
        self._update_camera_hint_label()

    def _update_camera_hint_label(self) -> None:
        parts: list[str] = []
        if self.combined_check is not None and self.combined_check.isChecked():
            parts.append("综合")
        indexes = _join_camera_indexes(self._selected_camera_labels)
        if indexes != "未选":
            parts.append(indexes)
        self.camera_hint_label.setText("已选 " + (" / ".join(parts) if parts else "未选"))

    def _visual_fps(self) -> float:
        if self.session is None:
            return 40.0
        if self.session.has_visual:
            return self.session.reference_visual_fps
        return self._mocap_fps()

    def _mocap_fps(self) -> float:
        if self.session is None or self.session.display_bvh is None:
            return 120.0
        return self.session.display_bvh.motion.raw_fps

    def _delta_fps(self) -> float:
        if self.session is None:
            return 40.0
        if self.session.has_visual:
            return self.session.reference_visual_fps
        return self._mocap_fps()

    def _configure_sliders(self) -> None:
        visual_frames = 0
        mocap_frames = 0
        delta_frames = 0

        if self.session is not None:
            visual_frames = int(round(max(self.session.preview_duration, 0.0) * self._visual_fps()))
            if self.session.display_bvh is not None:
                mocap_frames = max(0, len(self.session.display_bvh.motion.raw_frames) - 1)
            delta_frames = int(round(max(self._max_shift_seconds(), 1.0) * self._delta_fps()))

        self._set_slider_without_events(self.visual_slider, 0, visual_frames, 0)
        self._set_slider_without_events(self.mocap_slider, 0, mocap_frames, 0)
        self._set_slider_without_events(self.delta_slider, -delta_frames, delta_frames, 0)
        self.visual_slider.setEnabled(self.session is not None and self.session.has_visual)
        self.mocap_slider.setEnabled(self.session is not None and self.session.display_bvh is not None)
        self.delta_slider.setEnabled(self.session is not None and self.session.has_bvh)

    def _set_slider_without_events(self, slider: QtWidgets.QSlider, minimum: int, maximum: int, value: int) -> None:
        slider.blockSignals(True)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.blockSignals(False)

    def _max_shift_seconds(self) -> float:
        if self.session is None:
            return 0.0
        return max(self.session.camera_max_duration, self.session.bvh_raw_duration, 1.0)

    def _set_drag_state(self, target: str, active: bool) -> None:
        if active:
            self.stop_playback()
        if target == "visual":
            self._dragging_visual = active
        elif target == "mocap":
            self._dragging_mocap = active
        else:
            self._dragging_delta = active

        if not active:
            self._refresh_view(force_heavy=True)

    def _on_visual_slider_changed(self, value: int) -> None:
        self.state.preview_time = quantize_time(value / self._visual_fps(), self._visual_fps())
        lightweight = self.session is not None and self.session.runtime_options.defer_heavy_refresh and self._dragging_visual
        self._refresh_view(force_heavy=not lightweight)

    def _on_mocap_slider_changed(self, value: int) -> None:
        if self.session is None or self.session.display_bvh is None:
            return
        mocap_time = value / self._mocap_fps()
        preview_time = mocap_time + self.state.delta_t
        self.state.preview_time = quantize_time(max(0.0, preview_time), self._visual_fps())
        lightweight = self.session.runtime_options.defer_heavy_refresh and self._dragging_mocap
        self._refresh_view(force_heavy=not lightweight)

    def _on_delta_slider_changed(self, value: int) -> None:
        self.state.delta_t = quantize_time(value / self._delta_fps(), self._delta_fps())
        lightweight = self.session is not None and self.session.runtime_options.defer_heavy_refresh and self._dragging_delta
        self._refresh_view(force_heavy=not lightweight)

    def _on_camera_selection_changed(self) -> None:
        self._selected_camera_labels = tuple(
            label for label, checkbox in sorted(self.camera_checks.items()) if checkbox.isEnabled() and checkbox.isChecked()
        )
        self._update_camera_hint_label()
        self._refresh_view(force_heavy=False)

    def _show_combined_energy(self) -> bool:
        return self.combined_check is not None and self.combined_check.isChecked()

    def _selected_cameras(self):
        if self.session is None:
            return []
        labels = set(self._selected_camera_labels)
        return [camera for camera in self.session.cameras if camera.label in labels]

    def _combined_energy_for_display(self) -> np.ndarray:
        if self.session is None:
            return np.zeros(0, dtype=np.float64)
        if not self.session.cameras:
            return np.zeros(0, dtype=np.float64)
        combined, _ = build_combined_camera_energy(self.session.cameras, self.session.reference_visual_fps)
        return combined

    def _combined_energy_for_alignment(self) -> np.ndarray:
        if self.session is None:
            return np.zeros(0, dtype=np.float64)
        selected = self._selected_cameras()
        if not selected:
            if self._show_combined_energy() and self.session.cameras:
                selected = list(self.session.cameras)
            else:
                return np.zeros(0, dtype=np.float64)
        combined, _ = build_combined_camera_energy(selected, self.session.reference_visual_fps)
        return combined

    def auto_align(self, refresh_only: bool = False) -> None:
        if self.session is None:
            return

        combined_energy = self._combined_energy_for_alignment()
        if not refresh_only:
            selected_text = _join_camera_display_names(self._selected_camera_labels)
            if not self._selected_camera_labels and self._show_combined_energy():
                selected_text = "综合"
            if len(combined_energy) and self.session.alignment_bvh is not None:
                auto_delta_t, confidence = estimate_initial_offset(
                    combined_energy,
                    self.session.reference_visual_fps,
                    self.session.alignment_bvh.energy_visual,
                    self.session.reference_visual_fps,
                )
                auto_delta_t = quantize_time(auto_delta_t, self._delta_fps())
                self.state.auto_delta_t = auto_delta_t
                self.state.auto_confidence = confidence
                self.state.delta_t = auto_delta_t
                self._set_slider_without_events(
                    self.delta_slider,
                    self.delta_slider.minimum(),
                    self.delta_slider.maximum(),
                    int(round(auto_delta_t * self._delta_fps())),
                )
                self.state.status_message = f"已按 {selected_text} 自动估计偏移"
            elif self.session.alignment_bvh is None:
                self.state.auto_delta_t = 0.0
                self.state.auto_confidence = 0.0
                self.state.delta_t = 0.0
                self.state.status_message = "当前缺少可用于自动对齐的动捕曲线"
                self._set_slider_without_events(
                    self.delta_slider,
                    self.delta_slider.minimum(),
                    self.delta_slider.maximum(),
                    0,
                )
            else:
                self.state.auto_delta_t = 0.0
                self.state.auto_confidence = 0.0
                self.state.delta_t = 0.0
                self.state.status_message = "当前未勾选任何相机，无法自动对齐"
                self._set_slider_without_events(
                    self.delta_slider,
                    self.delta_slider.minimum(),
                    self.delta_slider.maximum(),
                    0,
                )
        self._refresh_view(force_heavy=True)

    def shift_delta_frames(self, frame_count: int) -> None:
        if self.session is None:
            return
        next_value = self.delta_slider.value() + frame_count
        next_value = max(self.delta_slider.minimum(), min(self.delta_slider.maximum(), next_value))
        self.delta_slider.setValue(next_value)

    def shift_visual_frames(self, frame_count: int) -> None:
        self.stop_playback()
        if self.session is None:
            return
        if self.visual_slider.isEnabled():
            next_value = self.visual_slider.value() + frame_count
            next_value = max(self.visual_slider.minimum(), min(self.visual_slider.maximum(), next_value))
            self.visual_slider.setValue(next_value)
            return
        if self.mocap_slider.isEnabled():
            next_value = self.mocap_slider.value() + frame_count
            next_value = max(self.mocap_slider.minimum(), min(self.mocap_slider.maximum(), next_value))
            self.mocap_slider.setValue(next_value)

    def _on_plot_clicked(self, x_value: float) -> None:
        if self.session is None:
            return
        self.stop_playback()
        limit = max(self.session.preview_duration, self.session.bvh_visual_duration)
        x_value = max(0.0, min(x_value, limit))
        self.state.preview_time = quantize_time(x_value, self._visual_fps())
        self._refresh_view(force_heavy=True)

    def _current_mark_info(self) -> dict | None:
        if self.session is None:
            return None
        visual_time = self.state.preview_time
        visual_frame = int(round(visual_time * self._visual_fps()))
        mocap_time = max(0.0, visual_time - self.state.delta_t)
        mocap_frame = int(round(mocap_time * self._mocap_fps()))
        return {
            "visual_time": visual_time,
            "visual_frame": visual_frame,
            "mocap_time": mocap_time,
            "mocap_frame": mocap_frame,
            "delta_t": self.state.delta_t,
        }

    def _record_mark_start(self) -> None:
        info = self._current_mark_info()
        if info is None:
            QtWidgets.QMessageBox.warning(self, "无法记录", "请先加载数据后再记录起点。")
            return
        self._mark_start = info
        self.mark_start_label.setText(
            f"起点：视 {info['visual_time']:.3f}s 帧{info['visual_frame']} | "
            f"动 {info['mocap_time']:.3f}s 帧{info['mocap_frame']}"
        )
        self.log(
            f"[起点] 视觉帧 {info['visual_frame']}（{info['visual_time']:.3f}s），"
            f"动捕帧 {info['mocap_frame']}（{info['mocap_time']:.3f}s）"
        )
        self._refresh_view(force_heavy=False)

    def _record_mark_end(self) -> None:
        info = self._current_mark_info()
        if info is None:
            QtWidgets.QMessageBox.warning(self, "无法记录", "请先加载数据后再记录终点。")
            return
        self._mark_end = info
        self.mark_end_label.setText(
            f"终点：视 {info['visual_time']:.3f}s 帧{info['visual_frame']} | "
            f"动 {info['mocap_time']:.3f}s 帧{info['mocap_frame']}"
        )
        self.log(
            f"[终点] 视觉帧 {info['visual_frame']}（{info['visual_time']:.3f}s），"
            f"动捕帧 {info['mocap_frame']}（{info['mocap_time']:.3f}s）"
        )
        self._refresh_view(force_heavy=False)

    def _export_clip_log(self) -> None:
        if self._mark_start is None and self._mark_end is None:
            QtWidgets.QMessageBox.warning(self, "无法导出", "请至少先记录一个起点或终点。")
            return

        results_dir = DEFAULT_OUTPUT_ROOT / "results_csv"
        results_dir.mkdir(parents=True, exist_ok=True)

        trial_code = "unknown"
        if self.session is not None:
            match = re.match(r"S(\d+)_(\d+)", self.session.session_id)
            if match:
                trial_code = f"S{match.group(1)}{match.group(2)}"
            else:
                trial_code = self.session.session_id

        default_path = str(results_dir / f"{trial_code}.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存裁剪记录",
            default_path,
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return

        cam_paths: dict[str, str] = {}
        if self.session is not None:
            for camera in sorted(self.session.cameras, key=lambda item: item.label):
                paths = camera.input_paths()
                if camera.source_kind == "frames" and paths:
                    cam_paths[camera.label] = str(paths[0].parent)
                else:
                    cam_paths[camera.label] = str(paths[0]) if paths else ""

        mocap_avi = ""
        if self.session is not None and self.session.mocap_subject is not None:
            avis = sorted(self.session.mocap_subject.glob("*.avi"))
            if avis:
                mocap_avi = str(avis[0])

        bvh_position = ""
        bvh_order = ""
        bvh_all = ""
        if self.session is not None:
            if self.session.position_bvh is not None:
                bvh_position = str(self.session.position_bvh.motion.path)
            if self.session.order_bvh is not None:
                bvh_order = str(self.session.order_bvh.motion.path)
            bvh_all = ";".join(str(source.motion.path) for source in self.session.bvh_sources)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cam_labels = sorted(cam_paths.keys())
        header = [
            "标记",
            "视觉时间(s)",
            "视觉帧",
            "动捕时间(s)",
            "动捕帧",
            "偏移量(s)",
        ] + [f"{label}_路径" for label in cam_labels] + [
            "mocap_avi路径",
            "bvh_position路径",
            "bvh_order路径",
            "bvh_全部路径",
            "会话编号",
            "导出时间",
        ]

        rows = [header]
        for tag, info in [("start", self._mark_start), ("end", self._mark_end)]:
            if info is None:
                row = [tag, "", "", "", "", ""]
                row += [""] * len(cam_labels)
                row += [mocap_avi, bvh_position, bvh_order, bvh_all, trial_code, timestamp]
            else:
                row = [
                    tag,
                    f"{info['visual_time']:.6f}",
                    info["visual_frame"],
                    f"{info['mocap_time']:.6f}",
                    info["mocap_frame"],
                    f"{info['delta_t']:.6f}",
                ]
                row += [cam_paths.get(label, "") for label in cam_labels]
                row += [mocap_avi, bvh_position, bvh_order, bvh_all, trial_code, timestamp]
            rows.append(row)

        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

        self.log(f"裁剪记录已导出：{path}")
        QtWidgets.QMessageBox.information(self, "导出完成", f"裁剪记录已保存到：\n{path}")

    def export_current_result(self) -> None:
        if self.session is None or not self.session.has_bvh:
            QtWidgets.QMessageBox.warning(self, "无法导出", "当前没有可导出的动捕数据。")
            return

        try:
            outputs = export_alignment_bundle(self.session, self.state.delta_t, figure=self.energy_canvas.figure)
        except Exception as exc:
            self.log(f"导出失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))
            return

        lines = [f"{key}: {path}" for key, path in outputs.items()]
        self.log("当前对齐结果已导出。")
        QtWidgets.QMessageBox.information(self, "导出完成", "\n".join(lines))

    def _refresh_camera_tiles(self) -> None:
        if self.session is None:
            for label, tile in self.camera_tiles.items():
                tile.set_identity(_camera_display_name(label))
                tile.set_meta("暂无图像")
                tile.set_status("未加载", "warning")
                tile.set_pixmap(None)
            return

        camera_map = {camera.label: camera for camera in self.session.cameras}
        for label, tile in self.camera_tiles.items():
            display_name = _camera_display_name(label)
            camera = camera_map.get(label)
            if camera is None:
                tile.set_identity(display_name)
                tile.set_meta("未参与加载")
                tile.set_status("缺失", "warning")
                tile.set_pixmap(None)
                continue

            frame_index = preview_time_to_camera_frame(camera, self.state.preview_time)
            frame = camera.get_frame(frame_index, scale=self.session.runtime_options.preview_scale)
            tile.set_identity(display_name)
            tile.set_meta(f"{camera.source_kind} | {camera.fps:.0f}fps | 帧 {frame_index}")
            tile.set_status("在线", "normal")
            tile.set_pixmap(rgb_array_to_qpixmap(frame))

    def _sync_timeline_sliders(self) -> None:
        if self.session is None:
            self.visual_label.setText("视觉进度：未加载")
            self.mocap_label.setText("动捕进度：未加载")
            self.delta_label.setText("对齐偏移：+0.000 秒")
            return

        visual_frame = int(round(self.state.preview_time * self._visual_fps()))
        visual_frame = max(self.visual_slider.minimum(), min(self.visual_slider.maximum(), visual_frame))
        self._set_slider_without_events(self.visual_slider, self.visual_slider.minimum(), self.visual_slider.maximum(), visual_frame)

        if self.session.display_bvh is not None:
            mocap_time = max(0.0, self.state.preview_time - self.state.delta_t)
            mocap_frame = int(round(mocap_time * self._mocap_fps()))
            mocap_frame = max(self.mocap_slider.minimum(), min(self.mocap_slider.maximum(), mocap_frame))
            self._set_slider_without_events(self.mocap_slider, self.mocap_slider.minimum(), self.mocap_slider.maximum(), mocap_frame)
            total_mocap = max(self.mocap_slider.maximum(), 0)
            self.mocap_label.setText(f"动捕进度：{mocap_time:.3f}s | 帧 {mocap_frame} / {total_mocap}")
        else:
            self.mocap_label.setText("动捕进度：未加载")

        if self.session.has_visual:
            total_visual = max(self.visual_slider.maximum(), 0)
            self.visual_label.setText(f"视觉进度：{self.state.preview_time:.3f}s | 帧 {visual_frame} / {total_visual}")
        else:
            self.visual_label.setText("视觉进度：未加载")

        self.delta_label.setText(
            f"对齐偏移：{self.state.delta_t:+.3f} 秒 | 自动估计 {self.state.auto_delta_t:+.3f} 秒 | 置信度 {self.state.auto_confidence:.3f}"
        )

    def _refresh_view(self, force_heavy: bool) -> None:
        if self.session is None:
            self.energy_canvas.render(None, 0.0, 0.0, np.zeros(0, dtype=np.float64), False, (), None, None)
            self.skeleton_canvas.render(None, 0.0, 0.0)
            self._refresh_skeleton_overlay()
            self._refresh_camera_tiles()
            self._sync_timeline_sliders()
            self._update_status_texts()
            return

        combined_energy = self._combined_energy_for_display()
        self.energy_canvas.render(
            self.session,
            self.state.preview_time,
            self.state.delta_t,
            combined_energy,
            self._show_combined_energy(),
            self._selected_camera_labels,
            self._mark_start["visual_time"] if self._mark_start is not None else None,
            self._mark_end["visual_time"] if self._mark_end is not None else None,
        )
        if force_heavy:
            self._refresh_camera_tiles()
            self.skeleton_canvas.render(self.session, self.state.preview_time, self.state.delta_t)
            self._refresh_skeleton_overlay()
        self._sync_timeline_sliders()
        self._update_status_texts()

    def _update_status_texts(self) -> None:
        if self.session is None:
            self._set_state_badge("未加载", "warning")
            self.visual_row.set_value("未提供")
            self.fps_row.set_value("--")
            self.delta_row.set_value("--")
            self.frame_row.set_value("--")
            self.skeleton_info_label.setText("当前骨架：未加载")
            return

        if self.session.has_visual and self.session.has_bvh:
            badge_text = "视觉与动捕已就绪"
            level = "normal"
        elif self.session.has_visual:
            badge_text = "仅加载视觉数据"
            level = "warning"
        else:
            badge_text = "仅加载动捕数据"
            level = "warning"
        self._set_state_badge(badge_text, level)

        camera_frame_text = "未加载"
        if self.session.cameras:
            first_camera = self.session.cameras[0]
            camera_frame_text = str(preview_time_to_camera_frame(first_camera, self.state.preview_time))

        display_frame_text = "未加载"
        if self.session.display_bvh is not None:
            display_frame_text = str(
                preview_time_to_bvh_frame(self.session, self.state.preview_time, self.state.delta_t, self.session.display_bvh)
            )

        if self.session.cameras:
            camera_fps_desc = " / ".join(
                f"{camera.label.replace('cam', '')}={camera.fps:.2f}"
                for camera in sorted(self.session.cameras, key=lambda item: item.label)
            )
            visual_desc = (
                f"来源：{self.session.source_mode} | "
                f"参考帧率：{self._visual_fps():.2f} fps | "
                f"各相机实际 FPS：{camera_fps_desc} | "
                f"已接入：{_join_camera_display_names(list(self.session.available_camera_labels))}"
            )
        else:
            visual_desc = "未加载视觉输入，仅可浏览动捕时间轴。"

        self.visual_row.set_value(visual_desc)
        if self.session.display_bvh is not None:
            self.skeleton_info_label.setText(
                f"当前骨架：{self.session.display_bvh.display_name} | 动捕帧 {display_frame_text} | 偏移 {self.state.delta_t:+.3f} 秒"
            )
        else:
            self.skeleton_info_label.setText("当前骨架：未提供")
        self.fps_row.set_value(f"视觉参考 {self._visual_fps():.2f} fps | 动捕原始 {self._mocap_fps():.2f} fps")
        self.delta_row.set_value(
            f"当前偏移 {self.state.delta_t:+.3f} 秒 | 自动估计 {self.state.auto_delta_t:+.3f} 秒 | "
            f"置信度 {self.state.auto_confidence:.3f} | {self.state.status_message or '等待调整'}"
        )
        self.frame_row.set_value(
            f"视觉帧 {camera_frame_text} | 动捕帧 {display_frame_text} | "
            f"当前显示相机：{_join_camera_display_names(self._selected_camera_labels)}"
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.camera_stage and event.type() == QtCore.QEvent.Resize and self.skeleton_overlay is not None:
            self.skeleton_overlay.clamp_to_parent()
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.session is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()
        step = 1
        if modifiers & QtCore.Qt.ShiftModifier:
            step = 10
        elif modifiers & QtCore.Qt.ControlModifier:
            step = 5

        if key == QtCore.Qt.Key_Space:
            if self.play_timer.isActive():
                self.stop_playback()
            else:
                self.start_playback()
            return
        if key == QtCore.Qt.Key_Left:
            self.shift_delta_frames(-step)
            return
        if key == QtCore.Qt.Key_Right:
            self.shift_delta_frames(step)
            return
        if key == QtCore.Qt.Key_Up:
            self.shift_visual_frames(step)
            return
        if key == QtCore.Qt.Key_Down:
            self.shift_visual_frames(-step)
            return
        if key == QtCore.Qt.Key_Home:
            self.auto_align()
            return
        if key == QtCore.Qt.Key_1:
            self._record_mark_start()
            return
        if key == QtCore.Qt.Key_2:
            self._record_mark_end()
            return
        if key in {QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter}:
            self.export_current_result()
            return
        if key == QtCore.Qt.Key_PageUp:
            self.navigate_prev()
            return
        if key == QtCore.Qt.Key_PageDown:
            self.navigate_next()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._pending_initial_layout:
            self._pending_initial_layout = False
            QtCore.QTimer.singleShot(0, self.apply_default_window_layout)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_playback()
        self._release_session()
        super().closeEvent(event)


def run_application(options: LaunchOptions) -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
        app.setApplicationName(APP_TITLE)
    app.setPalette(create_palette())
    app.setStyleSheet(app_stylesheet())

    window = MainWindow(options)
    window.showMaximized()
    return app.exec_() if owns_app else 0


def main(options: LaunchOptions) -> None:
    try:
        raise SystemExit(run_application(options))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
