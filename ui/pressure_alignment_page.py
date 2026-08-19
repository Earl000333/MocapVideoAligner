from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable
import csv
import re

import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from models import BVHMotion, SessionData
from ui.widgets import InfoRow, TechPanel, repolish
from config import DEFAULT_OUTPUT_ROOT, DEFAULT_RECONSTRUCTED_TACTILE_ROOT, DEFAULT_VISUAL_ALIGN_REVIEW_ROOT
from utils.bvh_parser import load_bvh_motion_preserve_frames
from utils.bvh_pose import compute_joint_positions, transform_display_positions
from utils.pressure_alignment import (
    MocapFootCurveSet,
    PressureAlignmentResult,
    PressureCurveSet,
    PressureMetaInfo,
    PressureSensorFrameSet,
    build_pressure_curve_set,
    estimate_pressure_alignment,
    load_mocap_foot_curves,
    load_pressure_curve_csv,
    load_pressure_meta,
    load_pressure_sensor_csv,
    load_reconstructed_pressure_sensors,
    load_visual_alignment_offset,
    resolve_reconstructed_rec_dir,
    resolve_visual_alignment_csv,
)
from utils.pressure_dynamics_alignment import DynamicsVgrfCurveSet, estimate_pressure_alignment_dynamics


_FOOT_LAYOUT_DIR = Path(__file__).resolve().parent.parent / "add" / "Gait-Data-Collector" / "foot_sensor_layout"


def _load_foot_layout(sensor_id: str) -> tuple[np.ndarray, np.ndarray]:
    foot = "leftfoot" if "left" in sensor_id else "rightfoot"
    outline = np.genfromtxt(_FOOT_LAYOUT_DIR / f"{foot}_curve.csv", delimiter=",", skip_header=1, usecols=(1, 2)).astype(np.float64)
    dots = np.genfromtxt(_FOOT_LAYOUT_DIR / f"{foot}_dots.csv", delimiter=",", skip_header=1, usecols=(1, 2)).astype(np.float64)
    outline[:, 0] = 1.0 - outline[:, 0]
    outline[:, 1] = 1.0 - outline[:, 1]
    dots[:, 0] = 1.0 - dots[:, 0]
    dots[:, 1] = 1.0 - dots[:, 1]
    return outline, dots


def _is_foot_joint(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("foot", "toe", "heel", "ankle", "ball"))


def _sample_pressure_values(
    data: PressureSensorFrameSet | None,
    preview_time: float,
) -> tuple[np.ndarray | None, bool | None]:
    """Return interpolated sensor values and nearest-frame valid flag."""
    if data is None or len(data.time_s) == 0 or len(data.sensor_values) == 0:
        return None, None
    if len(data.time_s) == 1:
        return data.sensor_values[0], data.is_valid_at(0)

    # Nearest frame for validity badge; interpolate values for smooth heatmap.
    nearest = int(np.argmin(np.abs(np.asarray(data.time_s, dtype=np.float64) - float(preview_time))))
    nearest = max(0, min(nearest, len(data.time_s) - 1))
    valid = data.is_valid_at(nearest)

    index = int(np.searchsorted(data.time_s, preview_time, side="right") - 1)
    index = max(0, min(index, len(data.time_s) - 2))
    start_time = float(data.time_s[index])
    end_time = float(data.time_s[index + 1])
    if end_time <= start_time:
        return data.sensor_values[index], valid
    alpha = np.clip((preview_time - start_time) / (end_time - start_time), 0.0, 1.0)
    values = data.sensor_values[index] * (1.0 - alpha) + data.sensor_values[index + 1] * alpha
    return values, valid


class FootPressureCanvas(FigureCanvas):
    def __init__(self, sensor_id: str, parent=None) -> None:
        self.figure = Figure(figsize=(4.2, 5.0), facecolor="#F3E6D7")
        super().__init__(self.figure)
        self.setParent(parent)
        self.sensor_id = sensor_id
        self._outline_raw, self._sensor_coords_raw = _load_foot_layout(sensor_id)
        self._outline = self._outline_raw.copy()
        self._sensor_coords = self._sensor_coords_raw.copy()
        self._values = np.zeros(len(self._sensor_coords), dtype=np.float64)
        self._frame_valid: bool | None = None
        self._ax = self.figure.add_subplot(111)
        self._ax.set_facecolor("#F7EEE4")
        self._ax.set_aspect("equal")
        self._ax.axis("off")
        self.setMinimumSize(240, 300)
        self._render_empty(active=False)

    def set_values(self, values: np.ndarray | None, valid: bool | None = None) -> None:
        self._frame_valid = valid
        if values is None or len(values) == 0:
            self._values = np.zeros(len(self._sensor_coords), dtype=np.float64)
            self._render_empty(active=False)
            return
        raw_vals = np.asarray(values, dtype=np.float64)
        self._values = np.zeros(len(self._sensor_coords), dtype=np.float64)
        k = min(len(raw_vals), len(self._values))
        if k:
            self._values[:k] = np.clip(raw_vals[:k], 0.0, 1.0)
        self._render_active()

    def _reset_canvas(self) -> None:
        self._ax.cla()
        self._ax.set_facecolor("#F7EEE4")
        self._ax.set_aspect("equal")
        self._ax.axis("off")
        xs, ys = self._outline[:, 0], self._outline[:, 1]
        margin = 0.02
        self._ax.set_xlim(xs.min() - margin, xs.max() + margin)
        self._ax.set_ylim(ys.min() - margin, ys.max() + margin)

    def _draw_outline(self, active: bool, fill: bool = True) -> None:
        outline_closed = np.vstack([self._outline, self._outline[0]])
        fill_color = "#1c3745" if active else "#182630"
        fill_alpha = 0.20 if active else 0.13
        edge_color = "#9fe3ff" if active else "#7aa7bf"
        crisp_color = "#edfaff" if active else "#bdd7e7"
        if fill:
            self._ax.fill(self._outline[:, 0], self._outline[:, 1], color=fill_color, alpha=fill_alpha, zorder=1)
        self._ax.plot(outline_closed[:, 0], outline_closed[:, 1], color=edge_color, linewidth=2.4, alpha=0.76, zorder=3)
        self._ax.plot(outline_closed[:, 0], outline_closed[:, 1], color=crisp_color, linewidth=0.8, alpha=0.70, zorder=4)

    def _draw_points(self, active: bool) -> None:
        self._ax.scatter(
            self._sensor_coords[:, 0],
            self._sensor_coords[:, 1],
            s=95,
            c=self._values,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            edgecolors="#f5fbff",
            linewidths=0.7,
            alpha=0.95 if active else 0.70,
            zorder=5,
        )
        self._ax.scatter(
            self._sensor_coords[:, 0],
            self._sensor_coords[:, 1],
            s=22,
            c="#355166" if active else "#7e9eb2",
            edgecolors="none",
            alpha=0.10,
            zorder=4,
        )

    def _render_empty(self, active: bool) -> None:
        self._reset_canvas()
        self._draw_outline(active=active)
        self._ax.text(
            0.5,
            0.5,
            "等待当前会话触觉数据",
            transform=self._ax.transAxes,
            ha="center",
            va="center",
            color="#5E5045",
        )
        self.draw_idle()

    def _render_active(self) -> None:
        self._reset_canvas()
        self._draw_outline(active=True)
        self._draw_points(active=True)
        self._draw_outline(active=True, fill=False)
        if self._frame_valid is True:
            self._ax.text(
                0.04,
                0.96,
                "Valid",
                transform=self._ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="#0F766E",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#D9F3EE",
                    "edgecolor": "#0F766E",
                    "linewidth": 1.0,
                    "alpha": 0.92,
                },
                zorder=10,
            )
        self.draw_idle()


class FootMotionCanvas(FigureCanvas):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(5.6, 5.0), facecolor="#F3E6D7")
        super().__init__(self.figure)
        self.setParent(parent)
        self._ax = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self._motion: BVHMotion | None = None
        self._axis_preset = "zup"
        self._preview_time = 0.0
        self._fake_segments: list[dict] = []
        self._pending_fake_start: dict | None = None
        self._frame_index = 0
        self._foot_joint_mask: np.ndarray | None = None
        self._edge_artists = []
        self._scatter = None
        self._foot_scatter = None
        self.setMinimumSize(340, 300)
        self._show_placeholder("请加载对齐 BVH")

    def set_motion(self, motion: BVHMotion | None, axis_preset: str = "zup") -> None:
        self._motion = motion
        self._axis_preset = axis_preset
        self._foot_joint_mask = None
        self._edge_artists = []
        self._scatter = None
        self._foot_scatter = None
        self.render(self._preview_time)

    def set_preview_time(self, preview_time: float) -> None:
        self._preview_time = max(0.0, float(preview_time))
        self.render(self._preview_time)

    def _show_placeholder(self, text: str) -> None:
        self._ax.clear()
        self._ax.set_facecolor("#F3E6D7")
        self._ax.text2D(0.5, 0.5, text, transform=self._ax.transAxes, ha="center", va="center", color="#241C17")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_zticks([])
        self._ax.set_xlabel("")
        self._ax.set_ylabel("")
        self._ax.set_zlabel("")
        self.draw_idle()

    def _ensure_artists(self) -> None:
        if self._motion is None:
            return
        if self._edge_artists and self._scatter is not None:
            return
        self._ax.clear()
        self._ax.set_facecolor("#F3E6D7")
        self._ax.view_init(elev=18, azim=-62)
        self._ax.set_xlabel("")
        self._ax.set_ylabel("")
        self._ax.set_zlabel("")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_zticks([])
        self._ax.grid(False)
        self._edge_artists = []
        for _ in self._motion.edges:
            artist, = self._ax.plot([], [], [], color="#AF5A3B", linewidth=2.0)
            self._edge_artists.append(artist)
        self._scatter = self._ax.scatter([], [], [], color="#355166", s=18)
        self._foot_joint_mask = np.array([_is_foot_joint(joint.name) for joint in self._motion.joints], dtype=bool)
        self._foot_scatter = self._ax.scatter(
            [],
            [],
            [],
            s=115,
            c="#D97706",
            alpha=0.96,
            edgecolors="#fff6ec",
            linewidths=0.9,
            depthshade=False,
        )

    def render(self, preview_time: float) -> None:
        self._preview_time = max(0.0, float(preview_time))
        if self._motion is None or len(self._motion.raw_frames) == 0:
            self._show_placeholder("请加载对齐 BVH")
            return

        self._ensure_artists()
        if self._motion is None:
            self._show_placeholder("请加载对齐 BVH")
            return

        frame_index = int(round(self._preview_time * self._motion.raw_fps))
        frame_index = max(0, min(frame_index, max(len(self._motion.raw_frames) - 1, 0)))
        self._frame_index = frame_index
        positions = compute_joint_positions(self._motion, frame_index)
        positions = transform_display_positions(positions, self._axis_preset)

        center = positions.mean(axis=0)
        span = float(np.max(np.ptp(positions, axis=0))) if len(positions) else 1.0
        radius = max(1.0, span * 0.65)
        self._ax.set_xlim(center[0] - radius, center[0] + radius)
        self._ax.set_ylim(center[1] - radius, center[1] + radius)
        self._ax.set_zlim(center[2] - radius, center[2] + radius)

        for artist, (parent, child) in zip(self._edge_artists, self._motion.edges):
            points = positions[[parent, child]]
            artist.set_data(points[:, 0], points[:, 1])
            artist.set_3d_properties(points[:, 2])

        if self._scatter is not None:
            self._scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        if self._foot_joint_mask is not None and np.any(self._foot_joint_mask):
            foot_positions = positions[self._foot_joint_mask]
            if self._foot_scatter is not None:
                self._foot_scatter._offsets3d = (
                    foot_positions[:, 0],
                    foot_positions[:, 1],
                    foot_positions[:, 2],
                )

        self.draw_idle()


class PressureCurveCanvas(FigureCanvas):
    # Seek preview playhead by clicking / dragging on the curve, same as visual page.
    clicked = QtCore.pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(10.8, 4.6), facecolor="#F3E6D7")
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.95)
        self.ax = self.figure.add_subplot(111)
        self._series_visible = {
            "mocap_left": True,
            "mocap_right": True,
            "pressure_left": True,
            "pressure_right": True,
            "mocap_vgrf": True,
            "tactile_vgrf": True,
        }
        self._dragging_playhead = False
        self.mpl_connect("button_press_event", self._on_button_press)
        self.mpl_connect("motion_notify_event", self._on_motion_notify)
        self.mpl_connect("button_release_event", self._on_button_release)
        self._render_empty(ylabel="归一化")

    def _style_axis(self) -> None:
        self.ax.set_facecolor("#F7EEE4")
        self.ax.tick_params(colors="#5E5045", labelsize=12)
        for spine in self.ax.spines.values():
            spine.set_color("#D4BFAE")
        self.ax.title.set_color("#241C17")
        self.ax.xaxis.label.set_color("#241C17")
        self.ax.yaxis.label.set_color("#241C17")
        self.ax.grid(alpha=0.28, color="#D2BCA9")

    def _event_time(self, event) -> float | None:
        if event is None or event.xdata is None:
            return None
        # Accept clicks on main axis or dynamics twin axis.
        if event.inaxes is None:
            return None
        if event.inaxes is not self.ax and event.inaxes not in getattr(self.figure, "axes", []):
            return None
        return float(event.xdata)

    def _on_button_press(self, event) -> None:
        if getattr(event, "button", None) not in (1, None):
            return
        x_value = self._event_time(event)
        if x_value is None:
            return
        self._dragging_playhead = True
        self.clicked.emit(x_value)

    def _on_motion_notify(self, event) -> None:
        if not getattr(self, "_dragging_playhead", False):
            return
        x_value = self._event_time(event)
        if x_value is None:
            return
        self.clicked.emit(x_value)

    def _on_button_release(self, event) -> None:
        if getattr(event, "button", None) not in (1, None):
            return
        self._dragging_playhead = False

    def _reset_axes(self) -> None:
        """Clear main axis and drop any leftover twin axes from prior modes."""
        for extra_ax in list(self.figure.axes):
            if extra_ax is not self.ax:
                self.figure.delaxes(extra_ax)
        self.ax.clear()
        self.ax.set_title("")
        # Restore default single-axis margins after dynamics twin-axis layout.
        self.figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.95)

    def set_series_visibility(
        self,
        *,
        mocap_left: bool | None = None,
        mocap_right: bool | None = None,
        pressure_left: bool | None = None,
        pressure_right: bool | None = None,
        mocap_vgrf: bool | None = None,
        tactile_vgrf: bool | None = None,
    ) -> None:
        if mocap_left is not None:
            self._series_visible["mocap_left"] = bool(mocap_left)
        if mocap_right is not None:
            self._series_visible["mocap_right"] = bool(mocap_right)
        if pressure_left is not None:
            self._series_visible["pressure_left"] = bool(pressure_left)
        if pressure_right is not None:
            self._series_visible["pressure_right"] = bool(pressure_right)
        if mocap_vgrf is not None:
            self._series_visible["mocap_vgrf"] = bool(mocap_vgrf)
        if tactile_vgrf is not None:
            self._series_visible["tactile_vgrf"] = bool(tactile_vgrf)

    def _render_empty(self, *, ylabel: str = "归一化") -> None:
        self._reset_axes()
        self._style_axis()
        self.ax.text(0.5, 0.5, "请先加载用于对齐的数据", transform=self.ax.transAxes, ha="center", va="center")
        self.ax.set_xlabel("时间（秒）")
        self.ax.set_ylabel(ylabel)
        self.draw_idle()

    def _plot_series(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        label: str,
        color: str,
        linestyle: str = "-",
        visible: bool = True,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        if not visible or len(x) == 0 or len(y) == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        if valid_mask is None or len(valid_mask) == 0:
            self.ax.plot(x, y, color=color, linewidth=2.0, linestyle=linestyle, label=label)
            return

        # valid_mask=1 (original/source) => solid; reconstructed/other(0) => dashed
        mask = (np.asarray(valid_mask[:n]) != 0)
        labeled = False
        for is_valid, style in ((True, "-"), (False, "--")):
            selected = mask if is_valid else ~mask
            if not np.any(selected):
                continue
            idx = np.flatnonzero(selected)
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for group in np.split(idx, splits):
                if len(group) == 0:
                    continue
                start_i = int(group[0])
                end_i = int(group[-1]) + 1
                # Share one neighbor sample so segment junctions stay continuous.
                left = max(0, start_i - (1 if start_i > 0 else 0))
                right = min(n, end_i + (1 if end_i < n else 0))
                # Prefer endpoints inside the selected run, but keep at least 2 points.
                if right - left < 2:
                    left = max(0, start_i)
                    right = min(n, end_i)
                    if right - left < 2 and start_i > 0:
                        left = start_i - 1
                    if right - left < 2 and end_i < n:
                        right = end_i + 1
                seg_x = x[left:right]
                seg_y = y[left:right]
                if len(seg_x) < 2:
                    continue
                use_label = label if not labeled else None
                self.ax.plot(
                    seg_x,
                    seg_y,
                    color=color,
                    linewidth=2.0,
                    linestyle=style,
                    label=use_label,
                )
                labeled = True


    def _draw_fake_markers(
        self,
        *,
        fake_segments: list[dict] | None,
        pending_fake_start: float | None,
    ) -> None:
        """Shade completed fake segments and mark pending/open endpoints."""
        labeled_start = False
        labeled_end = False
        labeled_span = False
        for seg in fake_segments or []:
            start_t = seg.get("start_time_s")
            end_t = seg.get("end_time_s")
            if start_t is None or end_t is None:
                continue
            start_t = float(start_t)
            end_t = float(end_t)
            lo, hi = (start_t, end_t) if start_t <= end_t else (end_t, start_t)
            self.ax.axvspan(
                lo,
                hi,
                color="#F59E0B",
                alpha=0.16,
                zorder=0,
                label=("Fake 片段" if not labeled_span else None),
            )
            labeled_span = True
            self.ax.axvline(
                start_t,
                color="#16A34A",
                linewidth=1.8,
                alpha=0.9,
                label=("Fake起点" if not labeled_start else None),
            )
            labeled_start = True
            self.ax.axvline(
                end_t,
                color="#2563EB",
                linewidth=1.8,
                linestyle="--",
                alpha=0.9,
                label=("Fake终点" if not labeled_end else None),
            )
            labeled_end = True
        if pending_fake_start is not None:
            self.ax.axvline(
                float(pending_fake_start),
                color="#16A34A",
                linewidth=2.0,
                alpha=0.95,
                label=("Fake起点(待定终点)" if not labeled_start else None),
            )


    def _draw_strike_events(
        self,
        *,
        dynamics: DynamicsVgrfCurveSet,
        delta_t2: float,
        show_mocap: bool,
        show_tactile: bool,
    ) -> None:
        """Mark strike / loading-onset events on the new-mechanism preview curves."""
        mocap_t = getattr(dynamics, "mocap_strike_times_s", None)
        press_t = getattr(dynamics, "pressure_strike_times_s", None)
        labeled_m = False
        labeled_p = False

        if show_tactile and press_t is not None and len(press_t):
            y_src = np.asarray(dynamics.tactile_vgrf_bw, dtype=np.float64)
            x_src = np.asarray(dynamics.time_s, dtype=np.float64)
            if len(x_src) and len(y_src):
                xs = np.asarray(press_t, dtype=np.float64)
                ys = np.interp(xs, x_src, y_src)
                self.ax.scatter(
                    xs,
                    ys,
                    s=42,
                    c="#A94F5B",
                    marker="o",
                    edgecolors="#5C2430",
                    linewidths=0.8,
                    zorder=6,
                    label="触觉触地事件",
                )
                labeled_p = True

        if show_mocap and mocap_t is not None and len(mocap_t):
            y_src = np.asarray(dynamics.mocap_vgrf_bw, dtype=np.float64)
            x_src = np.asarray(dynamics.time_s, dtype=np.float64)
            if len(x_src) and len(y_src):
                xs = np.asarray(mocap_t, dtype=np.float64) + float(delta_t2)
                ys = np.interp(xs - float(delta_t2), x_src, y_src)
                # Prefer the twin axis when it exists so markers share the mocap scale.
                axes = list(getattr(self.figure, "axes", []) or [])
                target_ax = axes[-1] if len(axes) > 1 else self.ax
                target_ax.scatter(
                    xs,
                    ys,
                    s=48,
                    c="#355166",
                    marker="D",
                    edgecolors="#1B2C38",
                    linewidths=0.8,
                    zorder=6,
                    label="动捕触地事件",
                )
                labeled_m = True

        # No-op if neither side had events; keep signature side-effect free otherwise.
        _ = (labeled_m, labeled_p)


    def render(
        self,
        mocap: MocapFootCurveSet | None,
        pressure: PressureCurveSet | None,
        preview_time: float,
        delta_t2: float,
        *,
        mode: str = "legacy",
        dynamics: DynamicsVgrfCurveSet | None = None,
        fake_segments: list[dict] | None = None,
        pending_fake_start: float | None = None,
    ) -> None:
        # 新机制：只显示两条总曲线（动捕 触地代理 vs 触觉 压力总和）
        # 使用双纵轴，避免两条曲线振幅差一个数量级时，动捕曲线在视觉上被压成水平线。
        if mode == "dynamics":
            if dynamics is None:
                self._render_empty(ylabel="对齐曲线")
                return
            self._reset_axes()
            self._style_axis()
            # Leave room for the right-side twin y-axis labels.
            self.figure.subplots_adjust(left=0.07, right=0.90, bottom=0.14, top=0.90)

            shifted_time = dynamics.time_s + delta_t2
            show_mocap = self._series_visible.get("mocap_vgrf", True)
            show_tactile = self._series_visible.get("tactile_vgrf", True)
            handles = []
            labels = []

            if show_tactile and len(dynamics.time_s) and len(dynamics.tactile_vgrf_bw):
                tactile_valid = getattr(dynamics, "tactile_valid", None)
                if tactile_valid is not None and len(tactile_valid):
                    self._plot_series(
                        dynamics.time_s,
                        dynamics.tactile_vgrf_bw,
                        label="触觉 压力总和",
                        color="#A94F5B",
                        visible=True,
                        valid_mask=tactile_valid,
                    )
                    handles_labels = self.ax.get_legend_handles_labels()
                    if handles_labels[0]:
                        handles.append(handles_labels[0][-1])
                        labels.append("触觉 压力总和")
                else:
                    (line_t,) = self.ax.plot(
                        dynamics.time_s,
                        dynamics.tactile_vgrf_bw,
                        color="#A94F5B",
                        linewidth=2.0,
                        label="触觉 压力总和",
                    )
                    handles.append(line_t)
                    labels.append("触觉 压力总和")
                self.ax.set_ylabel("触觉 压力总和", color="#A94F5B")
                self.ax.tick_params(axis="y", colors="#A94F5B")

            if show_mocap and len(shifted_time) and len(dynamics.mocap_vgrf_bw):
                if show_tactile:
                    ax_m = self.ax.twinx()
                    ax_m.set_facecolor("none")
                    (line_m,) = ax_m.plot(
                        shifted_time,
                        dynamics.mocap_vgrf_bw,
                        color="#355166",
                        linewidth=2.0,
                        linestyle="--",
                        label="动捕 触地代理",
                    )
                    ax_m.set_ylabel("动捕 触地代理", color="#355166")
                    ax_m.tick_params(axis="y", colors="#355166")
                    # Keep twin-axis spines readable on the shared style.
                    for spine in ax_m.spines.values():
                        spine.set_color("#D4BFAE")
                else:
                    (line_m,) = self.ax.plot(
                        shifted_time,
                        dynamics.mocap_vgrf_bw,
                        color="#355166",
                        linewidth=2.0,
                        linestyle="--",
                        label="动捕 触地代理",
                    )
                    self.ax.set_ylabel("动捕 触地代理", color="#355166")
                    self.ax.tick_params(axis="y", colors="#355166")
                handles.append(line_m)
                labels.append("动捕 触地代理")

            self._draw_fake_markers(fake_segments=fake_segments, pending_fake_start=pending_fake_start)
            self._draw_strike_events(
                dynamics=dynamics,
                delta_t2=delta_t2,
                show_mocap=show_mocap,
                show_tactile=show_tactile,
            )
            self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
            self.ax.set_xlabel("时间（秒）")
            # Merge line / marker labels across primary and twin axes.
            merged_h, merged_l = [], []
            seen = set()
            for ax in getattr(self.figure, "axes", []) or [self.ax]:
                for h, lab in zip(*ax.get_legend_handles_labels()):
                    if not lab or lab in seen:
                        continue
                    seen.add(lab)
                    merged_h.append(h)
                    merged_l.append(lab)
            if merged_h:
                self.ax.legend(merged_h, merged_l, loc="upper right", fontsize=11, ncol=2)
            elif labels:
                self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
            else:
                self.ax.text(0.5, 0.5, "??????", transform=self.ax.transAxes, ha="center", va="center")

# Strike-estimator diagnostics (offset is event-based, not correlation).
            if len(dynamics.mocap_vgrf_bw) and len(dynamics.tactile_vgrf_bw):
                n_evt = int(getattr(dynamics, "event_n", 0) or 0)
                scatter = getattr(dynamics, "event_scatter", float("nan"))
                ok = bool(getattr(dynamics, "event_ok", False))
                note = getattr(dynamics, "event_note", "") or ""
                scatter_txt = f"{scatter:.4f}s" if scatter == scatter else "nan"
                msg = (
                    f"触地事件 n={n_evt} | scatter={scatter_txt} | ok={ok} | "
                    f"scale→m={float(getattr(dynamics, 'length_scale_to_m', 1.0)):g}"
                    + (f" | {note}" if note else "")
                )
                self.ax.set_title(msg, fontsize=10, color="#5E5045", pad=8)

            max_x = max(
                float(dynamics.time_s[-1]) if len(dynamics.time_s) else 1.0,
                float(dynamics.time_s[-1] + delta_t2) if len(dynamics.time_s) else 1.0,
                preview_time,
                1.0,
            )
            self.ax.set_xlim(min(0.0, delta_t2), max_x)
            self.draw_idle()
            return

        # 旧机制：足底贴地 / 压力总和 的 L/R 四条曲线
        if mocap is None and pressure is None:
            self._render_empty(ylabel="归一化")
            return

        self._reset_axes()
        self._style_axis()
        if pressure is not None:
            self._plot_series(
                pressure.time_s,
                pressure.left_sum,
                label="触觉-L",
                color="#A94F5B",
                visible=self._series_visible["pressure_left"],
                valid_mask=getattr(pressure, "left_valid", None),
            )
            self._plot_series(
                pressure.time_s,
                pressure.right_sum,
                label="触觉-R",
                color="#D97706",
                visible=self._series_visible["pressure_right"],
                valid_mask=getattr(pressure, "right_valid", None),
            )
        if mocap is not None:
            shifted_time = mocap.time_s + delta_t2
            self._plot_series(
                shifted_time,
                mocap.left_sum,
                label="动捕-L",
                color="#355166",
                linestyle="--",
                visible=self._series_visible["mocap_left"],
            )
            self._plot_series(
                shifted_time,
                mocap.right_sum,
                label="动捕-R",
                color="#0F766E",
                linestyle="--",
                visible=self._series_visible["mocap_right"],
            )
        self._draw_fake_markers(fake_segments=fake_segments, pending_fake_start=pending_fake_start)
        self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
        self.ax.set_xlabel("时间（秒）")
        self.ax.set_ylabel("归一化")
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
        else:
            self.ax.text(0.5, 0.5, "没有可见曲线", transform=self.ax.transAxes, ha="center", va="center")

        max_x = max(
            float(pressure.time_s[-1]) if pressure is not None and len(pressure.time_s) else 1.0,
            float(mocap.time_s[-1] + delta_t2) if mocap is not None and len(mocap.time_s) else 1.0,
            preview_time,
            1.0,
        )
        self.ax.set_xlim(min(0.0, delta_t2), max_x)
        self.draw_idle()


class PressureAlignmentPage(QtWidgets.QWidget):
    def __init__(
        self,
        session_getter: Callable[[], SessionData | None],
        log_callback: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_getter = session_getter
        self._log_callback = log_callback
        self._session: SessionData | None = None
        self._motion: BVHMotion | None = None
        self._mocap: MocapFootCurveSet | None = None
        self._pressure: PressureCurveSet | None = None
        self._pressure_left: PressureSensorFrameSet | None = None
        self._pressure_right: PressureSensorFrameSet | None = None
        self._meta: PressureMetaInfo | None = None
        self._result: PressureAlignmentResult | None = None
        self._preview_time = 0.0
        self._delta_t2 = 0.0
        self._manual_adjusted = False
        self._updating_controls = False
        self._alignment_mode = "legacy"
        self._visual_prior_source: Path | None = None
        self._review_root = DEFAULT_VISUAL_ALIGN_REVIEW_ROOT
        self._reconstructed_root: Path | None = (
            DEFAULT_RECONSTRUCTED_TACTILE_ROOT
            if DEFAULT_RECONSTRUCTED_TACTILE_ROOT.exists()
            else None
        )
        self._pressure_source_label = "未加载"
        self._dynamics_curves: DynamicsVgrfCurveSet | None = None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._reset_view()

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(10)
        self.body_splitter.addWidget(self._build_workspace())
        sidebar = self._build_sidebar()
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(460)
        self.body_splitter.addWidget(sidebar)
        self.body_splitter.setStretchFactor(0, 7)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([1640, 360])
        root.addWidget(self.body_splitter)

    def _build_workspace(self) -> QtWidgets.QWidget:
        workspace = TechPanel()
        layout = QtWidgets.QVBoxLayout(workspace)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("动捕-触觉对齐")
        title.setProperty("panelTitle", True)
        layout.addWidget(title)

        self.preview_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setHandleWidth(10)
        self.preview_splitter.addWidget(self._build_tactile_panel())
        self.preview_splitter.addWidget(self._build_motion_panel())
        self.preview_splitter.setStretchFactor(0, 5)
        self.preview_splitter.setStretchFactor(1, 4)
        self.preview_splitter.setSizes([1040, 620])
        layout.addWidget(self.preview_splitter, 1)

        layout.addWidget(self._build_curve_panel(), 0)
        return workspace

    def _build_tactile_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        header = QtWidgets.QLabel("触觉展示")
        header.setProperty("panelTitle", True)
        layout.addWidget(header)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        for sensor_id, label_text in (("left_foot", "左脚"), ("right_foot", "右脚")):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(6)
            label = QtWidgets.QLabel(label_text)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setProperty("muted", True)
            col.addWidget(label)
            canvas = FootPressureCanvas(sensor_id)
            setattr(self, f"{sensor_id}_canvas", canvas)
            col.addWidget(canvas, 1)
            row.addLayout(col, 1)
        layout.addLayout(row, 1)
        return panel

    def _build_motion_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        header = QtWidgets.QLabel("动捕预览")
        header.setProperty("panelTitle", True)
        layout.addWidget(header)
        self.motion_info_label = QtWidgets.QLabel("未加载")
        self.motion_info_label.setProperty("metricValue", True)
        layout.addWidget(self.motion_info_label)
        self.motion_canvas = FootMotionCanvas()
        layout.addWidget(self.motion_canvas, 1)
        return panel

    def _build_curve_panel(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)
        header = QtWidgets.QLabel("对齐曲线")
        header.setProperty("panelTitle", True)
        top_row.addWidget(header, 1)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(10)
        selector_label = QtWidgets.QLabel("显示曲线")
        selector_label.setProperty("metricTitle", True)
        selector_row.addWidget(selector_label)

        # 旧机制：L / R 成对显示（触觉 + 动捕同步勾选）
        self.pair_left_check = QtWidgets.QCheckBox("左脚 L（触觉+动捕）")
        self.pair_left_check.setChecked(True)
        self.pair_left_check.stateChanged.connect(self._on_curve_visibility_changed)
        selector_row.addWidget(self.pair_left_check)

        self.pair_right_check = QtWidgets.QCheckBox("右脚 R（触觉+动捕）")
        self.pair_right_check.setChecked(True)
        self.pair_right_check.stateChanged.connect(self._on_curve_visibility_changed)
        selector_row.addWidget(self.pair_right_check)

        # Aliases kept for readability at older call sites.
        self.mocap_left_check = self.pair_left_check
        self.mocap_right_check = self.pair_right_check
        self.pressure_left_check = self.pair_left_check
        self.pressure_right_check = self.pair_right_check

        # 新机制只使用两条总曲线；默认隐藏，由模式切换显示。
        self.mocap_vgrf_check = QtWidgets.QCheckBox("动捕 触地代理")
        self.mocap_vgrf_check.setChecked(True)
        self.mocap_vgrf_check.stateChanged.connect(self._on_curve_visibility_changed)
        self.mocap_vgrf_check.setVisible(False)
        selector_row.addWidget(self.mocap_vgrf_check)

        self.tactile_vgrf_check = QtWidgets.QCheckBox("触觉 压力总和")
        self.tactile_vgrf_check.setChecked(True)
        self.tactile_vgrf_check.stateChanged.connect(self._on_curve_visibility_changed)
        self.tactile_vgrf_check.setVisible(False)
        selector_row.addWidget(self.tactile_vgrf_check)

        selector_row.addStretch(1)
        self.curve_hint_label = QtWidgets.QLabel("当前默认显示全部曲线")
        self.curve_hint_label.setProperty("metricValue", True)
        selector_row.addWidget(self.curve_hint_label)
        top_row.addLayout(selector_row, 0)
        layout.addLayout(top_row)

        self.curve_canvas = PressureCurveCanvas()
        self.curve_canvas.clicked.connect(self._on_curve_clicked)
        layout.addWidget(self.curve_canvas, 1)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setSpacing(10)
        self.play_button = QtWidgets.QPushButton("开始播放")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_playback)
        control_row.addWidget(self.play_button)
        self.prev_button = QtWidgets.QPushButton("上一帧")
        self.prev_button.clicked.connect(lambda: self.shift_preview_frames(-1))
        control_row.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton("下一帧")
        self.next_button.clicked.connect(lambda: self.shift_preview_frames(1))
        control_row.addWidget(self.next_button)
        self.delta_label = QtWidgets.QLabel("对齐偏移")
        self.delta_label.setProperty("metricTitle", True)
        control_row.addWidget(self.delta_label)

        self.delta_minus_10 = QtWidgets.QPushButton("对齐 -10 帧")
        self.delta_minus_1 = QtWidgets.QPushButton("对齐 -1 帧")
        self.delta_plus_1 = QtWidgets.QPushButton("对齐 +1 帧")
        self.delta_plus_10 = QtWidgets.QPushButton("对齐 +10 帧")
        for step_button in (self.delta_minus_10, self.delta_minus_1, self.delta_plus_1, self.delta_plus_10):
            step_button.setEnabled(False)
            control_row.addWidget(step_button)

        self.delta_minus_10.clicked.connect(lambda: self._delta_step(-10))
        self.delta_minus_1.clicked.connect(lambda: self._delta_step(-1))
        self.delta_plus_1.clicked.connect(lambda: self._delta_step(1))
        self.delta_plus_10.clicked.connect(lambda: self._delta_step(10))

        self.delta_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.delta_slider.setEnabled(False)
        self.delta_slider.valueChanged.connect(self._on_delta_slider_changed)
        control_row.addWidget(self.delta_slider, 1)
        self.delta_spin = QtWidgets.QDoubleSpinBox()
        self.delta_spin.setEnabled(False)
        self.delta_spin.setRange(-1000.0, 1000.0)
        self.delta_spin.setDecimals(3)
        self.delta_spin.setSingleStep(0.025)
        self.delta_spin.setSuffix(" s")
        self.delta_spin.valueChanged.connect(self._on_delta_spin_changed)
        self.delta_spin.setMaximumWidth(120)
        control_row.addWidget(self.delta_spin)
        layout.addLayout(control_row)

        fake_row = QtWidgets.QHBoxLayout()
        fake_row.setSpacing(8)
        self.fake_start_btn = QtWidgets.QPushButton("记录Fake起点")
        self.fake_end_btn = QtWidgets.QPushButton("记录Fake终点")
        self.fake_undo_btn = QtWidgets.QPushButton("撤销上一段")
        self.fake_export_btn = QtWidgets.QPushButton("导出Fake帧CSV")
        for btn in (self.fake_start_btn, self.fake_end_btn, self.fake_undo_btn, self.fake_export_btn):
            btn.setEnabled(False)
            fake_row.addWidget(btn)
        self.fake_start_btn.clicked.connect(self._record_fake_start)
        self.fake_end_btn.clicked.connect(self._record_fake_end)
        self.fake_undo_btn.clicked.connect(self._undo_fake_segment)
        self.fake_export_btn.clicked.connect(self._export_fake_frames_csv)
        self.fake_status_label = QtWidgets.QLabel("Fake片段：0 段")
        self.fake_status_label.setProperty("metricValue", True)
        fake_row.addWidget(self.fake_status_label, 1)
        layout.addLayout(fake_row)
        return panel

    def _build_sidebar(self) -> QtWidgets.QWidget:
        panel = TechPanel()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        info_group = QtWidgets.QGroupBox("当前状态")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(14, 14, 14, 14)
        info_layout.setSpacing(14)
        self.session_row = InfoRow("会话")
        self.source_row = InfoRow("输入文件")
        self.tactile_source_row = InfoRow("触觉来源")
        self.coarse_row = InfoRow("粗偏移")
        self.visual_prior_row = InfoRow("视频对齐先验")
        self.left_row = InfoRow("左脚结果")
        self.right_row = InfoRow("右脚结果")
        self.final_row = InfoRow("最终 delta_t2")
        self.status_row = InfoRow("校验状态")
        for row in (
            self.session_row,
            self.source_row,
            self.tactile_source_row,
            self.coarse_row,
            self.visual_prior_row,
            self.left_row,
            self.right_row,
            self.final_row,
            self.status_row,
        ):
            info_layout.addWidget(row)
        layout.addWidget(info_group)

        mode_group = QtWidgets.QGroupBox("对齐机制")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(14, 14, 14, 14)
        mode_layout.setSpacing(8)
        self.alignment_mode_group = QtWidgets.QButtonGroup(self)
        self.alignment_mode_group.setExclusive(True)
        # 互斥一步切换：点选即自动重估，不再需要“重新评估”按钮。
        self.mode_legacy_btn = QtWidgets.QPushButton("旧机制：足底贴地曲线互相关")
        self.mode_dynamics_btn = QtWidgets.QPushButton("新机制：触地事件对齐")
        for btn in (self.mode_legacy_btn, self.mode_dynamics_btn):
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumHeight(34)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.mode_legacy_btn.setChecked(True)
        self.alignment_mode_group.addButton(self.mode_legacy_btn, 0)
        self.alignment_mode_group.addButton(self.mode_dynamics_btn, 1)
        self.mode_legacy_btn.toggled.connect(self._on_alignment_mode_toggled)
        self.mode_dynamics_btn.toggled.connect(self._on_alignment_mode_toggled)
        mode_layout.addWidget(self.mode_legacy_btn)
        mode_layout.addWidget(self.mode_dynamics_btn)
        self._update_alignment_mode_button_styles()
        self.alignment_mode_hint = QtWidgets.QLabel("点选即自动对齐（无需再点重估）")
        self.alignment_mode_hint.setWordWrap(True)
        self.alignment_mode_hint.setProperty("metricValue", True)
        mode_layout.addWidget(self.alignment_mode_hint)
        layout.addWidget(mode_group)

        action_group = QtWidgets.QGroupBox("辅助初始化")
        action_layout = QtWidgets.QVBoxLayout(action_group)
        action_layout.setContentsMargins(14, 14, 14, 14)
        action_layout.setSpacing(10)
        self.import_reconstructed_btn = QtWidgets.QPushButton("导入重建触觉数据")
        self.import_reconstructed_btn.setToolTip(
            "选择 reconstruction_<timestamp> 文件夹（如 reconstruction_20260816_222245）。"
            "之后触觉对齐优先使用该目录中按 rec.../pressure_*_t*.csv 重建的数据，"
            "匹配不到时再回退相机目录中的 pressure_left/right.csv。"
        )
        self.import_reconstructed_btn.clicked.connect(self._import_reconstructed_tactile_root)
        action_layout.addWidget(self.import_reconstructed_btn)
        self.import_visual_align_btn = QtWidgets.QPushButton("导入视频-动捕对齐结果辅助")
        self.import_visual_align_btn.setToolTip(
            "选择包含多个视频-动捕对齐 CSV 的文件夹（例如 AlignReviews_csv），"
            "再按当前会话编号匹配对应文件中的“偏移量(s)”，并用该偏移初始化动捕-触觉 delta_t2。"
        )
        self.import_visual_align_btn.clicked.connect(self._import_visual_alignment_prior)
        action_layout.addWidget(self.import_visual_align_btn)
        layout.addWidget(action_group)

        log_group = QtWidgets.QGroupBox("操作日志")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(14, 14, 14, 14)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(220)
        self.log_view.setMinimumHeight(220)
        self.log_view.setProperty("console", True)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 1)
        return panel

    def _log(self, message: str) -> None:
        if not message:
            return
        timestamp = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")
        self._log_callback(message)

    def _set_status(self, text: str, level: str = "warning") -> None:
        self.status_row.set_value(text)
        self.status_row.value_label.setProperty("level", level)
        repolish(self.status_row.value_label)

    def _reset_view(self) -> None:
        self.stop_playback()
        self._session = None
        self._motion = None
        self._mocap = None
        self._pressure = None
        self._pressure_left = None
        self._pressure_right = None
        self._meta = None
        self._result = None
        self._delta_t2 = 0.0
        self._manual_adjusted = False
        self._visual_prior_source = None
        self._dynamics_curves = None
        self._fake_segments = []
        self._pending_fake_start = None
        self._set_controls_enabled(False)
        if hasattr(self, "fake_status_label"):
            self.fake_status_label.setText("Fake片段：0 段")
        # Keep L/R (and dynamics) curve checkbox state across trial switches.
        self.left_foot_canvas.set_values(None)
        self.right_foot_canvas.set_values(None)
        self.motion_canvas.set_motion(None)
        if hasattr(self, "curve_canvas"):
            self._apply_curve_series_visibility()
            self.curve_canvas.render(
                None,
                None,
                0.0,
                0.0,
                mode=self._current_alignment_mode(),
                dynamics=None,
            )
        self.session_row.set_value("--")
        self.source_row.set_value("--")
        self.coarse_row.set_value("--")
        self.visual_prior_row.set_value("--")
        self.left_row.set_value("--")
        self.right_row.set_value("--")
        self.final_row.set_value("--")
        self._set_status("等待当前会话")
        self.motion_info_label.setText("未加载")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.delta_slider.setEnabled(enabled)
        self.delta_spin.setEnabled(enabled)
        for button in (self.delta_minus_10, self.delta_minus_1, self.delta_plus_1, self.delta_plus_10):
            button.setEnabled(enabled)
        for button in (
            getattr(self, "fake_start_btn", None),
            getattr(self, "fake_end_btn", None),
            getattr(self, "fake_undo_btn", None),
            getattr(self, "fake_export_btn", None),
        ):
            if button is not None:
                button.setEnabled(enabled)

    def set_session_context(self, session: SessionData | None, *, force: bool = False) -> None:
        if session is None:
            self._reset_view()
            return

        if not force and self._session is session:
            return
        self._session = session
        self._load_from_session(session)

    def set_preview_time(self, preview_time: float) -> None:
        self._preview_time = max(0.0, float(preview_time))
        if self._motion is not None:
            self.motion_canvas.set_preview_time(self._preview_time)
        self._refresh_pressure_canvases()
        if self._mocap is not None or self._pressure is not None:
            self.curve_canvas.render(
                self._mocap,
                self._pressure,
                self._preview_time,
                self._delta_t2,
                mode=self._current_alignment_mode(),
                dynamics=self._dynamics_curves,
                **self._fake_render_args(),
            )

    def _set_play_button_state(self, playing: bool) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.blockSignals(False)
        self.play_button.setText("暂停播放" if playing else "开始播放")
        self.play_button.setProperty("playing", playing)
        repolish(self.play_button)

    def _playback_fps(self) -> float:
        if self._pressure is not None and self._pressure.sample_fps > 0:
            return self._pressure.sample_fps
        if self._motion is not None and self._motion.raw_fps > 0:
            return self._motion.raw_fps
        if self._mocap is not None and self._mocap.reference_fps > 0:
            return self._mocap.reference_fps
        return 0.0

    def _playback_duration(self) -> float:
        return max(
            float(self._pressure.time_s[-1]) if self._pressure is not None and len(self._pressure.time_s) else 0.0,
            float(self._mocap.time_s[-1] + self._delta_t2)
            if self._mocap is not None and len(self._mocap.time_s)
            else 0.0,
        )

    def _toggle_playback(self, checked: bool) -> None:
        if checked:
            self.start_playback()
        else:
            self.stop_playback()

    def start_playback(self) -> None:
        fps = self._playback_fps()
        if fps <= 0 or self._session is None:
            self._set_play_button_state(False)
            return
        self._play_timer.start(max(20, int(round(1000.0 / fps))))
        self._set_play_button_state(True)

    def stop_playback(self) -> None:
        self._play_timer.stop()
        if hasattr(self, "play_button"):
            self._set_play_button_state(False)

    def _advance_playback(self) -> None:
        fps = self._playback_fps()
        if fps <= 0:
            self.stop_playback()
            return
        next_time = self._preview_time + 1.0 / fps
        duration = self._playback_duration()
        if next_time > duration + 1e-9:
            self.set_preview_time(0.0)
            self.stop_playback()
            return
        self.set_preview_time(next_time)

    def shift_preview_frames(self, frame_count: int) -> None:
        self.stop_playback()
        fps = self._playback_fps()
        if fps <= 0:
            return
        duration = self._playback_duration()
        next_time = self._preview_time + frame_count / fps
        self.set_preview_time(max(0.0, min(next_time, duration)))

    def _fake_render_args(self) -> dict:
        pending_t = None
        if self._pending_fake_start is not None:
            pending_t = float(self._pending_fake_start.get("preview_time_s", 0.0))
        return {
            "fake_segments": list(self._fake_segments),
            "pending_fake_start": pending_t,
        }

    def _update_fake_status_label(self) -> None:
        if not hasattr(self, "fake_status_label"):
            return
        n = len(self._fake_segments)
        if self._pending_fake_start is None:
            self.fake_status_label.setText(f"Fake片段：{n} 段")
        else:
            t0 = float(self._pending_fake_start.get("preview_time_s", 0.0))
            left_f = self._pending_fake_start.get("left_frame_idx", "?")
            self.fake_status_label.setText(
                f"Fake片段：{n} 段 | 待定起点 t={t0:.3f}s L#{left_f}"
            )

    @staticmethod
    def _nearest_frame_index(time_s: np.ndarray | None, target_t: float) -> int | None:
        if time_s is None:
            return None
        arr = np.asarray(time_s, dtype=np.float64)
        if len(arr) == 0:
            return None
        return int(np.argmin(np.abs(arr - float(target_t))))

    def _current_fake_mark_info(self) -> dict | None:
        """Snapshot current playhead mapped onto left/right tactile frames.

        Left foot is the discrete time base. Right foot uses the nearest frame
        to the same absolute time on the shared pressure timeline.
        """
        if self._pressure_left is None or len(self._pressure_left.time_s) == 0:
            return None
        preview_t = float(self._preview_time)
        left_idx = self._nearest_frame_index(self._pressure_left.time_s, preview_t)
        if left_idx is None:
            return None
        left_t = float(self._pressure_left.time_s[left_idx])
        right_idx = None
        right_t = None
        if self._pressure_right is not None and len(self._pressure_right.time_s):
            right_idx = self._nearest_frame_index(self._pressure_right.time_s, left_t)
            if right_idx is not None:
                right_t = float(self._pressure_right.time_s[right_idx])
        return {
            "preview_time_s": preview_t,
            "left_frame_idx": int(left_idx),
            "left_time_s": left_t,
            "right_frame_idx": int(right_idx) if right_idx is not None else "",
            "right_time_s": right_t if right_t is not None else "",
            "delta_t2": float(self._delta_t2),
            "session_id": self._session.session_id if self._session is not None else "",
        }

    def _record_fake_start(self) -> None:
        info = self._current_fake_mark_info()
        if info is None:
            QtWidgets.QMessageBox.warning(self, "无法记录", "请先加载左脚触觉数据后再记录 Fake 起点。")
            return
        self._pending_fake_start = info
        self._update_fake_status_label()
        self._log(
            f"[Fake起点] 待定段#{len(self._fake_segments) + 1} | "
            f"t={info['preview_time_s']:.3f}s | L#{info['left_frame_idx']} ({info['left_time_s']:.3f}s) | "
            f"R#{info['right_frame_idx']} ({info['right_time_s'] if info['right_time_s'] != '' else 'NA'})"
        )
        self._refresh_views()

    def _record_fake_end(self) -> None:
        if self._pending_fake_start is None:
            QtWidgets.QMessageBox.warning(self, "无法记录", "请先记录 Fake 起点。")
            return
        info = self._current_fake_mark_info()
        if info is None:
            QtWidgets.QMessageBox.warning(self, "无法记录", "请先加载左脚触觉数据后再记录 Fake 终点。")
            return
        start = self._pending_fake_start
        end = info
        # Normalize order so start_time <= end_time while keeping endpoint metadata.
        if float(end["left_frame_idx"]) < float(start["left_frame_idx"]):
            start, end = end, start
        segment = {
            "segment_id": len(self._fake_segments) + 1,
            "start_time_s": float(start["preview_time_s"]),
            "end_time_s": float(end["preview_time_s"]),
            "start_left_frame_idx": int(start["left_frame_idx"]),
            "end_left_frame_idx": int(end["left_frame_idx"]),
            "start_left_time_s": float(start["left_time_s"]),
            "end_left_time_s": float(end["left_time_s"]),
            "start_right_frame_idx": start["right_frame_idx"],
            "end_right_frame_idx": end["right_frame_idx"],
            "start_right_time_s": start["right_time_s"],
            "end_right_time_s": end["right_time_s"],
            "delta_t2": float(self._delta_t2),
            "session_id": self._session.session_id if self._session is not None else "",
        }
        self._fake_segments.append(segment)
        self._pending_fake_start = None
        self._update_fake_status_label()
        self._log(
            f"[Fake终点] 完成段#{segment['segment_id']} | "
            f"L#{segment['start_left_frame_idx']}→#{segment['end_left_frame_idx']} | "
            f"t={segment['start_time_s']:.3f}→{segment['end_time_s']:.3f}s"
        )
        self._refresh_views()

    def _undo_fake_segment(self) -> None:
        if self._pending_fake_start is not None:
            self._pending_fake_start = None
            self._update_fake_status_label()
            self._log("已取消待定 Fake 起点")
            self._refresh_views()
            return
        if not self._fake_segments:
            self._log("没有可撤销的 Fake 片段")
            return
        removed = self._fake_segments.pop()
        # reindex
        for i, seg in enumerate(self._fake_segments, start=1):
            seg["segment_id"] = i
        self._update_fake_status_label()
        self._log(f"已撤销 Fake 段#{removed.get('segment_id')}")
        self._refresh_views()

    def _export_fake_frames_csv(self) -> None:
        if not self._fake_segments:
            QtWidgets.QMessageBox.warning(self, "无法导出", "请至少完成一段 Fake 起点+终点标记。")
            return
        if self._pressure_left is None or len(self._pressure_left.time_s) == 0:
            QtWidgets.QMessageBox.warning(self, "无法导出", "当前没有左脚触觉帧数据。")
            return

        session_id = self._session.session_id if self._session is not None else "unknown"
        # Compact trial code like visual export: S5_1 -> S501? Actually visual uses S + num + trial.
        trial_code = session_id
        match = re.match(r"S(\d+)_(\d+)", session_id)
        if match:
            # Keep readable compact form S{subject}{trial} without forcing 2-digit action.
            trial_code = f"S{match.group(1)}{match.group(2)}"
        else:
            trial_code = re.sub(r"[^0-9A-Za-z_-]+", "_", session_id)

        out_dir = DEFAULT_OUTPUT_ROOT / "fake_tactile_csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(out_dir / f"{trial_code}_fake_frames.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出 Fake 帧 CSV",
            default_path,
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return

        left_times = np.asarray(self._pressure_left.time_s, dtype=np.float64)
        right_times = (
            np.asarray(self._pressure_right.time_s, dtype=np.float64)
            if self._pressure_right is not None and len(self._pressure_right.time_s)
            else np.zeros(0, dtype=np.float64)
        )

        # Inclusive left-frame coverage; right foot = nearest frame to left time.
        fake_left_to_seg: dict[int, int] = {}
        for seg in self._fake_segments:
            a = int(seg["start_left_frame_idx"])
            b = int(seg["end_left_frame_idx"])
            lo, hi = (a, b) if a <= b else (b, a)
            lo = max(0, lo)
            hi = min(len(left_times) - 1, hi)
            for idx in range(lo, hi + 1):
                fake_left_to_seg.setdefault(idx, int(seg["segment_id"]))

        header = [
            "segment_id",
            "left_frame_idx",
            "left_time_s",
            "right_frame_idx",
            "right_time_s",
        ]
        rows: list[list[object]] = [header]
        for left_idx in sorted(fake_left_to_seg):
            left_t = float(left_times[left_idx])
            if len(right_times):
                right_idx = int(np.argmin(np.abs(right_times - left_t)))
                right_t = f"{float(right_times[right_idx]):.6f}"
            else:
                right_idx = ""
                right_t = ""
            rows.append([
                fake_left_to_seg[left_idx],
                left_idx,
                f"{left_t:.6f}",
                right_idx,
                right_t,
            ])

        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

        self._log(f"??? Fake ? CSV?{path} | segments={len(self._fake_segments)} frames={len(fake_left_to_seg)}")
        QtWidgets.QMessageBox.information(
            self,
            "????",
            f"??? {len(self._fake_segments)} ? Fake ???? {len(fake_left_to_seg)} ??\n{path}",
        )
    def _on_curve_clicked(self, x_value: float) -> None:
        """Move the preview playhead by clicking/dragging the alignment curve."""
        if self._session is None and self._pressure is None and self._mocap is None:
            return
        self.stop_playback()
        duration = self._playback_duration()
        if duration <= 0.0:
            try:
                _x0, x1 = self.curve_canvas.ax.get_xlim()
                duration = max(float(x1), 0.0)
            except Exception:
                duration = 0.0
        x_value = float(x_value)
        if duration > 0.0:
            x_value = max(0.0, min(x_value, duration))
        else:
            x_value = max(0.0, x_value)
        fps = self._playback_fps()
        if fps > 0:
            x_value = round(x_value * fps) / fps
        self.set_preview_time(x_value)

    def _resolve_session_paths(
        self,
        session: SessionData,
    ) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None, str]:
        output_dir = session.output_dir
        mocap_path: Path | None = None
        if session.display_bvh is not None:
            expected = output_dir / f"{session.session_id}_{session.display_bvh.role}_aligned.bvh"
            if expected.exists():
                mocap_path = expected
        if mocap_path is None:
            aligned_candidates = sorted(output_dir.glob("*_aligned.bvh"))
            mocap_path = aligned_candidates[0] if aligned_candidates else None
        if mocap_path is None and session.display_bvh is not None:
            mocap_path = session.display_bvh.motion.path

        # Priority: reconstructed tactile dataset root -> camera/session local files.
        reconstructed_rec_dir = resolve_reconstructed_rec_dir(session.session_id, self._reconstructed_root)
        roots = [session.cam_session, output_dir, session.mocap_subject]
        roots = [root for index, root in enumerate(roots) if root is not None and root not in roots[:index]]
        pressure_path = next(
            (root / "contact_curve.csv" for root in roots if (root / "contact_curve.csv").exists()),
            None,
        )
        pressure_left_path = next(
            (root / "pressure_left.csv" for root in roots if (root / "pressure_left.csv").exists()),
            None,
        )
        pressure_right_path = next(
            (root / "pressure_right.csv" for root in roots if (root / "pressure_right.csv").exists()),
            None,
        )
        meta_path = next(
            (root / "meta.json" for root in roots if (root / "meta.json").exists()),
            None,
        )

        if reconstructed_rec_dir is not None:
            source_label = f"重建优先 | {reconstructed_rec_dir.name}"
        elif pressure_left_path is not None and pressure_right_path is not None:
            source_label = f"相机目录 | {pressure_left_path.parent.name}"
        elif pressure_path is not None:
            source_label = f"相机目录 | {pressure_path.name}"
        else:
            source_label = "未找到触觉数据"
        return (
            mocap_path,
            pressure_path,
            pressure_left_path,
            pressure_right_path,
            meta_path,
            reconstructed_rec_dir,
            source_label,
        )

    def _load_from_session(self, session: SessionData) -> None:
        (
            mocap_path,
            pressure_path,
            pressure_left_path,
            pressure_right_path,
            meta_path,
            reconstructed_rec_dir,
            source_label,
        ) = self._resolve_session_paths(session)
        self._reset_view()
        self._session = session
        self.session_row.set_value(session.session_id)
        self._pressure_source_label = source_label
        if hasattr(self, "tactile_source_row"):
            self.tactile_source_row.set_value(source_label)

        try:
            if mocap_path is not None:
                if session.display_bvh is not None and session.display_bvh.motion.path == mocap_path:
                    self._motion = session.display_bvh.motion
                else:
                    self._motion = load_bvh_motion_preserve_frames(mocap_path)
                self._mocap = load_mocap_foot_curves(
                    self._motion,
                    axis_preset=session.runtime_options.axis_preset,
                )
                self.motion_canvas.set_motion(self._motion, session.runtime_options.axis_preset)

            if reconstructed_rec_dir is not None:
                self._pressure_left, self._pressure_right = load_reconstructed_pressure_sensors(reconstructed_rec_dir)
                self._pressure = build_pressure_curve_set(
                    self._pressure_left,
                    self._pressure_right,
                    reference_fps=session.reference_visual_fps,
                )
                self._pressure_source_label = f"重建优先 | {reconstructed_rec_dir}"
                self._log(f"触觉数据来自重建目录：{reconstructed_rec_dir}")
            elif pressure_left_path is not None and pressure_right_path is not None:
                self._pressure_left = load_pressure_sensor_csv(pressure_left_path)
                self._pressure_right = load_pressure_sensor_csv(pressure_right_path)
                self._pressure = build_pressure_curve_set(
                    self._pressure_left,
                    self._pressure_right,
                    reference_fps=session.reference_visual_fps,
                )
                self._pressure_source_label = f"相机目录 | {pressure_left_path.parent}"
                self._log(f"触觉数据来自相机目录：{pressure_left_path.parent}")
            elif pressure_path is not None:
                self._pressure = load_pressure_curve_csv(pressure_path)
                self._pressure_source_label = f"相机目录 | {pressure_path}"
                self._log(f"触觉曲线来自相机目录：{pressure_path}")

            self._meta = load_pressure_meta(meta_path, mocap_path)
            self._preview_time = 0.0
            self._manual_adjusted = False
            self._refresh_views()

            if self._motion is None:
                self._set_status("当前会话没有可用动捕数据", "danger")
                return
            if self._pressure is None:
                self._set_status("当前记录没有重建触觉数据，也没有 pressure_left/right.csv", "warning")
                self._log(
                    f"已加载动捕，当前记录无触觉数据：{session.session_id} | "
                    f"recon_root={self._reconstructed_root}"
                )
                return

            auto_applied = False
            if self._review_root is not None and self._review_root.exists():
                # Apply the visual-mocap offset before estimating, so either
                # alignment mechanism uses it as the search centre immediately.
                auto_applied = self._try_apply_visual_alignment_prior(manual=False, reestimate=False)
            result = self._estimate_alignment_result(session.session_id)
            self._result = replace(result, session_id=session.session_id, manual_adjusted=False)
            self._delta_t2 = self._result.delta_t2
            self._manual_adjusted = False
            self._configure_delta_controls()
            self._refresh_views()
            self._update_rows()
            self._set_status("数据已加载", "normal")
            pressure_names = (
                f"recon:{reconstructed_rec_dir.name}"
                if reconstructed_rec_dir is not None
                else (
                    f"{pressure_left_path.name} + {pressure_right_path.name}"
                    if pressure_left_path is not None and pressure_right_path is not None
                    else (pressure_path.name if pressure_path is not None else "unknown_pressure")
                )
            )
            self._log(f"加载完成：{mocap_path.name} | {pressure_names} | source={self._pressure_source_label}")
            if auto_applied and self._visual_prior_source is not None:
                self._log(
                    f"已用视频-动捕对齐结果初始化 delta_t2={self._delta_t2:+.6f}s：{self._visual_prior_source.name}"
                )
        except Exception as exc:
            self._set_status(str(exc), "danger")
            self._log(f"加载失败：{exc}")

    def _configure_delta_controls(self) -> None:
        if self._mocap is None or self._pressure is None:
            self._set_controls_enabled(False)
            return
        duration = max(
            float(self._mocap.time_s[-1]) if len(self._mocap.time_s) else 1.0,
            float(self._pressure.time_s[-1]) if len(self._pressure.time_s) else 1.0,
            1.0,
        )
        self._updating_controls = True
        try:
            self.delta_slider.setRange(int(round(-duration * 1000.0)), int(round(duration * 1000.0)))
            self.delta_spin.setRange(-duration, duration)
            self.delta_slider.setValue(int(round(self._delta_t2 * 1000.0)))
            self.delta_spin.setValue(self._delta_t2)
        finally:
            self._updating_controls = False
        self._set_controls_enabled(True)

    def _on_delta_slider_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._set_delta_t2(value / 1000.0, manual=True)

    def _on_delta_spin_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self._set_delta_t2(float(value), manual=True)

    def _set_delta_t2(self, value: float, *, manual: bool) -> None:
        self._delta_t2 = float(value)
        self._updating_controls = True
        try:
            self.delta_slider.setValue(int(round(self._delta_t2 * 1000.0)))
            self.delta_spin.setValue(self._delta_t2)
        finally:
            self._updating_controls = False
        if manual:
            self._manual_adjusted = True
        if self._result is not None:
            self._result = replace(self._result, delta_t2=self._delta_t2, manual_adjusted=self._manual_adjusted)
        self._refresh_views()
        self._update_rows()

    def _refresh_views(self) -> None:
        if self._motion is not None:
            self.motion_canvas.render(self._preview_time)
            self.motion_info_label.setText(f"帧 {self.motion_canvas._frame_index} | {self._motion.path.name}")
        self._refresh_pressure_canvases()
        self._update_curve_selector_for_mode()
        if hasattr(self, "curve_canvas"):
            self._apply_curve_series_visibility()
            self.curve_canvas.render(
                self._mocap,
                self._pressure,
                self._preview_time,
                self._delta_t2,
                mode=self._current_alignment_mode(),
                dynamics=self._dynamics_curves,
                **self._fake_render_args(),
            )

    def _refresh_pressure_canvases(self) -> None:
        left_vals, left_valid = _sample_pressure_values(self._pressure_left, self._preview_time)
        right_vals, right_valid = _sample_pressure_values(self._pressure_right, self._preview_time)
        self.left_foot_canvas.set_values(left_vals, left_valid)
        self.right_foot_canvas.set_values(right_vals, right_valid)

    def _paired_left_visible(self) -> bool:
        return bool(self.pair_left_check.isChecked()) if hasattr(self, "pair_left_check") else True

    def _paired_right_visible(self) -> bool:
        return bool(self.pair_right_check.isChecked()) if hasattr(self, "pair_right_check") else True

    def _apply_curve_series_visibility(self) -> None:
        if not hasattr(self, "curve_canvas"):
            return
        left_on = self._paired_left_visible()
        right_on = self._paired_right_visible()
        self.curve_canvas.set_series_visibility(
            mocap_left=left_on,
            mocap_right=right_on,
            pressure_left=left_on,
            pressure_right=right_on,
            mocap_vgrf=self.mocap_vgrf_check.isChecked() if hasattr(self, "mocap_vgrf_check") else True,
            tactile_vgrf=self.tactile_vgrf_check.isChecked() if hasattr(self, "tactile_vgrf_check") else True,
        )

    def _on_curve_visibility_changed(self) -> None:
        if hasattr(self, "curve_canvas"):
            self._apply_curve_series_visibility()
            self.curve_canvas.render(
                self._mocap,
                self._pressure,
                self._preview_time,
                self._delta_t2,
                mode=self._current_alignment_mode(),
                dynamics=self._dynamics_curves,
                **self._fake_render_args(),
            )

    def _delta_step(self, frames: int) -> None:
        if self._pressure is not None and self._pressure.sample_fps > 0:
            step = frames / self._pressure.sample_fps
        elif self._mocap is not None and self._mocap.reference_fps > 0:
            step = frames / self._mocap.reference_fps
        else:
            step = frames / 40.0
        self._set_delta_t2(self._delta_t2 + step, manual=True)


    def _update_curve_selector_for_mode(self) -> None:
        dynamics = self._current_alignment_mode() == "dynamics"
        for widget in (self.pair_left_check, self.pair_right_check):
            widget.setVisible(not dynamics)
        if hasattr(self, "mocap_vgrf_check"):
            self.mocap_vgrf_check.setVisible(dynamics)
        if hasattr(self, "tactile_vgrf_check"):
            self.tactile_vgrf_check.setVisible(dynamics)
        if hasattr(self, "curve_hint_label"):
            if dynamics:
                self.curve_hint_label.setText("新机制：两条总曲线")
            else:
                self.curve_hint_label.setText("旧机制：L/R 成对显示（触觉+动捕同步）")

    def _current_alignment_mode(self) -> str:
        if hasattr(self, "mode_dynamics_btn") and self.mode_dynamics_btn.isChecked():
            return "dynamics"
        return "legacy"

    def _alignment_mode_label(self, mode: str | None = None) -> str:
        current = mode if mode is not None else self._current_alignment_mode()
        return "新机制" if current == "dynamics" else "旧机制"

    def _update_alignment_mode_button_styles(self) -> None:
        """Highlight the active alignment mechanism button."""
        active = (
            "QPushButton {"
            "background-color: #C4553B;"
            "color: #FFF8F1;"
            "border: 1px solid #9E3F2A;"
            "border-radius: 8px;"
            "font-weight: 700;"
            "padding: 8px 12px;"
            "text-align: left;"
            "}"
            "QPushButton:hover { background-color: #B04831; }"
            "QPushButton:pressed { background-color: #943C29; }"
        )
        idle = (
            "QPushButton {"
            "background-color: #F7EEE4;"
            "color: #3A2F27;"
            "border: 1px solid #D4BFAE;"
            "border-radius: 8px;"
            "font-weight: 600;"
            "padding: 8px 12px;"
            "text-align: left;"
            "}"
            "QPushButton:hover { background-color: #EFE0CF; border-color: #C9A98E; }"
            "QPushButton:pressed { background-color: #E5D2BC; }"
        )
        if hasattr(self, "mode_legacy_btn"):
            self.mode_legacy_btn.setStyleSheet(active if self.mode_legacy_btn.isChecked() else idle)
        if hasattr(self, "mode_dynamics_btn"):
            self.mode_dynamics_btn.setStyleSheet(active if self.mode_dynamics_btn.isChecked() else idle)

    def _on_alignment_mode_toggled(self, checked: bool) -> None:
        # Always refresh styles so both checked/unchecked transitions update color.
        self._update_alignment_mode_button_styles()
        # The exclusive button group emits once for the newly selected button.
        if not checked:
            return
        mode = self._current_alignment_mode()
        if mode == getattr(self, "_alignment_mode", None):
            # Still force a redraw so residual twin-axis curves cannot linger.
            self._update_curve_selector_for_mode()
            self._refresh_views()
            return
        self._alignment_mode = mode
        self._update_curve_selector_for_mode()
        if self._session is None or self._pressure is None or self._meta is None:
            self._log(f"已切换到{self._alignment_mode_label(mode)}（加载数据后将自动对齐）")
            if mode == "legacy":
                self._dynamics_curves = None
            self._refresh_views()
            return
        self._log(f"切换到{self._alignment_mode_label(mode)}，自动重估对齐")
        self._reestimate_alignment()

    def _estimate_alignment_result(self, session_id: str | None = None) -> PressureAlignmentResult:
        if self._pressure is None or self._meta is None:
            raise RuntimeError("缺少触觉曲线或 meta 信息，无法估计对齐。")
        mode = self._current_alignment_mode()
        if mode == "dynamics":
            if self._motion is None:
                raise RuntimeError("动力学对齐需要完整 BVH 轨迹。")
            left_total = None
            right_total = None
            pressure_time = None
            # Prefer raw (unnormalized) sensor channel sums for measured vGRF/BW.
            if self._pressure_left is not None and self._pressure_right is not None:
                left_total = self._pressure_left.sensor_totals
                right_total = self._pressure_right.sensor_totals
                # Align both feet onto a shared absolute timeline starting at 0.
                left_t = np.asarray(self._pressure_left.time_s, dtype=np.float64)
                right_t = np.asarray(self._pressure_right.time_s, dtype=np.float64)
                origin = float(min(left_t[0] if len(left_t) else 0.0, right_t[0] if len(right_t) else 0.0))
                # sensor loaders already rezero each foot independently; use pressure curve time if available.
                pressure_time = self._pressure.time_s if self._pressure is not None else left_t
                # Rebuild totals on each foot's own time, then let the dynamics builder resample.
                # If left/right times differ, pass left time with left totals and resample right separately via combined path:
                # simplest robust path: interpolate both onto the common pressure curve time grid.
                if self._pressure is not None and len(self._pressure.time_s):
                    pressure_time = self._pressure.time_s
                    left_total = np.interp(pressure_time, left_t, left_total) if len(left_t) else np.zeros_like(pressure_time)
                    right_total = np.interp(pressure_time, right_t, right_total) if len(right_t) else np.zeros_like(pressure_time)

            result, curves = estimate_pressure_alignment_dynamics(
                self._motion,
                self._pressure,
                self._meta,
                axis_preset=self._session.runtime_options.axis_preset if self._session is not None else "zup",
                search_window_ms=200,
                reference_fps=self._session.reference_visual_fps if self._session is not None else None,
                left_total=left_total,
                right_total=right_total,
                pressure_time_s=pressure_time,
            )
            self._dynamics_curves = curves
            note = getattr(curves, "event_note", "") or ""
            scatter = getattr(curves, "event_scatter", float("nan"))
            n_evt = int(getattr(curves, "event_n", 0) or 0)
            ok = bool(getattr(curves, "event_ok", False))
            self._log(
                "新机制(触地事件): "
                f"ok={ok} | n={n_evt} | scatter={scatter if scatter == scatter else float('nan'):.4f}s | "
                f"unit_scale={curves.length_scale_to_m:g} | "
                f"delta_t2={float(result.delta_t2):+.6f}s | t_coarse={float(result.t_coarse):+.6f}s"
                + (f" | note={note}" if note else "")
            )
        else:
            if self._mocap is None:
                raise RuntimeError("旧机制对齐需要足底贴地曲线。")
            self._dynamics_curves = None
            result = estimate_pressure_alignment(
                self._mocap,
                self._pressure,
                self._meta,
                search_window_ms=200,
            )
        if session_id:
            result = replace(result, session_id=session_id)
        return result

    def _reestimate_alignment(self) -> None:
        if self._session is None or self._pressure is None or self._meta is None:
            self._set_status("请先加载会话与触觉数据", "warning")
            return
        try:
            result = self._estimate_alignment_result(self._session.session_id)
            self._result = replace(result, manual_adjusted=False)
            self._manual_adjusted = False
            self._delta_t2 = self._result.delta_t2
            self._configure_delta_controls()
            self._refresh_views()
            self._update_rows()
            mode_name = self._alignment_mode_label()
            prior_note = f" | prior={self._visual_prior_source.name}" if self._visual_prior_source else ""
            self._set_status(f"{mode_name}已自动对齐", "normal")
            self._log(
                f"{mode_name}自动对齐：delta_t2={self._delta_t2:+.6f}s | "
                f"t_coarse={self._result.t_coarse:+.6f}s{prior_note}"
            )
        except Exception as exc:
            self._set_status(str(exc), "danger")
            self._log(f"对齐估计失败：{exc}")

    def _import_reconstructed_tactile_root(self) -> None:
        start_dir = str(
            self._reconstructed_root
            if self._reconstructed_root is not None and self._reconstructed_root.exists()
            else (
                DEFAULT_RECONSTRUCTED_TACTILE_ROOT.parent
                if DEFAULT_RECONSTRUCTED_TACTILE_ROOT.parent.exists()
                else Path.cwd()
            )
        )
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择重建触觉数据文件夹（reconstruction_<timestamp>）",
            start_dir,
        )
        if not folder:
            return
        root = Path(folder)
        has_manifest = any(root.rglob("reconstruction_manifest.csv"))
        has_legacy = any(root.rglob("pressure_left_t*.csv"))
        has_continuous = any(root.rglob("pressure_left.csv")) and any(root.rglob("pressure_right.csv"))
        if not has_manifest and not has_legacy and not has_continuous:
            self._set_status("所选文件夹中未找到重建触觉数据", "warning")
            self._log(
                "重建触觉导入失败：目录下没有 reconstruction_manifest.csv / "
                f"pressure_left.csv / pressure_left_t*.csv | {root}"
            )
            return
        self._reconstructed_root = root
        self._log(f"已设置重建触觉数据根目录：{root}")
        if self._session is not None:
            self._load_from_session(self._session)
            self._set_status("已切换到重建触觉数据源", "normal")
        else:
            self._set_status("重建触觉目录已设置，加载会话后将优先使用", "normal")

    def _import_visual_alignment_prior(self) -> None:
        if self._session is None:
            self._set_status("请先加载会话", "warning")
            return
        start_dir = str(self._review_root if self._review_root is not None and self._review_root.exists() else Path.cwd())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择视频-动捕对齐结果文件夹",
            start_dir,
        )
        if not folder:
            return
        try:
            self._review_root = Path(folder)
            csv_path = resolve_visual_alignment_csv(self._session.session_id, self._review_root)
            if csv_path is None:
                self._set_status("所选文件夹中未找到当前会话对应 CSV", "warning")
                self._log(
                    f"未在文件夹中匹配到视频对齐 CSV：session={self._session.session_id} | dir={self._review_root}"
                )
                return
            self._apply_visual_alignment_csv(csv_path, manual=False)
            self._reestimate_alignment()
            self._log(
                f"视频对齐辅助导入后已用{self._alignment_mode_label()}自动对齐："
                f"delta_t2={self._delta_t2:+.6f}s"
            )
        except Exception as exc:
            self._set_status(str(exc), "danger")
            self._log(f"导入视频对齐辅助失败：{exc}")

    def _try_apply_visual_alignment_prior(self, *, manual: bool, reestimate: bool = True) -> bool:
        if self._session is None:
            return False
        if self._review_root is None:
            return False
        csv_path = resolve_visual_alignment_csv(self._session.session_id, self._review_root)
        if csv_path is None:
            return False
        self._apply_visual_alignment_csv(csv_path, manual=manual)
        if reestimate:
            self._reestimate_alignment()
        return True

    def _apply_visual_alignment_csv(self, csv_path: Path, *, manual: bool) -> None:
        if self._session is None or self._pressure is None:
            raise RuntimeError("请先加载动捕与触觉数据，再导入视频对齐辅助。")

        delta_t = load_visual_alignment_offset(csv_path)
        # 先验：触觉时间 ≈ 视觉时间。导入后把该偏移写入 t_coarse，供当前机制自动搜索窗口使用。
        self._visual_prior_source = csv_path
        if self._review_root is None:
            self._review_root = csv_path.parent
        if self._meta is None:
            self._meta = PressureMetaInfo(
                source_path=None,
                started_at_iso=None,
                epoch_monotonic_us=None,
                t_coarse=delta_t,
                t_coarse_source="visual_alignment_csv",
            )
        else:
            self._meta = replace(
                self._meta,
                t_coarse=delta_t,
                t_coarse_source="visual_alignment_csv",
            )
        if self._result is None:
            self._result = PressureAlignmentResult(
                session_id=self._session.session_id,
                delta_t2=delta_t,
                delta_t2_left=delta_t,
                delta_t2_right=delta_t,
                peak_left=0.0,
                peak_right=0.0,
                t_coarse=delta_t,
                search_window_ms=200,
                manual_adjusted=manual,
                reference_fps=(
                    self._pressure.sample_fps
                    if self._pressure.sample_fps > 0
                    else (self._mocap.reference_fps if self._mocap is not None else 40.0)
                ),
                mocap_source_file=self._mocap.source_path.name if self._mocap is not None else "unknown_mocap",
                pressure_source_file=self._pressure.source_path.name,
                axis_preset=(
                    self._mocap.axis_preset
                    if self._mocap is not None
                    else self._session.runtime_options.axis_preset
                ),
                exported_at="",
            )
        else:
            self._result = replace(
                self._result,
                t_coarse=delta_t,
                manual_adjusted=manual,
            )
        # Display the imported visual-mocap prior until the automatic estimate
        # below replaces it with the result from the selected mechanism.
        self._delta_t2 = float(delta_t)
        self._manual_adjusted = bool(manual)
        self._log(f"视频对齐辅助已加载：{csv_path.name} | t_coarse={delta_t:+.6f}s")

    def _update_rows(self) -> None:
        if self._result is None:
            self.coarse_row.set_value("--")
            self.visual_prior_row.set_value("--")
            self.left_row.set_value("--")
            self.right_row.set_value("--")
            self.final_row.set_value("--")
            if hasattr(self, "tactile_source_row"):
                self.tactile_source_row.set_value(self._pressure_source_label or "--")
            return

        result = self._result
        self.coarse_row.set_value(f"{result.t_coarse:+.3f}s")
        if self._visual_prior_source is not None:
            self.visual_prior_row.set_value(
                f"{self._delta_t2:+.3f}s | {self._visual_prior_source.name}"
            )
        else:
            self.visual_prior_row.set_value("未导入")
        self.source_row.set_value(f"{result.mocap_source_file} | {result.pressure_source_file}")
        if hasattr(self, "tactile_source_row"):
            self.tactile_source_row.set_value(self._pressure_source_label or "--")
        self.left_row.set_value(f"{result.delta_t2_left:+.3f}s | peak {result.peak_left:.3f}")
        self.right_row.set_value(f"{result.delta_t2_right:+.3f}s | peak {result.peak_right:.3f}")
        self.final_row.set_value(f"{self._delta_t2:+.3f}s" + (" | 手动" if self._manual_adjusted else " | 自动"))

        delta_gap = abs(result.delta_t2_left - result.delta_t2_right)
        if delta_gap > 0.05:
            self._set_status(f"左右脚差异 {delta_gap * 1000:.1f}ms，建议人工介入", "danger")
        elif result.peak_left < 0.3 or result.peak_right < 0.3:
            self._set_status("互相关峰值低于 0.3，对齐可信度不足", "danger")
        else:
            self._set_status("左右脚结果一致", "normal")






