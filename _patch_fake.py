from pathlib import Path
import re

path = Path("ui/pressure_alignment_page.py")
text = path.read_text(encoding="utf-8")
nl = "\r\n" if "\r\n" in text else "\n"

def rep(old: str, new: str, label: str) -> None:
    global text
    variants = [old, old.replace("\n", "\r\n"), old.replace("\r\n", "\n")]
    for o in variants:
        if o in text:
            text = text.replace(o, new.replace("\n", nl), 1)
            print("ok", label)
            return
    raise SystemExit(f"missing: {label}")

# imports
if "import csv" not in text:
    rep("from dataclasses import replace\nfrom pathlib import Path\nfrom typing import Callable\n",
        "from dataclasses import replace\nfrom pathlib import Path\nfrom typing import Callable\nimport csv\nfrom datetime import datetime\n",
        "imports csv/datetime")

if "DEFAULT_OUTPUT_ROOT" not in text.split("from config import",1)[1][:200]:
    rep("from config import DEFAULT_RECONSTRUCTED_TACTILE_ROOT, DEFAULT_VISUAL_ALIGN_REVIEW_ROOT\n",
        "from config import DEFAULT_OUTPUT_ROOT, DEFAULT_RECONSTRUCTED_TACTILE_ROOT, DEFAULT_VISUAL_ALIGN_REVIEW_ROOT\n",
        "import DEFAULT_OUTPUT_ROOT")

# canvas render signature: add fake_segments parameter and draw them
rep(
'''    def render(
        self,
        mocap: MocapFootCurveSet | None,
        pressure: PressureCurveSet | None,
        preview_time: float,
        delta_t2: float,
        *,
        mode: str = "legacy",
        dynamics: DynamicsVgrfCurveSet | None = None,
    ) -> None:
''',
'''    def render(
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
''',
"render signature")

# helper method before render or after _plot_series - insert draw fake helper after _plot_series end is hard.
# Instead inject draw calls before axvline preview in both branches.

draw_helper = '''
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
'''

if "def _draw_fake_markers" not in text:
    rep(
'''    def render(
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
''',
draw_helper + '''
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
''',
"draw helper")

# dynamics branch axvline
rep(
'''            self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
            self.ax.set_xlabel("时间（秒）")
            if labels:
                self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
            else:
                self.ax.text(0.5, 0.5, "没有可见曲线", transform=self.ax.transAxes, ha="center", va="center")

            # Strike-estimator diagnostics (offset is event-based, not correlation).
''',
'''            self._draw_fake_markers(fake_segments=fake_segments, pending_fake_start=pending_fake_start)
            self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
            self.ax.set_xlabel("时间（秒）")
            # Merge line labels with fake marker labels.
            h2, l2 = self.ax.get_legend_handles_labels()
            if h2:
                self.ax.legend(h2, l2, loc="upper right", fontsize=11, ncol=2)
            elif labels:
                self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
            else:
                self.ax.text(0.5, 0.5, "没有可见曲线", transform=self.ax.transAxes, ha="center", va="center")

            # Strike-estimator diagnostics (offset is event-based, not correlation).
''',
"dynamics fake markers")

# legacy branch axvline
rep(
'''        self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
        self.ax.set_xlabel("时间（秒）")
        self.ax.set_ylabel("归一化")
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
        else:
            self.ax.text(0.5, 0.5, "没有可见曲线", transform=self.ax.transAxes, ha="center", va="center")
''',
'''        self._draw_fake_markers(fake_segments=fake_segments, pending_fake_start=pending_fake_start)
        self.ax.axvline(preview_time, color="#241C17", linewidth=1.5, alpha=0.52)
        self.ax.set_xlabel("时间（秒）")
        self.ax.set_ylabel("归一化")
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=2)
        else:
            self.ax.text(0.5, 0.5, "没有可见曲线", transform=self.ax.transAxes, ha="center", va="center")
''',
"legacy fake markers")

# page state init fields near _preview_time / _dynamics
rep(
'''        self._preview_time = 0.0
''',
'''        self._preview_time = 0.0
        self._fake_segments: list[dict] = []
        self._pending_fake_start: dict | None = None
''',
"state fields")

# control row buttons after delta controls - insert mark group before return panel of curve section
rep(
'''        control_row.addWidget(self.delta_spin)
        layout.addLayout(control_row)
        return panel
''',
'''        control_row.addWidget(self.delta_spin)
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
''',
"fake controls")

# enable fake buttons with controls
rep(
'''    def _set_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.delta_slider.setEnabled(enabled)
        self.delta_spin.setEnabled(enabled)
        for button in (self.delta_minus_10, self.delta_minus_1, self.delta_plus_1, self.delta_plus_10):
            button.setEnabled(enabled)
''',
'''    def _set_controls_enabled(self, enabled: bool) -> None:
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
''',
"enable fake buttons")

# reset fake state on _reset_view (trial switch)
rep(
'''        self._dynamics_curves = None
        self._set_controls_enabled(False)
        # Keep L/R (and dynamics) curve checkbox state across trial switches.
''',
'''        self._dynamics_curves = None
        self._fake_segments = []
        self._pending_fake_start = None
        self._set_controls_enabled(False)
        if hasattr(self, "fake_status_label"):
            self.fake_status_label.setText("Fake片段：0 段")
        # Keep L/R (and dynamics) curve checkbox state across trial switches.
''',
"reset fake state")

# helper to pass fake args in all curve_canvas.render calls - replace pattern
# We'll add a method _curve_render_kwargs and update call sites carefully.

# Add methods before _on_curve_clicked or after shift_preview_frames
rep(
'''    def _on_curve_clicked(self, x_value: float) -> None:
''',
'''    def _fake_render_args(self) -> dict:
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
            f"R#{info['right_frame_idx']} ({info['right_time_s'] if info['right_time_s'] !== '' else 'NA'})"
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
        left_path = str(self._pressure_left.source_path) if self._pressure_left is not None else ""
        right_path = str(self._pressure_right.source_path) if self._pressure_right is not None else ""
        exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build inclusive left-frame coverage from each completed segment.
        fake_left_to_seg: dict[int, int] = {}
        for seg in self._fake_segments:
            a = int(seg["start_left_frame_idx"])
            b = int(seg["end_left_frame_idx"])
            lo, hi = (a, b) if a <= b else (b, a)
            lo = max(0, lo)
            hi = min(len(left_times) - 1, hi)
            for idx in range(lo, hi + 1):
                # Keep first segment id if overlap.
                fake_left_to_seg.setdefault(idx, int(seg["segment_id"]))

        header = [
            "row_type",
            "session_id",
            "trial_code",
            "segment_id",
            "left_frame_idx",
            "left_time_s",
            "right_frame_idx",
            "right_time_s",
            "is_fake",
            "preview_start_s",
            "preview_end_s",
            "delta_t2",
            "left_source",
            "right_source",
            "exported_at",
            "note",
        ]
        rows: list[list[object]] = [header]

        # Segment summary rows first.
        for seg in self._fake_segments:
            rows.append([
                "segment",
                session_id,
                trial_code,
                int(seg["segment_id"]),
                f"{int(seg['start_left_frame_idx'])}-{int(seg['end_left_frame_idx'])}",
                f"{float(seg['start_left_time_s']):.6f}-{float(seg['end_left_time_s']):.6f}",
                f"{seg['start_right_frame_idx']}-{seg['end_right_frame_idx']}",
                f"{seg['start_right_time_s']}-{seg['end_right_time_s']}",
                1,
                f"{float(seg['start_time_s']):.6f}",
                f"{float(seg['end_time_s']):.6f}",
                f"{float(seg.get('delta_t2', self._delta_t2)):.6f}",
                left_path,
                right_path,
                exported_at,
                "fake_segment_range_left_based",
            ])

        # Frame-level rows: left base + nearest right.
        for left_idx in sorted(fake_left_to_seg):
            left_t = float(left_times[left_idx])
            if len(right_times):
                right_idx = int(np.argmin(np.abs(right_times - left_t)))
                right_t = float(right_times[right_idx])
            else:
                right_idx = ""
                right_t = ""
            seg_id = fake_left_to_seg[left_idx]
            rows.append([
                "frame",
                session_id,
                trial_code,
                seg_id,
                left_idx,
                f"{left_t:.6f}",
                right_idx,
                f"{right_t:.6f}" if right_t !== "" else "",
                1,
                "",
                "",
                f"{float(self._delta_t2):.6f}",
                left_path,
                right_path,
                exported_at,
                "left_base_right_nearest",
            ])

        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

        self._log(f"已导出 Fake 帧 CSV：{path} | segments={len(self._fake_segments)} frames={len(fake_left_to_seg)}")
        QtWidgets.QMessageBox.information(
            self,
            "导出完成",
            f"已导出 {len(self._fake_segments)} 个 Fake 片段，共 {len(fake_left_to_seg)} 个左脚基准帧。\\n{path}",
        )

    def _on_curve_clicked(self, x_value: float) -> None:
''',
"fake mark/export methods")

# Fix accidental JS-style !== in inserted code
text = text.replace("!==", "!=")

# Patch all curve_canvas.render call sites to include fake args.
# Safer: wrap via modifying _refresh_views and set_preview_time and _on_curve_visibility_changed

def add_fake_kwargs_to_render_blocks():
    global text
    pattern = re.compile(
        r"self\.curve_canvas\.render\(\s*"
        r"self\._mocap,\s*"
        r"self\._pressure,\s*"
        r"self\._preview_time,\s*"
        r"self\._delta_t2,\s*"
        r"mode=self\._current_alignment_mode\(\),\s*"
        r"dynamics=self\._dynamics_curves,\s*"
        r"\)",
        re.S,
    )
    repl = (
        "self.curve_canvas.render(\n"
        "                self._mocap,\n"
        "                self._pressure,\n"
        "                self._preview_time,\n"
        "                self._delta_t2,\n"
        "                mode=self._current_alignment_mode(),\n"
        "                dynamics=self._dynamics_curves,\n"
        "                **self._fake_render_args(),\n"
        "            )"
    )
    text2, n = pattern.subn(repl, text)
    if n == 0:
        raise SystemExit("no render call sites updated")
    text = text2
    print(f"ok updated {n} render call sites")

add_fake_kwargs_to_render_blocks()

# ensure re imported in page methods - top level has no re; export uses re
if "import re\n" not in text:
    rep("import csv\nfrom datetime import datetime\n", "import csv\nimport re\nfrom datetime import datetime\n", "import re")

path.write_text(text, encoding="utf-8")
print("wrote file")
