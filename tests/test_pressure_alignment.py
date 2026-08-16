from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.bvh_parser import load_bvh_motion_preserve_frames
from utils.pressure_alignment import (
    list_reconstructed_segments,
    load_reconstructed_pressure_sensors,
    resolve_reconstructed_rec_dir,
    build_pressure_curve_set,
    build_pressure_aligned_curve_matrix,
    estimate_pressure_alignment,
    export_pressure_alignment_bundle,
    load_mocap_foot_curves,
    load_pressure_curve_csv,
    load_pressure_meta,
    load_pressure_sensor_csv,
    load_visual_alignment_offset,
    normalize_trial_code,
    resolve_visual_alignment_csv,
)
from utils.session import find_cam_session


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PressureAlignmentTests(unittest.TestCase):
    def test_pressure_alignment_pipeline_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bvh_path = root / "S100_1_aligned.bvh"
            csv_path = root / "contact_curve.csv"
            meta_path = root / "meta.json"

            frame_lines = []
            for index in range(20):
                z_value = 0 if index == 2 else 1
                frame_lines.append(f"0 0 0 0 0 0 0 0 {z_value} 1 0 {z_value}")

            bvh_text = textwrap.dedent(
                """
                HIERARCHY
                ROOT Hips
                {
                    OFFSET 0 0 0
                    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
                    JOINT LeftFoot
                    {
                        OFFSET 0 0 0
                        CHANNELS 3 Xposition Yposition Zposition
                        End Site
                        {
                            OFFSET 0 0 0
                        }
                    }
                    JOINT RightFoot
                    {
                        OFFSET 0 0 0
                        CHANNELS 3 Xposition Yposition Zposition
                        End Site
                        {
                            OFFSET 0 0 0
                        }
                    }
                }
                MOTION
                Frames: 20
                Frame Time: 0.1
                """
            ).strip() + "\n" + "\n".join(frame_lines)
            _write_text(bvh_path, bvh_text)

            pressure_rows = ["time_s,left_sum,right_sum"]
            for index in range(20):
                value = 1 if index == 3 else 0
                pressure_rows.append(f"{index * 0.1:.1f},{value},{value}")
            pressure_csv = "\n".join(pressure_rows)
            _write_text(csv_path, pressure_csv)

            meta_json = textwrap.dedent(
                """
                {
                  "schema_version": 2,
                  "mode": "pressure_only",
                  "started_at_iso": "2026-08-14T10:00:00",
                  "epoch_monotonic_us": 123456789,
                  "subject": "demo"
                }
                """
            ).strip()
            _write_text(meta_path, meta_json)

            motion = load_bvh_motion_preserve_frames(bvh_path)
            mocap = load_mocap_foot_curves(motion, axis_preset="raw")
            pressure = load_pressure_curve_csv(csv_path)
            meta = load_pressure_meta(meta_path, bvh_path)

            result = estimate_pressure_alignment(mocap, pressure, meta, search_window_ms=0)
            self.assertAlmostEqual(result.delta_t2, 0.1, places=3)
            self.assertGreater(result.peak_left, 0.3)
            self.assertGreater(result.peak_right, 0.3)

            matrix, headers = build_pressure_aligned_curve_matrix(mocap, pressure, result.delta_t2)
            self.assertEqual(headers, ["time_s", "mocap_left", "mocap_right", "pressure_left", "pressure_right"])
            self.assertEqual(matrix.shape[1], 5)

            outputs = export_pressure_alignment_bundle(
                output_root=root / "output",
                result=result,
                mocap=mocap,
                pressure=pressure,
            )
            self.assertTrue(outputs["metadata"].exists())
            self.assertTrue(outputs["curves"].exists())
            payload = outputs["metadata"].read_text(encoding="utf-8")
            self.assertIn("_pressure_alignment.json", str(outputs["metadata"]))
            self.assertIn('"delta_t2"', payload)
            self.assertIn('"manual_adjusted"', payload)

    def test_raw_left_right_pressure_sensor_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left_path = root / "pressure_left.csv"
            right_path = root / "pressure_right.csv"
            header = "frame_idx,t_us," + ",".join(str(index) for index in range(1, 49))
            left_rows = [
                header,
                "0,46000," + ",".join("1" for _ in range(48)),
                "1,71000," + ",".join("2" for _ in range(48)),
            ]
            right_rows = [
                header,
                "0,46000," + ",".join("3" for _ in range(48)),
                "1,71000," + ",".join("4" for _ in range(48)),
            ]
            _write_text(left_path, "\n".join(left_rows))
            _write_text(right_path, "\n".join(right_rows))

            left = load_pressure_sensor_csv(left_path)
            right = load_pressure_sensor_csv(right_path)
            pressure = build_pressure_curve_set(left, right, reference_fps=40.0)

            self.assertEqual(left.sensor_values.shape, (2, 48))
            self.assertEqual(right.sensor_values.shape, (2, 48))
            self.assertAlmostEqual(float(left.time_s[0]), 0.0)
            self.assertEqual(len(pressure.left_sum), len(pressure.right_sum))
            self.assertGreaterEqual(float(pressure.left_sum[-1]), float(pressure.left_sum[0]))

    def test_find_cam_session_in_action_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            expected = root / "S1310" / "rec20260810_211816_demo_S1310_3"
            expected.mkdir(parents=True)

            resolved = find_cam_session("S1310_3", root)

            self.assertEqual(resolved, expected)

    def test_visual_alignment_csv_offset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "S10011.csv"
            _write_text(
                csv_path,
                "\n".join(
                    [
                        "标记,视觉时间(s),视觉帧,动捕时间(s),动捕帧,偏移量(s),会话编号,导出时间",
                        "start,0.128363,5,0.179708,22,-0.051345,S10011,2026-08-12 16:13:07",
                        "end,18.561292,723,18.612637,2234,-0.051345,S10011,2026-08-12 16:13:07",
                    ]
                ),
            )

            offset = load_visual_alignment_offset(csv_path)
            self.assertAlmostEqual(offset, -0.051345, places=6)
            self.assertEqual(normalize_trial_code("S1001_1"), "S10011")
            self.assertEqual(resolve_visual_alignment_csv("S1001_1", root), csv_path)
            self.assertIsNone(resolve_visual_alignment_csv("S9999_9", root))

    def test_mocap_foot_curve_grounded_high_lifted_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bvh_path = root / "foot_height.bvh"

            # Build two representations:
            # - zup: source is Y-up, height lives in Y, then transform maps it to display Z
            # - raw: source is Z-up, height lives in Z
            frame_lines = []
            for index in range(10):
                left_y = 0.0 if index < 5 else 1.0
                right_y = 1.0 if index < 5 else 0.0
                frame_lines.append(f"0 0 0 0 0 0 0 {left_y} 0 0 {right_y} 0")

            bvh_text = textwrap.dedent(
                """
                HIERARCHY
                ROOT Hips
                {
                    OFFSET 0 0 0
                    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
                    JOINT LeftFoot
                    {
                        OFFSET 0 0 0
                        CHANNELS 3 Xposition Yposition Zposition
                        End Site
                        {
                            OFFSET 0 0 0
                        }
                    }
                    JOINT RightFoot
                    {
                        OFFSET 0 0 0
                        CHANNELS 3 Xposition Yposition Zposition
                        End Site
                        {
                            OFFSET 0 0 0
                        }
                    }
                }
                MOTION
                Frames: 10
                Frame Time: 0.1
                """
            ).strip() + "\n" + "\n".join(frame_lines)
            _write_text(bvh_path, bvh_text)

            motion = load_bvh_motion_preserve_frames(bvh_path)
            mocap_zup = load_mocap_foot_curves(motion, axis_preset="zup")
            self.assertGreater(float(np.min(mocap_zup.left_sum[:5])), 0.9)
            self.assertLess(float(np.max(mocap_zup.left_sum[5:])), 0.1)
            self.assertLess(float(np.max(mocap_zup.right_sum[:5])), 0.1)
            self.assertGreater(float(np.min(mocap_zup.right_sum[5:])), 0.9)

            # raw mode expects upright on Z: rewrite frames so height is Z.
            frame_lines_raw = []
            for index in range(10):
                left_z = 0.0 if index < 5 else 1.0
                right_z = 1.0 if index < 5 else 0.0
                frame_lines_raw.append(f"0 0 0 0 0 0 0 0 {left_z} 0 0 {right_z}")
            bvh_path_raw = root / "foot_height_raw.bvh"
            _write_text(bvh_path_raw, bvh_text.split("MOTION")[0] + "MOTION\nFrames: 10\nFrame Time: 0.1\n" + "\n".join(frame_lines_raw))
            motion_raw = load_bvh_motion_preserve_frames(bvh_path_raw)
            mocap_raw = load_mocap_foot_curves(motion_raw, axis_preset="raw")
            self.assertGreater(float(np.min(mocap_raw.left_sum[:5])), 0.9)
            self.assertLess(float(np.max(mocap_raw.left_sum[5:])), 0.1)
            self.assertLess(float(np.max(mocap_raw.right_sum[:5])), 0.1)
            self.assertGreater(float(np.min(mocap_raw.right_sum[5:])), 0.9)



    def test_reconstructed_tactile_loader_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "reconstruction_20260816_222245"
            rec_dir = root / "20260810" / "S14" / "rec20260810_213839_demo_S1401_1"
            rec_dir.mkdir(parents=True)

            header = "frame_idx,t_us,valid_mask,source_frame_idx,source_t_us," + ",".join(str(i) for i in range(1, 49))
            # two segments with absolute t_us (us)
            left_t0 = [
                header,
                "0,1000,1,0,1000," + ",".join("1" for _ in range(48)),
                "1,26000,0,,," + ",".join("2" for _ in range(48)),
            ]
            right_t0 = [
                header,
                "0,2000,1,0,2000," + ",".join("3" for _ in range(48)),
                "1,27000,0,,," + ",".join("4" for _ in range(48)),
            ]
            left_t1 = [
                header,
                "0,51000,1,2,51000," + ",".join("5" for _ in range(48)),
                "1,76000,1,3,76000," + ",".join("6" for _ in range(48)),
            ]
            right_t1 = [
                header,
                "0,52000,1,2,52000," + ",".join("7" for _ in range(48)),
                "1,77000,1,3,77000," + ",".join("8" for _ in range(48)),
            ]
            _write_text(rec_dir / "pressure_left_t0.csv", "\n".join(left_t0))
            _write_text(rec_dir / "pressure_right_t0.csv", "\n".join(right_t0))
            _write_text(rec_dir / "pressure_left_t1.csv", "\n".join(left_t1))
            _write_text(rec_dir / "pressure_right_t1.csv", "\n".join(right_t1))
            _write_text(
                rec_dir / "reconstruction_manifest.csv",
                "segment_name,left_rows,right_rows,left_inserted,right_inserted,left_dt_hat_us,right_dt_hat_us\n"
                "t0,2,2,1,1,25000.0,25000.0\n"
                "t1,2,2,0,0,25000.0,25000.0\n",
            )

            self.assertEqual(list_reconstructed_segments(rec_dir), ["t0", "t1"])
            self.assertEqual(resolve_reconstructed_rec_dir("S1401_1", root), rec_dir)
            self.assertEqual(resolve_reconstructed_rec_dir("S14011", root), rec_dir)

            left, right = load_reconstructed_pressure_sensors(rec_dir)
            self.assertEqual(left.sensor_values.shape, (4, 48))
            self.assertEqual(right.sensor_values.shape, (4, 48))
            # earliest absolute time is left t0 start (0.001s), so left starts near 0
            self.assertAlmostEqual(float(left.time_s[0]), 0.0, places=6)
            self.assertGreater(float(right.time_s[0]), 0.0)
            # second segment continues after first
            self.assertGreater(float(left.time_s[2]), float(left.time_s[1]))

            pressure = build_pressure_curve_set(left, right, reference_fps=40.0)
            self.assertEqual(len(pressure.left_sum), len(pressure.right_sum))
            self.assertGreater(float(np.max(pressure.left_sum)), 0.0)

if __name__ == "__main__":
    unittest.main()
