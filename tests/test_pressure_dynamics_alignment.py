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
from utils.pressure_alignment import PressureCurveSet, load_pressure_meta
from utils.pressure_dynamics_alignment import (
    estimate_length_scale_to_meters,
    scale_joint_xyz,

    body_com,
    measured_vgrf_bw,
    predict_vgrf_bw,

    build_dynamics_vgrf_curves,

    estimate_delta_t2,
    estimate_pressure_alignment_dynamics,
    predict_vgrf_bw,
    measured_vgrf_bw,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class DynamicsAlignmentTests(unittest.TestCase):
    def test_estimate_delta_t2_recovers_known_lag(self) -> None:
        fs = 100.0
        t = np.arange(0.0, 8.0, 1.0 / fs)
        # Simple periodic GRF-like waveform in body weights.
        base = 1.0 + 0.35 * np.sin(2.0 * np.pi * 1.2 * t) + 0.08 * np.sin(2.0 * np.pi * 2.4 * t)
        true_lag = 0.07
        # Build synthetic CoM whose second derivative recovers base after +1 transform.
        # a/g + 1 = base  => a = (base-1)*g
        g = 9.80665
        a = (base - 1.0) * g
        # integrate twice with zero mean velocity/position bias
        v = np.cumsum(a) / fs
        v = v - np.mean(v)
        p = np.cumsum(v) / fs
        p = p - np.mean(p)
        com = np.column_stack([np.zeros_like(p), np.zeros_like(p), p])
        joint_xyz = {
            "Spine": com + np.array([0.0, 0.0, 0.2]),
            "Neck": com + np.array([0.0, 0.0, 0.5]),
            "Head": com + np.array([0.0, 0.0, 0.7]),
            "LeftArm": com + np.array([0.2, 0.0, 0.4]),
            "LeftForeArm": com + np.array([0.35, 0.0, 0.3]),
            "LeftHand": com + np.array([0.45, 0.0, 0.25]),
            "LeftHandEnd": com + np.array([0.5, 0.0, 0.25]),
            "RightArm": com + np.array([-0.2, 0.0, 0.4]),
            "RightForeArm": com + np.array([-0.35, 0.0, 0.3]),
            "RightHand": com + np.array([-0.45, 0.0, 0.25]),
            "RightHandEnd": com + np.array([-0.5, 0.0, 0.25]),
            "LeftUpLeg": com + np.array([0.1, 0.0, 0.0]),
            "LeftLeg": com + np.array([0.1, 0.0, -0.4]),
            "LeftFoot": com + np.array([0.1, 0.0, -0.8]),
            "LeftToeBase": com + np.array([0.15, 0.0, -0.85]),
            "RightUpLeg": com + np.array([-0.1, 0.0, 0.0]),
            "RightLeg": com + np.array([-0.1, 0.0, -0.4]),
            "RightFoot": com + np.array([-0.1, 0.0, -0.8]),
            "RightToeBase": com + np.array([-0.15, 0.0, -0.85]),
        }

        # Pressure total delayed by true_lag; split half/half.
        delay = int(round(true_lag * fs))
        delayed = np.roll(base, delay)
        delayed[:delay] = delayed[delay]
        left = delayed * 0.5
        right = delayed * 0.5

        metrics = estimate_delta_t2(
            joint_xyz,
            fs,
            left,
            right,
            fs,
            t_coarse=0.0,
            half_window_s=0.2,
            vertical=2,
            fc=8.0,
        )
        self.assertAlmostEqual(float(metrics["delta_t2"]), true_lag, delta=0.02)
        self.assertGreater(float(metrics["xcorr_peak"]), 0.2)

    def test_predict_and_measured_shapes(self) -> None:
        fs = 50.0
        com = np.zeros((100, 3), dtype=float)
        com[:, 2] = np.sin(np.linspace(0, 4 * np.pi, 100))
        pred = predict_vgrf_bw(com, fs, vertical=2, fc=8.0)
        meas = measured_vgrf_bw(np.ones(100) + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 100)), fs, fc=8.0)
        self.assertEqual(len(pred), 100)
        self.assertEqual(len(meas), 100)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(np.isfinite(meas).all())



    def test_dynamics_curves_are_two_totals(self) -> None:
        # pred_total and meas_total must be single total curves (no L/R split).
        fs = 40.0
        n = 200
        t = np.arange(n) / fs
        # synthetic joint trajectories for a simple vertical bounce
        z = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
        joints = {
            "Spine": np.column_stack([np.zeros(n), np.zeros(n), z]),
            "Neck": np.column_stack([np.zeros(n), np.zeros(n), z + 0.2]),
            "Head": np.column_stack([np.zeros(n), np.zeros(n), z + 0.4]),
            "LeftUpLeg": np.column_stack([np.zeros(n) - 0.1, np.zeros(n), z - 0.4]),
            "LeftLeg": np.column_stack([np.zeros(n) - 0.1, np.zeros(n), z - 0.8]),
            "LeftFoot": np.column_stack([np.zeros(n) - 0.1, np.zeros(n), z - 1.0]),
            "LeftToeBase": np.column_stack([np.zeros(n) - 0.1, np.zeros(n), z - 1.05]),
            "RightUpLeg": np.column_stack([np.zeros(n) + 0.1, np.zeros(n), z - 0.4]),
            "RightLeg": np.column_stack([np.zeros(n) + 0.1, np.zeros(n), z - 0.8]),
            "RightFoot": np.column_stack([np.zeros(n) + 0.1, np.zeros(n), z - 1.0]),
            "RightToeBase": np.column_stack([np.zeros(n) + 0.1, np.zeros(n), z - 1.05]),
            "LeftArm": np.column_stack([np.zeros(n) - 0.2, np.zeros(n), z + 0.1]),
            "LeftForeArm": np.column_stack([np.zeros(n) - 0.3, np.zeros(n), z]),
            "LeftHand": np.column_stack([np.zeros(n) - 0.35, np.zeros(n), z - 0.05]),
            "LeftHandEnd": np.column_stack([np.zeros(n) - 0.4, np.zeros(n), z - 0.1]),
            "RightArm": np.column_stack([np.zeros(n) + 0.2, np.zeros(n), z + 0.1]),
            "RightForeArm": np.column_stack([np.zeros(n) + 0.3, np.zeros(n), z]),
            "RightHand": np.column_stack([np.zeros(n) + 0.35, np.zeros(n), z - 0.05]),
            "RightHandEnd": np.column_stack([np.zeros(n) + 0.4, np.zeros(n), z - 0.1]),
        }
        pred = predict_vgrf_bw(body_com(joints), fs, vertical=2, fc=8.0)
        meas = measured_vgrf_bw(1.0 + 0.3 * np.sin(2 * np.pi * 1.2 * t), fs, fc=8.0)
        self.assertEqual(pred.ndim, 1)
        self.assertEqual(meas.ndim, 1)
        self.assertEqual(len(pred), n)
        self.assertEqual(len(meas), n)


    def test_length_scale_cm_to_meters(self) -> None:
        # Standing height ~170 cm should map to metres with scale 0.01.
        n = 20
        joints_cm = {
            "Hips": np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 100.0)]),
            "Head": np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 170.0)]),
            "LeftFoot": np.column_stack([np.full(n, -10.0), np.zeros(n), np.zeros(n)]),
            "RightFoot": np.column_stack([np.full(n, 10.0), np.zeros(n), np.zeros(n)]),
        }
        scale = estimate_length_scale_to_meters(joints_cm)
        self.assertAlmostEqual(scale, 0.01, places=6)
        joints_m = scale_joint_xyz(joints_cm, scale)
        height_m = float(joints_m["Head"][0, 2] - joints_m["LeftFoot"][0, 2])
        self.assertAlmostEqual(height_m, 1.7, places=6)

        # Already-metric data stays unchanged.
        joints_m0 = {
            "Hips": np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 1.0)]),
            "Head": np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 1.7)]),
            "LeftFoot": np.column_stack([np.full(n, -0.1), np.zeros(n), np.zeros(n)]),
            "RightFoot": np.column_stack([np.full(n, 0.1), np.zeros(n), np.zeros(n)]),
        }
        self.assertAlmostEqual(estimate_length_scale_to_meters(joints_m0), 1.0, places=6)

if __name__ == "__main__":
    unittest.main()
