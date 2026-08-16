from __future__ import annotations

import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fake_imread(path: str, flags: int | None = None) -> np.ndarray | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    matches = re.findall(r"(\d+)", image_path.stem)
    value = int(matches[-1]) if matches else 1
    return np.full((12, 12, 3), value % 255, dtype=np.uint8)


def _fake_cvt_color(image: np.ndarray, code: int) -> np.ndarray:
    if code == fake_cv2.COLOR_BGR2RGB:
        return image[:, :, ::-1]
    if code == fake_cv2.COLOR_RGB2GRAY:
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.tensordot(image[..., :3], weights, axes=([-1], [0])).astype(np.uint8)
    if code == fake_cv2.COLOR_BGR2GRAY:
        return _fake_cvt_color(image[:, :, ::-1], fake_cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported fake cv2 color code: {code}")


def _fake_resize(image: np.ndarray, size: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    target_width, target_height = size
    y_index = np.linspace(0, image.shape[0] - 1, target_height).astype(int)
    x_index = np.linspace(0, image.shape[1] - 1, target_width).astype(int)
    return image[np.ix_(y_index, x_index)]


class _FakeVideoCapture:
    def __init__(self, path: str):
        self.path = path

    def isOpened(self) -> bool:
        return False

    def get(self, prop: int) -> float:
        return 0.0

    def read(self):
        return False, None

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        return None


fake_cv2 = types.SimpleNamespace(
    imread=_fake_imread,
    cvtColor=_fake_cvt_color,
    resize=_fake_resize,
    VideoCapture=_FakeVideoCapture,
    IMREAD_COLOR=1,
    COLOR_BGR2RGB=1,
    COLOR_RGB2GRAY=2,
    COLOR_BGR2GRAY=3,
    INTER_AREA=0,
    CAP_PROP_FPS=5,
    CAP_PROP_FRAME_COUNT=7,
    CAP_PROP_POS_FRAMES=1,
)
sys.modules["cv2"] = fake_cv2

from models import BVHMotion, BVHJoint, BVHSourceData, ImageSequenceSource, RuntimeOptions, SessionData
from utils import session as session_module
from utils.alignment import aligned_start_frames, preview_time_to_camera_frame
from utils.bvh_pose import transform_display_positions
from utils.energy import image_sequence_energy, infer_frame_times_from_paths
from utils.session import (
    choose_bvh_paths,
    choose_bvh_roles,
    derive_session_id,
    enumerate_trials,
    find_mocap_subject,
    get_cam_frame_dirs,
    list_frame_paths,
    load_session_data,
    load_session_from_paths,
    resolve_source_mode,
)

FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"


class AlignmentFlowTests(unittest.TestCase):
    def test_transform_display_positions_zup(self) -> None:
        positions = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        transformed = transform_display_positions(positions, "zup")
        expected = np.array([[1.0, -3.0, 2.0], [-4.0, 6.0, 5.0]])
        np.testing.assert_allclose(transformed, expected)

    def test_aligned_start_frames_uses_each_bvh_fps(self) -> None:
        motion = BVHMotion(
            path=Path("dummy.bvh"),
            text="",
            frame_time_str="0.0083333333",
            raw_fps=120.0,
            raw_frames=np.zeros((10, 6), dtype=np.float64),
            joints=[BVHJoint("Hips", -1, np.zeros(3), [], [])],
            edges=[],
            channel_info=[],
        )
        camera = ImageSequenceSource(
            label="cam1",
            fps=40.0,
            frame_count=10,
            energy=np.zeros(9, dtype=np.float64),
            source_dir=Path("cam1"),
            frame_paths=(Path("cam1/frame_0001.png"),),
        )
        session = type(
            "SessionStub",
            (),
            {
                "cameras": [camera],
                "alignment_bvh": BVHSourceData("order", motion, np.zeros(0), np.zeros(0)),
            },
        )()

        bvh_start, cam_starts = aligned_start_frames(session, 0.05)
        self.assertEqual(bvh_start, 6)
        self.assertEqual(cam_starts["cam1"], 0)

        bvh_start, cam_starts = aligned_start_frames(session, -0.05)
        self.assertEqual(bvh_start, 0)
        self.assertEqual(cam_starts["cam1"], 2)

    def test_resolve_source_mode_prefers_frames_when_both_sources_exist(self) -> None:
        frame_dirs = {"cam1": Path("cam1")}
        videos = {f"cam{i}": Path(f"{i}_demo.mp4") for i in range(1, 5)}
        self.assertEqual(resolve_source_mode("auto", frame_dirs, videos), "frames")

    def test_choose_bvh_roles_prefers_skeleton0_and_skeleton1(self) -> None:
        files = (
            Path("S1011_Skeleton1.bvh"),
            Path("S1011_Skeleton0.bvh"),
            Path("S1011_Other.bvh"),
        )
        position_bvh, order_bvh = choose_bvh_roles(files)
        self.assertEqual(position_bvh.name, "S1011_Skeleton0.bvh")
        self.assertEqual(order_bvh.name, "S1011_Skeleton1.bvh")

    def test_choose_bvh_paths_keeps_extra_skeleton_files(self) -> None:
        files = (
            Path("S4011_Skeleton3.bvh"),
            Path("S4011_Skeleton0.bvh"),
            Path("S4011_Skeleton2.bvh"),
            Path("S4011_Skeleton1.bvh"),
        )
        position_bvh, order_bvh, extra_bvhs = choose_bvh_paths(files)
        self.assertEqual(position_bvh.name, "S4011_Skeleton0.bvh")
        self.assertEqual(order_bvh.name, "S4011_Skeleton1.bvh")
        self.assertEqual([path.name for path in extra_bvhs], ["S4011_Skeleton2.bvh", "S4011_Skeleton3.bvh"])

    def test_derive_session_id_prefers_embedded_session_pattern(self) -> None:
        session_id = derive_session_id(
            Path("manual_demo_S109_2"),
            Path("D:/dummy/S1092_skeleton.bvh"),
            None,
        )
        self.assertEqual(session_id, "S109_2")

    def test_list_frame_paths_uses_numeric_sort(self) -> None:
        frame_dir = FIXTURE_ROOT / "frame_sort"
        ordered = list_frame_paths(frame_dir)
        self.assertEqual([path.name for path in ordered], ["frame_1.png", "frame_2.png", "frame_10.png"])

    def test_list_frame_paths_sorts_timestamp_style_names(self) -> None:
        frame_dir = FIXTURE_ROOT / "frame_sort_timestamp"
        ordered = list_frame_paths(frame_dir)
        self.assertEqual(
            [path.name for path in ordered],
            ["1_162852.002.jpg", "1_162852.035.jpg", "1_162853.001.jpg"],
        )

    def test_frame_timestamps_infer_actual_fps_from_names(self) -> None:
        frame_dir = FIXTURE_ROOT / "frame_sort_timestamp"
        frame_paths = list_frame_paths(frame_dir)
        frame_times, fps = infer_frame_times_from_paths(frame_paths, 40.0)
        self.assertIsNotNone(frame_times)
        np.testing.assert_allclose(frame_times, np.array([0.0, 0.033, 0.999]), atol=1e-6)
        self.assertAlmostEqual(fps, 2.002002, places=5)

    def test_image_sequence_energy_skips_unreadable_frames(self) -> None:
        frame_dir = FIXTURE_ROOT / "image_sequence_with_bad_frame"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = (
            frame_dir / "bad_001.jpg",
            frame_dir / "good_002.jpg",
            frame_dir / "good_003.jpg",
        )
        for frame_path in frame_paths:
            frame_path.touch()

        with patch("utils.energy._read_image_bgr") as mocked_read:
            mocked_read.side_effect = [
                None,
                np.full((12, 12, 3), 2, dtype=np.uint8),
                np.full((12, 12, 3), 3, dtype=np.uint8),
            ]
            energy, fps, frame_count = image_sequence_energy(frame_paths, 40.0)

        self.assertEqual(frame_count, 3)
        self.assertEqual(fps, 40.0)
        self.assertEqual(len(energy), 1)

    def test_preview_time_uses_frame_timestamps_when_available(self) -> None:
        camera = ImageSequenceSource(
            label="cam1",
            fps=2.5,
            frame_count=3,
            energy=np.zeros(2),
            frame_times=np.array([0.0, 0.1, 0.4]),
            source_dir=Path("cam1"),
            frame_paths=(Path("a.jpg"), Path("b.jpg"), Path("c.jpg")),
        )
        self.assertEqual(preview_time_to_camera_frame(camera, 0.06), 1)
        self.assertEqual(preview_time_to_camera_frame(camera, 0.28), 2)

    def test_get_cam_frame_dirs_supports_numeric_directory_names(self) -> None:
        cam_session = FIXTURE_ROOT / "cam_root_numeric" / "demo_S100_1"
        frame_dirs = get_cam_frame_dirs(cam_session)
        self.assertEqual(sorted(frame_dirs), ["cam1", "cam2", "cam3", "cam4"])
        self.assertEqual(frame_dirs["cam1"].name, "1")
        self.assertEqual(frame_dirs["cam4"].name, "4")

    def test_mocap_trial_discovery_supports_group_subfolders(self) -> None:
        root = FIXTURE_ROOT / "mocap_nested"
        trials = enumerate_trials(root)
        subject_path = find_mocap_subject("S401_1", root)

        self.assertEqual([trial.display_name for trial in trials], ["对象4 动作01 第1次", "对象4 动作01 第2次"])
        self.assertEqual(subject_path.name, "S4011")

    def test_mocap_trial_discovery_supports_multi_digit_subjects(self) -> None:
        root = FIXTURE_ROOT / "mocap_multi_digit_subject"
        trial_dir = root / "S13011"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "S13011_Skeleton0.bvh").touch()

        trials = enumerate_trials(root)
        subject_path = find_mocap_subject("S1301_1", root)

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].subject, 13)
        self.assertEqual(trials[0].action, 1)
        self.assertEqual(trials[0].rep, 1)
        self.assertEqual(trials[0].mocap_folder_name, "S13011")
        self.assertEqual(trials[0].cam_session_suffix, "S1301_1")
        self.assertEqual(subject_path.name, "S13011")

    def test_load_session_data_prefers_frame_sequences_and_prepares_cache(self) -> None:
        cam_root = FIXTURE_ROOT / "cam_root"
        mocap_root = FIXTURE_ROOT / "mocap_root"
        cache_root = FIXTURE_ROOT / "cache_root"
        output_root = FIXTURE_ROOT / "output_root"

        with patch.object(session_module, "_load_cached_precompute", return_value=None), patch.object(
            session_module,
            "_save_cached_precompute",
        ) as mocked_save:
            session = load_session_data(
                session_id="S100_1",
                cam_root=cam_root,
                mocap_root=mocap_root,
                cache_root=cache_root,
                output_root=output_root,
                source_mode="auto",
            )

        self.assertEqual(session.source_mode, "frames")
        self.assertAlmostEqual(session.reference_visual_fps, 40.0)
        self.assertEqual(len(session.cameras), 4)
        self.assertIsNotNone(session.position_bvh)
        self.assertIsNone(session.order_bvh)
        self.assertTrue(mocked_save.called)
        saved_arrays = mocked_save.call_args.args[2]
        self.assertIn("combined_camera_energy", saved_arrays)
        self.assertIn("position_bvh_energy_visual", saved_arrays)

    def test_load_session_from_paths_supports_bvh_only(self) -> None:
        bvh_path = FIXTURE_ROOT / "mocap_root" / "S1001" / "skeleton.bvh"
        cache_root = FIXTURE_ROOT / "cache_root"
        output_root = FIXTURE_ROOT / "output_root"

        with patch.object(session_module, "_load_cached_precompute", return_value=None), patch.object(
            session_module,
            "_save_cached_precompute",
        ):
            session = load_session_from_paths(
                position_bvh_path=bvh_path,
                cache_root=cache_root,
                output_root=output_root,
            )

        self.assertFalse(session.has_visual)
        self.assertTrue(session.has_bvh)
        self.assertAlmostEqual(session.reference_visual_fps, 120.00000048, places=4)
        self.assertEqual(session.source_mode, "none")

    def test_session_prefers_dynamic_bvh_for_preview_and_alignment(self) -> None:
        static_motion = BVHMotion(
            path=Path("Skeleton0.bvh"),
            text="",
            frame_time_str="0.0083333333",
            raw_fps=120.0,
            raw_frames=np.array(
                [
                    [0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [10.0, 90.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            joints=[BVHJoint("Hips", -1, np.zeros(3), [], [])],
            edges=[],
            channel_info=[("Hips", "Xposition")] * 9,
        )
        dynamic_motion = BVHMotion(
            path=Path("Skeleton1.bvh"),
            text="",
            frame_time_str="0.0083333333",
            raw_fps=120.0,
            raw_frames=np.array(
                [
                    [0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                    [0.5, 90.0, 0.2, 1.0, 0.0, 0.3, 4.0, 5.0, 6.0],
                ],
                dtype=np.float64,
            ),
            joints=[BVHJoint("Hips", -1, np.zeros(3), [], [])],
            edges=[],
            channel_info=[("Hips", "Xposition")] * 9,
        )
        session = SessionData(
            session_id="S101_1",
            cam_session=None,
            mocap_subject=None,
            cameras=[],
            position_bvh=BVHSourceData("position", static_motion, np.zeros(1), np.zeros(1)),
            order_bvh=BVHSourceData("order", dynamic_motion, np.zeros(1), np.zeros(1)),
            combined_camera_energy=np.zeros(0),
            reference_visual_fps=40.0,
            output_dir=Path("out"),
            cache_dir=Path("cache"),
            source_mode="none",
            runtime_options=RuntimeOptions(source_mode="none", lite_mode=False, preview_scale=1.0, axis_preset="zup"),
        )
        self.assertEqual(session.display_bvh.motion.path.name, "Skeleton1.bvh")
        self.assertEqual(session.alignment_bvh.motion.path.name, "Skeleton1.bvh")

    def test_session_can_select_extra_bvh_for_preview_and_alignment(self) -> None:
        static_motion = BVHMotion(
            path=Path("Skeleton0.bvh"),
            text="",
            frame_time_str="0.0083333333",
            raw_fps=120.0,
            raw_frames=np.zeros((2, 9), dtype=np.float64),
            joints=[BVHJoint("Hips", -1, np.zeros(3), [], [])],
            edges=[],
            channel_info=[("Hips", "Xposition")] * 9,
        )
        extra_motion = BVHMotion(
            path=Path("Skeleton3.bvh"),
            text="",
            frame_time_str="0.0083333333",
            raw_fps=120.0,
            raw_frames=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 4.0, 2.0],
                ],
                dtype=np.float64,
            ),
            joints=[BVHJoint("Hips", -1, np.zeros(3), [], [])],
            edges=[],
            channel_info=[("Hips", "Xposition")] * 9,
        )
        session = SessionData(
            session_id="S401_1",
            cam_session=None,
            mocap_subject=None,
            cameras=[],
            position_bvh=BVHSourceData("position", static_motion, np.zeros(1), np.zeros(1)),
            order_bvh=None,
            combined_camera_energy=np.zeros(0),
            reference_visual_fps=40.0,
            output_dir=Path("out"),
            cache_dir=Path("cache"),
            source_mode="none",
            runtime_options=RuntimeOptions(source_mode="none", lite_mode=False, preview_scale=1.0, axis_preset="zup"),
            extra_bvhs=[BVHSourceData("skeleton3", extra_motion, np.zeros(1), np.zeros(1))],
        )
        self.assertEqual(session.display_bvh.motion.path.name, "Skeleton3.bvh")
        self.assertEqual(session.alignment_bvh.motion.path.name, "Skeleton3.bvh")


if __name__ == "__main__":
    unittest.main()
