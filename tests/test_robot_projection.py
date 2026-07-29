"""Tests for North URDF FK and head-camera projection."""

from pathlib import Path
import unittest

import numpy as np

from shadow_replay.robot_projection import CameraCalibration, NorthUrdfKinematics


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "asserts/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf"


class RobotProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = NorthUrdfKinematics(URDF)

    def test_north_65d_fk_contains_wrists_and_fingertips(self):
        landmarks = self.kinematics.landmark_positions(np.zeros(65))
        for side in ("left", "right"):
            self.assertIn(f"{side}_wrist", landmarks)
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                point = landmarks[f"{side}_{finger}_fingertip"]
                self.assertEqual(point.shape, (3,))
                self.assertTrue(np.isfinite(point).all())

    def test_camera_forward_axis_projects_to_principal_point(self):
        calibration = CameraCalibration.from_config(
            {
                "matrix": [[200.0, 0.0, 123.0], [0.0, 210.0, 234.0], [0.0, 0.0, 1.0]],
                "distortion": [],
                "mount": {
                    "link": "head_base_link",
                    "xyz": [0.066735, 0.03, 0.0],
                    "rpy": [-np.pi / 2, 0.0, -np.pi / 2],
                },
                "source": "test",
            }
        )
        poses = self.kinematics.link_poses(np.zeros(65))
        urdf_from_camera = np.linalg.inv(calibration.camera_from_urdf(poses))
        point_camera = np.asarray([0.0, 0.0, 1.0, 1.0])
        point_urdf = (urdf_from_camera @ point_camera)[:3]
        image, visible = calibration.project(point_urdf[None], poses)
        self.assertTrue(visible[0])
        np.testing.assert_allclose(image[0], [123.0, 234.0], atol=1e-8)

    def test_fk_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, r"shape \(65,\)"):
            self.kinematics.link_poses(np.zeros(64))


if __name__ == "__main__":
    unittest.main()
