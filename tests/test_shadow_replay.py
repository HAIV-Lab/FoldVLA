"""Unit tests for critical Shadow Replay interfaces and episode boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from shadow_replay.actions import ActionDimensionError, ActionNormalizer, NorthActionAdapter
from shadow_replay.dataset import EpisodeData, build_valid_chunk_mask
from shadow_replay.metrics import (
    build_ground_truth_chunks,
    chunk_consistency_metrics,
    evaluate_dataset_clipped_predictions,
    evaluate_action_predictions,
    gripper_classification_metrics,
    quaternion_geodesic_deg,
    stage_metrics,
)
from shadow_replay.model import DummyModelRunner
from shadow_replay.replay import ShadowReplay
from shadow_replay.safety import ActionSafetyChecker, north_dataset_joint_names


WORKSPACE = Path(__file__).resolve().parents[1]


class ModelOutputTests(unittest.TestCase):
    def test_single_frame_action_output(self) -> None:
        model = DummyModelRunner(action_dim=3, state_dim=3, chunk_length=1)
        output = model.predict(
            np.zeros((1, 1, 8, 6), dtype=np.float32),
            np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
            ["fold"],
            seed=1,
        )
        self.assertEqual(output.action_raw.shape, (1, 1, 3))
        np.testing.assert_allclose(output.action_raw[0, 0], [1.0, 2.0, 3.0])

    def test_action_chunk_output(self) -> None:
        model = DummyModelRunner(action_dim=2, state_dim=2, chunk_length=4)
        output = model.predict(
            np.zeros((2, 1, 8, 6), dtype=np.float32),
            np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32),
            ["a", "b"],
            seed=1,
        )
        self.assertEqual(output.action_raw.shape, (2, 4, 2))
        np.testing.assert_allclose(output.action_raw[1], [[3.0, 4.0]] * 4)


class EpisodeBoundaryTests(unittest.TestCase):
    def test_episode_tail_valid_mask(self) -> None:
        expected = np.asarray(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        np.testing.assert_array_equal(build_valid_chunk_mask(4, 4), expected)
        chunks, mask = build_ground_truth_chunks(
            np.arange(8, dtype=np.float32).reshape(4, 2), 4
        )
        self.assertEqual(chunks.shape, (4, 4, 2))
        np.testing.assert_array_equal(mask, expected)

    def test_empty_episode(self) -> None:
        episode = EpisodeData(
            episode_id=7,
            frame_indices=np.empty(0, dtype=np.int64),
            timestamps=np.empty(0),
            states=np.empty((0, 65), dtype=np.float32),
            actions=np.empty((0, 65), dtype=np.float32),
            instructions=[],
            optional_fields={},
            source_length=0,
            frame_stride=1,
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            episode.validate()

    def test_different_episode_lengths_do_not_cross(self) -> None:
        first = build_valid_chunk_mask(2, 3)
        second = build_valid_chunk_mask(5, 3)
        self.assertEqual(int(first.sum()), 3)
        self.assertEqual(int(second.sum()), 12)
        self.assertFalse(first[-1, 1])
        self.assertTrue(second[1, 2])

    def test_trex_tactile_mosaic_order(self) -> None:
        image = np.zeros((4, 10, 3), dtype=np.uint8)
        for row in range(2):
            for column in range(5):
                image[
                    row * 2 : (row + 1) * 2,
                    column * 2 : (column + 1) * 2,
                    0,
                ] = row * 5 + column
        tiles = ShadowReplay._split_tactile_deform(image)
        self.assertEqual(tiles.shape, (10, 2, 2))
        np.testing.assert_allclose(
            tiles[:, 0, 0], np.arange(10, dtype=np.float32) / 255.0
        )


class NormalizationAndAdapterTests(unittest.TestCase):
    def test_normalization_denormalization(self) -> None:
        normalizer = ActionNormalizer(
            stats={
                "min": np.asarray([-2.0, 0.0], dtype=np.float32),
                "max": np.asarray([2.0, 4.0], dtype=np.float32),
            },
            norm_type="minmax",
        )
        value = np.asarray([[0.0, 3.0], [1.0, 2.0]], dtype=np.float32)
        reconstructed = normalizer.denormalize(normalizer.normalize(value))
        np.testing.assert_allclose(reconstructed, value, atol=2e-6)

    def test_checkpoint_action_dimension_mismatch(self) -> None:
        with self.assertRaises(ActionDimensionError):
            NorthActionAdapter(
                north_dataset_joint_names(),
                model_action_dim=83,
                adapter="auto",
                hand_utils_json=WORKSPACE
                / "UniDex"
                / "src"
                / "assets"
                / "utils"
                / "hand_utils.json",
            )

    def test_unidex82_round_trip_applies_hands_only(self) -> None:
        adapter = NorthActionAdapter(
            north_dataset_joint_names(),
            model_action_dim=82,
            adapter="auto",
            hand_utils_json=WORKSPACE
            / "UniDex"
            / "src"
            / "assets"
            / "utils"
            / "hand_utils.json",
        )
        action = np.linspace(-0.2, 0.8, 65, dtype=np.float32)
        native = adapter.dataset_to_model(action)
        self.assertEqual(native.shape, (82,))
        state = np.zeros(65, dtype=np.float32)
        adapted = adapter.model_to_robot(native, state)
        hand_indices = adapter.raw_hand_indices["left"] + adapter.raw_hand_indices["right"]
        np.testing.assert_allclose(
            adapted.robot_joint_target[hand_indices], action[hand_indices], atol=1e-6
        )
        non_hand = sorted(set(range(65)) - set(hand_indices))
        np.testing.assert_allclose(adapted.robot_joint_target[non_hand], 0.0)
        self.assertIn("unidex_wrist_pose_not_applied_no_ik", adapted.limitations)


class MetricTests(unittest.TestCase):
    def test_quaternion_angle_error(self) -> None:
        identity = np.asarray([[0.0, 0.0, 0.0, 1.0]])
        half_turn_x = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(
            float(quaternion_geodesic_deg(identity, identity)[0]), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(quaternion_geodesic_deg(identity, half_turn_x)[0]), 180.0, places=7
        )
        np.testing.assert_allclose(
            quaternion_geodesic_deg(identity, -identity), [0.0], atol=1e-7
        )

    def test_gripper_classification_metrics(self) -> None:
        target = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0])
        predicted = np.asarray([0.0, 1.0, 1.0, 1.0, 0.0])
        metrics = gripper_classification_metrics(predicted, target)
        self.assertAlmostEqual(metrics["gripper_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["gripper_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["gripper_recall"], 1.0)
        self.assertAlmostEqual(metrics["gripper_f1"], 0.8)

    def test_replanning_overlap_consistency(self) -> None:
        length, horizon = 6, 3
        prediction = np.zeros((length, horizon, 1), dtype=np.float32)
        for step in range(length):
            for offset in range(horizon):
                prediction[step, offset, 0] = step + offset
        mask = build_valid_chunk_mask(length, horizon)
        metrics = chunk_consistency_metrics(prediction, mask)
        self.assertEqual(metrics["overlap_consistency_mae"], 0.0)
        self.assertGreater(metrics["overlap_comparison_count"], 0)

    def test_single_step_metric_shapes(self) -> None:
        prediction = np.asarray([[[1.0, 2.0]]])
        target = np.asarray([[[0.0, 2.0]]])
        mask = np.asarray([[True]])
        metrics = evaluate_action_predictions(
            prediction, target, mask, fps=10.0
        )
        self.assertAlmostEqual(metrics["action_mae"], 0.5)
        self.assertAlmostEqual(metrics["action_mse"], 0.5)
        self.assertEqual(metrics["per_dimension_mae"], [1.0, 0.0])

    def test_gt_and_prediction_derivatives_are_computed_independently(self) -> None:
        prediction = np.asarray([[[0.0], [1.0], [3.0]]], dtype=np.float32)
        target = np.asarray([[[0.0], [0.5], [1.0]]], dtype=np.float32)
        mask = np.ones((1, 3), dtype=bool)
        metrics = evaluate_action_predictions(prediction, target, mask, fps=2.0)
        derivatives = metrics["trajectory_derivatives"]
        self.assertAlmostEqual(
            derivatives["vla_prediction"]["acceleration_rad_s2"]["mean_abs"],
            4.0,
        )
        self.assertAlmostEqual(
            derivatives["ground_truth"]["acceleration_rad_s2"]["mean_abs"],
            0.0,
        )

    def test_dataset_bounds_clip_prediction_and_gt_before_error(self) -> None:
        prediction = np.asarray([[[-2.0, 3.0]]], dtype=np.float32)
        target = np.asarray([[[-1.0, 4.0]]], dtype=np.float32)
        mask = np.ones((1, 1), dtype=bool)
        metrics = evaluate_dataset_clipped_predictions(
            prediction,
            target,
            mask,
            lower=[0.0, 0.0],
            upper=[1.0, 2.0],
            fps=30.0,
        )
        self.assertEqual(metrics["action_mae"], 0.0)
        clipping = metrics["dataset_bound_clipping"]
        self.assertEqual(clipping["prediction_clipped_element_count"], 2)
        self.assertEqual(clipping["ground_truth_clipped_element_count"], 2)

    def test_stage_metrics_include_semantic_and_chunk_fields(self) -> None:
        prediction = np.zeros((3, 2, 9), dtype=np.float32)
        target = np.zeros_like(prediction)
        mask = np.ones((3, 2), dtype=bool)
        metrics = stage_metrics(
            prediction,
            target,
            mask,
            ["approach", "approach", None],
            fps=10.0,
            xyz_indices=[0, 1, 2],
            rotation_indices=[3, 4, 5, 6, 7, 8],
            rotation_representation="rotation6d",
            position_unit="m",
        )
        self.assertEqual(metrics["approach"]["sample_count"], 2)
        self.assertIn("position", metrics["approach"])
        self.assertIn("rotation", metrics["approach"])
        self.assertIn("overlap_consistency_mae", metrics["approach"])


class SafetyTests(unittest.TestCase):
    def test_horizon_zero_does_not_treat_tracking_error_as_velocity(self) -> None:
        checker = ActionSafetyChecker(
            {
                "enabled": True,
                "violation_behavior": "clip",
                "max_joint_step": 0.1,
                "max_joint_velocity": 0.2,
                "max_joint_acceleration": 1.0,
            }
        )
        target = np.ones(65, dtype=np.float32)
        result = checker.check(
            target,
            np.zeros(65, dtype=np.float32),
            previous_action=None,
            previous_velocity=None,
            dt=0.1,
        )
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.action_safe, target)

    def test_nan_inf_safety_check(self) -> None:
        checker = ActionSafetyChecker(
            {"enabled": True, "violation_behavior": "reject"}
        )
        action = np.zeros(65, dtype=np.float32)
        action[3] = np.nan
        result = checker.check(action, np.zeros(65, dtype=np.float32))
        self.assertTrue(result.nan_or_inf)
        self.assertTrue(result.rejected)
        self.assertIn("nan_or_inf", result.flags)
        np.testing.assert_allclose(result.action_safe, 0.0)

    def test_workspace_violation(self) -> None:
        checker = ActionSafetyChecker(
            {
                "enabled": True,
                "violation_behavior": "report_only",
                "workspace_min": [-0.1, -0.1, -0.1],
                "workspace_max": [0.1, 0.1, 0.1],
            }
        )
        eef = np.tile(
            np.asarray([0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            2,
        )
        result = checker.check(
            np.zeros(65, dtype=np.float32),
            np.zeros(65, dtype=np.float32),
            eef_action=eef,
        )
        self.assertFalse(result.valid)
        self.assertFalse(result.rejected)
        self.assertIn("workspace", result.flags)


if __name__ == "__main__":
    unittest.main()
