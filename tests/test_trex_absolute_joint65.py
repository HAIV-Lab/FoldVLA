"""Unit tests for the fail-closed T-Rex absolute_joint65 adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "T-Rex" / "hardware_code"))

from eval.absolute_joint65 import (  # noqa: E402
    AbsoluteJoint65Adapter,
    BodyJointBinding,
    Joint65CommandRejected,
    assemble_absolute_joint65,
    split_absolute_joint65,
)


class _FakeComponent:
    def __init__(self, names, lower=-2.0, upper=2.0):
        self.joint_name = list(names)
        self.joint_pos_limit = np.tile([lower, upper], (len(names), 1))
        self.positions = {name: 0.0 for name in names}

    def get_joint_pos_dict(self):
        return dict(self.positions)


class _FakeRobot:
    def __init__(self):
        self.components = {
            "left_arm": _FakeComponent([f"la{i}" for i in range(7)]),
            "right_arm": _FakeComponent([f"ra{i}" for i in range(7)]),
            "north_body": _FakeComponent([f"body{i}" for i in range(7)], -1.0, 1.0),
        }
        self.positions = {
            name: 0.0
            for component in self.components.values()
            for name in component.joint_name
        }

    def get_controllable_component_map(self):
        return self.components

    def get_joint_pos_dict(self, component):
        result = {}
        for component_name in component:
            for name in self.components[component_name].joint_name:
                result[name] = self.components[component_name].positions[name]
        return result


class _FakeHand:
    class _State:
        def __init__(self):
            self.angles = [0.0] * 22

    def get_states(self):
        return self._State()


def _binding(robot):
    return BodyJointBinding(
        robot,
        [f"north_body:body{i}" for i in range(7)],
    )


class AbsoluteJoint65Tests(unittest.TestCase):
    def test_exact_layout_round_trip(self):
        source = np.arange(65, dtype=np.float64) / 100.0
        parts = split_absolute_joint65(source)
        self.assertEqual(parts.left_arm.tolist(), source[0:7].tolist())
        self.assertEqual(parts.left_hand.tolist(), source[7:29].tolist())
        self.assertEqual(parts.right_arm.tolist(), source[29:36].tolist())
        self.assertEqual(parts.right_hand.tolist(), source[36:58].tolist())
        self.assertEqual(parts.body.tolist(), source[58:65].tolist())
        reconstructed = assemble_absolute_joint65(
            parts.left_arm,
            parts.left_hand,
            parts.right_arm,
            parts.right_hand,
            parts.body,
        )
        np.testing.assert_array_equal(reconstructed, source)

    def test_body_mapping_must_have_seven_unique_real_joints(self):
        robot = _FakeRobot()
        with self.assertRaisesRegex(Joint65CommandRejected, "exactly seven"):
            BodyJointBinding(robot, ["north_body:body0"])
        with self.assertRaisesRegex(Joint65CommandRejected, "unique"):
            BodyJointBinding(robot, ["north_body:body0"] * 7)
        with self.assertRaisesRegex(Joint65CommandRejected, "is not in component"):
            BodyJointBinding(
                robot,
                [*(f"north_body:body{i}" for i in range(6)), "north_body:missing"],
            )

    def test_prepare_preserves_absolute_targets(self):
        robot = _FakeRobot()
        adapter = AbsoluteJoint65Adapter(
            robot,
            _FakeHand(),
            _FakeHand(),
            _binding(robot),
            left_arm_joint_names=[f"la{i}" for i in range(7)],
            right_arm_joint_names=[f"ra{i}" for i in range(7)],
            max_step_rad=0.1,
            arm_hand_lower_limits=[-2.0] * 58,
            arm_hand_upper_limits=[2.0] * 58,
        )
        current = adapter.read_state()
        target = np.linspace(0.001, 0.05, 65)
        parts, body_targets = adapter.prepare(target, current)
        np.testing.assert_allclose(parts.left_arm, target[:7])
        np.testing.assert_allclose(parts.left_hand, target[7:29])
        np.testing.assert_allclose(parts.right_arm, target[29:36])
        np.testing.assert_allclose(parts.right_hand, target[36:58])
        np.testing.assert_allclose(body_targets["north_body"], target[58:65])

    def test_nan_step_and_limit_violations_are_rejected_not_clipped(self):
        robot = _FakeRobot()
        adapter = AbsoluteJoint65Adapter(
            robot,
            _FakeHand(),
            _FakeHand(),
            _binding(robot),
            left_arm_joint_names=[f"la{i}" for i in range(7)],
            right_arm_joint_names=[f"ra{i}" for i in range(7)],
            max_step_rad=0.1,
            arm_hand_lower_limits=[-2.0] * 58,
            arm_hand_upper_limits=[2.0] * 58,
        )
        current = np.zeros(65)
        target = np.zeros(65)
        target[0] = np.nan
        with self.assertRaisesRegex(Joint65CommandRejected, "NaN/Inf"):
            adapter.prepare(target, current)

        target = np.zeros(65)
        target[10] = 0.2
        with self.assertRaisesRegex(Joint65CommandRejected, "step rejected"):
            adapter.prepare(target, current)

        target = np.zeros(65)
        target[58] = 1.1
        with self.assertRaisesRegex(Joint65CommandRejected, "physical limits"):
            adapter.prepare(target, target)


if __name__ == "__main__":
    unittest.main()
