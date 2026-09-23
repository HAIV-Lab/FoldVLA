from types import SimpleNamespace

import pytest
import torch

from rl_isaaclab.utils.north_tactile import (
    NORTH_TACTILE_DIM,
    NORTH_TACTILE_FEATURE_NAMES,
    NORTH_TACTILE_SENSOR_BODIES,
    NorthTactileAdapter,
    NorthTactileAdapterConfig,
)


def _sensor(force, contact_position=None, sensor_position=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0)):
    force = torch.as_tensor(force, dtype=torch.float32).reshape(1, 1, 1, 3)
    kwargs = {
        "force_matrix_w": force,
        "net_forces_w": force[:, 0, :, :].sum(dim=1),
        "pos_w": torch.as_tensor(sensor_position, dtype=torch.float32).reshape(1, 1, 3),
        "quat_w": torch.as_tensor(quaternion, dtype=torch.float32).reshape(1, 1, 4),
    }
    if contact_position is not None:
        kwargs["contact_pos_w"] = torch.as_tensor(contact_position, dtype=torch.float32).reshape(1, 1, 1, 3)
    return SimpleNamespace(data=SimpleNamespace(**kwargs))


def test_dataset_schema_is_ten_wrenches_and_flattened_to_sixty_values():
    assert len(NORTH_TACTILE_SENSOR_BODIES) == 10
    assert len(NORTH_TACTILE_FEATURE_NAMES) == NORTH_TACTILE_DIM == 60
    sensors = [_sensor((0.0, 0.0, 0.0)) for _ in NORTH_TACTILE_SENSOR_BODIES]

    tactile_f6 = NorthTactileAdapter().convert(sensors, flatten=False)
    tactile_flat = NorthTactileAdapter().convert(sensors, flatten=True)

    assert tactile_f6.shape == (1, 10, 6)
    assert tactile_flat.shape == (1, 60)
    assert torch.equal(tactile_flat, tactile_f6.reshape(1, 60))


def test_filtered_forces_are_summed_and_rotated_to_sensor_frame():
    first = _sensor((1.0, 2.0, 3.0), contact_position=(0.0, 1.0, 0.0))
    # Add a second filtered contact target to exercise force aggregation.
    first.data.force_matrix_w = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]]]], dtype=torch.float32
    )
    first.data.contact_pos_w = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]]], dtype=torch.float32
    )
    sensors = [first] + [_sensor((0.0, 0.0, 0.0)) for _ in range(9)]

    tactile = NorthTactileAdapter().convert_f6(sensors)

    # Total force is [5, 2, 4].  The two contact moments are
    # [0, 0, -1] and [0, 8, 0], respectively.
    assert torch.allclose(tactile[0, 0, :3], torch.tensor([5.0, 2.0, 4.0]))
    assert torch.allclose(tactile[0, 0, 3:], torch.tensor([-0.0, 8.0, -1.0]))


def test_wrench_sign_and_calibration_scale_are_applied():
    sensors = [_sensor((0.0, 0.0, 2.0)) for _ in NORTH_TACTILE_SENSOR_BODIES]
    adapter = NorthTactileAdapter(
        NorthTactileAdapterConfig(wrench_sign=-1.0, force_scale=0.5, torque_scale=2.0)
    )

    tactile = adapter.convert_f6(sensors)

    assert torch.allclose(tactile[:, :, 2], torch.full((1, 10), -1.0))
    assert torch.allclose(tactile[:, :, 0:2], torch.zeros((1, 10, 2)))
    assert torch.allclose(tactile[:, :, 3:], torch.zeros((1, 10, 3)))


def test_missing_contact_pose_can_emit_zero_torque_for_force_only_safety():
    sensors = [_sensor((1.0, 2.0, 3.0)) for _ in NORTH_TACTILE_SENSOR_BODIES]
    tactile = NorthTactileAdapter().convert_f6(sensors)

    assert torch.allclose(tactile[0, :, :3], torch.tensor([[1.0, 2.0, 3.0]]).repeat(10, 1))
    assert torch.allclose(tactile[0, :, 3:], torch.zeros((10, 3)))


def test_wrong_sensor_count_is_rejected():
    with pytest.raises(ValueError, match="Expected 10 sensors"):
        NorthTactileAdapter().convert([])
