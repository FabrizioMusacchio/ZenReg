import csv

import numpy as np
import pytest

from zenreg.synthetic import (
    _write_3d_rigid_transform_table,
    _write_3d_slice_shift_table,
    _write_local_motion_table,
    _write_piecewise_anchor_shift_table,
    _write_time_rotation_table,
    _write_time_shift_table,
    _write_time_shift_zyx_table,
    create_2d_local_motion_distorted_stack,
    create_2d_time_piecewise_xy_motion_distorted_stack,
    create_2d_time_rotation_motion_distorted_stack,
    create_2d_time_translation_rotation_motion_distorted_stack,
    create_3d_motion_distorted_stack,
    create_3d_slice_motion_distorted_stack,
    create_3d_time_intra_motion_distorted_stack,
    create_3d_time_rigid_motion_distorted_stack,
    create_3d_time_sparse_puncta_rigid_motion_distorted_stack,
    create_3d_time_xy_motion_distorted_stack,
    create_3d_time_zyx_motion_distorted_stack,
)


def _read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_synthetic_generators_return_expected_shapes_and_gt():
    stack_3d_slice, slice_shifts = create_3d_slice_motion_distorted_stack(
        z_count=4,
        channel_count=3,
        shape_yx=(32, 32),
        noise_sigma=0,
    )
    assert stack_3d_slice.shape == (1, 4, 3, 32, 32)
    assert slice_shifts.shape == (1, 4, 2)

    stack_3d_xy, time_shifts_yx = create_3d_time_xy_motion_distorted_stack(
        time_count=3,
        z_count=4,
        channel_count=1,
        shape_yx=(32, 32),
        noise_sigma=0,
    )
    assert stack_3d_xy.shape == (3, 4, 1, 32, 32)
    assert time_shifts_yx.shape == (3, 2)

    stack_3d_intra, intra_shifts = create_3d_time_intra_motion_distorted_stack(
        time_count=3,
        z_count=4,
        channel_count=1,
        shape_yx=(32, 32),
        noise_sigma=0,
        time_varying_slice_shifts=False,
    )
    assert stack_3d_intra.shape == (3, 4, 1, 32, 32)
    np.testing.assert_allclose(intra_shifts[0], intra_shifts[1])

    stack_3d_zyx, time_shifts_zyx = create_3d_time_zyx_motion_distorted_stack(
        time_count=3,
        z_count=5,
        channel_count=1,
        shape_yx=(32, 32),
        noise_sigma=0,
    )
    assert stack_3d_zyx.shape == (3, 5, 1, 32, 32)
    assert time_shifts_zyx.shape == (3, 3)


def test_synthetic_rotation_and_local_generators_return_gt():
    stack_rot, shifts_yx, rotations = create_2d_time_rotation_motion_distorted_stack(
        time_count=4,
        channel_count=1,
        shape_yx=(32, 32),
        noise_sigma=0,
    )
    assert stack_rot.shape == (4, 1, 1, 32, 32)
    np.testing.assert_allclose(shifts_yx, 0)
    assert rotations.shape == (4,)

    stack_local, local_params = create_2d_local_motion_distorted_stack(
        time_count=12,
        channel_count=1,
        shape_yx=(40, 40),
        noise_sigma=0,
    )
    assert stack_local.shape == (12, 1, 1, 40, 40)
    assert local_params.shape == (12, 6)

    stack_trans_rot, trans_shifts, trans_rot = create_2d_time_translation_rotation_motion_distorted_stack(
        time_count=12,
        channel_count=1,
        shape_yx=(40, 40),
        noise_sigma=0,
    )
    assert stack_trans_rot.shape == (12, 1, 1, 40, 40)
    assert trans_shifts.shape == (12, 2)
    assert trans_rot.shape == (12,)

    stack_piecewise, anchor_shifts = create_2d_time_piecewise_xy_motion_distorted_stack(
        time_count=4,
        channel_count=1,
        shape_yx=(40, 40),
        grid_shape_yx=(2, 3),
        noise_sigma=0,
    )
    assert stack_piecewise.shape == (4, 1, 1, 40, 40)
    assert anchor_shifts.shape == (4, 2, 3, 2)


def test_synthetic_3d_rigid_generators_and_validation():
    stack_rigid, shifts, rotations, centers = create_3d_time_rigid_motion_distorted_stack(
        time_count=3,
        z_count=6,
        channel_count=1,
        shape_yx=(32, 32),
        rotation_mode="all",
        center_mode="inside_offset",
        noise_sigma=0,
    )
    assert stack_rigid.shape == (3, 6, 1, 32, 32)
    assert shifts.shape == rotations.shape == centers.shape == (3, 3)
    assert np.any(np.abs(rotations[:, 1]) > 0)

    stack_points, point_shifts, point_rotations, point_centers = (
        create_3d_time_sparse_puncta_rigid_motion_distorted_stack(
            time_count=3,
            z_count=8,
            channel_count=2,
            shape_yx=(40, 40),
            point_count=12,
            noise_sigma=0,
        )
    )
    assert stack_points.shape == (3, 8, 2, 40, 40)
    assert point_shifts.shape == point_rotations.shape == point_centers.shape == (3, 3)

    with pytest.raises(ValueError, match="rotation_mode"):
        create_3d_time_rigid_motion_distorted_stack(rotation_mode="bad")
    with pytest.raises(ValueError, match="center_mode"):
        create_3d_time_rigid_motion_distorted_stack(center_mode="bad")


def test_legacy_3d_motion_generator_shape():
    stack, time_shifts, z_shifts = create_3d_motion_distorted_stack(
        time_count=3,
        z_count=4,
        channel_count=2,
        shape_yx=(32, 32),
        noise_sigma=0,
    )

    assert stack.shape == (3, 4, 2, 32, 32)
    assert time_shifts.shape == (3, 2)
    assert z_shifts.shape == (3, 4, 2)


def test_synthetic_gt_table_writers(tmp_path):
    shifts_yx = np.asarray([[0.0, 0.0], [2.0, -3.0]], dtype=np.float32)
    shifts_zyx = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, -3.0]], dtype=np.float32)
    rotations = np.asarray([0.0, 4.0], dtype=np.float32)
    local_params = np.asarray([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]], dtype=np.float32)
    anchor_shifts = np.zeros((2, 2, 2, 2), dtype=np.float32)
    anchor_shifts[1, 1, 1] = [2, -3]

    time_path = _write_time_shift_table(tmp_path / "time.csv", shifts_yx, registration_stack=0)
    zyx_path = _write_time_shift_zyx_table(tmp_path / "zyx.csv", shifts_zyx, registration_stack=0)
    rotation_path = _write_time_rotation_table(tmp_path / "rot.csv", rotations, registration_stack=0)
    local_path = _write_local_motion_table(tmp_path / "local.csv", local_params)
    anchor_path = _write_piecewise_anchor_shift_table(tmp_path / "anchor.csv", anchor_shifts)
    slice_path = _write_3d_slice_shift_table(
        tmp_path / "slice.csv",
        time_shifts_yx=shifts_yx,
        z_shifts_yx=np.zeros((2, 2, 2), dtype=np.float32),
    )
    rigid_path = _write_3d_rigid_transform_table(
        tmp_path / "rigid.csv",
        shifts_zyx=shifts_zyx,
        rotations_zyx_deg=np.zeros((2, 3), dtype=np.float32),
        centers_zyx=np.ones((2, 3), dtype=np.float32),
        registration_stack=0,
    )

    assert _read_csv_rows(time_path)[1]["expected_registration_shift_x_ref_t0"] == "3.0"
    assert _read_csv_rows(zyx_path)[1]["expected_registration_shift_z_ref_t0"] == "-1.0"
    assert _read_csv_rows(rotation_path)[1]["expected_registration_rotation_deg_ref_t0"] == "-4.0"
    assert _read_csv_rows(local_path)[1]["motion_magnitude"] == "14.0"
    assert _read_csv_rows(anchor_path)[-1]["expected_anchor_correction_shift_x_ref_t0"] == "3.0"
    assert _read_csv_rows(slice_path)[0]["expected_local_z_correction_shift_y"] == "-0.0"
    assert _read_csv_rows(rigid_path)[1]["expected_registration_shift_x_ref_t0"] == "3.0"
