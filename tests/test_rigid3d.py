import numpy as np
import pytest

from zenreg import register_stack
from zenreg.synthetic import (
    create_3d_time_rigid_motion_distorted_stack,
    create_3d_time_sparse_puncta_rigid_motion_distorted_stack,
)


def test_register_stack_dispatches_to_simpleitk_rigid_3d_backend():
    pytest.importorskip("SimpleITK")
    stack, _, _, _ = create_3d_time_rigid_motion_distorted_stack(
        time_count=3,
        z_count=10,
        shape_yx=(48, 48),
        rotation_mode="z",
        noise_sigma=0.0,
        random_state=101,
    )

    registered, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        rotreg=True,
        rigid_3d_backend="simpleitk",
        rot_init_iterations=1,
        rot_iterations=3,
        rot_shrink_factors=(2, 1),
        rot_smoothing_sigmas=(1.0, 0.0),
        phase_cross_correlation_upsample_factor=5,
        verbose=False,
        return_details=True,
    )

    assert registered.shape == stack.shape
    assert details["rigid_3d_backend"] == "simpleitk"
    assert details["time_shifts_zyx"].shape == (stack.shape[0], 3)
    assert details["rotation_shifts_zyx_deg"].shape == (stack.shape[0], 3)
    assert details["rigid_3d_matrices_zyx"].shape == (stack.shape[0], 3, 3)
    np.testing.assert_allclose(details["time_shifts_zyx"][0], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(details["rotation_shifts_zyx_deg"][0], np.zeros(3), atol=1e-6)


def test_register_stack_dispatches_to_points_rigid_3d_backend():
    stack, _, _, _ = create_3d_time_sparse_puncta_rigid_motion_distorted_stack(
        time_count=3,
        z_count=14,
        shape_yx=(64, 64),
        point_count=45,
        noise_sigma=0.0,
        random_state=103,
    )

    registered, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        rotreg=True,
        rigid_3d_backend="points",
        rot_init_iterations=0,
        rot_points_max_points=80,
        rot_points_min_distance=2,
        rot_points_threshold_rel=0.2,
        rot_points_iterations=5,
        rot_points_max_match_distance=6.0,
        transform_order=0,
        verbose=False,
        return_details=True,
    )

    assert registered.shape == stack.shape
    assert details["rigid_3d_backend"] == "points"
    assert details["time_shifts_zyx"].shape == (stack.shape[0], 3)
    assert details["rotation_shifts_zyx_deg"].shape == (stack.shape[0], 3)
    assert np.max(np.abs(details["time_shifts_zyx"][1:])) > 0.1
    assert np.max(np.abs(details["rotation_shifts_zyx_deg"][1:])) > 0.1


def test_rigid_3d_mask_zero_clip_preserves_more_z_with_corner_invalidity():
    stack, _, _, _ = create_3d_time_sparse_puncta_rigid_motion_distorted_stack(
        time_count=3,
        z_count=14,
        shape_yx=(64, 64),
        point_count=45,
        noise_sigma=0.0,
        random_state=103,
    )

    registered, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        rotreg=True,
        rigid_3d_backend="points",
        rot_init_iterations=0,
        rot_points_max_points=80,
        rot_points_min_distance=2,
        rot_points_threshold_rel=0.2,
        rot_points_iterations=5,
        rot_points_max_match_distance=6.0,
        transform_order=0,
        zero_clip=True,
        zero_clip_margin=(0, 0, 0),
        verbose=False,
        return_details=True,
    )

    assert registered.shape[1] >= 8
    assert details["zero_clip_bounds"]["z_top"] + details["zero_clip_bounds"]["z_bottom"] <= 6


def test_rigid_3d_zero_clip_failure_warns_and_keeps_registered_stack():
    pytest.importorskip("SimpleITK")
    stack, _, _, _ = create_3d_time_sparse_puncta_rigid_motion_distorted_stack(
        time_count=3,
        z_count=14,
        shape_yx=(64, 64),
        point_count=45,
        noise_sigma=0.0,
        random_state=107,
    )

    with pytest.warns(RuntimeWarning, match="zero_clip=True was requested"):
        registered, details = register_stack(
            stack,
            registration_channel=0,
            method="phase_cross_correlation",
            time_registration_mode="full_3d",
            zreg=True,
            rotreg=True,
            rigid_3d_backend="simpleitk",
            rot_init_iterations=2,
            transform_order=0,
            zero_clip=True,
            zero_clip_margin=(20, 40, 40),
            verbose=False,
            return_details=True,
        )

    assert registered.shape == stack.shape
    assert details["zero_clip_bounds"] is None
    assert isinstance(details["zero_clip_failed_reason"], str)
