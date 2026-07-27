import numpy as np

from zenreg import register_stack, z_project
from zenreg.synthetic import create_2d_motion_distorted_stack


def test_phase_cross_correlation_matches_synthetic_gt_default_reference():
    stack, applied_shifts = create_2d_motion_distorted_stack(noise_sigma=0.0)

    _, estimated_shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        verbose=False,
        return_shifts=True,
    )

    expected_shifts = applied_shifts[0, :] - applied_shifts
    np.testing.assert_allclose(estimated_shifts, expected_shifts, atol=0.05)


def test_registration_stack_selects_reference_timepoint():
    stack, applied_shifts = create_2d_motion_distorted_stack(noise_sigma=0.0)
    registration_stack = 3

    _, estimated_shifts = register_stack(
        stack,
        registration_channel=0,
        registration_stack=registration_stack,
        method="phase_cross_correlation",
        verbose=False,
        return_shifts=True,
    )

    expected_shifts = applied_shifts[registration_stack, :] - applied_shifts
    np.testing.assert_allclose(estimated_shifts, expected_shifts, atol=0.05)
    np.testing.assert_array_equal(estimated_shifts[registration_stack], np.zeros(2, dtype=np.float32))


def test_z_project_supports_expected_projection_methods():
    stack = np.arange(2 * 3 * 1 * 4 * 5, dtype=np.float32).reshape(2, 3, 1, 4, 5)

    np.testing.assert_allclose(z_project(stack, projection_method="max"), np.max(stack, axis=1, keepdims=True))
    np.testing.assert_allclose(
        z_project(stack, projection_method="mean"),
        np.mean(stack, axis=1, keepdims=True),
    )
    np.testing.assert_allclose(
        z_project(stack, projection_method="median"),
        np.median(stack, axis=1, keepdims=True),
    )
    np.testing.assert_allclose(z_project(stack, projection_method="var"), np.var(stack, axis=1, keepdims=True))
    np.testing.assert_allclose(z_project(stack, projection_method="std"), np.std(stack, axis=1, keepdims=True))
