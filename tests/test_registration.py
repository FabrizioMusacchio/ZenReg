import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from zenreg import register_stack, z_project
import zenreg.registration as registration_module
from zenreg.synthetic import (
    create_2d_motion_distorted_stack,
    create_2d_time_rotation_motion_distorted_stack,
    create_3d_slice_motion_distorted_stack,
    create_3d_time_zyx_motion_distorted_stack,
)


def _gaussian_volume(shape_zyx=(9, 48, 48)):
    zz, yy, xx = np.indices(shape_zyx, dtype=np.float32)
    volume = np.zeros(shape_zyx, dtype=np.float32)
    for cz, cy, cx, sz, sy, sx in [
        (3.5, 15, 17, 1.2, 4.0, 5.0),
        (5.0, 31, 28, 1.0, 5.0, 4.0),
        (4.0, 22, 37, 1.4, 3.0, 3.0),
    ]:
        volume += np.exp(
            -(
                ((zz - cz) ** 2) / (2 * sz**2)
                + ((yy - cy) ** 2) / (2 * sy**2)
                + ((xx - cx) ** 2) / (2 * sx**2)
            )
        )
    return volume.astype(np.float32)


def _two_timepoint_3d_stack(applied_shift_zyx):
    base = _gaussian_volume()
    moved = ndi_shift(base, shift=applied_shift_zyx, order=1, mode="constant", cval=0.0)
    stack = np.zeros((2, base.shape[0], 1, base.shape[1], base.shape[2]), dtype=np.float32)
    stack[0, :, 0, :, :] = base
    stack[1, :, 0, :, :] = moved
    return stack


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


def test_2d_synthetic_shift_amplitudes_keep_default_and_allow_override():
    _, default_shifts = create_2d_motion_distorted_stack(time_count=9, noise_sigma=0.0)
    _, custom_shifts = create_2d_motion_distorted_stack(
        time_count=9,
        shift_amplitude_y=8.0,
        shift_amplitude_x=6.0,
        noise_sigma=0.0,
    )

    np.testing.assert_allclose(default_shifts, custom_shifts / 2.0, atol=1e-6)
    np.testing.assert_allclose(np.max(default_shifts[:, 0]), 4.0, atol=1e-6)
    np.testing.assert_allclose(np.min(default_shifts[:, 1]), -6.0, atol=1e-6)


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


def test_registration_template_time_range_builds_2d_time_template():
    stack, _ = create_2d_motion_distorted_stack(time_count=5, noise_sigma=0.0)

    registered, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        registration_template_time_range=(0, 3),
        projection_method="mean",
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    assert registered.shape == stack.shape
    assert details["registration_template_time_range"] == (0, 3)
    assert details["projection_method"] == "mean"
    assert details["time_shifts_zyx"].shape == (stack.shape[0], 3)


def test_registration_template_time_range_all_uses_all_frames():
    stack, _ = create_2d_motion_distorted_stack(time_count=5, noise_sigma=0.0)

    _, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        registration_template_time_range="all",
        projection_method="mean",
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    assert details["registration_template_time_range"] == (0, stack.shape[0])


def test_registration_template_time_range_rejects_previous_reference_mode():
    stack, _ = create_2d_motion_distorted_stack(time_count=5, noise_sigma=0.0)

    with pytest.raises(ValueError, match="registration_template_time_range requires"):
        register_stack(
            stack,
            registration_channel=0,
            method="phase_cross_correlation",
            time_reference_mode="previous",
            registration_template_time_range=(0, 3),
            verbose=False,
        )


def test_raw_time_shifts_are_reported_before_limit_clipping():
    stack = _two_timepoint_3d_stack((0.0, 4.0, -3.0))

    _, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        max_xy_shifts=(1, 1),
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    np.testing.assert_allclose(details["time_shifts_yx"][1], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(details["time_shifts_yx_raw"][1], [-4.0, 3.0], atol=0.05)


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


def test_full_3d_time_registration_estimates_zyx_shift():
    stack = _two_timepoint_3d_stack((1.0, 2.0, -3.0))

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shift_details["time_shifts_zyx"][1], [-1.0, -2.0, 3.0], atol=0.08)


def test_rigid_3d_spacing_defaults_to_omio_metadata(monkeypatch):
    stack = np.zeros((2, 3, 1, 8, 8), dtype=np.float32)
    metadata = {
        "PhysicalSizeZ": 2.5,
        "PhysicalSizeY": 0.4,
        "PhysicalSizeX": 0.3,
        "PhysicalSizeZUnit": "micron",
        "PhysicalSizeYUnit": "micron",
        "PhysicalSizeXUnit": "micron",
    }
    captured = {}

    def fake_register_stack_rigid_3d_from_main_wrapper(stack_arg, **kwargs):
        captured["rot_spacing_zyx"] = kwargs["rot_spacing_zyx"]
        captured["registration_settings"] = kwargs["registration_settings"]
        details = {
            "rot_spacing_zyx": kwargs["rot_spacing_zyx"],
            "rot_spacing_source": kwargs["registration_settings"]["rot_spacing_source"],
        }
        return stack_arg, details

    monkeypatch.setattr(
        registration_module,
        "_register_stack_rigid_3d_from_main_wrapper",
        fake_register_stack_rigid_3d_from_main_wrapper,
    )

    _, details = register_stack(
        stack,
        metadata=metadata,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        rotreg=True,
        rigid_3d_backend="simpleitk",
        verbose=False,
        return_details=True,
    )

    assert captured["rot_spacing_zyx"] == pytest.approx((2.5, 0.4, 0.3))
    assert captured["registration_settings"]["rot_spacing_source"] == "metadata"
    assert details["rot_spacing_source"] == "metadata"


def test_user_rot_spacing_overrides_metadata(monkeypatch):
    stack = np.zeros((2, 3, 1, 8, 8), dtype=np.float32)
    metadata = {"PhysicalSizeZ": 2.5, "PhysicalSizeY": 0.4, "PhysicalSizeX": 0.3}
    captured = {}

    def fake_register_stack_rigid_3d_from_main_wrapper(stack_arg, **kwargs):
        captured["rot_spacing_zyx"] = kwargs["rot_spacing_zyx"]
        captured["registration_settings"] = kwargs["registration_settings"]
        return stack_arg, kwargs["registration_settings"]

    monkeypatch.setattr(
        registration_module,
        "_register_stack_rigid_3d_from_main_wrapper",
        fake_register_stack_rigid_3d_from_main_wrapper,
    )

    register_stack(
        stack,
        metadata=metadata,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        rotreg=True,
        rigid_3d_backend="simpleitk",
        rot_spacing_zyx=(1.0, 1.1, 1.2),
        verbose=False,
        return_details=True,
    )

    assert captured["rot_spacing_zyx"] == pytest.approx((1.0, 1.1, 1.2))
    assert captured["registration_settings"]["rot_spacing_source"] == "user"


def test_projection_time_registration_can_estimate_z_shift_from_orthogonal_projections():
    stack = _two_timepoint_3d_stack((1.0, 0.0, 0.0))

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="projection",
        zreg=True,
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shift_details["time_shifts_zyx"][1], [-1.0, 0.0, 0.0], atol=0.08)


def test_shift_limits_clip_time_registration_shifts():
    stack = _two_timepoint_3d_stack((0.0, 5.0, -7.0))

    _, shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        max_xy_shifts=(2.0, 3.0),
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shifts[1], [-2.0, 3.0], atol=1e-6)


def test_skimage_transform_backend_runs_with_nearest_neighbor_order():
    stack = _two_timepoint_3d_stack((0.0, 2.0, -3.0))

    registered, shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        transform_backend="skimage",
        transform_order=0,
        verbose=False,
        return_shifts=True,
    )

    assert registered.shape == stack.shape
    np.testing.assert_allclose(shifts[1], [-2.0, 3.0], atol=0.08)


def test_standard_time_registration_n_jobs_matches_serial():
    stack, _ = create_3d_time_zyx_motion_distorted_stack(
        time_count=5,
        z_count=10,
        noise_sigma=0.0,
    )

    serial_registered, serial_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        n_jobs=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    parallel_registered, parallel_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        n_jobs=2,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    np.testing.assert_allclose(parallel_details["time_shifts_zyx"], serial_details["time_shifts_zyx"])
    np.testing.assert_allclose(parallel_registered, serial_registered)
    assert parallel_details["n_jobs"] == 2


def test_register_stack_can_run_intra_stack_only():
    stack = _two_timepoint_3d_stack((0.0, 0.0, 0.0))

    corrected, shifts = register_stack(
        stack,
        registration_channel=0,
        time_registration_mode="none",
        intra_stack=True,
        intra_stack_reference_mode="full_projection",
        verbose=False,
        return_shifts=True,
    )

    assert corrected.shape == stack.shape
    assert shifts.shape == (2, stack.shape[1], 2)


def test_intra_stack_n_jobs_matches_serial():
    stack, _ = create_3d_slice_motion_distorted_stack(
        z_count=8,
        noise_sigma=0.0,
    )

    serial_corrected, serial_shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="none",
        intra_stack=True,
        intra_stack_reference_mode="first_slice",
        n_jobs=1,
        verbose=False,
        return_shifts=True,
    )
    parallel_corrected, parallel_shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="none",
        intra_stack=True,
        intra_stack_reference_mode="first_slice",
        n_jobs=2,
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(parallel_shifts, serial_shifts)
    np.testing.assert_allclose(parallel_corrected, serial_corrected)


def test_full_3d_pystackreg_falls_back_to_projection_mode():
    stack = _two_timepoint_3d_stack((0.0, 1.0, -1.0))

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="pystackreg",
        time_registration_mode="full_3d",
        verbose=False,
        return_shifts=True,
    )

    assert shift_details["effective_time_registration_mode"] == "projection"


def test_previous_time_reference_accumulates_pairwise_shifts():
    base = _gaussian_volume()
    stack = np.zeros((3, base.shape[0], 1, base.shape[1], base.shape[2]), dtype=np.float32)
    for t, shift_y in enumerate([0.0, 2.0, 4.0]):
        stack[t, :, 0, :, :] = ndi_shift(
            base,
            shift=(0.0, shift_y, 0.0),
            order=1,
            mode="constant",
            cval=0.0,
        )

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_reference_mode="previous",
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shift_details["time_shifts_zyx"][:, 1], [0.0, -2.0, -4.0], atol=0.08)


def test_synthetic_3d_slice_example_matches_first_slice_intra_stack_gt():
    stack, applied_slice_shifts = create_3d_slice_motion_distorted_stack(
        z_count=8,
        noise_sigma=0.0,
    )

    _, estimated_shifts = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="none",
        intra_stack=True,
        intra_stack_reference_mode="first_slice",
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(estimated_shifts, -applied_slice_shifts, atol=0.15)


def test_synthetic_3d_time_zyx_example_matches_full_3d_gt():
    stack, applied_time_shifts = create_3d_time_zyx_motion_distorted_stack(
        time_count=5,
        z_count=14,
        noise_sigma=0.0,
    )

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shift_details["time_shifts_zyx"], applied_time_shifts[0] - applied_time_shifts, atol=0.4)


def test_zero_clip_crops_directional_zyx_translation_borders():
    stack = _two_timepoint_3d_stack((-1.0, -2.0, 3.0))

    clipped, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        time_registration_mode="full_3d",
        zreg=True,
        zero_clip=True,
        verbose=False,
        return_shifts=True,
    )

    assert clipped.shape == (2, stack.shape[1] - 1, 1, stack.shape[3] - 2, stack.shape[4] - 3)
    assert shift_details["zero_clip_bounds"] == {
        "z_top": 1,
        "z_bottom": 0,
        "y_top": 2,
        "y_bottom": 0,
        "x_left": 0,
        "x_right": 3,
    }


def test_registration_z_range_matches_legacy_aliases():
    stack = _two_timepoint_3d_stack((0.0, 2.0, -3.0))

    _, shifts_registration_z_range = register_stack(
        stack,
        registration_channel=0,
        registration_z_range=(1, stack.shape[1] - 1),
        verbose=False,
        return_shifts=True,
    )
    _, shifts_zrange = register_stack(
        stack,
        registration_channel=0,
        zrange=(1, stack.shape[1] - 1),
        verbose=False,
        return_shifts=True,
    )
    _, shifts_projection_range = register_stack(
        stack,
        registration_channel=0,
        projection_range=(1, stack.shape[1] - 1),
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(shifts_zrange, shifts_registration_z_range)
    np.testing.assert_allclose(shifts_projection_range, shifts_zrange)


def test_registration_z_range_rejects_conflicting_aliases():
    stack = _two_timepoint_3d_stack((0.0, 2.0, -3.0))

    with pytest.raises(ValueError, match="conflicting registration_z_range"):
        register_stack(
            stack,
            registration_channel=0,
            registration_z_range=(1, stack.shape[1] - 1),
            projection_range=(0, stack.shape[1]),
            verbose=False,
        )


def test_synthetic_rotation_example_matches_rotation_gt():
    stack, _, applied_rotations_deg = create_2d_time_rotation_motion_distorted_stack(
        time_count=5,
        noise_sigma=0.0,
    )

    _, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        rotreg=True,
        max_xy_shifts=(0, 0),
        max_rot_shifts=12,
        verbose=False,
        return_shifts=True,
    )

    np.testing.assert_allclose(
        shift_details["rotation_shifts_deg"],
        applied_rotations_deg[0] - applied_rotations_deg,
        atol=0.25,
    )


def test_rotation_n_jobs_matches_serial():
    stack, _, _ = create_2d_time_rotation_motion_distorted_stack(
        time_count=5,
        noise_sigma=0.0,
    )

    serial_registered, serial_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        rotreg=True,
        max_xy_shifts=(0, 0),
        max_rot_shifts=12,
        n_jobs=1,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    parallel_registered, parallel_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        rotreg=True,
        max_xy_shifts=(0, 0),
        max_rot_shifts=12,
        n_jobs=2,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    np.testing.assert_allclose(parallel_details["time_shifts_zyx"], serial_details["time_shifts_zyx"])
    np.testing.assert_allclose(parallel_details["rotation_shifts_deg"], serial_details["rotation_shifts_deg"])
    np.testing.assert_allclose(parallel_registered, serial_registered)


def test_rotation_zero_clip_auto_uses_mask_bounds():
    stack, _, _ = create_2d_time_rotation_motion_distorted_stack(
        time_count=5,
        noise_sigma=0.0,
    )

    clipped, shift_details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        rotreg=True,
        max_xy_shifts=(0, 0),
        zero_clip=True,
        verbose=False,
        return_shifts=True,
    )

    assert shift_details["zero_clip_mode"] == "mask"
    assert clipped.shape[3] < stack.shape[3]
    assert clipped.shape[4] < stack.shape[4]
    assert shift_details["zero_clip_bounds"]["y_top"] > 0
    assert shift_details["zero_clip_bounds"]["x_left"] > 0


def test_standard_registration_output_memmap_matches_numpy(tmp_path):
    stack, _ = create_2d_motion_distorted_stack(time_count=4, noise_sigma=0.0)

    registered_numpy, details_numpy = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    registered_memmap, details_memmap = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        output_use_memmap=True,
        output_memmap_folder=tmp_path / "standard_cache",
        output_memmap_name="registered_standard",
        n_jobs=2,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    assert registered_memmap.shape == registered_numpy.shape
    assert details_memmap["output_use_memmap"] is True
    np.testing.assert_allclose(details_memmap["time_shifts_zyx"], details_numpy["time_shifts_zyx"])
    np.testing.assert_allclose(np.asarray(registered_memmap), registered_numpy)


def test_standard_zero_clip_output_memmap_matches_numpy(tmp_path):
    stack, _ = create_2d_motion_distorted_stack(time_count=4, noise_sigma=0.0)

    clipped_numpy, details_numpy = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        zero_clip=True,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )
    clipped_memmap, details_memmap = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        zero_clip=True,
        output_use_memmap=True,
        output_memmap_folder=tmp_path / "zero_clip_cache",
        output_memmap_name="registered_zero_clip",
        n_jobs=2,
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    assert clipped_memmap.shape == clipped_numpy.shape
    assert details_memmap["zero_clip_bounds"] == details_numpy["zero_clip_bounds"]
    np.testing.assert_allclose(np.asarray(clipped_memmap), clipped_numpy)
