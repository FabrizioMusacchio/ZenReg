import numpy as np
from scipy.ndimage import map_coordinates, shift as ndi_shift

from zenreg import plot_normcorre_patch_overlay, register_stack, register_stack_normcorre


def _gaussian_image(shape_yx=(80, 80)):
    yy, xx = np.indices(shape_yx, dtype=np.float32)
    image = np.zeros(shape_yx, dtype=np.float32)
    for cy, cx, sy, sx in [
        (24, 24, 5, 7),
        (48, 52, 7, 5),
        (56, 25, 4, 4),
        (28, 58, 3, 6),
    ]:
        image += np.exp(-(((yy - cy) ** 2) / (2 * sy**2) + ((xx - cx) ** 2) / (2 * sx**2)))
    image -= image.min()
    image /= image.max()
    return image.astype(np.float32)


def _gaussian_volume(shape_zyx=(9, 64, 64)):
    zz, yy, xx = np.indices(shape_zyx, dtype=np.float32)
    volume = np.zeros(shape_zyx, dtype=np.float32)
    for cz, cy, cx, sz, sy, sx in [
        (2.5, 20, 22, 1.0, 5, 7),
        (5.5, 42, 44, 1.2, 7, 5),
        (6.0, 46, 24, 0.9, 4, 4),
    ]:
        volume += np.exp(
            -(
                ((zz - cz) ** 2) / (2 * sz**2)
                + ((yy - cy) ** 2) / (2 * sy**2)
                + ((xx - cx) ** 2) / (2 * sx**2)
            )
        )
    volume -= volume.min()
    volume /= volume.max()
    return volume.astype(np.float32)


def _two_channel_2d_stack(applied_shifts_yx):
    base0 = _gaussian_image()
    base1 = np.flipud(base0)
    stack = np.zeros((len(applied_shifts_yx), 1, 2, *base0.shape), dtype=np.float32)
    for t, shift_yx in enumerate(applied_shifts_yx):
        stack[t, 0, 0] = ndi_shift(base0, shift=shift_yx, order=1, mode="constant", cval=0.0)
        stack[t, 0, 1] = ndi_shift(base1, shift=shift_yx, order=1, mode="constant", cval=0.0)
    return stack


def _two_channel_3d_stack(applied_shifts_zyx):
    base0 = _gaussian_volume()
    base1 = base0[:, ::-1, :]
    stack = np.zeros((len(applied_shifts_zyx), base0.shape[0], 2, base0.shape[1], base0.shape[2]), dtype=np.float32)
    for t, shift_zyx in enumerate(applied_shifts_zyx):
        stack[t, :, 0] = ndi_shift(base0, shift=shift_zyx, order=1, mode="constant", cval=0.0)
        stack[t, :, 1] = ndi_shift(base1, shift=shift_zyx, order=1, mode="constant", cval=0.0)
    return stack


def _local_y_shift(image, *, amplitude=2.0):
    yy, xx = np.indices(image.shape, dtype=np.float32)
    shift_y = amplitude * np.sin(2 * np.pi * xx / image.shape[1])
    coords = [yy - shift_y, xx]
    return map_coordinates(image, coords, order=1, mode="constant", cval=0.0).astype(np.float32)


def test_plot_normcorre_patch_overlay_writes_png(tmp_path):
    stack = _two_channel_3d_stack(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
    metadata = {
        "Annotations": {
            "original_filename": "source.ome.tif",
            "original_parentfolder": str(tmp_path),
        }
    }

    output_path = plot_normcorre_patch_overlay(
        stack,
        metadata,
        registration_channel=0,
        registration_stack=0,
        nc_strides=(3, 24, 24),
        nc_overlaps=(2, 12, 12),
        projection_method="mean",
        projection_range=(1, 10),
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0


def test_normcorre_2d_translation_estimates_correction_shift():
    applied = np.asarray([[0.0, 0.0], [2.0, -3.0], [-1.5, 2.5]], dtype=np.float32)
    stack = _two_channel_2d_stack(applied)

    registered, details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=False,
        max_shifts=(5, 5),
        upsample_factor=10,
        verbose=False,
    )

    assert registered.shape == stack.shape
    np.testing.assert_allclose(details["time_shifts_yx"], applied[0] - applied, atol=0.12)
    assert np.mean(np.abs(registered[1, 0, 0] - registered[0, 0, 0])) < np.mean(
        np.abs(stack[1, 0, 0] - stack[0, 0, 0])
    )


def test_register_stack_dispatches_to_normcorre_backend():
    applied = np.asarray([[0.0, 0.0], [2.0, -3.0], [-1.5, 2.5]], dtype=np.float32)
    stack = _two_channel_2d_stack(applied)

    registered, details = register_stack(
        stack,
        registration_channel=0,
        registration_stack=0,
        method="normcorre",
        time_registration_mode="projection",
        projection_method="max",
        max_xy_shifts=(5, 5),
        phase_cross_correlation_upsample_factor=10,
        phase_cross_correlation_normalization=None,
        nc_pw_rigid=False,
        verbose=False,
        return_details=True,
    )

    assert registered.shape == stack.shape
    assert details["method"] == "normcorre"
    assert details["nc_pw_rigid"] is False
    np.testing.assert_allclose(details["time_shifts_yx"], applied[0] - applied, atol=0.12)


def test_normcorre_correction_iterations_accumulate_residual_shifts():
    applied = np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    stack = _two_channel_2d_stack(applied)

    one_pass, one_details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=False,
        max_shifts=(2, 2),
        correction_iterations=1,
        upsample_factor=10,
        verbose=False,
    )
    two_pass, two_details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=False,
        max_shifts=(2, 2),
        correction_iterations=2,
        upsample_factor=10,
        verbose=False,
    )

    one_error = np.mean(np.abs(one_pass[1, 0, 0] - one_pass[0, 0, 0]))
    two_error = np.mean(np.abs(two_pass[1, 0, 0] - two_pass[0, 0, 0]))
    assert two_error < one_error
    np.testing.assert_allclose(two_details["time_shifts_yx"][1], [-4.0, 0.0], atol=0.2)
    assert two_details["rigid_shifts_by_correction"].shape == (2, 2, 2)
    np.testing.assert_allclose(one_details["time_shifts_yx"][1], [-2.0, 0.0], atol=0.2)


def test_normcorre_piecewise_global_translation_stays_near_rigid_solution():
    applied = np.asarray([[0.0, 0.0], [2.0, -3.0], [-1.5, 2.5]], dtype=np.float32)
    stack = _two_channel_2d_stack(applied)

    rigid_registered, _ = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=False,
        max_shifts=(5, 5),
        upsample_factor=10,
        verbose=False,
    )
    piecewise_registered, piecewise_details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=True,
        strides=(24, 24),
        overlaps=(12, 12),
        max_shifts=(5, 5),
        max_deviation_rigid=2,
        upsample_factor=10,
        verbose=False,
    )

    raw_error = np.mean(np.abs(stack[1, 0, 0] - stack[0, 0, 0]))
    rigid_error = np.mean(np.abs(rigid_registered[1, 0, 0] - rigid_registered[0, 0, 0]))
    piecewise_error = np.mean(np.abs(piecewise_registered[1, 0, 0] - piecewise_registered[0, 0, 0]))
    patch_shifts = piecewise_details["patch_shifts"][1].reshape(-1, 2)
    expected_shift = applied[0] - applied[1]
    np.testing.assert_allclose(piecewise_details["time_shifts_yx"][1], expected_shift, atol=0.12)
    assert np.all(np.abs(patch_shifts - expected_shift) < np.asarray([2.01, 2.01]))
    assert piecewise_error < raw_error * 0.5
    assert rigid_error < piecewise_error


def test_normcorre_3d_translation_estimates_zyx_correction_shift():
    applied = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, -3.0], [-1.0, -1.5, 2.0]], dtype=np.float32)
    stack = _two_channel_3d_stack(applied)

    registered, details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=True,
        pw_rigid=False,
        max_shifts=(2, 4, 4),
        upsample_factor=10,
        verbose=False,
    )

    assert registered.shape == stack.shape
    np.testing.assert_allclose(details["time_shifts_zyx"], applied[0] - applied, atol=0.15)


def test_normcorre_auto_uses_3d_for_multiz_but_explicit_false_projects():
    applied = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, -3.0]], dtype=np.float32)
    stack = _two_channel_3d_stack(applied)

    _, auto_details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=None,
        pw_rigid=False,
        max_shifts=(2, 4, 4),
        upsample_factor=10,
        verbose=False,
    )
    projected_registered, projected_details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        projection_method="mean",
        pw_rigid=False,
        max_shifts=(4, 4),
        upsample_factor=10,
        verbose=False,
    )

    assert auto_details["time_registration_mode"] == "full_3d"
    assert auto_details["rigid_shifts"].shape == (2, 3)
    assert projected_registered.shape == stack.shape
    assert projected_details["time_registration_mode"] == "projection"
    assert projected_details["projection_method"] == "mean"
    assert projected_details["rigid_shifts"].shape == (2, 2)


def test_normcorre_piecewise_mode_reports_patch_shift_grid():
    base = _gaussian_image()
    moved = _local_y_shift(base, amplitude=1.8)
    stack = np.zeros((2, 1, 1, *base.shape), dtype=np.float32)
    stack[0, 0, 0] = base
    stack[1, 0, 0] = moved

    registered, details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=True,
        strides=(24, 24),
        overlaps=(12, 12),
        max_shifts=(4, 4),
        max_deviation_rigid=3,
        upsample_factor=5,
        verbose=False,
    )

    assert registered.shape == stack.shape
    assert details["patch_shifts"].shape == (2, 3, 3, 2)
    assert float(np.std(details["patch_shifts"][1, :, :, 0])) > 0.05
    assert np.mean(np.abs(registered[1, 0, 0] - registered[0, 0, 0])) < np.mean(
        np.abs(stack[1, 0, 0] - stack[0, 0, 0])
    )


def test_normcorre_caiman_port_options_run_and_report_settings():
    applied = np.asarray([[0.0, 0.0], [1.2, -1.8], [-0.8, 1.0]], dtype=np.float32)
    stack = _two_channel_2d_stack(applied)

    registered, details = register_stack_normcorre(
        stack,
        registration_channel=0,
        is3d=False,
        pw_rigid=True,
        strides=(32, 32),
        overlaps=(16, 16),
        max_shifts=(4, 4),
        max_deviation_rigid=2,
        n_iterations=2,
        niter_rig=1,
        template_init_mode="rigid_median",
        template_update_method="caiman",
        splits=2,
        gSig_filt=(3, 3),
        shift_interpolation="resize",
        border_nan="copy",
        transform_order=3,
        upsample_factor=5,
        verbose=False,
    )

    assert registered.shape == stack.shape
    assert details["template_init_mode"] == "rigid_median"
    assert details["template_update_method"] == "caiman"
    assert details["splits"] == 2
    assert details["gSig_filt"] == (3.0, 3.0)
    assert details["shift_interpolation"] == "resize"
    assert details["border_nan"] == "copy"
