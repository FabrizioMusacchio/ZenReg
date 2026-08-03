import numpy as np
import pytest

from zenreg import (
    load_expected_rigid_z_rotation,
    load_expected_rigid_corrections,
    load_expected_slice_registration_shifts,
    load_expected_time_registration_rotations,
    load_expected_time_registration_shifts,
    open_in_napari,
    print_caiman_patch_summary,
    print_local_patch_summary,
    print_residual_mae_summary,
    print_rigid_comparison,
    print_shift_comparison,
    show_before_after,
    show_projection,
    show_residual_comparison,
    show_residual_comparison_multi,
    show_slices,
    show_timepoints,
)


def test_load_expected_time_registration_shifts(tmp_path):
    path = tmp_path / "time_shifts.csv"
    path.write_text(
        "t,expected_registration_shift_y_ref_t0,expected_registration_shift_x_ref_t0\n"
        "0,0.0,0.0\n"
        "1,-2.5,1.25\n"
    )

    shifts = load_expected_time_registration_shifts(path, registration_stack=0, axes="yx")

    np.testing.assert_allclose(shifts, [[0.0, 0.0], [-2.5, 1.25]])
    assert shifts.dtype == np.float32


def test_load_expected_slice_registration_shifts_supports_current_columns(tmp_path):
    path = tmp_path / "slice_shifts.csv"
    path.write_text(
        "t,z,expected_registration_shift_y,expected_registration_shift_x\n"
        "0,0,0.0,0.0\n"
        "0,1,1.5,-2.0\n"
    )

    shifts = load_expected_slice_registration_shifts(path)

    assert shifts.shape == (1, 2, 2)
    np.testing.assert_allclose(shifts[0, 1], [1.5, -2.0])


def test_load_expected_slice_registration_shifts_supports_legacy_columns(tmp_path):
    path = tmp_path / "legacy_slice_shifts.csv"
    path.write_text(
        "t,z,expected_local_z_correction_shift_y,expected_local_z_correction_shift_x\n"
        "0,0,0.0,0.0\n"
        "0,1,-1.0,2.5\n"
    )

    shifts = load_expected_slice_registration_shifts(path)

    assert shifts.shape == (1, 2, 2)
    np.testing.assert_allclose(shifts[0, 1], [-1.0, 2.5])


def test_load_expected_rigid_corrections(tmp_path):
    path = tmp_path / "rigid.csv"
    path.write_text(
        "t,"
        "expected_registration_shift_z_ref_t0,"
        "expected_registration_shift_y_ref_t0,"
        "expected_registration_shift_x_ref_t0,"
        "expected_registration_rotation_z_deg_ref_t0,"
        "expected_registration_rotation_y_deg_ref_t0,"
        "expected_registration_rotation_x_deg_ref_t0\n"
        "0,0.0,0.0,0.0,0.0,0.0,0.0\n"
        "1,-1.0,2.0,-3.0,1.5,-2.5,3.5\n"
    )

    shifts, rotations = load_expected_rigid_corrections(path, registration_stack=0)

    np.testing.assert_allclose(shifts, [[0.0, 0.0, 0.0], [-1.0, 2.0, -3.0]])
    np.testing.assert_allclose(rotations, [[0.0, 0.0, 0.0], [1.5, -2.5, 3.5]])


def test_load_expected_rotation_helpers(tmp_path):
    rigid_path = tmp_path / "rigid_z.csv"
    rigid_path.write_text(
        "t,"
        "expected_registration_shift_z_ref_t0,"
        "expected_registration_shift_y_ref_t0,"
        "expected_registration_shift_x_ref_t0,"
        "expected_registration_rotation_z_deg_ref_t0\n"
        "0,0,0,0,0\n"
        "1,-1,2,-3,4.5\n"
    )
    rotation_path = tmp_path / "rotation.csv"
    rotation_path.write_text(
        "t,expected_registration_rotation_deg_ref_t0\n"
        "0,0\n"
        "1,-3.25\n"
    )

    shifts, rotations_z = load_expected_rigid_z_rotation(rigid_path, registration_stack=0)
    rotations = load_expected_time_registration_rotations(rotation_path, registration_stack=0)

    np.testing.assert_allclose(shifts, [[0, 0, 0], [-1, 2, -3]])
    np.testing.assert_allclose(rotations_z, [0, 4.5])
    np.testing.assert_allclose(rotations, [0, -3.25])


def test_show_before_after_writes_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    stack = np.zeros((2, 1, 1, 12, 12), dtype=np.float32)
    registered = np.zeros_like(stack)
    stack[0, 0, 0, 4:8, 4:8] = 1.0
    stack[1, 0, 0, 5:9, 5:9] = 1.0
    registered[0, 0, 0, 4:8, 4:8] = 1.0
    registered[1, 0, 0, 4:8, 4:8] = 1.0

    output_path = tmp_path / "before_after.png"
    show_before_after(
        stack,
        registered,
        title="Before after test",
        save_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_show_timepoints_and_slices_write_pngs(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    stack = np.zeros((2, 3, 1, 12, 12), dtype=np.float32)
    stack[0, 0, 0, 4:8, 4:8] = 1.0
    stack[1, 1, 0, 5:9, 5:9] = 1.0

    show_timepoints(
        stack,
        title="Timepoints test",
        moving_time=1,
        projection_z_range=(0, 2),
        save_dir=tmp_path,
    )
    show_slices(stack, title="Slices test", save_dir=tmp_path, z0=0, z1=99)

    assert (tmp_path / "timepoints_test_c0_t0_t1_max_z0-2_timepoints.png").exists()
    assert (tmp_path / "slices_test_c0_z0_z2_slices.png").exists()


def test_show_projection_returns_yx_and_writes_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    stack = np.zeros((3, 2, 2, 8, 9), dtype=np.float32)
    stack[0, 0, 1, 2:4, 3:5] = 1.0
    stack[1, 1, 1, 4:6, 5:7] = 2.0

    result = show_projection(
        stack,
        title="Projection preview",
        registration_channel=1,
        registration_template_time_range=(0, 2),
        registration_z_range="all",
        projection_method="max",
        save_dir=tmp_path,
    )
    projection = show_projection(
        stack,
        title="Projection preview returned",
        registration_channel=1,
        registration_template_time_range=(0, 2),
        registration_z_range="all",
        projection_method="max",
        return_projection=True,
    )

    assert result is None
    assert projection.shape == (8, 9)
    assert float(np.max(projection)) == 2.0
    assert (tmp_path / "projection_preview_c1_max_t0-2_zall_projection.png").exists()


def test_show_timepoints_skips_single_timepoint(capsys):
    stack = np.zeros((1, 1, 1, 6, 6), dtype=np.float32)

    show_timepoints(stack, title="single")

    assert "T < 2" in capsys.readouterr().out


def test_residual_comparison_helpers_write_figures_without_errors():
    import matplotlib

    matplotlib.use("Agg", force=True)
    raw = np.zeros((2, 1, 1, 10, 10), dtype=np.float32)
    reg1 = raw.copy()
    reg2 = raw.copy()
    raw[1, 0, 0, 4:6, 4:6] = 1.0
    reg1[1, 0, 0, 3:5, 3:5] = 1.0
    reg2[1, 0, 0, 2:4, 2:4] = 1.0

    show_residual_comparison(raw, reg1, reg2, title="Residual compare")
    show_residual_comparison_multi(raw, (reg1, reg2), title="Residual multi", labels=("raw", "a", "b"))
    with pytest.raises(ValueError, match="labels"):
        show_residual_comparison_multi(raw, (reg1,), title="bad", labels=("raw", "a", "b"))


def test_print_summary_helpers_emit_expected_text(capsys):
    print_shift_comparison(
        "shift summary",
        np.asarray([[0.0, 0.0], [1.0, -2.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.5, -1.5]], dtype=np.float32),
    )
    details = {
        "rigid_3d_backend": "points",
        "time_shifts_zyx": np.asarray([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
        "rotation_shifts_zyx_deg": np.asarray([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
    }
    print_rigid_comparison(
        "rigid summary",
        details,
        np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
    )
    print_local_patch_summary(
        "local summary",
        {
            "time_shifts_yx": np.asarray([[0, 0], [1, 2]], dtype=np.float32),
            "patch_shifts": np.ones((2, 2, 2, 2), dtype=np.float32),
        },
        t=1,
    )
    print_caiman_patch_summary(
        "caiman summary",
        {
            "time_shifts_raw_caiman": np.asarray([[0, 0], [1, 2]], dtype=np.float32),
            "x_shifts_els": np.ones((2, 2, 2), dtype=np.float32),
            "y_shifts_els": np.ones((2, 2, 2), dtype=np.float32) * 2,
        },
        t=1,
    )
    print_residual_mae_summary(
        np.zeros((2, 1, 1, 5, 5), dtype=np.float32),
        np.ones((2, 1, 1, 5, 5), dtype=np.float32),
        labels=("raw", "registered"),
    )

    output = capsys.readouterr().out
    assert "shift summary" in output
    assert "rigid summary" in output
    assert "local summary" in output
    assert "caiman summary" in output
    assert "Residual MAE" in output


def test_open_in_napari_respects_enabled_false(monkeypatch):
    def fail_import(*args, **kwargs):
        raise AssertionError("OMIO should not be imported when enabled=False")

    monkeypatch.setattr("builtins.__import__", fail_import)
    open_in_napari(np.zeros((1, 1, 1, 2, 2), dtype=np.float32), {}, fname="off", enabled=False)
