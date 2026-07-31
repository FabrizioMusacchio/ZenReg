import numpy as np

from zenreg import (
    load_expected_rigid_corrections,
    load_expected_slice_registration_shifts,
    load_expected_time_registration_shifts,
    show_before_after,
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
