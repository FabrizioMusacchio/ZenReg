import csv

import numpy as np

from zenreg.reporting import (
    _projection_range_label,
    _raw_estimate_label,
    _settings_annotation,
    write_registration_outputs,
    write_registration_summary_plot,
)


def test_write_registration_outputs_accepts_legacy_shift_array(tmp_path):
    registered = np.zeros((2, 1, 1, 8, 8), dtype=np.float32)
    registered[:, 0, 0, 2:6, 2:6] = 1.0
    shifts_yx = np.asarray([[0.0, 0.0], [-1.5, 2.0]], dtype=np.float32)

    paths = write_registration_outputs(
        tmp_path / "legacy_registered.ome.tif",
        registered,
        shifts_yx,
    )

    assert paths["csv"] == tmp_path / "legacy_registered_registration_shifts.csv"
    assert paths["yaml"].exists()
    assert paths["plot"].exists()
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["shift_y"] == "-1.5"
    assert rows[1]["shift_x"] == "2"
    assert "registration_settings:" in paths["yaml"].read_text(encoding="utf-8")


def test_write_registration_summary_plot_only_writes_png(tmp_path):
    registered = np.zeros((2, 1, 1, 8, 8), dtype=np.float32)
    registered[:, 0, 0, 2:6, 2:6] = 1.0
    details = {
        "registration_channel": 0,
        "registration_stack": 0,
        "method": "phase_cross_correlation",
        "time_registration_mode": "projection",
        "effective_time_registration_mode": "projection",
        "time_reference_mode": "template",
        "projection_method": "max",
        "time_shifts_zyx": np.asarray([[0, 0, 0], [0, -1, 2]], dtype=np.float32),
        "pearson_correlations_before": np.asarray([1.0, 0.5], dtype=np.float32),
    }

    plot_path = write_registration_summary_plot(
        tmp_path / "preview" / "registration_summary.png",
        registered,
        details,
    )

    assert plot_path == tmp_path / "preview" / "registration_summary.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert not (tmp_path / "preview" / "registration_shifts.csv").exists()
    assert not (tmp_path / "preview" / "registration_settings.yaml").exists()


def test_write_registration_outputs_includes_optional_quality_metrics(tmp_path):
    registered = np.zeros((3, 1, 1, 8, 8), dtype=np.float32)
    registered[:, 0, 0, 2:6, 2:6] = 1.0
    details = {
        "registration_channel": 0,
        "registration_stack": 0,
        "method": "phase_cross_correlation",
        "time_registration_mode": "projection",
        "effective_time_registration_mode": "projection",
        "time_reference_mode": "template",
        "projection_method": "max",
        "calc_SNR": True,
        "calc_CNR": True,
        "SNR_sampling_step": 2,
        "CNR_sampling_step": 2,
        "snr_before": np.asarray([4.0, 3.5, 3.0], dtype=np.float32),
        "cnr_before": np.asarray([2.0, 1.5, 1.0], dtype=np.float32),
        "time_shifts_zyx": np.zeros((3, 3), dtype=np.float32),
        "pearson_correlations_before": np.asarray([1.0, 0.9, 0.8], dtype=np.float32),
    }

    paths = write_registration_outputs(
        tmp_path / "quality_registered.ome.tif",
        registered,
        details,
    )

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["snr_before"] == "4"
    assert rows[0]["cnr_before"] == "2"
    yaml_text = paths["yaml"].read_text(encoding="utf-8")
    assert "calc_SNR: true" in yaml_text
    assert "snr_before_mean" in yaml_text
    assert paths["plot"].stat().st_size > 0


def test_write_registration_outputs_with_prefix_intra_stack_and_rotation(tmp_path):
    registered = np.zeros((3, 2, 1, 8, 8), dtype=np.float32)
    registered[:, :, 0, 2:6, 2:6] = 1.0
    details = {
        "registration_channel": 0,
        "registration_stack": 0,
        "method": "phase_cross_correlation",
        "time_registration_mode": "full_3d",
        "effective_time_registration_mode": "full_3d",
        "time_reference_mode": "template",
        "registration_z_range": (0, 2),
        "projection_range": (0, 2),
        "projection_method": "mean",
        "zreg": True,
        "rotreg": True,
        "max_xy_shifts": (2.0, 3.0),
        "max_z_shifts": 1,
        "max_rot_shifts": 4.0,
        "transform_backend": "skimage",
        "transform_order": 1,
        "time_shifts_zyx": np.asarray(
            [[0, 0, 0], [0.5, -1, 2], [-0.5, 1, -2]],
            dtype=np.float32,
        ),
        "time_shifts_zyx_raw": np.asarray(
            [[0, 0, 0], [0.5, -3, 2], [-0.5, 1, -2]],
            dtype=np.float32,
        ),
        "rotation_shifts_zyx_deg": np.asarray(
            [[0, 0, 0], [1, 2, 3], [-1, -2, -3]],
            dtype=np.float32,
        ),
        "rotation_shifts_zyx_deg_raw": np.asarray(
            [[0, 0, 0], [5, 2, 3], [-1, -2, -3]],
            dtype=np.float32,
        ),
        "intra_stack_shifts_yx": np.ones((3, 2, 2), dtype=np.float32),
        "pearson_correlations_before": np.asarray([1.0, 0.4, 0.5], dtype=np.float32),
    }

    paths = write_registration_outputs(
        tmp_path / "registered.ome.tif",
        registered,
        details,
        report_prefix=tmp_path / "reports" / "custom_prefix",
    )

    assert paths["csv"] == tmp_path / "reports" / "custom_prefix_registration_shifts.csv"
    assert paths["yaml"] == tmp_path / "reports" / "custom_prefix_registration_settings.yaml"
    assert paths["plot"] == tmp_path / "reports" / "custom_prefix_registration_summary.png"
    csv_text = paths["csv"].read_text(encoding="utf-8")
    yaml_text = paths["yaml"].read_text(encoding="utf-8")
    assert "intra_stack" in csv_text
    assert "rotation_x_deg" in csv_text
    assert "shift_y_raw" in csv_text
    assert "rotation_z_limit_exceeded" in csv_text
    assert "True" in csv_text
    assert "correlation_before_mean" in yaml_text
    assert "registration_z_range: [0, 2]" in yaml_text
    assert paths["plot"].stat().st_size > 0


def test_summary_annotation_labels_template_time_and_singleton_z():
    registered = np.zeros((8, 1, 1, 8, 8), dtype=np.float32)
    details = {
        "registration_channel": 0,
        "registration_stack": 0,
        "registration_template_time_range": (0, 6),
        "method": "phase_cross_correlation",
        "time_registration_mode": "projection",
        "effective_time_registration_mode": "projection",
        "time_reference_mode": "template",
        "projection_method": "median",
        "intra_stack": False,
        "zreg": False,
        "rotreg": False,
        "transform_backend": "skimage",
        "transform_order": 1,
    }

    annotation = _settings_annotation(details, registered)

    assert "shape_TZCYX before registration=(8, 1, 1, 8, 8) | after=(8, 1, 1, 8, 8)" in annotation
    assert "template_t=0:6" in annotation
    assert "projection=median" in annotation
    assert "registration_z_range=Z_N=1" in annotation


def test_summary_annotation_keeps_yx_shift_labels_explicit():
    registered = np.zeros((4, 1, 1, 8, 8), dtype=np.float32)
    details = {
        "registration_channel": 0,
        "registration_stack": 0,
        "method": "phase_cross_correlation",
        "time_registration_mode": "projection",
        "effective_time_registration_mode": "projection",
        "time_reference_mode": "template",
        "projection_method": "max",
        "intra_stack": False,
        "zreg": False,
        "rotreg": False,
        "transform_backend": "skimage",
        "transform_order": 1,
        "max_xy_shifts": (60, 30),
        "time_shifts_zyx_raw": np.asarray(
            [
                [0, 0, 0],
                [0, 51, 12],
                [0, -20, -29],
                [0, 4, 8],
            ],
            dtype=np.float32,
        ),
    }

    raw_label = _raw_estimate_label(details, registered.shape[0])
    annotation = _settings_annotation(details, registered)

    assert "max_raw_shift[y=51, x=29]" in raw_label
    assert "max_y=60, max_x=30" in annotation


def test_projection_range_label_reports_all_slices_for_z_stacks():
    assert _projection_range_label({"registration_z_range": None}, 5) == "all slices (0:5)"
    assert _projection_range_label({"registration_z_range": (1, 4)}, 5) == "1:4"
