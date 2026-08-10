from pathlib import Path

from zenreg import discover_bids_like_batch_images, register_bids_like_batch
from zenreg.synthetic import write_batch_example_project


def test_discover_bids_like_batch_images_supports_nested_token_levels(tmp_path):
    image_path = tmp_path / "ID000001" / "DC000_FOV1" / "TL_000" / "image_01.ome.tif"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("dummy")
    ignored = tmp_path / "ID000001" / "DC000_FOV1" / "TL_000" / "ROIMask.raw"
    ignored.write_text("not an image")

    records = discover_bids_like_batch_images(
        tmp_path,
        subject_ids=None,
        subject_prefix="ID",
        tag_folder_levels=(("DC000_FOV",), ("TL_000",)),
        image_patterns=("*.ome.tif", "*.raw"),
    )

    assert len(records) == 1
    assert records[0].subject_id == "ID000001"
    assert records[0].tag_folders == ("DC000_FOV1", "TL_000")
    assert records[0].experiment_tag == "DC000_FOV1"
    assert records[0].image_path == image_path
    assert records[0].output_scope_dir == tmp_path / "ID000001" / "DC000_FOV1"


def test_register_bids_like_batch_processes_synthetic_project(tmp_path):
    write_batch_example_project(
        tmp_path,
        subject_ids=("ID000001",),
        experiment_tags=("TP000",),
    )

    result = register_bids_like_batch(
        tmp_path,
        subject_ids=("ID000001",),
        tag_folder_levels=(("TP000",),),
        image_patterns=("*.ome.tif",),
        register_kwargs={
            "registration_channel": 0,
            "method": "phase_cross_correlation",
            "time_registration_mode": "projection",
            "projection_method": "max",
            "zreg": False,
            "zero_clip": False,
            "verbose": False,
        },
        save_kwargs={"verbose": False},
        use_memmap=False,
        verbose=False,
    )

    assert len(result.processed) == 1
    assert len(result.skipped) == 0
    assert result.processed[0].output_path.exists()
    assert result.processed[0].output_path.name == "image_01_zenreg_registered.ome.tif"


def test_register_bids_like_batch_reports_load_none(monkeypatch, tmp_path):
    image_path = tmp_path / "ID000001" / "TP000" / "broken.raw"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("dummy")

    import zenreg.batch as batch_module

    monkeypatch.setattr(batch_module, "load_stack", lambda *args, **kwargs: (None, None))

    result = register_bids_like_batch(
        tmp_path,
        subject_ids=("ID000001",),
        tag_folder_levels=(("TP000",),),
        image_patterns=("*.raw",),
        register_kwargs={"registration_channel": 0, "verbose": False},
        use_memmap=False,
        verbose=False,
    )

    assert len(result.processed) == 0
    assert len(result.skipped) == 1
    assert result.skipped[0].stage == "load"
    assert result.root_error_report_path is not None
    report_text = result.root_error_report_path.read_text()
    assert str(image_path) in report_text
    assert "'template_metadata': {" in report_text
    assert "'T': 1" in report_text
