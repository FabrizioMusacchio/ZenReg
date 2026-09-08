from pathlib import Path

import numpy as np

from zenreg import (
    batch_create_thorlabs_raw_yaml_templates,
    discover_bids_like_batch_images,
    register_bids_like_batch,
)
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


def test_discover_bids_like_batch_images_supports_folder_stack_groups(tmp_path):
    fov_dir = tmp_path / "ID25068" / "FOV1_pre"
    for ov_name in ("OV_10", "OV_2", "OV_1"):
        ov_dir = fov_dir / ov_name
        ov_dir.mkdir(parents=True)
        (ov_dir / "slice_0001.tif").write_text("dummy")
    ignored_dir = fov_dir / "Other_1"
    ignored_dir.mkdir()
    (ignored_dir / "slice_0001.tif").write_text("dummy")

    records = discover_bids_like_batch_images(
        tmp_path,
        subject_prefix="ID",
        tag_folder_levels=(("FOV",),),
        image_patterns=("*.tif",),
        stack_folder_tag="OV",
        stack_folder_merge_axis="T")

    assert len(records) == 1
    assert records[0].subject_id == "ID25068"
    assert records[0].tag_folders == ("FOV1_pre",)
    assert records[0].input_kind == "folder_stack"
    assert records[0].stack_folder_tag == "OV"
    assert records[0].stack_folder_merge_axis == "T"
    assert records[0].image_path == fov_dir / "OV_1"
    assert records[0].stack_folder_paths == (
        fov_dir / "OV_1",
        fov_dir / "OV_2",
        fov_dir / "OV_10")
    assert records[0].output_scope_dir == fov_dir
    assert records[0].output_name_stem == "FOV1_pre_OV_merged_T"


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
    assert result.root_run_report_yaml_path is not None
    assert result.root_run_report_txt_path is not None
    assert result.root_run_report_yaml_path.exists()
    assert result.root_run_report_txt_path.exists()
    assert "image_01.ome.tif" in result.root_run_report_txt_path.read_text()
    assert "REGISTERED" in result.root_run_report_txt_path.read_text()


def test_register_bids_like_batch_loads_folder_stack_record(monkeypatch, tmp_path):
    fov_dir = tmp_path / "ID25068" / "FOV1_pre"
    for ov_name in ("OV_1", "OV_2"):
        ov_dir = fov_dir / ov_name
        ov_dir.mkdir(parents=True)
        (ov_dir / "slice_0001.tif").write_text("dummy")

    seen = {}

    import zenreg.batch as batch_module

    def fake_load_stack(path, **kwargs):
        seen.setdefault("paths", []).append(Path(path))
        seen.setdefault("kwargs", []).append(dict(kwargs))
        stack = np.full((1, 1, 1, 8, 8), len(seen["paths"]), dtype=np.float32)
        metadata = {"axes": "TZCYX", "shape": stack.shape}
        return stack, metadata

    def fake_register_stack(stack, **kwargs):
        seen["register_kwargs"] = dict(kwargs)
        seen["registered_shape"] = tuple(stack.shape)
        return stack, {"stack_shape_tzcyx": tuple(stack.shape)}

    def fake_save_stack(path, stack, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("registered")
        return path

    monkeypatch.setattr(batch_module, "load_stack", fake_load_stack)
    monkeypatch.setattr(batch_module, "register_stack", fake_register_stack)
    monkeypatch.setattr(batch_module, "save_stack", fake_save_stack)

    result = register_bids_like_batch(
        tmp_path,
        subject_prefix="ID",
        tag_folder_levels=(("FOV",),),
        image_patterns=("*.tif",),
        stack_folder_tag="OV",
        stack_folder_merge_axis="T",
        register_kwargs={"registration_channel": 0, "verbose": False},
        save_kwargs={"verbose": False},
        use_memmap=True,
        memmap_folder_name="omio_memmap_cache",
        verbose=False)

    assert len(result.processed) == 1
    assert seen["paths"] == [
        fov_dir / "OV_1" / "slice_0001.tif",
        fov_dir / "OV_2" / "slice_0001.tif"]
    assert all("folder_stacks" not in kwargs for kwargs in seen["kwargs"])
    assert all("merge_folder_stacks" not in kwargs for kwargs in seen["kwargs"])
    assert all("merge_along_axis" not in kwargs for kwargs in seen["kwargs"])
    memmap_folders = [Path(kwargs["memmap_folder"]) for kwargs in seen["kwargs"]]
    assert len(set(memmap_folders)) == 2
    assert all("stack_folder_sources" in folder.parts for folder in memmap_folders)
    assert seen["registered_shape"] == (2, 1, 1, 8, 8)
    assert result.processed[0].output_path == (
        fov_dir / "zenreg_output" / "FOV1_pre_OV_merged_T_zenreg_registered.ome.tif")
    report_text = result.root_run_report_txt_path.read_text()
    assert "OV_* folder stack" in report_text
    assert "merge_axis: T" in report_text


def test_register_bids_like_batch_appends_run_report_history(tmp_path):
    write_batch_example_project(
        tmp_path,
        subject_ids=("ID000001",),
        experiment_tags=("TP000",),
    )
    kwargs = dict(
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
        save_kwargs={"verbose": False, "overwrite": True},
        skip_registered=False,
        use_memmap=False,
        verbose=False,
    )

    first = register_bids_like_batch(tmp_path, **kwargs)
    second = register_bids_like_batch(tmp_path, **kwargs)

    import zenreg.batch as batch_module

    payload = batch_module._load_run_report(second.root_run_report_yaml_path, tmp_path)
    image_key = "ID000001/TP000/image_01.ome.tif"
    assert first.root_run_report_yaml_path == second.root_run_report_yaml_path
    assert image_key in payload["files"]
    assert len(payload["files"][image_key]["runs"]) == 2
    assert all(run["status"] == "processed" for run in payload["files"][image_key]["runs"])

    skipped = register_bids_like_batch(
        tmp_path,
        **{
            **kwargs,
            "skip_registered": True,
            "save_kwargs": {"verbose": False, "overwrite": False},
        },
    )
    report_text = skipped.root_run_report_txt_path.read_text()
    already_registered_lines = [
        line for line in report_text.splitlines() if "| skipped/already registered" in line
    ]
    assert already_registered_lines
    assert "[REGISTERED]" in report_text
    assert "output=ID000001/TP000/zenreg_output/image_01_zenreg_registered.ome.tif" not in already_registered_lines[-1]
    assert "Registered output already exists" not in already_registered_lines[-1]


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


def test_batch_create_thorlabs_raw_yaml_templates(monkeypatch, tmp_path):
    raw_path = tmp_path / "ID000001" / "DC000_FOV1" / "TL_000" / "broken.raw"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("raw")
    report_path = tmp_path / "zenreg_batch_error_report_2026-08-10_12-00-00.txt"
    report_path.write_text(
        "\n".join(
            [
                "ZENREG_BATCH_SKIPPED_RAW_FILES = {",
                f"    {str(raw_path)!r}: {{",
                "        'reason': 'metadata missing',",
                "        'stage': 'load',",
                "        'subject_id': 'ID000001',",
                "        'tag_folders': ('DC000_FOV1', 'TL_000'),",
                "        'template_metadata': {",
                "            'T': 11,",
                "            'Z': 3,",
                "            'C': 2,",
                "            'Y': 64,",
                "            'X': 65,",
                "            'bits': 16,",
                "            'pixelunit': 'micron',",
                "            'physicalsize_xyz': (0.5, 0.5, 1.0),",
                "            'time_increment': 1.0,",
                "            'time_increment_unit': 'seconds',",
                "        },",
                "    },",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    calls = []

    class DummyOmio:
        @staticmethod
        def create_thorlabs_raw_yaml(fname_raw, **template_metadata):
            calls.append((Path(fname_raw), template_metadata))
            Path(fname_raw).with_suffix(".yaml").write_text("template")

    import zenreg.batch as batch_module

    monkeypatch.setattr(batch_module, "_import_omio", lambda: DummyOmio)

    result = batch_create_thorlabs_raw_yaml_templates(
        tmp_path,
        report_name=report_path.name,
        verbose=False,
    )

    assert len(result.created) == 1
    assert len(result.skipped) == 0
    assert calls[0][0] == raw_path
    assert calls[0][1]["T"] == 11
    assert calls[0][1]["Y"] == 64
    assert raw_path.with_suffix(".yaml").exists()
