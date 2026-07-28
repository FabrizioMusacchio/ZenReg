import numpy as np

from zenreg import (
    cleanup_omio_cache,
    create_empty_stack,
    create_stack_metadata,
    load_stack,
    register_stack,
    save_stack,
    update_stack_metadata,
)


def test_omio_roundtrip_preserves_tzcyx_shape_and_metadata(tmp_path):
    stack = np.arange(1 * 2 * 1 * 8 * 9, dtype=np.float32).reshape(1, 2, 1, 8, 9)
    metadata = create_stack_metadata(stack, annotations={"test": "roundtrip"}, verbose=False)

    output_path = save_stack(
        tmp_path / "roundtrip.ome.tif",
        stack,
        metadata=metadata,
        overwrite=True,
        verbose=False,
    )
    loaded_stack, loaded_metadata = load_stack(output_path, return_metadata=True, verbose=False)

    assert output_path.exists()
    assert loaded_stack.shape == stack.shape
    assert loaded_metadata["axes"] == "TZCYX"
    assert loaded_metadata["SizeT"] == 1
    assert loaded_metadata["SizeZ"] == 2
    assert loaded_metadata["SizeC"] == 1
    np.testing.assert_allclose(loaded_stack, stack)


def test_update_stack_metadata_syncs_modified_shape():
    stack = np.zeros((2, 3, 1, 8, 9), dtype=np.float32)
    metadata = create_stack_metadata(stack, annotations={"test": "crop"}, verbose=False)
    cropped_stack = stack[:, 1:, :, 2:7, 3:8]

    cropped_metadata = update_stack_metadata(metadata, cropped_stack, verbose=False)

    assert cropped_metadata["axes"] == "TZCYX"
    assert cropped_metadata["shape"] == cropped_stack.shape
    assert cropped_metadata["SizeT"] == 2
    assert cropped_metadata["SizeZ"] == 2
    assert cropped_metadata["SizeC"] == 1
    assert cropped_metadata["SizeY"] == 5
    assert cropped_metadata["SizeX"] == 5


def test_load_stack_supports_omio_disk_memmap(tmp_path):
    stack = np.arange(1 * 2 * 1 * 8 * 9, dtype=np.float32).reshape(1, 2, 1, 8, 9)
    metadata = create_stack_metadata(stack, annotations={"test": "memmap"}, verbose=False)
    output_path = save_stack(
        tmp_path / "memmap_source.ome.tif",
        stack,
        metadata=metadata,
        overwrite=True,
        verbose=False,
    )
    cache_folder = tmp_path / "omio_cache"

    loaded_stack, loaded_metadata = load_stack(
        output_path,
        return_metadata=True,
        use_memmap=True,
        memmap_folder=cache_folder,
        verbose=False,
    )

    assert loaded_stack.shape == stack.shape
    assert loaded_metadata["axes"] == "TZCYX"
    assert "omio_cache_folder" in loaded_metadata
    assert "omio_zarr_store_path" in loaded_metadata
    np.testing.assert_allclose(np.asarray(loaded_stack), stack)
    cleanup_omio_cache(cache_folder, verbose=False)


def test_create_empty_stack_supports_omio_disk_memmap(tmp_path):
    cache_folder = tmp_path / "created_cache"

    stack, metadata = create_empty_stack(
        shape=(2, 3, 1, 8, 9),
        dtype=np.float32,
        use_memmap=True,
        memmap_folder=cache_folder,
        memmap_name="empty_stack",
        return_metadata=True,
        verbose=False,
    )

    assert stack.shape == (2, 3, 1, 8, 9)
    assert metadata["axes"] == "TZCYX"
    assert "omio_cache_folder" in metadata
    assert "omio_zarr_store_path" in metadata
    stack[0, 0, 0, 2:4, 3:5] = 1.0
    np.testing.assert_allclose(np.asarray(stack[0, 0, 0, 2:4, 3:5]), np.ones((2, 2)))
    cleanup_omio_cache(metadata["omio_cache_folder"], verbose=False)


def test_save_stack_writes_registration_report_sidecars(tmp_path):
    stack = np.zeros((2, 1, 1, 16, 16), dtype=np.float32)
    stack[:, 0, 0, 4:10, 5:11] = 1.0
    registered, details = register_stack(
        stack,
        registration_channel=0,
        method="phase_cross_correlation",
        verbose=False,
        return_shifts=True,
        return_details=True,
    )

    output_path = save_stack(
        tmp_path / "registered.ome.tif",
        registered,
        metadata=create_stack_metadata(stack, verbose=False),
        registration_details=details,
        overwrite=True,
        verbose=False,
    )

    csv_path = tmp_path / "registered_registration_shifts.csv"
    yaml_path = tmp_path / "registered_registration_settings.yaml"
    plot_path = tmp_path / "registered_registration_summary.png"
    assert output_path.exists()
    assert csv_path.exists()
    assert yaml_path.exists()
    assert plot_path.exists()
    assert "pearson_correlation" in csv_path.read_text(encoding="utf-8")
    assert "registration_settings:" in yaml_path.read_text(encoding="utf-8")
    assert plot_path.stat().st_size > 0
