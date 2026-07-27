import numpy as np

from zenreg import create_stack_metadata, load_stack, save_stack, update_stack_metadata


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
