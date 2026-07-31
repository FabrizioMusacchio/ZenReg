import numpy as np
import pytest

from zenreg._axes import ensure_tzcyx_stack, normalize_zrange, promote_to_tzcyx, restore_promoted_shape
from zenreg.filters import apply_filters, max_z_project, z_project


def test_promote_and_restore_yx_zyx_and_tzcyx_shapes():
    image_yx = np.arange(12, dtype=np.float32).reshape(3, 4)
    promoted_yx, ndim_yx = promote_to_tzcyx(image_yx)
    assert promoted_yx.shape == (1, 1, 1, 3, 4)
    np.testing.assert_array_equal(restore_promoted_shape(promoted_yx, ndim_yx), image_yx)

    volume_zyx = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    promoted_zyx, ndim_zyx = promote_to_tzcyx(volume_zyx)
    assert promoted_zyx.shape == (1, 2, 1, 3, 4)
    np.testing.assert_array_equal(restore_promoted_shape(promoted_zyx, ndim_zyx), volume_zyx)

    stack_tzcyx = np.zeros((2, 3, 1, 4, 5), dtype=np.float32)
    promoted_tzcyx, ndim_tzcyx = promote_to_tzcyx(stack_tzcyx)
    assert promoted_tzcyx is stack_tzcyx
    assert restore_promoted_shape(promoted_tzcyx, ndim_tzcyx) is stack_tzcyx


def test_axis_helpers_reject_invalid_shapes_and_ranges():
    with pytest.raises(ValueError, match="5 dimensions"):
        ensure_tzcyx_stack(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="YX, ZYX, or TZCYX"):
        promote_to_tzcyx(np.zeros((1, 2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="exactly two"):
        normalize_zrange((1, 2, 3), 5)
    with pytest.raises(ValueError, match="0 <= start < stop"):
        normalize_zrange((-1, 6), 5, strict=True)


def test_normalize_zrange_clamps_swaps_and_expands_empty_ranges():
    assert normalize_zrange(None, 5) == (0, 5)
    assert normalize_zrange((-10, 3), 5) == (0, 3)
    assert normalize_zrange((4, 2), 5) == (2, 4)
    assert normalize_zrange((99, 99), 5) == (4, 5)
    assert normalize_zrange((2, 2), 5) == (2, 3)


def test_apply_filters_supports_sequences_time_parameters_and_3d_mode():
    stack = np.zeros((2, 3, 1, 9, 9), dtype=np.float32)
    stack[:, :, :, 4, 4] = 1.0
    stack[1, :, :, 2, 2] = 0.5

    filtered = apply_filters(
        stack,
        filters=("median", "gaussian"),
        median_size=(1, 3),
        gaussian_sigma=(0.5, 0.8),
        apply_3d=False,
    )
    filtered_3d = apply_filters(stack, filters="gaussian", gaussian_sigma=0.8, apply_3d=True)

    assert filtered.shape == stack.shape
    assert filtered.dtype == np.float32
    assert filtered_3d.shape == stack.shape
    assert np.max(filtered_3d) < 1.0


def test_apply_filters_and_projection_validate_inputs():
    stack = np.zeros((1, 2, 1, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one"):
        apply_filters(stack, filters=[])
    with pytest.raises(ValueError, match="Unsupported filter"):
        apply_filters(stack, filters="bilateral")
    with pytest.raises(ValueError, match="median_size"):
        apply_filters(stack, filters="median", median_size=0)
    with pytest.raises(ValueError, match="gaussian_sigma"):
        apply_filters(stack, filters="gaussian", gaussian_sigma=0)
    with pytest.raises(ValueError, match="Unsupported projection_method"):
        z_project(stack, projection_method="sum")


def test_z_project_supports_simple_inputs_and_max_wrapper():
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    projected = z_project(volume, zrange=(1, 3), projection_method="mean")
    np.testing.assert_allclose(projected, np.mean(volume[1:3], axis=0, keepdims=True))

    image = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    np.testing.assert_allclose(z_project(image, projection_method="std"), np.zeros_like(image))
    np.testing.assert_allclose(max_z_project(volume), np.max(volume, axis=0, keepdims=True))
