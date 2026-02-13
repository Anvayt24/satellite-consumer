"""Unit tests for the quality_metrics module."""

import unittest
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import xarray as xr

from satellite_consumer.quality_metrics import (
    compute_channel_metrics,
    compute_data_quality_metrics,
    compute_sequence_metrics,
)


def _make_earth_mask(shape: tuple[int, int], fraction: float = 1.0) -> np.ndarray:
    """Create a boolean earth mask with a given fraction of on-earth pixels."""
    mask = np.zeros(shape, dtype=bool)
    n_earth = int(shape[0] * shape[1] * fraction)
    flat_mask = mask.ravel()
    flat_mask[:n_earth] = True
    return mask.reshape(shape)


def _make_mock_area_def(y_size: int, x_size: int) -> MagicMock:
    """Create a mock AreaDefinition that returns all-finite lons (all on-earth)."""
    mock = MagicMock()
    lons = da.from_array(np.ones((y_size, x_size)), chunks=500)  # type: ignore[no-untyped-call]
    mock.get_lonlats.return_value = (lons, lons)
    return mock


class TestComputeChannelMetrics(unittest.TestCase):
    """Test per-channel metric computation."""

    def setUp(self) -> None:
        # 10x10 grid where center 6x6 is on-earth (36 pixels)
        self.earth_mask = np.zeros((10, 10), dtype=bool)
        self.earth_mask[2:8, 2:8] = True

    def test_valid_data(self) -> None:
        data = np.ones((10, 10), dtype=np.float32) * 5.0
        result = compute_channel_metrics(data, self.earth_mask)
        self.assertAlmostEqual(result["nan_ratio"], 0.0)
        self.assertAlmostEqual(result["valid_pixel_ratio"], 1.0)
        self.assertAlmostEqual(result["mean_intensity"], 5.0)
        self.assertAlmostEqual(result["std_intensity"], 0.0)
        self.assertAlmostEqual(result["min_intensity"], 5.0)
        self.assertAlmostEqual(result["max_intensity"], 5.0)
        # All values must be plain Python floats for JSON serialization
        for v in result.values():
            self.assertIsInstance(v, float)

    def test_all_nan(self) -> None:
        data = np.full((10, 10), np.nan, dtype=np.float32)
        result = compute_channel_metrics(data, self.earth_mask)
        self.assertAlmostEqual(result["nan_ratio"], 1.0)
        self.assertAlmostEqual(result["valid_pixel_ratio"], 0.0)

    def test_partial_nan(self) -> None:
        data = np.ones((10, 10), dtype=np.float32) * 3.0
        # Set half of the 36 on-earth pixels to NaN
        data[2:5, 2:8] = np.nan  # 18 NaN pixels
        result = compute_channel_metrics(data, self.earth_mask)
        self.assertAlmostEqual(result["nan_ratio"], 0.5)
        self.assertAlmostEqual(result["valid_pixel_ratio"], 0.5)

    def test_empty_earth_mask(self) -> None:
        data = np.ones((10, 10), dtype=np.float32)
        empty_mask = np.zeros((10, 10), dtype=bool)
        result = compute_channel_metrics(data, empty_mask)
        self.assertAlmostEqual(result["nan_ratio"], 1.0)
        self.assertAlmostEqual(result["valid_pixel_ratio"], 0.0)


class TestComputeSequenceMetrics(unittest.TestCase):
    """Test sequence-level metric computation."""

    def test_basic(self) -> None:
        ds = xr.Dataset(
            coords={
                "time": [np.datetime64("2024-01-01T00:00", "ns")],
                "y_geostationary": np.arange(10, dtype=float),
                "x_geostationary": np.arange(10, dtype=float),
                "channel": ["VIS006", "IR_108"],
            },
            data_vars={
                "data": (
                    ["time", "y_geostationary", "x_geostationary", "channel"],
                    np.random.rand(1, 10, 10, 2).astype(np.float32),
                ),
            },
        )
        result = compute_sequence_metrics(ds)
        self.assertEqual(result["channel_count"], 2)
        self.assertEqual(result["y_pixels"], 10)
        self.assertEqual(result["x_pixels"], 10)
        self.assertIn("timestamp_iso", result)


class TestComputeDataQualityMetrics(unittest.TestCase):
    """Test the main orchestrator function."""

    def test_disabled_returns_empty(self) -> None:
        result = compute_data_quality_metrics(
            ds=MagicMock(),
            area_def=MagicMock(),
            enable=False,
        )
        self.assertEqual(result, {})

    def test_enabled_pre_stack_layout(self) -> None:
        """Test metrics with pre-stack layout (each channel as separate data_var)."""
        ds = xr.Dataset(
            coords={
                "time": [np.datetime64("2024-01-01T00:00", "ns")],
                "y": np.arange(10, dtype=float),
                "x": np.arange(10, dtype=float),
            },
            data_vars={
                "VIS006": (["time", "y", "x"], np.random.rand(1, 10, 10).astype(np.float32)),
                "IR_108": (["time", "y", "x"], np.random.rand(1, 10, 10).astype(np.float32)),
            },
        )
        area_def = _make_mock_area_def(10, 10)
        result = compute_data_quality_metrics(ds=ds, area_def=area_def, enable=True)
        self.assertIn("per_channel", result)
        self.assertIn("sequence_level", result)
        self.assertIn("VIS006", result["per_channel"])
        self.assertIn("IR_108", result["per_channel"])
        self.assertIn("mean_nan_ratio_all_channels", result["sequence_level"])

    def test_enabled_post_stack_layout(self) -> None:
        """Test metrics with post-stack layout (single 'data' var with channel dim)."""
        ds = xr.Dataset(
            coords={
                "time": [np.datetime64("2024-01-01T00:00", "ns")],
                "y_geostationary": np.arange(10, dtype=float),
                "x_geostationary": np.arange(10, dtype=float),
                "channel": ["VIS006", "IR_108"],
            },
            data_vars={
                "data": (
                    ["time", "y_geostationary", "x_geostationary", "channel"],
                    np.random.rand(1, 10, 10, 2).astype(np.float32),
                ),
            },
        )
        area_def = _make_mock_area_def(10, 10)
        result = compute_data_quality_metrics(ds=ds, area_def=area_def, enable=True)
        self.assertIn("per_channel", result)
        self.assertEqual(len(result["per_channel"]), 2)

    def test_metrics_are_json_serializable(self) -> None:
        """Verify all metrics can be serialized to JSON (no numpy scalars)."""
        import json

        ds = xr.Dataset(
            coords={
                "time": [np.datetime64("2024-01-01T00:00", "ns")],
                "y": np.arange(10, dtype=float),
                "x": np.arange(10, dtype=float),
            },
            data_vars={
                "VIS006": (["time", "y", "x"], np.random.rand(1, 10, 10).astype(np.float32)),
            },
        )
        area_def = _make_mock_area_def(10, 10)
        result = compute_data_quality_metrics(ds=ds, area_def=area_def, enable=True)
        # This will raise if any numpy scalars are present
        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)
