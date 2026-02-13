"""Lightweight data quality metrics for satellite imagery.

Computes per-channel and sequence-level metrics during ingestion. All returned
values are JSON-serializable Python primitives (float, int, str, dict, list).
"""

import logging
from typing import Any

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

log = logging.getLogger("sat_consumer")


def _build_earth_mask(
    area_def: AreaDefinition,
    y_size: int,
    x_size: int,
    chunksize: int = 500,
) -> np.ndarray:
    """Build a boolean mask where True indicates an on-earth-disk pixel.

    Uses the same lon-lat approach as ``process._get_earthdisk_nan_frac``,
    but returns the mask for reuse across channels.

    Args:
        area_def: The area definition from satpy.
        y_size: Number of pixels in the y dimension.
        x_size: Number of pixels in the x dimension.
        chunksize: Chunk size for the lon-lat computation.

    Returns:
        2D boolean ndarray (y, x) where True = on earth-disk.
    """
    chunks = [
        [min(chunksize, size - i * chunksize) for i in range(int(np.ceil(size / chunksize)))]
        for size in [y_size, x_size]
    ]
    lons, _ = area_def.get_lonlats(chunks=chunks)  # type: ignore[no-untyped-call]
    return np.isfinite(lons).compute()  # type: ignore[no-any-return]


def compute_channel_metrics(
    da: np.ndarray,
    earth_mask: np.ndarray,
) -> dict[str, float]:
    """Compute per-channel quality metrics on earth-disk pixels only.

    Args:
        da: 2D array (y, x) of pixel values for one channel.
        earth_mask: 2D boolean mask (y, x). True = on-earth-disk.

    Returns:
        Dict of metric_name to float value.
    """
    earth_pixels = da[earth_mask]
    n_earth = earth_pixels.size

    if n_earth == 0:
        return {
            "nan_ratio": 1.0,
            "valid_pixel_ratio": 0.0,
            "mean_intensity": float("nan"),
            "std_intensity": float("nan"),
            "min_intensity": float("nan"),
            "max_intensity": float("nan"),
        }

    is_nan = np.isnan(earth_pixels)
    nan_count = int(np.sum(is_nan))
    valid_mask = ~is_nan & np.isfinite(earth_pixels)
    valid_pixels = earth_pixels[valid_mask]
    n_valid = valid_pixels.size

    nan_ratio = float(nan_count / n_earth)
    valid_pixel_ratio = float(n_valid / n_earth)

    if n_valid == 0:
        return {
            "nan_ratio": nan_ratio,
            "valid_pixel_ratio": 0.0,
            "mean_intensity": float("nan"),
            "std_intensity": float("nan"),
            "min_intensity": float("nan"),
            "max_intensity": float("nan"),
        }

    mean_val = float(np.mean(valid_pixels))
    std_val = float(np.std(valid_pixels))
    min_val = float(np.min(valid_pixels))
    max_val = float(np.max(valid_pixels))

    return {
        "nan_ratio": nan_ratio,
        "valid_pixel_ratio": valid_pixel_ratio,
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "min_intensity": min_val,
        "max_intensity": max_val,
    }


def compute_sequence_metrics(ds: xr.Dataset) -> dict[str, Any]:
    """Compute sequence-level (per-timestep) quality metrics.

    Args:
        ds: The xarray Dataset for the current timestep.

    Returns:
        Dict of sequence-level metrics.
    """
    metrics: dict[str, Any] = {}

    if "channel" in ds.dims:
        metrics["channel_count"] = int(ds.sizes["channel"])

    if "y_geostationary" in ds.dims and "x_geostationary" in ds.dims:
        metrics["y_pixels"] = int(ds.sizes["y_geostationary"])
        metrics["x_pixels"] = int(ds.sizes["x_geostationary"])
    elif "y" in ds.dims and "x" in ds.dims:
        metrics["y_pixels"] = int(ds.sizes["y"])
        metrics["x_pixels"] = int(ds.sizes["x"])

    if "time" in ds.coords:
        time_val = ds.coords["time"].values
        if hasattr(time_val, "__len__") and len(time_val) > 0:
            metrics["timestamp_iso"] = str(np.datetime_as_string(time_val[0], unit="s"))
        else:
            metrics["timestamp_iso"] = str(np.datetime_as_string(time_val, unit="s"))

    return metrics


def compute_data_quality_metrics(
    ds: xr.Dataset,
    area_def: AreaDefinition,
    enable: bool,
) -> dict[str, Any]:
    """Compute all data quality metrics for a single timestep.

    When ``enable`` is False, returns an empty dict and does zero computation.

    Args:
        ds: The xarray Dataset for one timestep. Can be pre-stack (each channel
            as a separate data_var with dims y/x) or post-stack (single ``data``
            var with a ``channel`` dim).
        area_def: The satpy AreaDefinition for computing the earth-disk mask.
        enable: Whether to compute metrics.

    Returns:
        Nested dict with ``per_channel`` and ``sequence_level`` keys, or empty dict.
    """
    if not enable:
        return {}

    log.debug("Computing data quality metrics")

    # Determine spatial dimension names
    if "y" in ds.dims:
        y_dim, x_dim = "y", "x"
    else:
        y_dim, x_dim = "y_geostationary", "x_geostationary"

    earth_mask = _build_earth_mask(
        area_def=area_def,
        y_size=ds.sizes[y_dim],
        x_size=ds.sizes[x_dim],
    )

    per_channel: dict[str, dict[str, float]] = {}

    if "channel" in ds.dims and "data" in ds.data_vars:
        # Post-stacking: channels are along the 'channel' dim of the 'data' var
        data_arr = ds.data_vars["data"]
        for i, ch_name in enumerate(ds.coords["channel"].values):
            ch_data = (
                data_arr.isel(time=0, channel=i).values
                if "time" in data_arr.dims
                else (data_arr.isel(channel=i).values)
            )
            per_channel[str(ch_name)] = compute_channel_metrics(ch_data, earth_mask)
    else:
        # Pre-stacking: each data_var is a channel
        for var_name in ds.data_vars:
            da = ds.data_vars[var_name]
            if set(da.dims) >= {y_dim, x_dim}:
                ch_data = da.isel(time=0).values if "time" in da.dims else da.values
                per_channel[str(var_name)] = compute_channel_metrics(ch_data, earth_mask)

    sequence_level = compute_sequence_metrics(ds)

    if per_channel:
        nan_ratios = [m["nan_ratio"] for m in per_channel.values()]
        sequence_level["mean_nan_ratio_all_channels"] = float(np.mean(nan_ratios))

    log.debug("Data quality metrics computed: %d channels", len(per_channel))

    return {
        "per_channel": per_channel,
        "sequence_level": sequence_level,
    }
