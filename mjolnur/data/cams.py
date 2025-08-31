# mjolnur/data/cams.py
from __future__ import annotations
import xarray as xr, gcsfs, pandas as pd, numpy as np


def open_cams_zarr(zarr_uri: str) -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="google_default")
    ds = xr.open_zarr(fs.get_mapper(zarr_uri), consolidated=True)
    if np.issubdtype(ds.time.dtype, np.integer):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s"))
    return ds.sortby("time")


# mjolnur/data/cams.py - update get_pm_slice
def get_pm_slice(ds: xr.Dataset, when) -> xr.Dataset:
    when_ts = pd.Timestamp(when)

    # Check if exact time exists
    if when_ts in ds.time.values:
        return ds[["pm2p5", "pm10"]].sel(time=when_ts)

    # Otherwise use nearest with tolerance
    return ds[["pm2p5", "pm10"]].sel(time=when_ts, method="nearest", tolerance="45min")


# mjolnur/data/cams.py - update create_mock_cams_dataset
def create_mock_cams_dataset(start: str, stop: str, step_hours: int = 6) -> xr.Dataset:
    """Create mock CAMS PM data for testing"""
    import numpy as np

    # Add padding for t+6h lookups
    start_padded = pd.Timestamp(start) - pd.Timedelta(hours=12)
    stop_padded = pd.Timestamp(stop) + pd.Timedelta(hours=12)

    times = pd.date_range(start=start_padded, end=stop_padded, freq=f"{step_hours}h")
    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(0.5, 360.5, 1)

    np.random.seed(42)
    pm25 = np.random.lognormal(2.5, 1.0, (len(times), len(lat), len(lon))).astype(
        np.float32
    )
    pm10 = np.random.lognormal(3.0, 1.0, (len(times), len(lat), len(lon))).astype(
        np.float32
    )
    pm10 = np.maximum(pm10, pm25 * 1.5)

    ds = xr.Dataset(
        data_vars={
            "pm2p5": (["time", "lat", "lon"], pm25),
            "pm10": (["time", "lat", "lon"], pm10),
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
        },
    )

    # Ensure time is sorted
    ds = ds.sortby("time")

    print(f"Created mock CAMS with {len(times)} timesteps: {times[0]} to {times[-1]}")
    return ds