# mjolnur/utils/graphcast_adapter.py
from __future__ import annotations
import xarray as xr
import pandas as pd
import gcsfs


def open_zarr(zarr_uri: str) -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="google_default")
    return xr.open_zarr(fs.get_mapper(zarr_uri), consolidated=True)


def load_gc_decoded_from_cache(
    zarr_uri: str, tplus6: str, channels: list[str]
) -> xr.Dataset:
    """Read decoded GC fields at exact t+6 from a consolidated Zarr."""
    ds = open_zarr(zarr_uri)
    ds = ds.sel(time=pd.Timestamp(tplus6))
    keep = [c for c in channels if c in ds]
    return ds[keep].transpose("time", "lat", "lon")


# --- OPTIONAL: run GraphCast on the fly (you can wire this to your existing code) ---
# mjolnur/utils/graphcast_adapter.py
def run_graphcast_tplus6_on_the_fly(t0: str, channels: list[str]) -> xr.Dataset:
    """Mock implementation for testing the training pipeline"""
    import numpy as np
    import pandas as pd
    import xarray as xr

    # Create mock data matching your CAMS grid (180x360 for 1-degree)
    time = pd.Timestamp(t0) + pd.Timedelta(hours=6)
    lat = np.arange(-89.5, 90, 1)  # 180 points
    lon = np.arange(0.5, 360.5, 1)  # 360 points

    # Create mock dataset with requested channels
    data_vars = {}
    np.random.seed(42)  # For reproducibility during testing

    for ch in channels:
        if ch == "10m_u_component_of_wind":
            # Mock wind U component (-10 to 10 m/s)
            data = np.random.uniform(-10, 10, (1, len(lat), len(lon))).astype(
                np.float32
            )
        elif ch == "10m_v_component_of_wind":
            # Mock wind V component (-10 to 10 m/s)
            data = np.random.uniform(-10, 10, (1, len(lat), len(lon))).astype(
                np.float32
            )
        elif ch == "2m_temperature":
            # Mock temperature (250-320 K)
            data = np.random.uniform(250, 320, (1, len(lat), len(lon))).astype(
                np.float32
            )
        elif ch == "mean_sea_level_pressure":
            # Mock pressure (98000-105000 Pa)
            data = np.random.uniform(98000, 105000, (1, len(lat), len(lon))).astype(
                np.float32
            )
        elif ch == "total_precipitation_6hr":
            # Mock precipitation (0-10 mm)
            data = np.random.exponential(0.5, (1, len(lat), len(lon))).astype(
                np.float32
            )
        else:
            # Generic mock data for any other channel
            data = np.random.randn(1, len(lat), len(lon)).astype(np.float32)

        data_vars[ch] = (["time", "lat", "lon"], data)

    ds = xr.Dataset(
        data_vars=data_vars, coords={"time": [time], "lat": lat, "lon": lon}
    )
    return ds