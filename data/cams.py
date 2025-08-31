# mjolnur/data/cams.py
from __future__ import annotations
import xarray as xr, gcsfs, pandas as pd, numpy as np


def open_cams_zarr(zarr_uri: str) -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="google_default")
    ds = xr.open_zarr(fs.get_mapper(zarr_uri), consolidated=True)
    if np.issubdtype(ds.time.dtype, np.integer):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s"))
    return ds.sortby("time")


def get_pm_slice(ds: xr.Dataset, when) -> xr.Dataset:
    return ds[["pm2p5", "pm10"]].sel(
        time=pd.Timestamp(when), method="nearest", tolerance="45min"
    )
