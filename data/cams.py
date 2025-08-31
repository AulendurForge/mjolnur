import xarray as xr, pandas as pd, numpy as np, gcsfs


def open_cams_zarr(zarr_uri: str) -> xr.Dataset:
    fs = gcsfs.GCSFileSystem(token="google_default")
    ds = xr.open_zarr(fs.get_mapper(zarr_uri), consolidated=True)
    # many CAMS stores use epoch seconds—normalize to datetime if needed
    if np.issubdtype(ds.time.dtype, np.integer):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s"))
    return ds.sortby("time")


def get_pm_slice(ds: xr.Dataset, when) -> xr.Dataset:
    """Return global PM2.5/PM10 (µg m⁻³) at a datetime (nearest within ≤30min)."""
    return ds[["pm2p5", "pm10"]].sel(
        time=pd.Timestamp(when), method="nearest", tolerance="30min"
    )
