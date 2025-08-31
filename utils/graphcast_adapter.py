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
def run_graphcast_tplus6_on_the_fly(t0: str, channels: list[str]) -> xr.Dataset:
    """
    Placeholder: implement with your ERA5 prep + graphcast.run(...) that yields a
    Dataset with decoded vars and time dim == [t0+6h]. Return exactly the requested channels.
    """
    raise NotImplementedError(
        "Hook up to your GraphCast runner (ERA5→GC→decoded @ t+6)."
    )
