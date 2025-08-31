# mjolnur/data/dataloader.py
from __future__ import annotations
import numpy as np, pandas as pd, xarray as xr
from .cams import get_pm_slice
from mjolnur.utils.graphcast_adapter import load_gc_decoded_from_cache


def times_6h(start: str, stop: str, step_hours=6):
    t = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    while t <= stop:
        yield t
        t += pd.Timedelta(hours=step_hours)


def build_example_at_time(cams_ds: xr.Dataset, cfg: dict, t0: pd.Timestamp):
    t6 = t0 + pd.Timedelta(hours=6)

    # CAMS inputs/labels
    pm_t = get_pm_slice(cams_ds, t0)
    pm_t6 = get_pm_slice(cams_ds, t6)

    # GraphCast decoded at t+6
    if cfg["use_decoded_cache"]:
        gc = load_gc_decoded_from_cache(
            cfg["gc_decoded_zarr"], str(t6), cfg["gc_channels"]
        )
    else:
        from mjolnur.utils.graphcast_adapter import run_graphcast_tplus6_on_the_fly

        gc = run_graphcast_tplus6_on_the_fly(str(t0), cfg["gc_channels"])

    # to numpy BHWC
    def hwc(*das):
        return np.stack([np.asarray(d.data) for d in das], axis=-1)[None, ...]

    x_pm = hwc(pm_t.pm2p5, pm_t.pm10)  # (1,H,W,2)
    x_gc = np.stack(
        [np.asarray(gc[v].isel(time=0).data) for v in cfg["gc_channels"]], axis=-1
    )[None, ...]
    y_np = hwc(pm_t6.pm2p5, pm_t6.pm10)

    return x_pm, x_gc, y_np, dict(t0=str(t0), t6=str(t6))
