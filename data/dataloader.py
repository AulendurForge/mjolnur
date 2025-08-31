import numpy as np, xarray as xr
from .cams import get_pm_slice

GC_U = "10m_u_component_of_wind"
GC_V = "10m_v_component_of_wind"
# optional decoded fields to include as channels at t+6:
EXTRA_GC = [
    GC_U,
    GC_V,
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]


def make_example(trainer, cams_ds, t0: str):
    """
    t0 is a 6-hour UTC stamp (e.g., '2020-03-01T06:00:00').
    Uses your GraphCastAeroTrainer to roll once (t0->t0+6h) and build a (X, y) example.
    """
    # Run GC with 1 lead step so predictions.time[0] == t0+6h
    era = trainer.prepare_era5_for_graphcast(t0, num_steps=1)
    preds = trainer.run_graphcast_prediction(era)  # xr.Dataset with time dim len=1
    t6 = preds.time.values[0]  # exact stamp for labels & winds

    pm_t = get_pm_slice(cams_ds, t0)  # (lat, lon)
    pm_t6 = get_pm_slice(cams_ds, t6)  # label

    # Stack inputs: PM(t) and selected decoded fields at t+6
    fields = []
    for v in EXTRA_GC:
        if v in preds:
            fields.append(preds[v].isel(time=0).transpose("lat", "lon"))
    X_fields = xr.concat(fields, dim="channel")  # (channel, lat, lon)
    X_pm = xr.concat([pm_t.pm2p5, pm_t.pm10], dim="channel")  # (2, lat, lon)
    X = xr.concat([X_pm, X_fields], dim="channel")  # (C, H, W)

    y = xr.concat([pm_t6.pm2p5, pm_t6.pm10], dim="channel")  # (2, H, W)
    return X, y, preds
