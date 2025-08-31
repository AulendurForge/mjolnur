#!/usr/bin/env python3
"""
cams_to_zarr.py
Builds/appends a conservative-regridded 1°×1° CAMS PM Zarr on GCS from monthly NetCDFs.

Source:  gs://{BUCKET}/cams/raw/cams_eac4_pm_YYYYMM.nc
Target:  gs://{BUCKET}/cams/zarr/pm_1deg_conservative.zarr

Requires: xarray, zarr, dask, gcsfs, xesmf, esmpy, netCDF4 or h5netcdf
Auth:     Application Default Credentials (gcloud auth application-default login)
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import xarray as xr
import dask
import gcsfs
import logging
from contextlib import nullcontext
try:
    from dask.diagnostics import ProgressBar
except Exception:
    ProgressBar = None

# -------- Defaults --------
DEFAULT_BUCKET = os.getenv("MJOLNUR_BUCKET", "mjolnur-cams-data")
DEFAULT_RAW_PREFIX = "cams/raw"
DEFAULT_ZARR_PATH = f"gs://{DEFAULT_BUCKET}/cams/zarr/pm_1deg_conservative.zarr"
DEFAULT_LON_CONV = "-180-180"   # or "0-360"
DEFAULT_DLAT = 1.0
DEFAULT_DLON = 1.0
CHUNKS = {"time": 4, "lat": 90, "lon": 180}
LOCK_OBJ_NAME = "cams/zarr/pm_1deg_conservative.zarr.lock"

def configure_logging(log_file=None, verbose=False):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=(logging.DEBUG if verbose else logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.getLogger("gcsfs").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return logging.getLogger("cams_to_zarr")

def parse_args():
    p = argparse.ArgumentParser(description="CAMS monthly NetCDF → 1° Zarr (conservative regrid)")
    p.add_argument("--start", required=True, help="Start month YYYY-MM")
    p.add_argument("--end",   required=True, help="End month YYYY-MM (inclusive)")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name")
    p.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX, help="Prefix to monthly NetCDFs")
    p.add_argument("--zarr-path", default=DEFAULT_ZARR_PATH, help="Output Zarr path (gs://...)")
    p.add_argument("--lon-convention", default=DEFAULT_LON_CONV, choices=["-180-180","0-360"])
    p.add_argument("--dlat", type=float, default=DEFAULT_DLAT, help="Target grid lat spacing")
    p.add_argument("--dlon", type=float, default=DEFAULT_DLON, help="Target grid lon spacing")
    p.add_argument("--weights-path", default=os.path.expanduser("~/.cache/mjolnur/cams_to_1deg_weights.nc"),
                   help="Local file for xESMF weights")
    p.add_argument("--min-age-minutes", type=int, default=2, help="Only process files older than this")
    p.add_argument("--dry-run", action="store_true", help="List months found & exit")
    p.add_argument("--force-recompute-weights", action="store_true", help="Ignore cached weights and recompute")
    p.add_argument("--engine", default="auto", choices=["auto","h5netcdf","netcdf4","scipy"], help="NetCDF engine")
    p.add_argument("--log-file", default=None, help="Also write logs here")
    p.add_argument("--progress", action="store_true", help="Show Dask progress bars during writes")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    return p.parse_args()

def daterange_months(start_ym: str, end_ym: str):
    fy, fm = map(int, start_ym.split("-"))
    ty, tm = map(int, end_ym.split("-"))
    start = pd.Period(f"{fy:04d}-{fm:02d}", "M")
    end   = pd.Period(f"{ty:04d}-{tm:02d}", "M")
    for per in pd.period_range(start, end, freq="M"):
        yield per.year, per.month

def cams_month_path(bucket, raw_prefix, y, m):
    return f"gs://{bucket}/{raw_prefix}/cams_eac4_pm_{y:04d}{m:02d}.nc"

def open_nc_any(path, engine_pref="auto", chunks_time=8, log=None):
    tried = []
    engines = ["h5netcdf","netcdf4","scipy"] if engine_pref=="auto" else [engine_pref]
    for eng in engines:
        try:
            ds = xr.open_dataset(path, engine=eng, chunks={"time": chunks_time}, decode_cf=True)
            if log: log.info(f"[open] {path}  (engine='{eng}')")
            return ds
        except Exception as e:
            tried.append((eng, str(e)))
    msg = "\n".join([f"  - {eng}: {e}" for eng,e in tried])
    raise RuntimeError("Failed to open dataset with available engines:\n" + msg)

def normalize_geo(ds: xr.Dataset, lon_convention="-180-180"):
    ren = {}
    if "lon" not in ds.coords:
        for cand in ("longitude","LONGITUDE","x"):
            if cand in ds.coords: ren[cand] = "lon"; break
    if "lat" not in ds.coords:
        for cand in ("latitude","LATITUDE","y"):
            if cand in ds.coords: ren[cand] = "lat"; break
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ren["valid_time"] = "time"
    if ren:
        ds = ds.rename(ren)
    if "lat" in ds.coords and ds.lat.size>1 and float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        lon = ds["lon"]
        if lon_convention == "-180-180":
            lon2 = ((lon + 180) % 360) - 180
        else:
            lon2 = lon % 360
        ds = ds.assign_coords(lon=lon2).sortby("lon")
    if "time" not in ds.coords:
        for name in list(ds.coords) + list(ds.data_vars):
            if name in ds and np.issubdtype(ds[name].dtype, np.datetime64):
                ds = ds.set_coords(name).rename({name:"time"})
                break
    return ds

def standardize_pm_vars(ds: xr.Dataset, force_to_ug=False):
    def _to_micrograms(da):
        units = (da.attrs.get("units","") or "").lower()
        if force_to_ug or ("kg" in units and "m-3" in units) or ("kg m**-3" in units) or ("kg m-3" in units):
            da = da * 1e9
            da.attrs["units"] = "µg m^-3"
        return da
    for v in ("pm2p5","pm10"):
        if v in ds.data_vars:
            ds[v] = _to_micrograms(ds[v])
            ds[v].attrs.setdefault("long_name", f"{v} (surface)")
            ds[v].attrs.setdefault("standard_name", v)
            ds[v].encoding.clear()
    return ds

def make_target_grid(lon_convention="-180-180", dlat=1.0, dlon=1.0):
    if lon_convention == "-180-180":
        lons = np.arange(-180+dlon/2, 180, dlon)
    else:
        lons = np.arange(0+dlon/2, 360, dlon)
    lats = np.arange(-90+dlat/2, 90, dlat)
    return xr.Dataset(coords=dict(lon=("lon", lons), lat=("lat", lats)))

def _bounds_from_centers(centers):
    centers = np.asarray(centers)
    step = np.diff(centers).mean()
    edges = np.concatenate([[centers[0]-step/2], centers[:-1]+step/2, [centers[-1]+step/2]])
    left = edges[:-1]; right = edges[1:]
    return np.vstack([left,right]).T

def conservative_regrid_to_target(
    ds_src: xr.Dataset,
    lon_convention="-180-180",
    dlat=1.0,
    dlon=1.0,
    weights_path=None,
    force_recompute=False,
):
    """
    Conservative regrid using xESMF with 2-D rectilinear grids built from
    *bounds* and *step*, compatible with xesmf versions that require the
    6-arg grid_2d signature.
    """
    try:
        import xesmf as xe
    except Exception as e:
        raise RuntimeError(
            "xesmf is required for conservative regridding. Install xesmf & esmpy."
        ) from e

    # Normalize coords (time/lat/lon, sorted, lon in requested convention)
    ds_src = normalize_geo(ds_src, lon_convention=lon_convention)

    # Infer native spacing from the source coords
    src_lon = np.asarray(ds_src["lon"].values)
    src_lat = np.asarray(ds_src["lat"].values)
    if src_lon.ndim != 1 or src_lat.ndim != 1:
        raise ValueError("Expected 1-D lat/lon coords on source grid.")
    dlon_src = float(np.round(np.median(np.diff(src_lon)), 6))
    dlat_src = float(np.round(np.median(np.diff(src_lat)), 6))
    lon0b_src = float(src_lon.min() - dlon_src / 2.0)
    lon1b_src = float(src_lon.max() + dlon_src / 2.0)
    lat0b_src = float(src_lat.min() - dlat_src / 2.0)
    lat1b_src = float(src_lat.max() + dlat_src / 2.0)

    # Target grid bounds from requested spacing
    tgt = make_target_grid(lon_convention, dlat=dlat, dlon=dlon)
    tgt_lon = np.asarray(tgt["lon"].values)
    tgt_lat = np.asarray(tgt["lat"].values)
    dlon_tgt = float(dlon)
    dlat_tgt = float(dlat)
    lon0b_tgt = float(tgt_lon.min() - dlon_tgt / 2.0)
    lon1b_tgt = float(tgt_lon.max() + dlon_tgt / 2.0)
    lat0b_tgt = float(tgt_lat.min() - dlat_tgt / 2.0)
    lat1b_tgt = float(tgt_lat.max() + dlat_tgt / 2.0)

    # Build 2-D rectilinear grids (this signature: lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat)
    src_grid = xe.util.grid_2d(
        lon0b_src, lon1b_src, dlon_src, lat0b_src, lat1b_src, dlat_src
    )
    tgt_grid = xe.util.grid_2d(
        lon0b_tgt, lon1b_tgt, dlon_tgt, lat0b_tgt, lat1b_tgt, dlat_tgt
    )

    # Handle weights cache
    if weights_path and force_recompute and os.path.exists(weights_path):
        try:
            os.remove(weights_path)
        except Exception:
            pass
    reuse = bool(weights_path and os.path.exists(weights_path))

    # Create regridder (periodic wrap in longitude here)
    regridder = xe.Regridder(
        src_grid,
        tgt_grid,
        method="conservative",
        filename=weights_path,
        reuse_weights=reuse,
        periodic=True,
    )

    # Apply to variables; ensure (time, lat, lon) order for clarity
    out = {}
    for v in ("pm2p5", "pm10"):
        if v in ds_src:
            da_in = ds_src[v].transpose("time", "lat", "lon")
            da_out = regridder(da_in)
            if {"y","x"}.issubset(set(da_out.dims)):
                da_out = da_out.rename({"y":"lat","x":"lon"})
            da_out = da_out.transpose("time","lat","lon")
            da_out.attrs = ds_src[v].attrs
            out[v] = da_out

    # Package with 1-D target coords + original time
    return xr.Dataset(
        out,
        coords=dict(time=ds_src["time"], lon=("lon", tgt_lon), lat=("lat", tgt_lat)),
    )


def ensure_parent_dir(path_str: str):
    d = os.path.dirname(os.path.expanduser(path_str))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def acquire_lock(fs, bucket, log):
    lock_uri = f"gs://{bucket}/{LOCK_OBJ_NAME}"
    if fs.exists(lock_uri):
        raise RuntimeError("Zarr build already running (lock present).")
    with fs.open(lock_uri, "wb") as f:
        f.write(b"locked")
    log.info(f"[lock] acquired {lock_uri}")
    return lock_uri

def release_lock(fs, lock_uri, log):
    try:
        fs.rm(lock_uri)
        log.info(f"[lock] released {lock_uri}")
    except Exception:
        pass

def object_is_stable(fs, path, min_age_minutes=2):
    if not fs.exists(path):
        return False
    info = fs.info(path)
    ts = info.get("updated") or info.get("mtime")
    if ts is None:
        return True
    updated = pd.to_datetime(ts, utc=True)
    return (pd.Timestamp.utcnow() - updated) >= pd.Timedelta(minutes=min_age_minutes)

def main():
    args = parse_args()
    log = configure_logging(args.log_file, args.verbose)
    dask.config.set({"array.slicing.split_large_chunks": True})

    fs = gcsfs.GCSFileSystem(token="google_default")
    months = [(y,m) for (y,m) in daterange_months(args.start, args.end)]

    if args.dry_run:
        log.info("Dry run. Checking which months exist and are stable:")
        for y,m in months:
            p = cams_month_path(args.bucket, args.raw_prefix, y, m)
            ok = fs.exists(p)
            stable = object_is_stable(fs, p, args.min_age_minutes) if ok else False
            log.info(f"  {y:04d}-{m:02d}  exists={ok} stable={stable} path={p}")
        return 0

    ensure_parent_dir(args.weights_path)

    lock_uri = None
    try:
        lock_uri = acquire_lock(fs, args.bucket, log)
        store_exists = fs.exists(args.zarr_path)

        for (y,m) in months:
            src_path = cams_month_path(args.bucket, args.raw_prefix, y, m)
            if not object_is_stable(fs, src_path, args.min_age_minutes):
                log.info(f"[skip] not ready/stable: {src_path}")
                continue

            try:
                ds = open_nc_any(src_path, engine_pref=args.engine, chunks_time=8, log=log)
            except Exception as e:
                log.error(f"[error] open failed: {src_path} -> {e}")
                continue

            ds = normalize_geo(ds, lon_convention=args.lon_convention)
            ds = standardize_pm_vars(ds, force_to_ug=False)

            try:
                ds_rg = conservative_regrid_to_target(
                    ds, lon_convention=args.lon_convention, dlat=args.dlat, dlon=args.dlon,
                    weights_path=args.weights_path, force_recompute=args.force_recompute_weights
                )
            except Exception as e:
                log.error(f"[error] regrid failed for {src_path}: {e}")
                continue

            ds_rg = ds_rg.chunk(CHUNKS)
            mode = "w" if not store_exists else "a"
            log.info(f"[write:{mode}] {args.zarr_path}")

            ctx = ProgressBar() if (args.progress and ProgressBar is not None) else nullcontext()
            with ctx:
                if mode == "w":
                    ds_rg.to_zarr(
                        args.zarr_path,
                        mode="w",
                        consolidated=True,
                        zarr_version=2,
                        align_chunks=True,
                    )
                    store_exists = True
                else:
                    ds_rg.to_zarr(
                        args.zarr_path,
                        mode="a",
                        append_dim="time",
                        consolidated=True,
                        zarr_version=2,
                        align_chunks=True,
                    )

        # consolidate metadata
        try:
            import zarr
            mapper = fs.get_mapper(args.zarr_path)
            zarr.consolidate_metadata(mapper)
            log.info("[done] metadata consolidated")
        except Exception as e:
            log.warning(f"[warn] consolidate_metadata failed: {e}")

        log.info("[done] Zarr build/append complete.")
        return 0

    finally:
        if lock_uri:
            release_lock(fs, lock_uri, log)

if __name__ == "__main__":
    sys.exit(main())
