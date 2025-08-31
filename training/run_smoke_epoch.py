# mjolnur/training/run_smoke_epoch.py
from __future__ import annotations
import argparse, yaml, numpy as np, jax, jax.numpy as jnp, optax
import xarray as xr, pandas as pd
from mjolnur.data.cams import open_cams_zarr
from mjolnur.data.dataloader import times_6h, build_example_at_time
from mjolnur.models.sidecar import ACUNet
from mjolnur.models.advection import advect_pm

def main():
    def area_weights(H):
        lat = jnp.linspace(-89.5, 89.5, H)
        w = jnp.cos(jnp.deg2rad(lat))
        return (w / w.mean())[None, :, None, None]


    def persistence(pm_t):
        return pm_t


    def rmse(a, b, w=None):
        e = (a - b) ** 2
        if w is not None:
            e = w * e
        return jnp.sqrt(jnp.mean(e))


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True)
        args = ap.parse_args()
        cfg = yaml.safe_load(open(args.config))

        # data
        cams = open_cams_zarr(cfg["cams_zarr"])
        ts = list(
            times_6h(
                cfg["times"]["start"], cfg["times"]["stop"], cfg["times"]["step_hours"]
            )
        )

        # model/opt
        key = jax.random.PRNGKey(cfg["train"]["seed"])
        model = ACUNet(base=32, levels=3)
        # init shapes from first example
        pm_t, gc_t6, y, meta = build_example_at_time(cams, cfg, ts[0])
        params = model.init(key, jnp.array(pm_t), jnp.array(gc_t6))["params"]
        tx = optax.adamw(cfg["train"]["lr"], weight_decay=1e-4)
        opt_state = tx.init(params)

        def loss_step(params, batch):
            pm_t, gc_t6, y = batch
            yhat, aux = model.apply({"params": params}, pm_t, gc_t6)
            wlat = area_weights(pm_t.shape[1])
            mse = jnp.mean(wlat * (yhat - y) ** 2)
            l1 = jnp.mean(jnp.abs(aux["delta"]))
            loss = mse + 1e-4 * l1
            logs = {"rmse": jnp.sqrt(mse)}
            return loss, (logs, yhat)

        @jax.jit
        def train_step(params, opt_state, batch):
            (loss, (logs, yhat)), grads = jax.value_and_grad(loss_step, has_aux=True)(
                params, batch
            )
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, logs, yhat

        # loop
        for step_i, t0 in enumerate(ts[: cfg["train"]["steps"]], start=1):
            pm_t, gc_t6, y, meta = build_example_at_time(cams, cfg, t0)
            pm_t = jnp.array(pm_t)
            gc_t6 = jnp.array(gc_t6)
            y = jnp.array(y)

            params, opt_state, loss, logs, yhat = train_step(
                params, opt_state, (pm_t, gc_t6, y)
            )

            # baselines
            u10, v10 = gc_t6[..., 0], gc_t6[..., 1]
            adv = advect_pm(pm_t, u10, v10, 6.0)
            wlat = area_weights(pm_t.shape[1])
            rmse_model = float(logs["rmse"])
            rmse_pers = float(rmse(persistence(pm_t), y, wlat))
            rmse_adv = float(rmse(adv, y, wlat))

            if (step_i % cfg["train"]["log_every"]) == 0:
                print(
                    f"[{step_i:03d}] t0={meta['t0']}  loss={float(loss):.3f}  "
                    f"rmse(model)={rmse_model:.3f}  rmse(pers)={rmse_pers:.3f}  rmse(adv)={rmse_adv:.3f}"
                )


    if __name__ == "__main__":
        main()
