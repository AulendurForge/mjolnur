import jax, jax.numpy as jnp


def advect_pm(pm_t, u10_t6, v10_t6, dt_hours=6.0):
    """
    pm_t:   [B,H,W,2],   u10_t6,v10_t6: [B,H,W] on 1° grid
    returns advected pm at t+6: [B,H,W,2]
    """
    B, H, W, _ = pm_t.shape
    deg2m = 111e3
    dt_s = dt_hours * 3600.0
    lat = jnp.linspace(-89.5, 89.5, H)[None, :, None]
    coslat = jnp.cos(jnp.deg2rad(lat))

    jj, ii = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing="xy")
    lon = jj.astype(jnp.float32) + 0.5
    latg = ii.astype(jnp.float32) + 0.5

    def bilinear(img, y, x):
        x0 = jnp.floor(x)
        x1 = x0 + 1
        y0 = jnp.floor(y)
        y1 = y0 + 1
        wx = x - x0
        wy = y - y0
        x0m = (x0 % W).astype(jnp.int32)
        x1m = (x1 % W).astype(jnp.int32)
        y0c = jnp.clip(y0, 0, H - 1).astype(jnp.int32)
        y1c = jnp.clip(y1, 0, H - 1).astype(jnp.int32)
        Ia, Ib = img[y0c, x0m], img[y0c, x1m]
        Ic, Id = img[y1c, x0m], img[y1c, x1m]
        return (
            ((1 - wx) * (1 - wy))[..., None] * Ia
            + (wx * (1 - wy))[..., None] * Ib
            + ((1 - wx) * wy)[..., None] * Ic
            + (wx * wy)[..., None] * Id
        )

    def adv_one(pm, u, v):
        dlon = (u * dt_s) / (deg2m * coslat.squeeze(0))
        dlat = (v * dt_s) / deg2m
        src_x = (lon - dlon) % W
        src_y = jnp.clip(latg - dlat, 0.0, H - 1.001)
        return bilinear(pm, src_y, src_x)

    return jax.vmap(adv_one)(pm_t, u10_t6, v10_t6)
