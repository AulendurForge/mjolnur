# mjolnur/models/sidecar.py
from __future__ import annotations
import flax.linen as nn, jax.numpy as jnp
import jax
from .advection import advect_pm


class Conv(nn.Module):
    ch: int
    k: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.ch, (self.k, self.k), padding="SAME")(x)
        return nn.gelu(x)


class Block(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        h = Conv(self.ch)(x)
        h = Conv(self.ch)(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1))(x)
        return x + h


class ACUNet(nn.Module):
    base: int = 32
    levels: int = 3

    @nn.compact
    def __call__(self, pm_t, gc_t6):
        # pm_t: (B,H,W,2); gc_t6: (B,H,W,F) where first 2 are u10,v10
        u10, v10 = gc_t6[..., 0], gc_t6[..., 1]
        adv = advect_pm(pm_t, u10, v10, 6.0)  # (B,H,W,2)
        x = jnp.concatenate([pm_t, adv, gc_t6], axis=-1)

        skips = []
        h = Conv(self.base)(x)
        for i in range(self.levels):
            h = Block(self.base * (2**i))(h)
            skips.append(h)
            h = nn.avg_pool(h, (2, 2), (2, 2))
        for i in reversed(range(self.levels)):
            h = jax.image.resize(
                h,
                (h.shape[0], skips[i].shape[1], skips[i].shape[2], h.shape[3]),
                method="nearest",
            )
            h = jnp.concatenate([h, skips[i]], axis=-1)
            h = Block(self.base * (2**i))(h)
        delta = nn.Conv(2, (1, 1))(h)
        pm_hat = nn.softplus(adv + delta)
        return pm_hat, {"delta": delta}
