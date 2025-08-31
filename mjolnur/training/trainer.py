import jax, jax.numpy as jnp, optax


def area_weights(H):
    lat = jnp.linspace(-89.5, 89.5, H)
    w = jnp.cos(jnp.deg2rad(lat))
    return (w / w.mean())[None, :, None, None]


def make_train_step(model, lr=2e-4):
    opt = optax.adamw(lr, weight_decay=1e-4)

    @jax.jit
    def step(params, opt_state, batch):
        pm_t, gc_t6, y = batch["pm_t"], batch["gc_t6"], batch["y"]
        wlat = area_weights(pm_t.shape[1])

        def loss_fn(p):
            yhat, aux = model.apply({"params": p}, pm_t, gc_t6)
            mse = jnp.mean(wlat * (yhat - y) ** 2)
            l1 = jnp.mean(jnp.abs(aux["delta"]))
            return mse + 1e-4 * l1, {"rmse": jnp.sqrt(mse)}

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, float(loss), {k: float(v) for k, v in logs.items()}

    return step
