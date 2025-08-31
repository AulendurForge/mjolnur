# In: mjolnur/scripts/run_graphcast_and_cache.py

import os
import yaml
import fsspec
import xarray as xr
import pandas as pd
import jax
import haiku as hk
from google.cloud import storage
from datetime import datetime
from mjolnur.utils.graphcast_utils import (
    get_graphcast_model_and_params,
    load_era5_data_for_graphcast,
)  # Assuming you have a function like this
from mjolnur.utils.graphcast_adapter import (
    extract_latents_from_graphcast,
)  # We will define this


# This is the key function we need to add to your graphcast_adapter
def extract_latents_and_predictions(
    model: hk.Transformed,
    params: hk.Params,
    rng: jax.random.PRNGKey,
    inputs: xr.Dataset,
) -> xr.Dataset:
    """Runs GraphCast and returns both decoded predictions and pre-decoder latents."""

    def forward_pass_with_latents(inputs):
        # This inner function re-runs the model's forward pass
        # but intercepts the output of the 'grid2grid' module (the latents).
        model_forward = model.apply
        grid_module = model_forward.grid_module  # Adjust if module name is different
        decoder = model_forward.decoder  # Adjust if module name is different

        # Run encoder and processor
        latents = grid_module(inputs)

        # Run decoder
        predictions = decoder(latents)
        return predictions, latents

    # JIT the forward pass for performance
    jitted_forward_with_latents = jax.jit(
        hk.transform_with_state(forward_pass_with_latents).apply
    )

    # Run the model
    (predictions, latents), _ = jitted_forward_with_latents(params, {}, rng, inputs)

    # Convert predictions and latents to xarray Datasets and merge
    # ... (code to format predictions and latents into a single xr.Dataset) ...
    # This part is highly dependent on your graphcast_utils, but the concept is to return
    # an xr.Dataset with variables for both predictions and latents.

    # Placeholder for actual conversion logic
    # For now, let's assume this returns a merged dataset.
    print("WARNING: Latent and prediction extraction logic is a placeholder.")
    return xr.Dataset()  # Replace with actual merged dataset


def run(config_path: str):
    """Main function to run caching job."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Loading GraphCast model...")
    graphcast, params = get_graphcast_model_and_params(
        model_name=cfg["graphcast"]["model_name"]
    )
    rng = jax.random.PRNGKey(0)

    # Define time range
    time_range = pd.date_range(
        start=cfg["times"]["start"],
        end=cfg["times"]["stop"],
        freq=f"{cfg['times']['step_hours']}H",
    )

    output_zarr_path = cfg["output"]["zarr_path"]
    print(f"Will write outputs to: {output_zarr_path}")

    for i, t0 in enumerate(time_range):
        t0_str = t0.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Processing timestep {i + 1}/{len(time_range)}: {t0_str}")

        try:
            # 1. Load ERA5 data required by GraphCast for this timestep
            # This is where your ERA5 retriever is called.
            # You need a function that returns an xarray.Dataset in the exact
            # format GraphCast expects.
            graphcast_inputs = load_era5_data_for_graphcast(t0)

            # 2. Run GraphCast and extract outputs
            # NOTE: We need to implement `extract_latents_and_predictions`
            gc_outputs = extract_latents_and_predictions(
                graphcast, params, rng, graphcast_inputs
            )

            # 3. Save to Zarr
            # Set the time coordinate to t0 to represent the forecast *initialization* time
            gc_outputs = gc_outputs.assign_coords(time=[t0])

            if i == 0:
                # First write, create the zarr store
                gc_outputs.to_zarr(output_zarr_path, mode="w", consolidated=True)
            else:
                # Subsequent writes, append along the time dimension
                gc_outputs.to_zarr(
                    output_zarr_path, mode="a", append_dim="time", consolidated=True
                )

            print(f"Successfully processed and saved {t0_str}")

        except Exception as e:
            print(f"ERROR processing {t0_str}: {e}")
            continue

    print("Caching complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )
    args = parser.parse_args()
    run(args.config)
