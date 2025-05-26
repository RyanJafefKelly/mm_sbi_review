from mm_sbi_review.examples.turin import turin, compute_turin_summaries_with_max
import torch
import numpy as np

from rsnl.inference import run_rsnl
from rsnl.model import get_robust_model
from rsnl.visualisations import plot_and_save_all
import numpyro
import jax.numpy as jnp
import jax.random as random
import os
import arviz as az
import pickle as pkl


def run_turin_rsnl():
    folder_name = "data/turin_rsnl/"
    N = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_data_full = (
        torch.tensor(np.load("data/turin_obs.npy")).float().reshape(N, 801).to(device)
    )

    def compute_turin_summaries_with_max_jax_wrapper(x_data_full):
        x_data_torch = torch.as_tensor(np.array(x_data_full))
        delta_f = 5000000.0
        out_torch = compute_turin_summaries_with_max(x_data_torch, delta_f)
        return jnp.array(out_torch.cpu().numpy())

    def turin_sim_jax_wrapper(key, *theta):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert the JAX array 'theta' to a Torch tensor on the correct device.
        # If theta is shape (4,) this will work fine; if it’s just floats, reshape or adjust accordingly.
        theta_torch = torch.as_tensor(
            np.array(theta), device=device, dtype=torch.float32
        )

        # Pass it to your existing 'turin_sim' function, which expects torch tensors.
        out_torch = turin_sim(
            theta_torch
        )  # turin_sim is your instance of the turin class

        # Convert the PyTorch result back to jnp
        return jnp.array(out_torch.cpu().numpy())

    B = 4e9
    Ns = 801
    x_obs = compute_turin_summaries_with_max_jax_wrapper(x_data_full)
    turin_sim = turin(B=B, Ns=801, N=N, tau0=0)

    num_rounds = 4
    num_sims = 500

    model = get_robust_model
    prior = numpyro.distributions.Uniform(
        jnp.array([1e-9, 1e-9, 1e7, 1e-10]), jnp.array([1e-8, 1e-8, 5e9, 1e-9])
    )

    sim_fn = turin_sim_jax_wrapper
    sum_fn = compute_turin_summaries_with_max_jax_wrapper
    dummy_params = jnp.array([2e-9, 2e-9, 1e8, 2e-10])
    rng_key = random.PRNGKey(0)

    # TODO: ON NEXT TRY: GET WORKING MCMC, ESPECIALLY WITH ADJ_PARAMS ... scale_adj_var_x_obs CHECK UP ON
    # TODO: DEBUG: CHECK TO GET ROUND 2 MCMC
    # NOTE: SHOULD BE SAVING SIMS
    mcmc = run_rsnl(
        model,
        prior,
        sim_fn,
        sum_fn,
        rng_key,
        x_obs,
        num_sims_per_round=num_sims,
        num_rounds=num_rounds,
        jax_parallelise=False,
        true_params=dummy_params,
        save_each_round=True,
        # scale_adj_var_x_obs=scale_adj_var_x_obs,
        theta_dims=4,
    )

    mcmc.print_summary()
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    inference_data = az.from_numpyro(mcmc)

    with open(f"{folder_name}rsnl_max_theta_posterior_samples.pkl", "wb") as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f"{folder_name}rsnl_max_adj_params_posterior_samples.pkl", "wb") as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, dummy_params, folder_name=folder_name)


if __name__ == "__main__":
    run_turin_rsnl()
