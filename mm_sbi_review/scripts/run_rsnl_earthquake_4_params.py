from mm_sbi_review.examples.earthquake import (
    earthquake_sim_fn_4_param,
    sum_fn,
    early_return,
)
from mm_sbi_review.scripts.utils import (
    download_file,
    extract_tar_gz,
    combine_ascii_files,
)
import numpy as np
import torch
from torch import tensor

# from torch.distributions import Distribution, Uniform, Gamma
from etas.inversion import round_half_up, branching_ratio
from etas.simulation import generate_catalog, generate_background_events
import json
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
from rsnl.inference import run_rsnl
from rsnl.model import get_robust_model
import jax.random as random
import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from numpyro.distributions import constraints
from numpyro.distributions import Gamma, Uniform, Independent, Distribution
from numpyro.distributions.util import is_prng_key, validate_sample, promote_shapes
import jax
import numpyro
import os
import pickle as pkl
import jax.numpy as jnp
from numpyro.distributions.util import clamp_probs
from numpyro.distributions import constraints
from rsnl.utils import FlowNumpyro
from typing import Optional


def parameter_dict2array(parameters):
    order = [
        "log10_mu",
        "log10_iota",
        "log10_k0",
        "a",
        "log10_c",
        "omega",
        "log10_tau",
        "log10_d",
        "gamma",
        "rho",
    ]

    if "alpha" in parameters:
        order.insert(0, "alpha")

    return np.array([parameters.get(key, None) for key in order])


def run_earthquake():
    np.random.seed(123)
    torch.manual_seed(321)

    # true_mu = 2e-05
    # true_k = 0.2
    # true_c = 0.5
    # true_p = 1.5

    # true_params = {'log10_mu': np.log10(true_mu), 'log10_k0': np.log10(true_k),
    #             #    'a': true_alpha,
    #                'log10_c': np.log10(true_c), 'rho': true_p}
    # TODO: first - set up simulator to take in parameters
    # TODO: SNPE
    # TODO: Priors
    # beta = 2.302585092994046
    # beta = 2.4
    url = "https://scedc.caltech.edu/ftp/catalogs/SCEC_DC/SCEDC_catalogs.tar.gz"
    local_filename = "SCEDC_catalogs.tar.gz"
    extract_path = "./"

    # Download and extract the file
    download_file(url, local_filename)
    extract_tar_gz(local_filename, extract_path)
    combine_ascii_files("./raw")
    raw_catalog = pd.read_csv("raw/SCEDC_catalog.csv")
    raw_catalog["time"] = pd.to_datetime(raw_catalog["time"])
    raw_catalog = raw_catalog.sort_values(by="time")
    raw_catalog = raw_catalog[["time", "longitude", "latitude", "magnitude"]].dropna()
    raw_catalog.reset_index(drop=False, inplace=True)

    polygon_coords = np.load("data/SCEDC_shape.npy")

    poly = Polygon(polygon_coords)
    gdf = gpd.GeoDataFrame(
        raw_catalog,
        geometry=gpd.points_from_xy(raw_catalog.latitude, raw_catalog.longitude),
    )

    catalog = gdf[gdf.intersects(poly)].copy()
    catalog.drop("geometry", axis=1, inplace=True)
    len_trunc_x = len(catalog)
    print("Removed", len(raw_catalog) - len_trunc_x, "events outside polygon")

    # filter events within timewindow
    auxiliary_start = "1985-01-01 00:00:00"
    test_nll_end = "2014-01-11 00:00:00"
    catalog = catalog[catalog["time"] >= auxiliary_start]
    catalog = catalog[catalog["time"] < test_nll_end]
    len_trunc_t = len(catalog)
    print("Removed", len_trunc_x - len_trunc_t, "events outside timewindow")

    M_cut = 3.0
    catalog = catalog[catalog["magnitude"] >= M_cut]
    len_trunc_m = len(catalog)
    print("Removed", len_trunc_t - len_trunc_m, "events below Mcut")

    observed_summaries = sum_fn(catalog)

    # prior = ETASPrior()  # TODO
    # prior = dist.Uniform(low=jnp.repeat(0.0, 5),
    #                      high=jnp.repeat(1.0, 5))
    # x_test = jnp.array([[0.0005, 0.1, 0.05, 0.5, 1.5]])
    # lp_val = prior.log_prob(x_test)

    # prior = Uniform(low=jnp.array([0., 0., 0.0, 1.0]),
    #                 high=jnp.array([0.0001, 0.005, 1.0, 2.0]))
    # prior = Independent(prior, 1)  # TODO: CHECK IF THIS IS NEEDED
    def get_custom_robust_model(
        x_obs: jnp.ndarray,
        prior: dist,
        flow: Optional[FlowNumpyro] = None,
        scale_adj_var: Optional[jnp.ndarray] = None,
        standardisation_params=None,
    ) -> jnp.ndarray:
        """Get robust numpyro model."""
        laplace_mean = jnp.zeros(len(x_obs))
        laplace_var = jnp.ones(len(x_obs))
        # if scale_adj_var is None:
        #     scale_adj_var = jnp.ones(len(x_obs))

        mu = numpyro.sample("mu", dist.Gamma(concentration=0.1, rate=10.0))
        print("mu: ", mu)
        # Then rescale to get the target range/mean
        c = numpyro.sample("c", dist.Uniform(0, 1))
        k0 = numpyro.sample("k0", dist.Uniform(0, 1))
        # mu = numpyro.deterministic("mu", 0.1 * mu_raw)
        rho = numpyro.sample("rho", dist.Uniform(1, 2))

        theta = jnp.array([mu, k0, c, rho])

        theta_standard = numpyro.deterministic(
            "theta_standard",
            (theta - standardisation_params["theta_mean"])
            / standardisation_params["theta_std"],
        )
        # print('theta_standard: ', theta_standard.shape)
        # Note: better sampling if use standard laplace then scale
        adj_params = numpyro.sample(
            "adj_params", dist.Laplace(laplace_mean, laplace_var)
        )
        scaled_adj_params = numpyro.deterministic(
            "adj_params_scaled", adj_params * scale_adj_var
        )
        x_adj = numpyro.deterministic("x_adj", x_obs - scaled_adj_params)
        # print("x_adj: ", x_adj.shape)
        if flow is not None:  # i.e. if not first round
            x_adj_sample = numpyro.sample(
                "x_adj_sample", FlowNumpyro(flow, theta=theta_standard), obs=x_adj
            )
        else:
            x_adj_sample = x_adj

        # print("x_adj_sample: ", x_adj_sample.shape)

        return x_adj_sample

    model = get_custom_robust_model
    sim_fn = earthquake_sim_fn_4_param
    summ_fn = sum_fn

    # Define the Gamma prior for mu
    mu_prior = Gamma(concentration=0.1, rate=10)

    # Define the Uniform priors for the remaining parameters
    other_priors = Uniform(
        low=jnp.array([0.0, 0.0, 1.0]), high=jnp.array([1.0, 1.0, 2.0])
    )

    # Combine the priors into a joint distribution
    class CustomPrior(Distribution):
        support = constraints.real_vector

        def sample(self, key, sample_shape=()):
            mu_sample = mu_prior.sample(key, sample_shape)
            other_samples = other_priors.sample(key, sample_shape)
            return jnp.concatenate([mu_sample[..., None], other_samples], axis=-1)

        def log_prob(self, value):
            if value.ndim == 0:
                # Handle scalar case directly
                mu_log_prob = mu_prior.log_prob(value)
                other_log_prob = 0.0  # No other parameters to account for
            else:
                # Expected case for vector inputs
                mu_log_prob = mu_prior.log_prob(value[..., 0])
                other_log_prob = other_priors.log_prob(value[..., 1:])
            return mu_log_prob + other_log_prob

    prior = CustomPrior()

    # true_params = jnp.array([true_mu, true_k, true_c, true_p])
    def valid_fn(thetas):
        with open("data/config/SCEDC_30.json", "r") as f:
            simulation_config = json.load(f)
        catalog_params = simulation_config["theta_0"].copy()
        if len(thetas) == 5:
            mu, a, k0, c, rho = thetas
        if len(thetas) == 4:
            mu, k0, c, rho = thetas
            a = simulation_config["beta"]

        try:
            params_dict = dict(
                {
                    "log10_mu": np.log10(mu),
                    "log10_k0": np.log10(k0),
                    "a": np.array(a),
                    "log10_c": np.log10(c),
                    "rho": rho,
                }
            )  # Attempt conversion
            params_dict["a"] = params_dict["a"].item()
            params_dict["rho"] = params_dict["rho"].item()
        except Exception as e:
            print(e)

        catalog_params.update(params_dict)

        # Note: in four param model version, alpha ("a") fixed to beta
        if "a" not in catalog_params:
            catalog_params["a"] = simulation_config["beta"]

        if early_return(catalog_params, simulation_config["beta"]):
            return False
        else:
            return True

    print("observed_summaries: ", observed_summaries)
    folder_name = "res/earthquake_4_param/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    rng_key = random.PRNGKey(1)

    num_rounds_run_already = 4

    with open(f"res/earthquake_4_param/thetas_all_round_{str(2)}.pkl", "rb") as f:
        thetas_all = pkl.load(f)

    with open(f"res/earthquake_4_param/x_sims_all_round_{str(2)}.pkl", "rb") as f:
        x_sims_all = pkl.load(f)
    model_param_names = ["mu", "k0", "c", "rho"]
    num_sims_per_round = 3000
    theta_dims = 4
    num_rounds = 5 - num_rounds_run_already
    mcmc = run_rsnl(
        model,
        prior,
        sim_fn,
        summ_fn,
        rng_key,
        observed_summaries,
        num_sims_per_round=num_sims_per_round,
        num_rounds=num_rounds,
        true_params=None,  # NOTE: dummy
        theta_dims=theta_dims,
        jax_parallelise=False,
        num_chains=1,
        target_accept_prob=0.8,
        only_valid_sims_in_first_round=True,
        save_each_round=False,
        valid_fn=valid_fn,
        previous_thetas=thetas_all,
        previous_x_sims=x_sims_all,
        folder_name=folder_name,
        model_param_names=model_param_names,
    )
    mcmc.print_summary()
    if model_param_names is None:
        thetas = mcmc.get_samples()["theta"]
    else:
        thetas = jnp.squeeze(
            jnp.array([mcmc.get_samples()[name_i] for name_i in model_param_names])
        ).T
        thetas = thetas.reshape((num_sims_per_round, theta_dims))

    with open("res/earthquake_4_param/thetas_all_final.pkl", "wb") as f:
        pkl.dump(thetas, f)


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    run_earthquake()
