from mm_sbi_review.examples.earthquake import earthquake_sim_fn, sum_fn, early_return
from mm_sbi_review.scripts.utils import download_file, extract_tar_gz, combine_ascii_files
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
from rsnl.inference import run_snl
from rsnl.model import get_standard_model
import jax.random as random
import jax.numpy as jnp
from numpyro.distributions import Distribution as dist
from numpyro.distributions import constraints
from numpyro.distributions import Gamma, Uniform, Independent
from numpyro.distributions.util import is_prng_key, validate_sample, promote_shapes
import jax
import numpyro

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

def parameter_dict2array(param_dict):
    # Example placeholder
    return np.array([
        param_dict["log10_mu"],
        param_dict["log10_k0"],
        param_dict["a"],
        param_dict["log10_c"],
        param_dict["rho"],
    ], dtype=float)


class SBEtasPrior(dist):
    support = constraints.real_vector
    # TODO! Idea ... bare  bones uniform
    # TOOD! Add a validation function, early return -1s if bad

    def branching_ratio_helper(self, mu, K, alpha, c, p, config):
        """
        Placeholder for your real branching ratio calculation.
        Modify as needed.
        """
        # Possibly do transformations, e.g.:
        log10_mu = np.log10(mu) if mu > 0 else -np.inf
        log10_k0 = np.log10(K)  if K  > 0 else -np.inf
        log10_c  = np.log10(c)  if c  > 0 else -np.inf
        param_dict = {
            "log10_mu": log10_mu,
            'log10_iota': None,
            "log10_k0": log10_k0,
            "a": alpha,
            "log10_c": log10_c,
            "rho": p,
        }
        # Merge with config
        cat_params = dict(config["theta_0"])
        cat_params.update(param_dict)

        # Do your real ratio computation, or just return a placeholder:
        # e.g. br_val = some_function(cat_params, config["beta"])
        # For demo, we set 0.5 so it always passes
        branch_arr = [cat_params["log10_mu"], cat_params["log10_iota"],
                      cat_params["log10_k0"], cat_params["a"],
                      cat_params["log10_c"],
                      cat_params["omega"], cat_params["log10_tau"],
                      cat_params["log10_d"], cat_params["gamma"],
                      cat_params["rho"]]
        br_val = branching_ratio(branch_arr, config["beta"])
        return br_val

    @property
    def event_shape(self):
        return (5,)

    def __init__(self, beta=None, mu_max=0.0015, validate_args=None):
        """
        :param beta: scalar for sub-critical constraint.
        :param mu_max: upper bound for mu, used in sample_with_constraints only.
        """
        super().__init__(batch_shape=(), event_shape=(5,), validate_args=validate_args)

        with open("data/config/SCEDC_30.json", 'r') as f:
            self.simulation_config = json.load(f)

        if beta is None:
            self.beta = self.simulation_config["beta"]
        else:
            self.beta = beta

        self.mu_max = mu_max

        # 1) Gamma(0.1, 10) for mu (constrained sampling)
        self.mu_dist = Gamma(0.1, 10)

        # mu ~ Uniform(0, 0.002)
        self.mu_low, self.mu_high = 0.0, 0.02
        # alpha ~ Uniform(0, beta)
        self.alpha_low, self.alpha_high = 0.0, self.beta
        # c ~ Uniform(0, 1)
        self.c_low, self.c_high = 0.0, 1.0
        # p ~ Uniform(1, 2)
        self.p_low, self.p_high = 1.0, 2.0
        
        # without mu, used constrained sampling
        self.uni_dist = Uniform(low=jnp.array([0.0, 0.0, 0.0, 0.0]),
                                high=jnp.array([self.beta, 1.0, 2.0, 1.0]))
        

    @validate_sample
    def log_prob(self, value):
        if value.shape[-1] != 5:
            raise ValueError("SubcriticalUniform expects last dimension=5")

        mu     = value[..., 0]
        K      = value[..., 1]
        alpha  = value[..., 2]
        c      = value[..., 3]
        p      = value[..., 4]

        # We want to compute the log probability for each dimension
        # under the "conditional uniform" construction:
        # 1) mu ~ Uniform(0,0.002)
        # 2) alpha ~ Uniform(0,beta)
        # 3) K | alpha ~ Uniform(0, (beta - alpha)/beta)
        # 4) c ~ Uniform(0,1)
        # 5) p ~ Uniform(1,2)

        # Step by step:

        # A) Check bounds
        mu_in_range = (mu >= self.mu_low) & (mu <= self.mu_high)
        alpha_in_range = (alpha >= self.alpha_low) & (alpha <= self.alpha_high)
        c_in_range = (c >= self.c_low) & (c <= self.c_high)
        p_in_range = (p >= self.p_low) & (p <= self.p_high)

        # For K in [0, (beta - alpha)/beta], the upper bound is (beta - alpha)/beta
        # but only valid if alpha < beta, so (beta - alpha) > 0
        # We'll define:
        k_low = 0.0
        k_high = (self.beta - alpha) / self.beta   # shape (...)
        # Must also ensure alpha < beta => (beta - alpha) > 0 => alpha < beta
        k_in_range = (K >= k_low) & (K <= k_high) & (alpha < self.beta)

        # B) If out of range => log_prob = -inf
        is_valid = mu_in_range & alpha_in_range & c_in_range & p_in_range & k_in_range
        lp = jnp.full(value.shape[:-1], -jnp.inf)

        # C) For points inside all bounds, compute log_prob = sum of logs of side lengths
        # Because it's a product of conditionals:
        #   p(mu) = 1/(0.002 - 0) = 1/0.002 = 500
        #   p(alpha) = 1/beta
        #   p(K|alpha) = 1 / ((beta - alpha)/beta) = beta / (beta - alpha)
        #   p(c) = 1/(1-0) = 1
        #   p(p) = 1/(2-1) = 1
        # => total log_prob = log(500) + log(1/beta) + log(beta/(beta-alpha)) + 0 + 0
        # => = log(500) + (-log beta) + (log beta - log (beta-alpha))
        # => = log(500) - log((beta - alpha)).

        # We'll define it explicitly:
        mu_lp     = jnp.log(1.0 / (self.mu_high - self.mu_low))  # log(1/0.002)= log(500)
        alpha_lp  = jnp.log(1.0 / (self.alpha_high - self.alpha_low))  # log(1/beta)
        # For K =>  log(1 / ((beta - alpha)/beta))= log(beta/(beta-alpha))= log beta - log(beta-alpha)
        # c => uniform(0,1) => 0
        # p => uniform(1,2) => log(1/(2-1))= log(1)=0
        # So total is:
        log_p_K_given_alpha = jnp.log(self.beta) - jnp.log(self.beta - alpha)
        # Combine everything:
        local_lp = (mu_lp + alpha_lp + log_p_K_given_alpha)
        # c, p each add 0

        # So for valid points:
        lp = jnp.where(is_valid, local_lp, lp)
        return lp

    def sample(self, key, sample_shape=()):
        """
        Draw samples from the 5D uniform in a single call.
        sample_shape is a tuple indicating how many samples you want.
        Returns shape sample_shape + (5,).
        """
        # Just use the underlying distribution's sample,
        # which will produce shape (sample_shape..., 5).
        # Flatten sample_shape
        n = 1
        for s in sample_shape:
            n *= s

        # We do it dimension by dimension:
        rng_mu, rng_alpha, rng_k, rng_c, rng_p = jax.random.split(key, 5)

        # 1) mu
        mu_samps = jax.random.uniform(rng_mu, shape=(n,),
                                      minval=self.mu_low, maxval=self.mu_high)

        # 2) alpha
        alpha_samps = jax.random.uniform(rng_alpha, shape=(n,),
                                         minval=self.alpha_low, maxval=self.alpha_high)

        # 3) K, depends on alpha => we must pass alpha_samps in
        # K in [0, (beta - alpha)/beta]. We'll build the max array:
        k_upper = (self.beta - alpha_samps) / self.beta
        # So sample from Uniform(0, k_upper) for each row.
        # We'll do jax.vmap or a simple approach: jax.random.uniform can accept per-element minval/maxval in vmap style
        # We'll define a helper function:

        def sample_k(rng, up):
            return jax.random.uniform(rng, minval=0.0, maxval=up)

        # We can do a vmap over alpha_samps
        rngs_k = jax.random.split(rng_k, n)  # one rng per sample
        K_samps = jax.vmap(sample_k)(rngs_k, k_upper)

        # 4) c ~ Uniform(0,1)
        c_samps = jax.random.uniform(rng_c, shape=(n,), minval=0.0, maxval=1.0)

        # 5) p ~ Uniform(1,2)
        p_samps = jax.random.uniform(rng_p, shape=(n,), minval=1.0, maxval=2.0)

        # Combine => shape (n, 5)
        out = jnp.column_stack([mu_samps, K_samps, alpha_samps, c_samps, p_samps])

        # Reshape to sample_shape + (5,)
        out = out.reshape(*sample_shape, 5)
        return out

    @validate_sample
    def log_prob_old(self, value):
        """
        Evaluate log prior for (..., 5) => [mu, K, alpha, c, p],
        ignoring branching ratio and ignoring mu < mu_max.
        We do:
          - mu >= 0
          - (K, alpha, c, p) in uniform box
          - sub-critical => (K * beta) < (beta - alpha)
        """
        if value.shape[-1] != 5:
            raise ValueError("SBEtasPrior: expected last dimension=5")

        mu     = value[..., 0]
        K      = value[..., 1]
        alpha  = value[..., 2]
        c      = value[..., 3]
        p      = value[..., 4]

        # Start with 0
        lp = jnp.zeros(value.shape[:-1])

        gamma_lp = self.mu_dist.log_prob(mu)
        # No top truncation, just mu >= 0
        valid_mu = (mu >= 0.0)
        gamma_lp = jnp.where(valid_mu, gamma_lp, -jnp.inf)
        lp += gamma_lp

        # 2) Uniform for [K, alpha, c, p]
        other_params = jnp.stack([K, alpha, c, p], axis=-1)  # shape (...,4)
        uni_lp = self.uni_dist.log_prob(other_params).sum(axis=-1)
        lp += uni_lp

        # 3) sub-critical => (K * beta) < (beta - alpha)
        subcrit_mask = (K * self.beta) < (self.beta - alpha)
        lp = jnp.where(subcrit_mask, lp, -jnp.inf)

        return lp

    def sample_old(self, key, sample_shape=()):
        """
        Minimal naive sampling via rejection in Python, ignoring mu_max
        and branching ratio. Only:
          mu >= 0, sub-critical => (K*beta) < (beta-alpha)
        """
        num = 1
        for s in sample_shape:
            num *= s

        out = []
        batch_size = max(2 * num, 1000)
        rng = key

        while len(out) < num:
            rng, rng_mu, rng_uni = jax.random.split(rng, 3)
            # 1) Sample mu from Gamma(0.1,10)
            mu_cand = self.mu_dist.sample(rng_mu, (batch_size,))
            # 2) Sample (K, alpha, c, p) from Uniform
            uni_cand = self.uni_dist.sample(rng_uni, (batch_size,))

            # shape => (batch_size,5)
            cand = jnp.column_stack((mu_cand, uni_cand))

            # basic constraints in JAX
            mu_ok = (mu_cand >= 0.0)
            subcrit_ok = (uni_cand[:,0] * self.beta) < (self.beta - uni_cand[:,1])
            mask = mu_ok & subcrit_ok

            cand_ok = np.array(cand[mask])  # to CPU
            out.extend(cand_ok)

        out_array = np.stack(out[:num], axis=0)
        out_array = out_array.reshape(*sample_shape, 5)
        return jax.device_put(out_array)

    def sample_with_constraints(self, key, sample_shape=()):
        """
        Naive Python-based rejection sampling from the prior,
        including a final check that branching_ratio(...) < 1.05.
        """
        # Flatten sample_shape => total number
        num = 1
        for s in sample_shape:
            num *= s

        out = []
        batch_size = max(2 * num, 1000)
        rng = key

        while len(out) < num:
            # 1) Draw candidate mu
            rng, rng_mu, rng_uni = jax.random.split(rng, 3)
            mu_cand = self.mu_dist.sample(rng_mu, (batch_size,))
            # 2) Draw candidate (K, alpha, c, p)
            uni_cand = self.uni_dist.sample(rng_uni, (batch_size,))
            # shape => (batch_size, 5)
            cand = jnp.column_stack((mu_cand, uni_cand))

            # Filter by sub-critical & truncation in JAX
            mu_ok = (mu_cand >= 0.0) & (mu_cand < self.mu_max)
            subcrit_ok = (uni_cand[:, 0] * self.beta) < (self.beta - uni_cand[:, 1])
            mask = mu_ok & subcrit_ok

            # Convert to NumPy for branching ratio check
            cand_ok = np.array(cand[mask])

            for row in cand_ok:
                mu_i, K_i, alpha_i, c_i, p_i = row
                # Check branching ratio in Python
                br_val = self.branching_ratio_helper(mu_i, K_i, alpha_i, c_i, p_i, self.simulation_config)
                if br_val < 1.05:
                    out.append(row)

        # Take first 'num' valid
        out_array = np.stack(out[:num], axis=0)
        out_array = out_array.reshape(*sample_shape, 5)
        return jax.device_put(out_array)

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample
from numpyro.distributions import Gamma  # for mu

class ETASPrior(dist):
    """
    5-parameter ETAS prior:
      mu ~ Gamma(0.1, 10)
      alpha ~ Uniform(0, beta)
      K | alpha ~ Uniform(0, (beta - alpha)/beta)
      c ~ Uniform(0, 1)
      p ~ Uniform(1, 2)
    """

    support = constraints.real_vector  # we have a 5D vector

    @property
    def event_shape(self):
        return (5,)

    def __init__(self, beta=None, validate_args=None):
        """
        :param beta: Upper bound for alpha, also used in sub-critical constraint for K.
                     e.g. alpha in [0, beta], K < (beta - alpha)/beta
        """
        super().__init__(batch_shape=(), event_shape=(5,), validate_args=validate_args)
        if beta is None:
            with open("data/config/SCEDC_30.json", 'r') as f:
                self.simulation_config = json.load(f)

                self.beta = self.simulation_config["beta"]

        else:
            self.beta = beta

        # 1) mu ~ Gamma(0.1, 0.1)
        self.mu_shape = 0.1
        self.mu_rate  = 10  # TODO: CHECK ?

        # 2) alpha ~ Uniform(0, beta)
        self.alpha_low, self.alpha_high = 0.0, self.beta

        # 3) K | alpha ~ Uniform(0, (beta - alpha)/beta)
        # We'll handle in sampling & log_prob.
        self.k_low, self.k_high = 0.0, 1.0

        # 4) c ~ Uniform(0,1)
        self.c_low, self.c_high = 0.0, 1.0

        # 5) p ~ Uniform(1,2)
        self.p_low, self.p_high = 1.0, 2.0

    @validate_sample
    def log_prob(self, value):
        """
        value.shape = (..., 5) => [mu, alpha, K, c, p].
        Returns log p(value).
        """
        if value.shape[-1] != 5:
            raise ValueError("ETASPrior: expected last dimension=5")

        mu     = value[..., 0]
        alpha  = value[..., 1]
        K      = value[..., 2]
        c      = value[..., 3]
        p      = value[..., 4]

        # Start with zeros
        lp = jnp.zeros(value.shape[:-1])

        # ----------------------------
        # 1) mu ~ Gamma(0.1, 10)
        #    => log_prob from a standard NumPyro Gamma
        from numpyro.distributions import Gamma
        mu_dist = Gamma(self.mu_shape, self.mu_rate)
        mu_lp = mu_dist.log_prob(mu)
        # no top/bottom except mu>0
        lp = lp + mu_lp

        # ----------------------------
        # 2) alpha ~ Uniform(0, beta)
        alpha_in_range = (alpha >= self.alpha_low) & (alpha <= self.alpha_high)
        # log-prob for alpha if in range => -log(beta - 0) = -log(beta)
        alpha_lp = -jnp.log(self.alpha_high - self.alpha_low)
        # if out of range => -inf
        alpha_lp = jnp.where(alpha_in_range, alpha_lp, -jnp.inf)
        lp = lp + alpha_lp

        # ----------------------------
        # 3) K | alpha ~ Uniform(0, (beta - alpha)/beta)
        # => bounding box => 0 <= K <= (beta - alpha)/beta
        # => log p(K|alpha) = - log( (beta - alpha)/beta )
        # only if alpha < beta => (beta - alpha) > 0
        # check:
        k_lp = -jnp.log(self.k_high - self.k_low)
        k_in_range = (K >= self.k_low) & (K <= self.k_high) & (alpha < self.beta)
        k_lp = jnp.where(k_in_range, k_lp, -jnp.inf)
        lp = lp + k_lp

        # ----------------------------
        # 4) c ~ Uniform(0,1)
        c_in_range = (c >= self.c_low) & (c <= self.c_high)
        # log p(c) = -log(1 - 0) = 0 for valid
        c_lp = jnp.where(c_in_range, 0.0, -jnp.inf)
        lp = lp + c_lp

        # ----------------------------
        # 5) p ~ Uniform(1,2)
        p_in_range = (p >= self.p_low) & (p <= self.p_high)
        # log p(p) = -log(2 - 1)= -log(1)=0, or -inf if out of range
        p_lp = jnp.where(p_in_range, 0.0, -jnp.inf)
        lp = lp + p_lp

        return lp

    def sample(self, key, sample_shape=()):
        """
        Generate samples. We'll do:
          mu ~ Gamma(0.1,10)
          alpha ~ Uniform(0,beta)
          K | alpha ~ Uniform(0, (beta - alpha)/beta)
          c ~ Uniform(0,1)
          p ~ Uniform(1,2)
        """
        # Flatten sample_shape
        n = 1
        for s in sample_shape:
            n *= s

        rng_mu, rng_alpha, rng_k, rng_c, rng_p = jax.random.split(key, 5)

        # 1) mu
        from numpyro.distributions import Gamma
        mu_dist = Gamma(self.mu_shape, self.mu_rate)
        mu_samples = mu_dist.sample(rng_mu, (n,))  # shape (n,)

        # 2) alpha
        alpha_samples = jax.random.uniform(rng_alpha, shape=(n,),
                                           minval=self.alpha_low,
                                           maxval=self.alpha_high)

        # 3) K, depends on alpha => in [0, (beta - alpha)/beta]
        # We'll define a helper
        def sample_k(rng_k_i, alpha_i):
            upper = (self.beta - alpha_i)/self.beta
            return jax.random.uniform(rng_k_i, minval=0.0, maxval=upper)

        rngs_k = jax.random.split(rng_k, n)
        K_samples = jax.vmap(sample_k)(rngs_k, alpha_samples)

        # 4) c ~ Uniform(0,1)
        c_samples = jax.random.uniform(rng_c, shape=(n,),
                                       minval=0.0, maxval=1.0)

        # 5) p ~ Uniform(1,2)
        p_samples = jax.random.uniform(rng_p, shape=(n,),
                                       minval=1.0, maxval=2.0)

        # Combine => shape (n,5)
        out = jnp.column_stack([mu_samples, alpha_samples, K_samples,
                                c_samples, p_samples])

        # Reshape => sample_shape + (5,)
        out = out.reshape(*sample_shape, 5)
        return out


def run_earthquake():
    np.random.seed(123)
    torch.manual_seed(321)
    true_mu = 4.42e-06
    true_alpha = 1.13
    true_k = 5.44e-04
    true_c = 0.169
    true_p = 1.182

    true_params = {'log10_mu': np.log10(true_mu), 'log10_k0': np.log10(true_k),
                   'a': true_alpha,
                   'log10_c': np.log10(true_c), 'rho': true_p}
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
    combine_ascii_files('./raw')
    raw_catalog = pd.read_csv("raw/SCEDC_catalog.csv")
    raw_catalog['time'] = pd.to_datetime(raw_catalog['time'])
    raw_catalog = raw_catalog.sort_values(by='time')
    raw_catalog = raw_catalog[["time", "longitude", "latitude","magnitude"]].dropna()
    raw_catalog.reset_index(drop=False, inplace=True)

    polygon_coords = np.load('data/SCEDC_shape.npy')

    poly = Polygon(polygon_coords)
    gdf = gpd.GeoDataFrame(
        raw_catalog,
        geometry=gpd.points_from_xy(
            raw_catalog.latitude, raw_catalog.longitude),)

    catalog = gdf[gdf.intersects(poly)].copy()
    catalog.drop("geometry", axis=1, inplace=True)
    len_trunc_x = len(catalog)
    print('Removed',len(raw_catalog)-len_trunc_x, 'events outside polygon')

    # filter events within timewindow
    auxiliary_start = '1985-01-01 00:00:00'
    test_nll_end = '2014-01-11 00:00:00'
    catalog = catalog[catalog['time']>=auxiliary_start]
    catalog = catalog[catalog['time']<test_nll_end]
    len_trunc_t = len(catalog)
    print('Removed',len_trunc_x-len_trunc_t, 'events outside timewindow')

    M_cut = 3.0
    catalog = catalog[catalog['magnitude']>=M_cut]
    len_trunc_m = len(catalog)
    print('Removed',len_trunc_t-len_trunc_m, 'events below Mcut')

    observed_summaries = sum_fn(catalog)

    # prior = ETASPrior()  # TODO
    prior = Uniform(low=jnp.array([0., 0., 0., 0.0, 1.0]),
                    high=jnp.array([0.0001, 2.4, 1.0, 1.0, 2.0]))
    prior = Independent(prior, 1)
    # prior = dist.Uniform(low=jnp.repeat(0.0, 5),
    #                      high=jnp.repeat(1.0, 5))
    # x_test = jnp.array([[0.0005, 0.1, 0.05, 0.5, 1.5]])
    # lp_val = prior.log_prob(x_test)

    model = get_standard_model
    sim_fn = earthquake_sim_fn
    summ_fn = sum_fn
    rng_key = random.PRNGKey(0)

    # TODO: just dummy values atm
    true_params = jnp.array([true_mu, true_alpha, true_k, true_c, true_p])

    # sim_test = sim_fn(rng_key, *true_params)
    # observed_summaries = summ_fn(sim_test)  # NOTE: overrides
    lp = prior.log_prob(true_params)

    def valid_fn(thetas):
        mu, a, k0, c, rho = thetas
        with open("data/config/SCEDC_30.json", 'r') as f:
            simulation_config = json.load(f)
        catalog_params = simulation_config["theta_0"].copy()

        try:
            params_dict = dict(
                {
                    'log10_mu': np.log10(mu),
                    'log10_k0': np.log10(k0),
                    'a': a,
                    'log10_c': np.log10(c),
                    'rho': rho
                }
            )  # Attempt conversion
            params_dict["a"] = params_dict["a"].item()
            params_dict['rho'] = params_dict['rho'].item()
        except Exception as e:
            print(e)

        catalog_params.update(params_dict)

        # shape_coords = "data/input_data/california_shape.npy"  # used to be simulation_config["shape_coords"]

        # np.random.seed(777)
        # Note: in four param model version, alpha ("a") fixed to beta
        if "a" not in catalog_params:
            catalog_params["a"] = simulation_config["beta"]

        if early_return(catalog_params, simulation_config["beta"]):
            return False
        else:
            return True

    v = valid_fn(true_params)

    # TODO: check sim and obs summaries are well and misspecified
    mcmc = run_snl(model, prior, sim_fn, summ_fn, rng_key,
                   observed_summaries,
                   num_sims_per_round=600,
                   num_rounds=5,
                   true_params=true_params,
                   theta_dims=5,
                   jax_parallelise=False,
                   num_chains=4,
                   only_valid_sims_in_first_round=True,
                   valid_fn=valid_fn
                   )
    mcmc.print_summary()
    # num_prior_samples = 10
    # prior_samples = prior.sample((num_prior_samples,))

    # num_summaries = 39
    # ssx = np.ones((num_prior_samples, num_summaries))
    # for i in range(num_prior_samples):
    #     print(f"Sample {i}, {prior_samples[i, :]}")
    #     catalog = earthquake_sim_fn(prior_samples[i, :])
    #     ssx[i, :] = sum_fn(catalog)
    #     print(ssx[i, :])


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    run_earthquake()
