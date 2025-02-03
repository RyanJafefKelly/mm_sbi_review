from mm_sbi_review.examples.earthquake import earthquake_sim_fn, sum_fn
from mm_sbi_review.scripts.utils import download_file, extract_tar_gz, combine_ascii_files
import numpy as np
import torch
from torch import tensor
from torch.distributions import Distribution, Uniform, Gamma
from etas.inversion import round_half_up, branching_ratio
from etas.simulation import generate_catalog, generate_background_events
import json
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd


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


class SBEtasPrior(Distribution):
    """SB-ETAS prior with sub-critical constraint K * beta < beta - alpha."""
    def check_branching_ratio(self, mu, K, alpha, c, p):
        mu_np = mu.cpu().numpy()
        K_np = K.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        c_np = c.cpu().numpy()
        p_np = p.cpu().numpy()

        mask_list = []
        for i in range(len(mu_np)):
            param_dict = {
                "log10_mu" : np.log10(mu_np[i]) if mu_np[i] > 0 else -np.inf,
                "log10_k0" : np.log10(K_np[i])  if K_np[i]  > 0 else -np.inf,
                "a"        : alpha_np[i],
                "log10_c"  : np.log10(c_np[i])  if c_np[i]  > 0 else -np.inf,
                "rho"      : p_np[i]
            }
            # Build final parameter array/dict as your `branching_ratio()` function expects:
            catalog_params = self.simulation_config["theta_0"].copy()
            catalog_params.update(param_dict)

            # Possibly your function needs a direct array:
            theta = parameter_dict2array(catalog_params)  # adjust for your usage

            # Evaluate branching ratio:
            print("Theta: ", theta)
            br = branching_ratio(theta, self.simulation_config["beta"])
            mask_list.append(bool(br < 1.1))
            if bool(br < 1.1):
                print(f"Branching ratio: {br}")
                print(f"mu: {mu_np[i]}, K: {K_np[i]}, alpha: {alpha_np[i]}, c: {c_np[i]}, p: {p_np[i]}")

        # Convert to torch bool tensor, same device as mu
        return torch.tensor(mask_list, dtype=torch.bool, device=mu.device)


    def __init__(self, beta: float = None):
        """
        :param beta: scalar value of beta to enforce sub-critical region K beta < beta^alpha
        """
        super().__init__()

        self.mu = Gamma(concentration=0.1, rate=10)

        # Low and high bounds for (mu, K, alpha, c, p):

        with open("data/config/SCEDC_30.json", 'r') as f:
            self.simulation_config = json.load(f)
            if beta is None:
                self.beta = self.simulation_config["beta"]
            else:
                self.beta = beta

        self.low = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.high = torch.tensor([1.0, self.beta, 1.0, 2.0])  # NOTE: changed p from high 10 to high 2, alpha to beta, BIG. c from 10 to 0.05, K to 1
        self._uniform = Uniform(self.low, self.high)


    # def log_prob(self, theta):
    #     """Returns log of prior density.
    #        If sub-critical constraint is violated, return -inf."""
    #     # Check if within the bounding box:
    #     in_range = torch.all((theta >= self.low) & (theta <= self.high), dim=-1)
    #     lp = torch.zeros(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
    #     lp[~in_range] = -torch.inf

    #     # For points in box, add uniform log-prob:
    #     box_lp = self._uniform.log_prob(theta).sum(dim=-1)

    #     # Enforce sub-critical condition: K * beta < beta-alpha
    #     mu = theta[..., 0]
    #     K = theta[..., 1]
    #     alpha = theta[..., 2]
    #     sub_crit = (K * self.beta) < (self.beta - alpha)

    #     # Combine
    #     lp[sub_crit] = box_lp[sub_crit]
    #     lp[~sub_crit] = -torch.inf
    #     return lp

    def sample(self, sample_shape=torch.Size()):
        """Naive rejection sampling. For large sample_shape,
        you might want a more efficient approach."""
        # Compute total number of samples n from sample_shape tuple.
        n = 1
        for s in sample_shape:
            n *= s

        samples = []
        # Collect valid samples until we have n:
        while len(samples) < n:
            # Oversample to reduce the number of loops:
            mu = self.mu.sample((2 * n,))
            candidate_uniform = self._uniform.sample((2 * n,))
            K = candidate_uniform[..., 0]
            alpha = candidate_uniform[..., 1]
            c = candidate_uniform[..., 2]
            p = candidate_uniform[..., 3]

            # Sub-critical mask: K * beta < beta - alpha
            # TODO! NEXT STEP RUN FUNCTION FOR EACH VALUE
            br_mask = self.check_branching_ratio(mu, K, alpha, c, p)
            mask = (((K * self.beta) < (self.beta - alpha)) & (mu < 0.0015))  # 0.002 somewhat arbitrary, done to set expected num background events low enough to run
            final_mask = mask & br_mask
            candidate = torch.column_stack([mu, candidate_uniform])
            valid = candidate[final_mask]
            samples.extend(valid)

        # Slice the first n valid samples, and reshape to match sample_shape
        samples = torch.stack(samples[:n], dim=0)
        samples = samples.view(*sample_shape, 5)  # 5 = number of parameters
        return samples


def run_earthquake():
    np.random.seed(123)
    torch.manual_seed(321)
    true_mu = 1.58e-06
    true_k = 0.0025
    true_alpha = 1.3
    true_c = 0.003
    true_p = 0.66  # or 0.66 or 1.05

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
    auxiliary_start = '1981-01-01 00:00:00'
    test_nll_end = '2020-01-17 00:00:00'
    catalog = catalog[catalog['time']>=auxiliary_start]
    catalog = catalog[catalog['time']<test_nll_end]
    len_trunc_t = len(catalog)
    print('Removed',len_trunc_x-len_trunc_t, 'events outside timewindow')

    M_cut = 3.0
    catalog = catalog[catalog['magnitude']>=M_cut]
    len_trunc_m = len(catalog)
    print('Removed',len_trunc_t-len_trunc_m, 'events below Mcut')

    observed_summaries = sum_fn(catalog)

    prior = SBEtasPrior()
    num_prior_samples = 10
    prior_samples = prior.sample((num_prior_samples,))
    # TODO: Summaries
    # TODO: real-data?
    num_summaries = 39
    ssx = np.ones((num_prior_samples, num_summaries))
    for i in range(num_prior_samples):
        print(f"Sample {i}, {prior_samples[i, :]}")
        catalog = earthquake_sim_fn(prior_samples[i, :])
        ssx[i, :] = sum_fn(catalog)
        print(ssx[i, :])


if __name__ == "__main__":
    run_earthquake()
