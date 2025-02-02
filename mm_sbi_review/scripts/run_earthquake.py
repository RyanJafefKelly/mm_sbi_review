from mm_sbi_review.examples.earthquake import earthquake_sim_fn
import numpy as np
import torch
from torch import tensor
from torch.distributions import Distribution, Uniform, Gamma
from etas.inversion import round_half_up, branching_ratio
from etas.simulation import generate_catalog, generate_background_events
import json

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
    """SB-ETAS prior with sub-critical constraint K * beta < beta^alpha."""
    def check_branching_ratio(self, mu, K, alpha, c, p):
        with open("data/config/SCEDC_30.json", 'r') as f:
            simulation_config = json.load(f)
        param_dict = {"log10_mu": np.log10(mu), "log10_k0": np.log10(K),
                      "a": alpha,
                      "log10_c": np.log10(c),
                      "rho": p,
                      }
        catalog_params = simulation_config["theta_0"].copy()
        catalog_params.update(param_dict)
        theta = parameter_dict2array(param_dict)
        br = branching_ratio(theta, simulation_config["beta"])
        return br < 1.1  # TODO: EXPERIMENTAL - set arbitrarily

    
    def __init__(self, beta: float):
        """
        :param beta: scalar value of beta to enforce sub-critical region K beta < beta^alpha
        """
        super().__init__()
        self.beta = torch.tensor(beta)

        self.mu = Gamma(concentration=0.1, rate=10)

        # Low and high bounds for (mu, K, alpha, c, p):
        self.low = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.high = torch.tensor([1.0, beta, 1.0, 2.0])  # NOTE: changed p from high 10 to high 2, alpha to beta, BIG. c from 10 to 0.05, K to 1

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
            K = candidate_uniform[..., 1]
            alpha = candidate_uniform[..., 2]

            # Sub-critical mask: K * beta < beta - alpha
            # TODO! NEXT STEP RUN FUNCTION FOR EACH VALUE
            br_bool = self.check_branching_ratio(mu, K, alpha, candidate_uniform[..., 2], candidate_uniform[..., 3])
            mask = (((K * self.beta) < (self.beta - alpha)) & (mu < 0.002)) & br_bool  # 0.002 somewhat arbitrary, done to set expected num background events low enough to run
            candidate = torch.column_stack([mu, candidate_uniform])
            valid = candidate[mask]
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
    beta = 2.302585092994046
    # beta = 2.4
    prior = SBEtasPrior(beta=beta)
    num_prior_samples = 1_000
    prior_samples = prior.sample((num_prior_samples,))
    # TODO: Summaries
    # TODO: real-data?
    for i in range(num_prior_samples):
        print(f"Sample {i}, {prior_samples[i, :]}")
        obs_data = earthquake_sim_fn(prior_samples[i, :])




if __name__ == "__main__":
    run_earthquake()
