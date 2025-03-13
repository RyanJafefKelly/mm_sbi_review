from mm_sbi_review.examples.turin import turin, compute_turin_summaries
import torch
import numpy as np
from sbi.inference import NLE
from sbi.utils import BoxUniform
from mm_sbi_review.examples.turin import TurinSummary
import matplotlib.pyplot as plt
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


def run_turin_snl():
    N = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_data_full = (
        torch.tensor(np.load("data/turin_obs.npy")).float().reshape(N, 801).to(device)
    )
    B = 4e9
    Ns = 801
    x_obs = compute_turin_summaries(x_data_full, delta_f=(B / (Ns - 1)))
    turin_sim = turin(B=B, Ns=801, N=N, tau0=0)

    def simulator(thetas):
        num_sims = thetas.shape[0]
        x_data_full = torch.zeros(num_sims, N, 801)
        x_sim = torch.zeros(num_sims, 12)
        for i in range(num_sims):
            x_data_full = turin_sim(thetas[i, :])
            x_sim[i, :] = compute_turin_summaries(x_data_full, delta_f=(B / (Ns - 1)))
        # x_sim = torch.squeeze(x_sim)
        return x_sim

    # test_theta = torch.tensor([10 ** (-8.4), 7.8e-9, 1e9, 2.8e-10])
    # x_sim = simulator(test_theta)

    prior = BoxUniform(
        low=torch.tensor([1e-9, 1e-9, 1e7, 1e-10]),
        high=torch.tensor([1e-8, 1e-8, 5e9, 1e-9]),
    )
    check_sbi_inputs(simulator, prior)

    inference = NLE(prior)
    proposal = prior
    num_rounds = 5
    num_sims = 500

    num_rounds = 3
    num_sims = 1_000
    for _ in range(num_rounds):
        theta = proposal.sample((num_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(x_obs)

    posterior_samples = posterior.sample((1000,), x=x_obs)
    posterior_samples = posterior_samples.cpu().numpy()
    np.save("data/posterior_samples.npy", posterior_samples)

    pass


if __name__ == "__main__":
    run_turin_snl()
