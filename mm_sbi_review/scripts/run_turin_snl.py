from mm_sbi_review.examples.turin import (
    turin,
    compute_turin_summaries,
    compute_turin_summaries_with_max,
)
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
    x_obs_max = compute_turin_summaries_with_max(x_data_full, delta_f=(B / (Ns - 1)))
    turin_sim = turin(B=B, Ns=801, N=N, tau0=0)

    def simulator(thetas):
        num_sims = thetas.shape[0]
        x_data_full = torch.zeros(num_sims, N, 801)
        x_sim = torch.zeros(num_sims, 9)
        for i in range(num_sims):
            x_data_full = turin_sim(thetas[i, :])
            x_sim[i, :] = compute_turin_summaries(x_data_full, delta_f=(B / (Ns - 1)))
        # x_sim = torch.squeeze(x_sim)
        return x_sim

    def simulator_with_max(thetas):
        num_sims = thetas.shape[0]
        x_data_full = torch.zeros(num_sims, N, 801)
        x_sim = torch.zeros(num_sims, 12)
        for i in range(num_sims):
            x_data_full = turin_sim(thetas[i, :])
            x_sim[i, :] = compute_turin_summaries_with_max(
                x_data_full, delta_f=(B / (Ns - 1))
            )
        # x_sim = torch.squeeze(x_sim)
        return x_sim

    # test_theta = torch.tensor([10 ** (-8.4), 7.8e-9, 1e9, 2.8e-10])
    # x_sim = simulator(test_theta)

    prior = BoxUniform(
        low=torch.tensor([1e-9, 1e-9, 1e7, 1e-10]),
        high=torch.tensor([1e-8, 1e-8, 5e9, 1e-9]),
    )
    check_sbi_inputs(simulator, prior)
    check_sbi_inputs(simulator_with_max, prior)

    inference = NLE(prior)
    inference_with_max = NLE(prior)

    proposal = prior
    proposal_max = prior
    num_rounds = 3
    num_sims = 1_000
    for round_i in range(num_rounds):
        x_data_full = torch.zeros(num_sims, N, 801)
        x_data_full_with_max = torch.zeros(num_sims, N, 801)
        if round_i == 0:
            theta = proposal.sample((num_sims,))
            theta_max = theta
        else:
            theta = proposal.sample((num_sims,))
            theta_max = proposal_max.sample((num_sims,))
        # theta = proposal.sample((num_sims,))
        num_sims = theta.shape[0]
        x_sim = torch.zeros(num_sims, 9)
        x_sim_max = torch.zeros(num_sims, 12)
        for i in range(num_sims):
            if i % 100 == 0:
                print(f"Simulating {i} out of {num_sims}")
            x_data_full = turin_sim(theta[i, :])
            if round_i == 0:
                x_data_full_with_max = x_data_full
            else:
                x_data_full_with_max = turin_sim(theta_max[i, :])
            x_sim[i, :] = compute_turin_summaries(x_data_full, delta_f=(B / (Ns - 1)))
            x_sim_max[i, :] = compute_turin_summaries_with_max(
                x_data_full_with_max, delta_f=(B / (Ns - 1))
            )
        # x = simulator(theta)
        # x_with_max = simulator_with_max(theta)

        _ = inference.append_simulations(theta, x_sim).train()
        _ = inference_with_max.append_simulations(theta, x_sim_max).train()
        posterior = inference.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal = posterior.set_default_x(x_obs)

        posterior_with_max = inference_with_max.build_posterior(
            mcmc_method="slice_np_vectorized",
            mcmc_parameters={"num_chains": 20, "thin": 5},
        )
        proposal_max = posterior_with_max.set_default_x(x_obs_max)

    posterior_samples = posterior.sample((1000,), x=x_obs)
    posterior_samples = posterior_samples.cpu().numpy()
    np.save("data/snl_posterior_samples.npy", posterior_samples)

    posterior_samples_with_max = posterior_with_max.sample((1000,), x=x_obs_max)
    posterior_samples_with_max = posterior_samples_with_max.cpu().numpy()
    np.save("data/snl_posterior_samples_with_max.npy", posterior_samples_with_max)

    pass


if __name__ == "__main__":
    run_turin_snl()
