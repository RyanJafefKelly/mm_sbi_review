"""from robust-sbi package"""

import torch


from mm_sbi_review.examples.turin import turin, TurinSummary

# from sbi import SNPE, BoxUniform
from pyro.distributions import Uniform

from sbi.inference import SNPE_C as SNPE
from mm_sbi_review.utils.user_input_checks import *
from sbi.simulators.simutils import simulate_in_batches
import pickle
import os
import argparse
import numpy as np

# from networks.flow import build_made, build_maf, build_nsf
from torch import Tensor, nn, relu, tanh, tensor, uint8
from typing import Callable, Optional
from sbi.neural_nets.net_builders.flow import build_maf, build_nsf


def posterior_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the posterior.

    This function will usually be used for SNPE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                embedding_net,
                num_components,
            ),
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model == "made":
            return build_made(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf":
            return build_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    return build_fn


def simulate_for_sbi(
    simulator: Callable,
    proposal: Any,
    num_simulations: int,
    num_workers: int = 1,
    simulation_batch_size: int = 1,
    show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.

    This function performs two steps:

    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$.

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel workers to use for simulations.
        simulation_batch_size: Number of parameter sets that the simulator
            maps to data x at once. If None, we simulate all parameter sets at the
            same time. If >= 1, the simulator has to process data of shape
            (simulation_batch_size, parameter_dimension).
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """

    theta = proposal.sample((num_simulations,))

    x = simulate_in_batches(
        simulator, theta, simulation_batch_size, num_workers, show_progress_bar
    )

    return theta, x


def run_turin_npers(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    distance = args.distance
    beta = args.beta
    num_simulations = args.num_simulations
    # theta_gt = args.theta
    N = args.N
    seed = args.seed

    task_name = f"{distance}_beta={beta}_num={num_simulations}_N={N}_tau0/{seed}"
    root_name = "objects/turin/" + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [
        Uniform(1e-9 * torch.ones(1).to(device), 1e-8 * torch.ones(1).to(device)),
        Uniform(1e-9 * torch.ones(1).to(device), 1e-8 * torch.ones(1).to(device)),
        Uniform(1e7 * torch.ones(1).to(device), 5e9 * torch.ones(1).to(device)),
        Uniform(1e-10 * torch.ones(1).to(device), 1e-9 * torch.ones(1).to(device)),
    ]

    simulator, prior = prepare_for_sbi(turin(B=4e9, Ns=801, N=100, tau0=0), prior)

    sum_net = TurinSummary(input_size=1, hidden_dim=4, N=N).to(device)
    neural_posterior = posterior_nn(
        model="maf", embedding_net=sum_net, hidden_features=100, num_transforms=5
    )

    inference = SNPE(
        prior=prior, density_estimator=neural_posterior, device=str(device)
    )

    x_obs = (
        torch.tensor(np.load("data/turin_obs.npy"))
        .float()
        .reshape(-1, N, 801)
        .to(device)
    )

    if args.pre_generated_sim:
        theta = torch.tensor(np.load("data/turin_theta_2000_tau0.npy"))
        x = torch.tensor(np.load("data/turin_x_2000_tau0.npy")).reshape(
            num_simulations, N, 801
        )
    else:
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)
    x = x.to(device)
    theta = theta.to(device)

    inference = inference.append_simulations(theta, x.unsqueeze(1))
    density_estimator = inference.train()

    prior_new = [
        Uniform(1e-10 * torch.ones(1).to(device), 1e-7 * torch.ones(1).to(device)),
        Uniform(1e-10 * torch.ones(1).to(device), 1e-7 * torch.ones(1).to(device)),
        Uniform(1e6 * torch.ones(1).to(device), 1e10 * torch.ones(1).to(device)),
        Uniform(1e-11 * torch.ones(1).to(device), 1e-8 * torch.ones(1).to(device)),
    ]

    simulator, prior_new = prepare_for_sbi(
        turin(B=4e9, Ns=801, N=100, tau0=0), prior_new
    )
    posterior_new = inference.build_posterior(density_estimator, prior=prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
    with open(root_name + "/posterior_new.pkl", "wb") as handle:
        pickle.dump(posterior_new, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0, help="regularization weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--distance", type=str, default="mmd", choices=["euclidean", "none", "mmd"]
    )
    parser.add_argument(
        "--num_simulations", type=int, default=500, help="number of simulations"
    )
    parser.add_argument(
        "--theta", type=list, default=[10 ** (-8.4), 7.8e-9, 1e9, 2.8e-10]
    )
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument(
        "--pre-generated-sim",
        action="store_true",
        help="generate simulation data online or not",
    )
    args = parser.parse_args()
    run_turin_npers(args)
