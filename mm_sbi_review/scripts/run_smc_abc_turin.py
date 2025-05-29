import numpy as np
import elfi
from mm_sbi_review.examples.turin_elfi import build_turin_elfi
import pickle as pkl


def run_smc_abc_turin():
    """
    Run SMC ABC inference on the Turin dataset using ELFI.
    This function loads the observed power-delay profile data, builds the ELFI model,
    and performs inference using adaptive SMC.
    """
    # 1.  Load *observed* power-delay profile you already simulated with torch/JAX
    y_obs = np.load("data/turin_obs.npy")  # shape (N, Ns) = (100, 801)
    # elfi.set_client(elfi.clients.multiprocessing.Client(num_processes=3))
    # elfi.set_client("multiprocessing", num_processes=3)  # optional, for parallelism

    dirname = ""

    # 2.  Build the model
    m = build_turin_elfi(y_obs)

    # TODO: debug with script .py

    # 3.  Inference: adaptive SMC
    smc = elfi.AdaptiveDistanceSMC(m["d"], batch_size=10, seed=0)
    posterior_samples = smc.sample(200, rounds=3, bar=True)
    print(posterior_samples)

    with open(dirname + "adaptive_smc_samples.pkl", "wb") as f:
        pkl.dump(posterior_samples.samples_array, f)


if __name__ == "__main__":
    run_smc_abc_turin()
