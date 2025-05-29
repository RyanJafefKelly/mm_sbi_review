# turin_elfi.py  – minimal NumPy/ELFI port
import elfi
import numpy as np
from functools import partial


# ---------------------------------------------------------------------
# 1.  Low-level simulator (unchanged math, no torch / jax)
# ---------------------------------------------------------------------
def _turin_single(G0, T, lambda_0, sigma2_N, *, B, Ns, N, tau0, rng):
    """Simulate ONE realisation (all inputs are scalars)."""
    delta_f = B / (Ns - 1)
    t_max = 1.0 / delta_f

    H = np.zeros((N, Ns), dtype=np.complex128)
    mu = lambda_0 * t_max

    for j in range(N):
        n_pts = rng.poisson(mu)
        delays = np.sort(rng.uniform(0.0, t_max, n_pts))

        sigma2 = G0 * np.exp(-delays / T) / lambda_0 * B
        alpha = rng.normal(0, np.sqrt(sigma2 / 2)) + 1j * rng.normal(
            0, np.sqrt(sigma2 / 2)
        )

        H[j] = (
            np.exp(-1j * 2 * np.pi * delta_f * np.outer(np.arange(Ns), delays)) @ alpha
        )

    noise = rng.normal(0, np.sqrt(sigma2_N / 2), size=(N, Ns)) + 1j * rng.normal(
        0, np.sqrt(sigma2_N / 2), size=(N, Ns)
    )

    Y = H + noise
    y = np.fft.ifft(Y, axis=1)
    p = np.abs(y) ** 2
    return 10.0 * np.log10(p)


# ---------------------------------------------------------------------
# 2.  ELFI-compliant *batch* simulator
# ---------------------------------------------------------------------
def turin_simulator(
    G0,
    T,
    lambda_0,
    sigma2_N,
    *,
    B=4e9,
    Ns=801,
    N=100,
    tau0=0,
    batch_size=1,
    random_state=None
):
    """
    Parameters come in as 0-D (scalar) **or** 1-D (batch,) arrays.
    We iterate over the batch dimension and call _turin_single
    with scalar arguments.
    """
    rng = random_state or np.random

    # broadcast every parameter to 1-D array of length batch_size
    def to_1d(x):
        x = np.asarray(x)
        return np.full(batch_size, x) if x.ndim == 0 else x

    G0, T, lambda_0, sigma2_N = map(to_1d, (G0, T, lambda_0, sigma2_N))

    out = []
    for i in range(batch_size):
        out.append(
            _turin_single(
                G0[i],
                T[i],
                lambda_0[i],
                sigma2_N[i],
                B=B,
                Ns=Ns,
                N=N,
                tau0=tau0,
                rng=rng,
            )
        )

    # ELFI expects (batch, output_dim)
    return np.asarray(out).reshape(batch_size, -1)


# ---------------------------------------------------------------------
# 2.  Summary statistic (NumPy, vectorised over batch)
# ---------------------------------------------------------------------
def turin_summaries(flat_power_db, delta_f, N=100, Ns=801):
    """Batch-wise replica of `compute_turin_summaries`, but in NumPy.

    Parameters
    ----------
    flat_power_db : (batch, N*Ns) array  – output of `turin_simulator`
    delta_f       : scalar               – spacing = B/(Ns-1)
    """
    batch = flat_power_db.shape[0]

    # ---- reshape to (batch, N, Ns) and convert dB ➜ linear --------------
    power = 10.0 ** (flat_power_db / 10.0).reshape(batch, N, Ns)

    # ---- time grid ------------------------------------------------------
    t_max = 1.0 / delta_f
    t = np.linspace(0.0, t_max, Ns)  # (Ns,)
    dt = t[1] - t[0] if Ns > 1 else t_max

    # ---- raw moments per replica ---------------------------------------
    m0 = np.sum(power * dt, axis=2)  # (batch, N)
    m1 = np.sum(power * t * dt, axis=2)
    m2 = np.sum(power * t**2 * dt, axis=2)

    # optional scaling (kept from original Torch version)
    scale = 1e9
    m0 *= scale
    m1 *= scale
    m2 *= scale

    # ---- sample means across replicas ----------------------------------
    mean_m0 = m0.mean(axis=1)  # (batch,)
    mean_m1 = m1.mean(axis=1)
    mean_m2 = m2.mean(axis=1)

    # ---- sample covariances (unbiased, divisor N-1) ---------------------
    def sample_cov(x, y):
        # x, y : (batch, N)
        mean_x = x.mean(axis=1, keepdims=True)
        mean_y = y.mean(axis=1, keepdims=True)
        num = np.sum((x - mean_x) * (y - mean_y), axis=1)  # (batch,)
        denom = max(N - 1, 1)  # handle N=1
        return num / denom

    cov_m0_m0 = sample_cov(m0, m0)
    cov_m0_m1 = sample_cov(m0, m1)
    cov_m0_m2 = sample_cov(m0, m2)
    cov_m1_m1 = sample_cov(m1, m1)
    cov_m1_m2 = sample_cov(m1, m2)
    cov_m2_m2 = sample_cov(m2, m2)

    # ---- power extremes (max / median / min over time grid) -------------
    # First average over replicas, then take extrema over Ns
    power_avg = power.mean(axis=1)  # (batch, Ns)
    max_power = power_avg.max(axis=1)
    median_power = np.median(power_avg, axis=1)
    min_power = power_avg.min(axis=1)

    # ---- assemble summary vector ---------------------------------------
    eps = 1e-30
    summaries = np.stack(
        [
            mean_m0,
            mean_m1,
            mean_m2,
            max_power,
            median_power,
            min_power,
            np.log(cov_m0_m0 + eps),
            np.log(cov_m0_m1 + eps),
            np.log(cov_m0_m2 + eps),
            np.log(cov_m1_m1 + eps),
            np.log(cov_m1_m2 + eps),
            np.log(cov_m2_m2 + eps),
        ],
        axis=1,  # (batch, 12)
    )

    return summaries


# ---------------------------------------------------------------------
# 3.  Convenience builder for an ELFI model
# ---------------------------------------------------------------------
def build_turin_elfi(observed_power_db, B=4e9, Ns=801, N=100, tau0=0):
    """Create an `elfi.ElfiModel` with Turin simulator & summaries."""
    delta_f = B / (Ns - 1)
    m = elfi.new_model()

    # --- priors (use the same ranges as in the NPE example) ------------
    G0 = elfi.Prior("uniform", 1e-9, 9e-9, model=m, name="G0")
    T = elfi.Prior("uniform", 1e-9, 9e-9, model=m, name="T")
    lambda_0 = elfi.Prior("uniform", 1e7, 9.9e9, model=m, name="lambda_0")
    sigma2_N = elfi.Prior("uniform", 1e-10, 9e-10, model=m, name="sigma2_N")

    # --- simulator -----------------------------------------------------
    sim_fn = partial(turin_simulator, B=B, Ns=Ns, N=N, tau0=tau0)
    elfi.Simulator(
        sim_fn,
        G0,
        T,
        lambda_0,
        sigma2_N,
        observed=observed_power_db.reshape(1, -1),  # ELFI wants (1, dim)
        name="TurinSim",
    )

    # --- summary & distance -------------------------------------------
    sum_fn = partial(turin_summaries, delta_f=delta_f, N=N, Ns=Ns)
    S = elfi.Summary(sum_fn, m["TurinSim"], name="S")
    elfi.AdaptiveDistance(S, name="d")  # TODO: test different distance choice

    return m
