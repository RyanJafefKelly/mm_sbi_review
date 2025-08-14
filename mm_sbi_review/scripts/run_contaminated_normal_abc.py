import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


# --- summary statistics ---------------------------------------------------- #
def calc_summary(x):
    """Return mean and unbiased variance of a 1‑D sample as a 2‑vector."""
    return np.array([x.mean(), x.var(ddof=1)])


def block_summary(x, block_size=10):
    """Split sample into blocks and compute (mean,var) per block."""
    n_blocks = len(x) // block_size
    x = x[: n_blocks * block_size].reshape(n_blocks, block_size)
    return np.vstack([calc_summary(block) for block in x])


# --- distances ------------------------------------------------------------- #
def median_heuristic(X):
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)).ravel()
    return np.sqrt(np.median(d) / 2.0)


def unbiased_mmd(npe_posterior_samples, exact_posterior_samples, lengthscale: int = 1):
    m = npe_posterior_samples.shape[0]
    n = exact_posterior_samples.shape[0]

    xx = np.sum(npe_posterior_samples**2, axis=1)[:, None]
    yy = np.sum(exact_posterior_samples**2, axis=1)[None, :]
    xy = np.dot(npe_posterior_samples, exact_posterior_samples.T)

    k_simulated = np.exp(
        -(xx + xx.T - 2 * np.dot(npe_posterior_samples, npe_posterior_samples.T))
        / (2 * lengthscale**2)
    )
    k_obs = np.exp(
        -(yy + yy.T - 2 * np.dot(exact_posterior_samples, exact_posterior_samples.T))
        / (2 * lengthscale**2)
    )
    k_sim_obs = np.exp(-(xx + yy - 2 * xy) / (2 * lengthscale**2))

    k_simulated[np.diag_indices(m)] = 0
    k_obs[np.diag_indices(n)] = 0

    k_sim_term = np.sum(k_simulated) / (m * (m - 1))
    k_obs_term = np.sum(k_obs) / (n * (n - 1))
    k_sim_obs_term = -2 * np.sum(k_sim_obs) / (m * n)

    mmd_value = k_sim_term + k_obs_term + k_sim_obs_term

    return mmd_value


def kl_1nn(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, d = X.shape
    m = Y.shape[0]
    tree_X = KDTree(X)
    tree_Y = KDTree(Y)
    r = tree_X.query(X, k=2, eps=0.01, p=2)[0][:, 1]
    s = tree_Y.query(X, k=1, eps=0.01, p=2)[0]
    return (-np.log(r / s).sum() * d / n) + np.log(m / (n - 1.0))


# --- contaminated normal model -------------------------------------------- #
def true_dgp(theta, stdev_err=10.0, n_obs=100):
    w = 0.8
    sd = np.random.choice([1.0, stdev_err], size=n_obs, p=[w, 1 - w])
    return np.random.normal(0, 1, n_obs) * sd + theta


def assumed_dgp(theta, n_obs=100):
    return np.random.normal(theta, 1.0, n_obs)


# -------------------------------------------------------------------------- #
np.random.seed(0)
theta_true = 2.0
n_obs = 100
block_size = 10
obs = true_dgp(theta_true, 2.0, n_obs)
# S_obs = block_summary(obs, block_size=block_size)

# ell = median_heuristic(S_obs)

# analytic posterior
prior_var = 100.0
true_post_var = 1 / (1 / prior_var + n_obs / 1)
true_post_mu = true_post_var * (obs.mean() * n_obs)
true_post_std = np.sqrt(true_post_var)

# ABC
N = 100_000
thetas = np.random.normal(0, 3, N)
kl_dist = np.empty(N)
mmd_dist = np.empty(N)

# calcule median heuristic
test_thetas = np.random.normal(0, 3, 1000)
assumed_sims = np.array([assumed_dgp(th, n_obs) for th in test_thetas])
# S_test = [block_summary(sim, block_size) for sim in assumed_sims]
S_test = np.vstack(assumed_sims)
ell = median_heuristic(np.vstack([obs, assumed_sims]))

for i, th in enumerate(thetas):
    sim = assumed_dgp(th, n_obs)
    # S_sim = block_summary(sim, block_size)
    kl_dist[i] = kl_1nn(obs, sim)
    ell = median_heuristic(np.vstack([obs, sim]))
    mmd_dist[i] = unbiased_mmd(sim, obs, ell)

eps_kl = np.quantile(kl_dist, 0.001)
eps_mmd = np.quantile(mmd_dist, 0.001)
post_kl = thetas[kl_dist < eps_kl]
post_mmd = thetas[mmd_dist < eps_mmd]

print("KL accepted", post_kl.size, "MMD accepted", post_mmd.size)

# plot
xs = np.linspace(
    true_post_mu - 4 * true_post_std, true_post_mu + 4 * true_post_std, 300
)
true_pdf = (1 / (true_post_std * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((xs - true_post_mu) / true_post_std) ** 2
)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax[0].hist(
    post_kl,
    bins=30,
    density=True,
    alpha=0.7,
    color="#4C72B0",
    edgecolor="white",
    label="KL‑ABC",
)
ax[0].plot(xs, true_pdf, color="#333333", lw=2, label="Analytical")
ax[0].axvline(
    2.0,
    color="#333333",
    lw=2,
    ls="--",
    label=r"$\mu_{\mathrm{post}}$",
)
ax[0].set_title("KL distance (summary blocks)")
ax[0].set_xlabel(r"$\theta$")
ax[0].set_ylabel("Density")
ax[0].legend(frameon=False)

ax[1].hist(
    post_mmd,
    bins=30,
    density=True,
    alpha=0.7,
    color="#DD8452",
    edgecolor="white",
    label="MMD‑ABC",
)
ax[1].plot(xs, true_pdf, color="#333333", lw=2, label="Analytical")
ax[1].axvline(
    2.0,
    color="#333333",
    lw=2,
    ls="--",
    label=r"$\mu_{\mathrm{post}}$",
)
ax[1].set_title("MMD distance (summary blocks)")
ax[1].set_xlabel(r"$\theta$")
ax[1].legend(frameon=False)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.savefig("abc_summary_block_comparison.pdf", bbox_inches="tight")
plt.show()
