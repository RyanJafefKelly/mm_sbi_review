"""Copied across from robust-sbi.

See https://github.com/huangdaolang/robust-sbi/blob/main/simulators/turin.py
"""

import torch
import numpy as np
import torch.nn as nn


class turin:
    def __init__(self, B=4e9, Ns=801, N=100, tau0=0):
        self.B = B
        self.Ns = Ns
        self.N = N
        self.tau0 = tau0

    def __call__(self, theta, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(theta.shape) == 1:
            G0 = theta[0].to(device)
            T = theta[1].to(device)
            lambda_0 = theta[2].to(device)
            sigma2_N = theta[3].to(device)
        else:
            G0 = theta[0, 0].to(device)
            T = theta[0, 1].to(device)
            lambda_0 = theta[0, 2].to(device)
            sigma2_N = theta[0, 3].to(device)

        sigma2_N = sigma2_N if sigma2_N > 0 else torch.tensor(1e-10)

        nRx = self.N

        delta_f = self.B / (self.Ns - 1)  # Frequency step size
        t_max = 1 / delta_f

        tau = torch.linspace(0, t_max, self.Ns)

        H = torch.zeros((nRx, self.Ns), dtype=torch.cfloat)
        if lambda_0 < 0:
            lambda_0 = torch.tensor(1e7)
        mu_poisson = lambda_0 * t_max  # Mean of Poisson process

        for jR in range(nRx):

            n_points = int(
                torch.poisson(mu_poisson)
            )  # Number of delay points sampled from Poisson process

            delays = (torch.rand(n_points) * t_max).to(
                device
            )  # Delays sampled from a 1-dimensional Poisson point process

            delays = torch.sort(delays)[0]

            alpha = torch.zeros(n_points, dtype=torch.cfloat).to(
                device
            )  # Initialising vector of gains of length equal to the number of delay points

            sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * self.B

            for l in range(n_points):
                if delays[l] < self.tau0:
                    alpha[l] = 0
                else:
                    std = (
                        torch.sqrt(sigma2[l] / 2)
                        if torch.sqrt(sigma2[l] / 2) > 0
                        else torch.tensor(1e-7)
                    )
                    alpha[l] = torch.normal(0, std) + torch.normal(0, std) * 1j

            H[jR, :] = torch.matmul(
                torch.exp(
                    -1j
                    * 2
                    * torch.pi
                    * delta_f
                    * (torch.ger(torch.arange(self.Ns), delays))
                ),
                alpha,
            )

        # Noise power by setting SNR
        Noise = torch.zeros((nRx, self.Ns), dtype=torch.cfloat).to(device)

        for j in range(nRx):
            normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))
            Noise[j, :] = normal.sample([self.Ns]) + normal.sample([self.Ns]) * 1j

        # Received signal in frequency domain

        Y = H + Noise

        y = torch.zeros(Y.shape, dtype=torch.cfloat).to(device)
        p = torch.zeros(Y.shape).to(device)
        lens = len(Y[:, 0])

        for i in range(lens):
            y[i, :] = torch.fft.ifft(Y[i, :])

            p[i, :] = torch.abs(y[i, :]) ** 2

        return 10 * torch.log10(p)

    def get_name(self):
        return "turin"


# Numpy implementation
def TurinModel(G0, T, lambda_0, sigma2_N, B=4e9, Ns=801, N=100, tau0=0):

    nRx = N

    delta_f = B / (Ns - 1)  # Frequency step size
    t_max = 1 / delta_f

    tau = np.linspace(0, t_max, Ns)

    H = np.zeros((nRx, Ns), dtype=complex)

    mu_poisson = lambda_0 * t_max  # Mean of Poisson process

    for jR in range(nRx):

        n_points = np.random.poisson(
            mu_poisson
        )  # Number of delay points sampled from Poisson process

        delays = np.random.uniform(
            0, t_max, n_points
        )  # Delays sampled from a 1-dimensional Poisson point process

        delays = np.sort(delays)

        alpha = np.zeros(
            n_points, dtype=complex
        )  # Initialising vector of gains of length equal to the number of delay points

        sigma2 = G0 * np.exp(-delays / T) / lambda_0 * B

        for l in range(n_points):
            alpha[l] = (
                np.random.normal(0, np.sqrt(sigma2[l] / 2))
                + np.random.normal(0, np.sqrt(sigma2[l] / 2)) * 1j
            )

        H[jR, :] = (
            np.exp(-1j * 2 * np.pi * delta_f * (np.outer(np.arange(Ns), delays)))
            @ alpha
        )

    # Noise power by setting SNR
    Noise = np.zeros((nRx, Ns), dtype=complex)

    for j in range(nRx):
        Noise[j, :] = (
            np.random.normal(0, np.sqrt(sigma2_N / 2), Ns)
            + np.random.normal(0, np.sqrt(sigma2_N / 2), Ns) * 1j
        )

    # Received signal in frequency domain

    Y = H + Noise

    return Y


class TurinSummary(nn.Module):
    def __init__(self, input_size, hidden_dim, N):
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = 1
        self.lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_size, 8, 3, 3),
            nn.Conv1d(8, 16, 3, 3),
            nn.Conv1d(16, 32, 3, 3),
            nn.Conv1d(32, 64, 3, 3),
            nn.Conv1d(64, 8, 3, 3),
            nn.AvgPool1d(2),
        )

    def forward(self, Y):
        current_device = Y.device
        batch_size = Y.size(0)

        embeddings_conv = self.conv(Y.reshape(-1, 1, 801)).reshape(-1, self.N, 8)

        stat_conv = torch.mean(embeddings_conv, dim=1)

        return embeddings_conv, stat_conv

    def init_hidden(self, batch_size, current_device):
        hidden = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(
            current_device
        )
        c = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(
            current_device
        )
        return hidden, c


import torch


def compute_turin_summaries(power_data, delta_f):
    """
    Computes:
      - Three temporal moments (m0, m1, m2) for each realisation
      - The sample means of each moment
      - The pairwise covariances of these moments

    Args:
        power_data (torch.Tensor): shape (N_r, N_t),
            where N_r is number of realisations (or Rx)
            and N_t is number of time samples.
            power_data[i, :] = |y^i(t)|^2 in linear scale (not in dB).
        delta_f (float): frequency spacing used for the simulation
                         => t_max = 1 / delta_f

    Returns:
        summary_vector (torch.Tensor): length 10, containing
            [mean_m1, mean_m2, cov(m0,m0), cov(m0,m1), cov(m0,m2),
             cov(m1,m1), cov(m1,m2), cov(m2,m2)]
    """

    device = power_data.device
    N_r, N_t = power_data.shape

    # power_data = 10.0 ** (power_data / 10.0)

    # Time grid from 0 to t_max
    t_max = 1.0 / delta_f
    t = torch.linspace(0.0, t_max, N_t, device=device)
    # For a simple Riemann sum, define dt
    dt = t[1] - t[0] if N_t > 1 else t_max

    # 1) Compute the integrals (moments m_l^i) for l = 0,1,2.
    #    m_l^i = \int_0^{t_max} t^l |y^i(t)|^2 dt
    #    Approx. with discrete sum: sum_{k=0}^{N_t-1} [t[k]^l * power_data[i,k]] * dt
    m0 = torch.sum(power_data * dt, dim=1)  # l = 0
    m1 = torch.sum(t[None, :] * power_data * dt, dim=1)  # l = 1
    m2 = torch.sum((t[None, :] ** 2) * power_data * dt, dim=1)  # l = 2

    # scale_factor = 1e30  # NOTE: CHECK OKAY, VALUES TOO SMALL CAUSING ERRORS
    scale_factor = 1e9
    m0 = m0 * scale_factor
    m1 = m1 * scale_factor
    m2 = m2 * scale_factor

    # 2) Sample means across realisations:
    mean_m0 = torch.mean(m0)
    mean_m1 = torch.mean(m1)
    mean_m2 = torch.mean(m2)

    # 3) Sample covariances among (m0, m1, m2).
    #    cov(x, y) = (1/(N_r-1)) * sum( (x_i - mean_x)(y_i - mean_y) )
    def sample_cov(x, y):
        return torch.sum((x - torch.mean(x)) * (y - torch.mean(y))) / (N_r - 1)

    cov_m0_m0 = sample_cov(m0, m0)
    cov_m0_m1 = sample_cov(m0, m1)
    cov_m0_m2 = sample_cov(m0, m2)
    cov_m1_m1 = sample_cov(m1, m1)
    cov_m1_m2 = sample_cov(m1, m2)
    cov_m2_m2 = sample_cov(m2, m2)

    # power_data_avg = torch.mean(power_data, dim=0)
    # max_power = torch.max(power_data_avg)
    # median_power = torch.median(power_data_avg)
    # min_power = torch.min(power_data_avg)

    # 5) Construct the 10-element summary vector
    summary_vector = torch.tensor(
        [
            mean_m0.item(),  # 2) bar{m0}
            mean_m1.item(),  # 3) bar{m1}
            mean_m2.item(),  # 4) bar{m2}
            # max_power.item(),
            # median_power.item(),
            # min_power.item(),
            cov_m0_m0.item(),  # 5) cov(m0,m0)
            cov_m0_m1.item(),  # 6) cov(m0,m1)
            cov_m0_m2.item(),  # 7) cov(m0,m2)
            cov_m1_m1.item(),  # 8) cov(m1,m1)
            cov_m1_m2.item(),  # 9) cov(m1,m2)
            cov_m2_m2.item(),  # 10) cov(m2,m2)
        ],
        device=device,
    )

    eps = 1e-30
    summary_vector[6:] = torch.log(summary_vector[6:] + eps)  # log-scale cov summaries
    return summary_vector


def compute_turin_summaries_with_max(power_data, delta_f):
    """
    Computes:
      - Three temporal moments (m0, m1, m2) for each realisation
      - The sample means of each moment
      - The pairwise covariances of these moments

    Args:
        power_data (torch.Tensor): shape (N_r, N_t),
            where N_r is number of realisations (or Rx)
            and N_t is number of time samples.
            power_data[i, :] = |y^i(t)|^2 in linear scale (not in dB).
        delta_f (float): frequency spacing used for the simulation
                         => t_max = 1 / delta_f

    Returns:
        summary_vector (torch.Tensor): length 10, containing
            [mean_m1, mean_m2, cov(m0,m0), cov(m0,m1), cov(m0,m2),
             cov(m1,m1), cov(m1,m2), cov(m2,m2)]
    """

    device = power_data.device
    N_r, N_t = power_data.shape

    # power_data = 10.0 ** (power_data / 10.0)

    # Time grid from 0 to t_max
    t_max = 1.0 / delta_f
    t = torch.linspace(0.0, t_max, N_t, device=device)
    # For a simple Riemann sum, define dt
    dt = t[1] - t[0] if N_t > 1 else t_max

    # 1) Compute the integrals (moments m_l^i) for l = 0,1,2.
    #    m_l^i = \int_0^{t_max} t^l |y^i(t)|^2 dt
    #    Approx. with discrete sum: sum_{k=0}^{N_t-1} [t[k]^l * power_data[i,k]] * dt
    m0 = torch.sum(power_data * dt, dim=1)  # l = 0
    m1 = torch.sum(t[None, :] * power_data * dt, dim=1)  # l = 1
    m2 = torch.sum((t[None, :] ** 2) * power_data * dt, dim=1)  # l = 2

    # scale_factor = 1e30  # NOTE: CHECK OKAY, VALUES TOO SMALL CAUSING ERRORS
    scale_factor = 1e9
    m0 = m0 * scale_factor
    m1 = m1 * scale_factor
    m2 = m2 * scale_factor

    # 2) Sample means across realisations:
    mean_m0 = torch.mean(m0)
    mean_m1 = torch.mean(m1)
    mean_m2 = torch.mean(m2)

    # 3) Sample covariances among (m0, m1, m2).
    #    cov(x, y) = (1/(N_r-1)) * sum( (x_i - mean_x)(y_i - mean_y) )
    def sample_cov(x, y):
        return torch.sum((x - torch.mean(x)) * (y - torch.mean(y))) / (N_r - 1)

    cov_m0_m0 = sample_cov(m0, m0)
    cov_m0_m1 = sample_cov(m0, m1)
    cov_m0_m2 = sample_cov(m0, m2)
    cov_m1_m1 = sample_cov(m1, m1)
    cov_m1_m2 = sample_cov(m1, m2)
    cov_m2_m2 = sample_cov(m2, m2)

    power_data_avg = torch.mean(power_data, dim=0)
    max_power = torch.max(power_data_avg)
    median_power = torch.median(power_data_avg)
    min_power = torch.min(power_data_avg)

    # 5) Construct the 10-element summary vector
    summary_vector = torch.tensor(
        [
            mean_m0.item(),  # 2) bar{m0}
            mean_m1.item(),  # 3) bar{m1}
            mean_m2.item(),  # 4) bar{m2}
            max_power.item(),
            median_power.item(),
            min_power.item(),
            cov_m0_m0.item(),  # 5) cov(m0,m0)
            cov_m0_m1.item(),  # 6) cov(m0,m1)
            cov_m0_m2.item(),  # 7) cov(m0,m2)
            cov_m1_m1.item(),  # 8) cov(m1,m1)
            cov_m1_m2.item(),  # 9) cov(m1,m2)
            cov_m2_m2.item(),  # 10) cov(m2,m2)
        ],
        device=device,
    )

    eps = 1e-30
    summary_vector[6:] = torch.log(summary_vector[6:] + eps)  # log-scale cov summaries
    return summary_vector
