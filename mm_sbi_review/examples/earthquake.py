import json
import logging
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from etas import set_up_logger
from etas.inversion import round_half_up, branching_ratio
from etas.simulation import generate_catalog, generate_background_events

set_up_logger(level=logging.INFO)

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


def early_return(catalog_params, beta):
    early_return_bool = False
    # copied across from ETAS package run
    area = 406963
    timewindow_length = 10592.0
    expected_n_background = (
        np.power(10, catalog_params["log10_mu"]) * area * timewindow_length
    )

    if expected_n_background > 400_000:
        early_return_bool = True
    if expected_n_background < 20:
        early_return_bool = True
    theta = parameter_dict2array(catalog_params)
    print("Theta: ", theta)
    br = branching_ratio(theta, beta)
    print(f"Branching ratio: {br}")
    
    # subcritical check - k
    k = np.power(10, catalog_params["log10_k0"])
    alpha = catalog_params.get("alpha", 1.0)
    if k > 1 - (alpha/beta):
        early_return_bool = True

    if br > 1.05:  # TODO! EXPERIMENTAL - set arbitrarily
        early_return_bool = True

    return early_return_bool


def earthquake_sim_fn(key, mu, a, k0, c, rho):
    # TODO TESTING
    print('mu, a, k0, c, rho:', mu, a, k0, c, rho)
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
    region = Polygon(np.load(simulation_config["shape_coords"]))

    # np.random.seed(777)
    # Note: in four param model version, alpha ("a") fixed to beta
    if "a" not in catalog_params:
        catalog_params["a"] = simulation_config["beta"]

    if early_return(catalog_params, simulation_config["beta"]):
        print("stopping early")
        return None

    synthetic = generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime(simulation_config["timewindow_start"]),
        timewindow_end=pd.to_datetime(simulation_config["timewindow_end"]),
        parameters=catalog_params,
        mc=simulation_config["mc"],  # magnitude of completeness
        beta_main=simulation_config["beta"],  # Richter scale magnitude
        delta_m=simulation_config["delta_m"]  # bin size of magnitudes
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    # primary_start = simulation_config['primary_start']
    synthetic = synthetic.sort_values(by='time')
    return synthetic


def ripley_k_unmarked(times_days):
    n = len(times_days)
    T = times_days[-1] - times_days[0]
    # NOTE: RYAN constructed W_list
    W_list = np.concatenate((np.geomspace(0.001, 1, 10, endpoint=False), np.arange(1, 9)))
    Kvals = np.zeros(len(W_list))
    # For each w, store a pointer to the right
    pointers = [1] * len(W_list)
    sum_counts = np.zeros(len(W_list))

    for i in range(n):
        t_i = times_days[i]
        # Move each pointer as far as possible while t_j <= t_i + w
        for k, w in enumerate(W_list):
            while pointers[k] < n and times_days[pointers[k]] <= t_i + w:
                pointers[k] += 1
            # Count how many events are in (t_i, t_i + w], excluding i itself
            sum_counts[k] += (pointers[k] - i - 1)

    Kvals = (T * sum_counts) / (n * n)
    return Kvals


def ripley_k_thresholded(times_days, mags):
    n = len(times_days)
    T = times_days[-1] - times_days[0]
    W_list = [0.2, 0.5, 1.0, 3.0]
    M_T_list = [4.5, 5.0, 5.5, 6.0]

    # We store final results in [#thresholds, #w]
    Kvals = np.zeros((len(M_T_list), len(W_list)), dtype=np.float64)

    for i_T, M_T in enumerate(M_T_list):
        # Identify all events with m_i >= M_T
        large_indices = np.where(mags >= M_T)[0]
        nu = len(large_indices)
        if nu < 2:
            # If fewer than 2 "large" events, K^T is trivially zero or NaN
            continue

        # We'll accumulate counts for each w
        sum_counts_w = np.zeros(len(W_list), dtype=np.float64)

        # Option A: pointer approach for each w
        # But we must now do it for each i in large_indices
        pointers = [0] * len(W_list)  # pointer for each w
        # Start each pointer at 0 or at i+1?
        # We'll do a standard approach: pointer never moves backwards,
        # but we do re-init for each i. This is simpler to read but O(nu * n * len(W_list)).
        # If nu << n, this might be acceptable.
        
        # Option B (common): for each w, we do a binary search once for each i. 
        # We'll show Option B, which is often simpler to code:

        for i_large in large_indices:
            t_i = times_days[i_large]
            for k, w in enumerate(W_list):
                # Find how many events lie in (t_i, t_i + w].
                # We can do a binary search:
                # index j such that times[j] <= t_i + w
                # np.searchsorted returns the insertion index to keep sorted order
                j = np.searchsorted(times_days, t_i +w,  side='right')
                # count how many events are <= t_i + w, but strictly greater than i_large
                # i.e. count = j - (i_large + 1)
                count_in_window = max(0, j - (i_large + 1))
                sum_counts_w[k] += count_in_window

        # Normalise by nu^2
        Kvals[i_T, :] = (T*sum_counts_w) / (nu * nu)

    return Kvals

def sum_fn(catalog):
    if catalog is None:  # i.e., something gone wrong in simulation / invalid
        return -np.ones(39) + 0.1*np.random.normal(0, 1.0, 39)  # TODO: make this implausible from valid summaries - but not too wacky
    cat_sorted = catalog.sort_values("time")

    # S1: log num events
    S1 = np.log(len(catalog))

    # S2-S4: 20, 50, 90 quantiles of the inter-event time histogram
    # Convert to seconds since epoch, then take consecutive diffs:
    times_ns = cat_sorted.time.values.astype("datetime64[s]")
    # integer of seconds since epoch:
    times_sec = times_ns.astype("int64")
    times_days = times_sec / 86400.0
    dt_sec = np.diff(times_sec)

    # Convert to days (optional, depends on your preference):
    dt_days = dt_sec / 86400.0

    # --- S2, S3, S4: 20%, 50%, 90% quantiles ---
    S2, S3, S4 = np.quantile(dt_days, [0.2, 0.5, 0.9])

    # S5: ratio of the mean and median of the inter-event time histogram
    mean_dt = np.mean(dt_days)
    median_dt = np.median(dt_days)
    S5 = mean_dt / median_dt if median_dt > 0 else np.inf

    # S6-S23: Ripley's K statistic
    mags = cat_sorted.magnitude.values
    Kvals_unmarked = ripley_k_unmarked(times_days)

    # S24-S39: Magnitude thresholded Ripley's K statistic
    Kvals_thresholded = ripley_k_thresholded(times_days, mags)
    Kvals_thresholded_flat = Kvals_thresholded.flatten()

    summary_stats = np.concatenate([[S1], [S2], [S3], [S4], [S5],
                                    Kvals_unmarked,
                                    Kvals_thresholded_flat])

    return summary_stats
