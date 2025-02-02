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


def earthquake_sim_fn(params):
    # TODO TESTING
    with open("data/config/SCEDC_30.json", 'r') as f:
        simulation_config = json.load(f)
    catalog_params = simulation_config["theta_0"].copy()

    try:
        keys = ['log10_mu', 'log10_k0', "a", 'log10_c', 'rho']
        params = dict(zip(keys, np.log10(params[:2]).tolist() + [params[2]] + [np.log10(params[3]).tolist()] + [params[4]]))  # Attempt conversion
        params["a"] = params["a"].item()
        params['rho'] = params['rho'].item()
    except Exception as e:
        print(e)

    catalog_params.update(params)

    # shape_coords = "data/input_data/california_shape.npy"  # used to be simulation_config["shape_coords"]
    region = Polygon(np.load(simulation_config["shape_coords"]))

    # np.random.seed(777)
    # Note: in four param model version, alpha ("a") fixed to beta
    if "a" not in catalog_params:
        catalog_params["a"] = simulation_config["beta"]

    theta = parameter_dict2array(catalog_params)
    br = branching_ratio(theta, simulation_config["beta"])
    print(f"Branching ratio: {br}")

    if br > 1.1:  # TODO! EXPERIMENTAL - set arbitrarily
        return

    # TODO: EARLY RETURN
    catalog = generate_background_events(
        region,
        timewindow_start=pd.to_datetime(simulation_config["timewindow_start"]),
        timewindow_end=pd.to_datetime(simulation_config["timewindow_end"]),
        parameters=catalog_params,
        mc=simulation_config["mc"],  # magnitude of completeness
        beta=simulation_config["beta"],  # Richter scale magnitude
        delta_m=simulation_config["delta_m"]  # bin size of magnitudes
        # m_max=m_max,
        # background_lats=background_lats,
        # background_lons=background_lons,
        # background_probs=background_probs,
        # gaussian_scale=gaussian_scale,
    )
    print(f"Catalog length: {len(catalog)}")
    if len(catalog.index) > 800_000:
        return
    if len(catalog.index) == 0:
        return
    
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
