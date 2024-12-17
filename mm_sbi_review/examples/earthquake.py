import json
import logging
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from etas import set_up_logger
from etas.inversion import round_half_up
from etas.simulation import generate_catalog

set_up_logger(level=logging.INFO)


def earthquake_sim_fn():
    # TODO TESTING
    with open("data/config/SCEDC_30.json", 'r') as f:
        simulation_config = json.load(f)

    # shape_coords = "data/input_data/california_shape.npy"  # used to be simulation_config["shape_coords"]
    region = Polygon(np.load(simulation_config["shape_coords"]))

    # np.random.seed(777)

    synthetic = generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime(simulation_config["timewindow_start"]),
        timewindow_end=pd.to_datetime(simulation_config["timewindow_end"]),
        parameters=simulation_config["theta_0"],
        mc=simulation_config["mc"],
        beta_main=simulation_config["beta"],  # TODO! COPIED ACROSS ... LOOK INTO!
        delta_m=simulation_config["delta_m"]
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    primary_start = simulation_config['primary_start']
    fn_store = simulation_config['fn_store']
    os.makedirs(os.path.dirname(fn_store), exist_ok=True)
    synthetic[["latitude", "longitude", "time", "magnitude"]].query(
        "time>=@primary_start").to_csv(fn_store)
    print("\nDONE!")
