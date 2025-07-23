import json
import os
import networkx as nx
import numpy as np
from functools import partial
from benchmarks.cortical_culture.data_preprocessing import compute_band_power
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import solve_ivp
from benchmarks.wilson_cowan_graph import WilsonCowanGraph, create_MEA_graph
import time


def feature_dtypes(component):
    return [
        (feature_name, np.float32)
        for feature_name in component.config.dopt_params.objective_names
    ]


def obj_fun(params, targets, t_end):
    model = WilsonCowanGraph(
        G=create_MEA_graph(
            params, os.path.join(os.path.dirname(__file__), "system.json")
        ),
        dx=0.5,
        diffusion_strength=params.get("diffusion_strength"),
        diffusion_populations=["E"],
        spatial_kernels={
            ("E", "E"): "exponential",
            ("I", "E"): "gaussian",
            ("E", "I"): "exponential",
            ("I", "I"): "gaussian",
        },
        spatial_scales={
            ("E", "E"): params.get("E_E_scale", 1.0),
            ("I", "E"): params.get("I_E_scale", 0.5),
            ("E", "I"): params.get("E_I_scale", 1.0),
            ("I", "I"): params.get("I_I_scale", 0.8),
        },
        gain={"E": params.get("E_gain"), "I": params.get("I_gain")},
        tau={"E": params.get("E_tau"), "I": params.get("I_tau")},
        theta={"E": params.get("E_theta"), "I": params.get("I_theta")},
    )

    start_time = time.time()
    dt = 0.001
    t, solution = model.simulate(t_end, dt, method="DOP853")
    run_time = time.time() - start_time
    print(f"Simulation completed in {run_time:.3f} s")

    data = np.array(solution).T

    scale = 327.29  # scale to match experimental units
    offset = 0

    try:
        q = compute_band_power(data * scale + offset, fs=1 / dt)

        means = q.mean().to_dict()
        stds = q.std().to_dict()
    except:
        means = {}
        stds = {}

    objectives = []
    features = []
    for k, target in targets.items():
        if k.endswith("_mean"):
            observed = means.get(k.replace("_mean", ""), -1)
        elif k.endswith("_std"):
            observed = stds.get(k.replace("_std", ""), -1)

        objectives.append((observed - target) ** 2)
        features.append(observed)

    obj_values = np.asarray(objectives)
    features = np.asarray(
        [tuple(rf for rf in features)],
        dtype=np.dtype(
            [
                (k, np.float32)
                for k in [
                    "Delta_mean",
                    "Theta_mean",
                    "Alpha_mean",
                    "Beta_mean",
                    "Gamma_mean",
                    "Delta_std",
                    "Theta_std",
                    "Alpha_std",
                    "Beta_std",
                    "Gamma_std",
                ]
            ]
        ),
    )

    return obj_values, features


def obj_fun_init(worker=None):

    # load experimental data
    t_end = 60.0
    targets = {
        "Delta_mean": 23.491851806640625,
        "Theta_mean": 3.3645029067993164,
        "Alpha_mean": 0.62232905626297,
        "Beta_mean": 14.114240646362305,
        "Gamma_mean": 0.12134022265672684,
        "Delta_std": 29.584436416625977,
        "Theta_std": 3.922821044921875,
        "Alpha_std": 1.4222825765609741,
        "Beta_std": 99.20442962646484,
        "Gamma_std": 0.2624305188655853,
    }

    return partial(obj_fun, targets=targets, t_end=t_end)


if __name__ == "__main__":

    params = {
        "E_E_weight": 10.0,
        "E_I_weight": 4.0,
        "I_E_weight": 3.0,
        "I_I_weight": 1.0,
        "diffusion_strength": 1.0,
        "E_theta": 4.0,
        "I_theta": 3.0,
        "E_tau": 0.02,
        "I_tau": 0.01,
        "E_gain": 2.0,
        "I_gain": 3.0,
    }

    model = WilsonCowanGraph(
        G=create_MEA_graph(
            params, os.path.join(os.path.dirname(__file__), "system.json")
        ),
        dx=0.5,
        diffusion_strength=params.get("diffusion_strength"),
        diffusion_populations=["E"],
        spatial_kernels={
            ("E", "E"): "exponential",
            ("I", "E"): "gaussian",
            ("E", "I"): "exponential",
            ("I", "I"): "gaussian",
        },
        spatial_scales={
            ("E", "E"): 1.0,
            ("I", "E"): 0.5,
            ("E", "I"): 1.0,
            ("I", "I"): 0.8,
        },
        gain={"E": params.get("E_gain"), "I": params.get("I_gain")},
        tau={"E": params.get("E_tau"), "I": params.get("I_tau")},
        theta={"E": params.get("E_theta"), "I": params.get("I_theta")},
    )
    import time

    start_time = time.time()
    T = 60.0
    dt = 0.001
    t, solution = model.simulate(T, dt, method="DOP853")
    run_time = time.time() - start_time
    print(f"Simulation completed in {run_time:.3f} s")

    print(t.shape, solution.shape)

    model.plot_state(t, solution, time=T)
    plt.show()

    model.analyze_spectrum(
        t,
        solution,
        window_size=1024,  # Smaller window for better time resolution
        overlap=0.9,  # 75% overlap between windows
        max_freq=100,  # Only show frequencies up to 100 Hz
        magnitude_window="hann",
    )
    plt.show()
