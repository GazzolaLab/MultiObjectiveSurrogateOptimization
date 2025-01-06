import numpy as np


def feature_dtypes(component):
    return [
        (feature_name, np.float32)
        for feature_name in component.config.dopt_params.objective_names
    ]
