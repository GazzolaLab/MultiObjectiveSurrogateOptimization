import numpy as np
from scipy.stats import spearmanr


def mean_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Normalized Root Mean Squared Error (N-RMSE), averaged across all objectives.
    Normalization is done by the range (max - min) of the true values for each objective.

    Args:
        y_true (np.ndarray): Ground truth values. Shape: (n_samples, n_objectives).
        y_pred (np.ndarray): Predicted values. Shape: (n_samples, n_objectives).

    Returns:
        float: The mean N-RMSE score.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    y_range = np.max(y_true, axis=0) - np.min(y_true, axis=0)

    nrmse_per_objective = rmse / (y_range + 1e-8)

    return np.mean(nrmse_per_objective)


def mean_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE), averaged across all objectives.
    Returns the error as a percentage (e.g., 10.5 for 10.5%).

    Args:
        y_true (np.ndarray): Ground truth values. Shape: (n_samples, n_objectives).
        y_pred (np.ndarray): Predicted values. Shape: (n_samples, n_objectives).

    Returns:
        float: The mean MAPE score.
    """
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."

    mape_per_objective = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)), axis=0)

    return np.mean(mape_per_objective) * 100


def mean_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination), averaged across all objectives.

    Args:
        y_true (np.ndarray): Ground truth values. Shape: (n_samples, n_objectives).
        y_pred (np.ndarray): Predicted values. Shape: (n_samples, n_objectives).

    Returns:
        float: The mean R-squared score. Closer to 1 is better.
    """
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."

    ss_total = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    ss_residual = np.sum((y_true - y_pred) ** 2, axis=0)

    r2_per_objective = 1 - (ss_residual / (ss_total + 1e-8))

    return np.mean(r2_per_objective)


def mean_spearman_rank_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Spearman's rank correlation coefficient, averaged across all objectives.
    This measures how well the predicted ranks of solutions match the true ranks.

    Args:
        y_true (np.ndarray): Ground truth values. Shape: (n_samples, n_objectives).
        y_pred (np.ndarray): Predicted values. Shape: (n_samples, n_objectives).

    Returns:
        float: The mean Spearman correlation. Closer to 1 is better.
    """
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."

    n_objectives = y_true.shape[1]
    correlations = []

    for i in range(n_objectives):
        corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
        correlations.append(corr)

    return np.nanmean(correlations)


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if name == "nrmse":
        return mean_nrmse(y_true_clean, y_pred_clean)
    elif name == "mape":
        return mean_mape(y_true_clean, y_pred_clean)
    elif name == "r2":
        return mean_r2_score(y_true_clean, y_pred_clean)
    elif name == "spearman":
        return mean_spearman_rank_corr(y_true_clean, y_pred_clean)
    else:
        raise ValueError(f"Unknown metric name: {name}")
