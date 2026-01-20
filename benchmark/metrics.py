import numpy as np
from dysts.metrics import smape

def _rolling_apply(ts1, ts2, metric_fn):
    """Apply a metric on increasing prefixes of two time series."""
    n = min(ts1.shape[0], ts2.shape[0])
    if n <= 0:
        return np.array([])
    return np.array([metric_fn(ts1[:i], ts2[:i]) for i in range(1, n + 1)])


def smape_rolling(ts1, ts2):
    """Return the smape versus time for two time series."""
    return _rolling_apply(ts1, ts2, smape)

def vpt(arr, threshold=30):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.
    r
    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    all_exceed_times = list()
    for i in range(arr.shape[1]):
        exceed_times = np.where(arr[:, i] > threshold)[0]
        if len(exceed_times) == 0:
            tind = len(arr[:, i])
        else:
            tind = exceed_times[0]
        all_exceed_times.append(tind)
    return all_exceed_times

def mse(x, xhat):
    """
    Given a univariate forecast and ground truth, compute the MSE.
    
    Args:
        x (np.ndarray): The ground truth, a time series of shape (nt,).
        xhat (np.ndarray): The forecast, a time series of shape (nt,).

    Returns:
        float: The MSE.
    """
    return np.mean((x - xhat) ** 2)

def nrmse(x, xhat):
    """
    Given a univariate forecast and ground truth, compute the NRMSE.
    
    Args:
        x (np.ndarray): The ground truth, a time series of shape (nt, d).
        xhat (np.ndarray): The forecast, a time series of shape (nt, d).

    Returns:
        float: The NRMSE.
    """
    denom = np.var(x, axis=0)
    numerator = np.linalg.norm(x - xhat, axis=0) ** 2
    return np.sqrt(np.mean(numerator / denom, axis=0))


def horizoned_mse(x, xhat):
    """Given a horizoned forecast and ground truth, compute the NRMSE as a function of time"""
    return _rolling_apply(x, xhat, mse)

def horizoned_nrmse(x, xhat):
    """Given a horizoned forecast and ground truth, compute the NRMSE as a function of time"""
    return _rolling_apply(x, xhat, nrmse)

def horizoned_smape(x, xhat):
    """Given a horizoned forecast and ground truth, compute the SMAPE as a function of time"""
    return _rolling_apply(x, xhat, smape)

def vpt_smape(x, xhat, threshold=30):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        x (np.ndarray): The ground truth, a time series of shape (nt, 1).
        xhat (np.ndarray): The forecast, a time series of shape (nt, 1).
        threshold (float): The threshold to search for.

    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_smape(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind

def vpt_mse(x, xhat, threshold=2):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.
    r
    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_mse(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind

def vpt_nrmse(x, xhat, threshold=0.5):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.
    r
    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_nrmse(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind

def mse_rolling(ts1, ts2):
    return _rolling_apply(ts1, ts2, mse)

def mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def mae_rolling(ts1, ts2):
    return _rolling_apply(ts1, ts2, mae)