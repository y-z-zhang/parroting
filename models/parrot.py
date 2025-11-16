import numpy as np
from scipy.spatial.distance import cdist

def delay_embedding(data, D=30, tau=1):
    """
    Create a delay embedding of a time series

    Args:
        data: 1D array representing the context trajectory
        D: embedding dimension
        tau: delay used for the embedding

    Returns:
        embedded_data: Delay embedded data

    Usage:
        arr = np.array([1,2,3,4,5,6,7,8,9])
        delay_embedding(arr,5,1)
    """
    N = len(data)
    indices = np.arange(D) * tau + np.arange(N - (D - 1) * tau)[:, None]
    embedded_data = data[indices]

    return embedded_data

def embedding_distance(data, D=30, tau=1):
    """
    Calculate the L2 distance in a delay-embedded space

    Args:
        data (array-like): 1D array representing the context trajectory
        D (int): Motif length, embedding dimension (default 30).
        tau (int): delay used for the embedding (default 1).

    Returns:
        min_l2_distance: Minimum L2 distance in the embedding space
    """

    # Create delay embeddings
    embedded_data = delay_embedding(data, D=D, tau=tau)

    # Compute distance of other points to the last point in the embedding space using L2 norm
    last_point = embedded_data[-1]
    l2_distance = cdist(embedded_data[:-D * tau], last_point[None, :])

    return l2_distance

def context_parroting_forecast(data, D=30, tau=1, forecast_total_length=300):
    """
    Simple forecasting model based on context parroting.

    Args:
        data (array-like): The input 1D array representing the context trajectory.
        D (int): Motif length, embedding dimension (default 30).
        tau (int): delay used for the embedding (default 1).
        forecast_total_length (int): Desired length of the forecasted output (default 300).

    Returns:
        best_index (int): Index of the best-matching context point.
        min_embedding_distance (float): Minimum L2 distance in the embedding space.
        forecast (array-like): Forecasted sequence of the specified length.
    """

    # Calculate embedding distances
    embedding_distances = embedding_distance(data, D=D, tau=tau)
    min_embedding_distance = np.min(embedding_distances)

    # Find the index with the minimum embedding distance
    min_index = np.argmin(embedding_distances)
    best_index = min_index + (D-1)*tau + 1

    # Extract the motif starting from the best-matching index
    motif = data[best_index:-1]
    motif_length = len(motif)

    # Repeat the motif to create a forecast of the desired length
    num_repeats = forecast_total_length // motif_length + 1
    forecast = np.tile(motif, num_repeats)[:forecast_total_length]

    return best_index, min_embedding_distance, forecast


class ParrotForecaster:
    """
    A forecaster that uses context parroting to make predictions.
    """
    def __init__(self, D=30, tau=1):
        self.D = D
        self.tau = tau

    def fit(self, data):
        self.data = data

    def predict(self, data, forecast_total_length=300):
        return context_parroting_forecast(data, D=self.D, tau=self.tau, forecast_total_length=forecast_total_length)