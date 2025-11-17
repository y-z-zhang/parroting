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



import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SimplexForecaster:
    """
    Minimal simplex projection forecaster for a univariate time series.

    Args:
        m (int): Embedding dimension.
        tau (int): Delay (lag) between embedded coordinates.
        k (Optional[int]): Number of neighbors (default m+1). If larger than the
            available library, it is clipped automatically. Must be >= 1.

    Attributes:
        m (int): Embedding dimension.
        tau (int): Delay (lag) between embedded coordinates.
        k (int): Number of neighbors actually used.
        _X (np.ndarray): Library of embedded vectors, shape (L, m).
        _y (np.ndarray): Library of one-step-ahead targets, shape (L,).
        _context (np.ndarray): Original context series (float64).

    Example:
        ```python
        # Synthetic test: noisy sine, forecast 25 steps
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10 * np.pi, 400)
        series = np.sin(t) + 0.1 * rng.standard_normal(len(t))

        model = SimplexForecaster(m=3, tau=2).fit(series)
        yhat = model.forecast(horizon=25)
        print(yhat)
        ```
    """
    m: int = 30
    tau: int = 1
    k: Optional[int] = None

    def fit(self, context: np.ndarray) -> "SimplexForecaster":
        """
        Build the embedding library from a context series.

        Args:
            context (np.ndarray): 1D array of past observations.

        Returns:
            SimplexForecaster: Self.
        """
        x = np.asarray(context, dtype=np.float64).ravel()
        if x.ndim != 1:
            raise ValueError("context must be 1D")
        if self.m < 1 or self.tau < 1:
            raise ValueError("m and tau must be >= 1")

        X, y = self._embed(x, self.m, self.tau)
        if len(X) == 0:
            raise ValueError("Not enough data for embedding; need at least (m*tau + 1) points")

        self._X, self._y, self._context = X, y, x
        self.k = int(self.k) if self.k is not None else (self.m + 1)
        self.k = max(1, min(self.k, len(self._X)))
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """
        Roll out an autoregressive forecast.

        Args:
            horizon (int): Number of steps to forecast.

        Returns:
            np.ndarray: Forecasted values of length `horizon`.
        """
        if horizon < 1:
            return np.array([], dtype=np.float64)
        if not hasattr(self, "_X"):
            raise RuntimeError("Call fit(context) before forecast()")

        # Seed with the last m*tau values from context (most recent first in state)
        history = self._context.copy()
        preds = []
        for _ in range(horizon):
            state = self._current_state(history, self.m, self.tau)  # shape (m,)
            next_val = self._simplex_predict(state)
            preds.append(next_val)
            history = np.append(history, next_val)
        return np.asarray(preds, dtype=np.float64)

    # -------------------- internals --------------------

    @staticmethod
    def _embed(x: np.ndarray, m: int, tau: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create delay-coordinate embedding and one-step-ahead targets.
        """
        N = len(x)
        max_lag = (m - 1) * tau
        last_i = N - tau - 1
        if last_i - max_lag + 1 <= 0:
            return np.empty((0, m)), np.empty((0,))
        # Build embedded vectors X_t = [x_t, x_{t-tau}, ..., x_{t-(m-1)tau}]
        idxs = np.arange(max_lag, N - tau)
        X = np.column_stack([x[idxs - j * tau] for j in range(m)])
        y = x[idxs + tau]
        return X, y

    @staticmethod
    def _current_state(series: np.ndarray, m: int, tau: int) -> np.ndarray:
        """
        Build the current state vector from the tail of a series.
        """
        needed = (m - 1) * tau
        if len(series) - 1 - needed < 0:
            raise ValueError("Insufficient history to form the current state")
        base = len(series) - 1
        return np.array([series[base - j * tau] for j in range(m)], dtype=np.float64)

    def _simplex_predict(self, state: np.ndarray) -> float:
        """
        Predict one step ahead using k-nearest neighbors with exponential weights.
        """
        # Distances in embedding space
        dists = np.linalg.norm(self._X - state[None, :], axis=1)
        # Indices of k nearest neighbors
        idx = np.argpartition(dists, self.k - 1)[: self.k]
        nn_d = dists[idx]
        nn_y = self._y[idx]

        # Weights: Sugihara-style exponential scaling by nearest distance d1
        d1 = nn_d.min()
        if d1 == 0.0:
            # If the closest neighbor is identical, assign unit weight to exact matches
            mask = (nn_d == 0.0)
            w = mask.astype(np.float64)
            w /= w.sum()
        else:
            w = np.exp(-nn_d / d1)
            w /= w.sum()
        return float(np.dot(w, nn_y))


