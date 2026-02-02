from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Type

import numpy as np


class ForecastModel:
    """Unified forecast interface.

    All models accept a context array of shape (T, C) or (T,) and return
    a forecast of shape (H, C). For 1D context, C=1.
    """

    name: str = "base"

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.forecast(context, horizon)


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _ensure_2d(context: Any) -> np.ndarray:
    arr = _to_numpy(context)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"context must be 1D or 2D, got shape {arr.shape}")


def _validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")


def _per_channel_forecast(
    context: np.ndarray,
    horizon: int,
    forecaster: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    preds = []
    for dim in range(context.shape[1]):
        pred = np.asarray(forecaster(context[:, dim], horizon)).reshape(-1)
        if pred.shape[0] != horizon:
            raise ValueError(f"Expected horizon {horizon}, got {pred.shape[0]}")
        preds.append(pred)
    return np.stack(preds, axis=0).T


class DynaMixModel(ForecastModel):
    name = "dynamix"

    def __init__(
        self,
        model_id: str = "dynamix-3d-alrnn-v1.0",
        standardize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        dynamix_root = Path(__file__).resolve().parent / "DynaMix"
        if dynamix_root.exists():
            sys_path = str(dynamix_root)
            if sys_path not in _get_sys_path():
                _get_sys_path().append(sys_path)

        import torch
        from src.model.forecaster import DynaMixForecaster
        from src.utilities.utilities import load_hf_model

        self._torch = torch
        self._standardize = standardize
        model = load_hf_model(model_id)
        model.eval()
        self._device = device
        if self._device is not None:
            model = model.to(self._device)
        self._forecaster = DynaMixForecaster(model)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        context_tensor = self._torch.tensor(
            context_2d, dtype=self._torch.float32, device=self._device
        )
        with self._torch.no_grad():
            reconstruction = self._forecaster.forecast(
                context=context_tensor,
                horizon=horizon,
                standardize=self._standardize,
            )
        return _to_numpy(reconstruction)


class ChronosModel(ForecastModel):
    name = "chronos"

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-base",
        device_map: str = "cpu",
        torch_dtype: str = "bfloat16",
        num_samples: int = 20,
        limit_prediction_length: bool = False,
    ) -> None:
        import torch
        from chronos import BaseChronosPipeline

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self._torch = torch
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self._num_samples = num_samples
        self._limit_prediction_length = limit_prediction_length

    def _predict_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        forecast = self._pipeline.predict(
            context=self._torch.tensor(series, dtype=self._torch.float32),
            prediction_length=horizon,
            num_samples=self._num_samples,
            limit_prediction_length=self._limit_prediction_length,
        )
        samples = _to_numpy(forecast[0])
        if samples.ndim == 1:
            return samples
        return samples.mean(axis=0)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        return _per_channel_forecast(context_2d, horizon, self._predict_univariate)


class ChronosBoltModel(ChronosModel):
    name = "chronos_bolt"

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        device_map: str = "cpu",
        torch_dtype: str = "bfloat16",
        num_samples: int = 20,
        limit_prediction_length: bool = False,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            num_samples=num_samples,
            limit_prediction_length=limit_prediction_length,
        )


class TimesFMModel(ForecastModel):
    name = "timesfm"

    def __init__(
        self,
        horizon_len: int = 300,
        context_len: int = 2048,
        per_core_batch_size: int = 32,
        backend: str = "cpu",
        num_layers: int = 50,
        use_positional_embedding: bool = False,
        checkpoint_repo_id: str = "google/timesfm-2.0-500m-pytorch",
    ) -> None:
        import timesfm

        self._horizon_len = horizon_len
        hparams = timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=per_core_batch_size,
            horizon_len=horizon_len,
            num_layers=num_layers,
            use_positional_embedding=use_positional_embedding,
            context_len=context_len,
        )
        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint_repo_id)
        self._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        point_forecast, _ = self._model.forecast(context_2d.T)
        preds = _to_numpy(point_forecast).T
        if horizon > self._horizon_len:
            raise ValueError(
                f"TimesFM configured for horizon_len={self._horizon_len}, got {horizon}"
            )
        return preds[:horizon]


class TimeMoEModel(ForecastModel):
    name = "timemoe"

    def __init__(
        self,
        model_id: str = "Maple728/TimeMoE-200M",
        device_map: str = "cpu",
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self._torch = torch
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        context_tensor = self._torch.tensor(
            context_2d.T, dtype=self._torch.float32, device=self._model.device
        )
        output = self._model.generate(context_tensor, max_new_tokens=horizon)
        forecast = output[:, -horizon:]
        return _to_numpy(forecast).T


class Moirai2Model(ForecastModel):
    name = "moirai2"

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        max_context: int = 1680,
        context_length: Optional[int] = None,
    ) -> None:
        from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast

        self._model_id = model_id
        self._max_context = max_context
        self._context_length = context_length
        self._Moirai2Module = Moirai2Module
        self._Moirai2Forecast = Moirai2Forecast
        self._model_cache: Dict[tuple[int, int], Any] = {}

    def _get_model(self, context_length: int, horizon: int) -> Any:
        key = (context_length, horizon)
        if key not in self._model_cache:
            model = self._Moirai2Forecast(
                module=self._Moirai2Module.from_pretrained(self._model_id),
                prediction_length=horizon,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self._model_cache[key] = model
        return self._model_cache[key]

    def _predict_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        from einops import rearrange
        import torch

        ctx = (
            self._context_length
            if self._context_length is not None
            else min(len(series), self._max_context)
        )
        model = self._get_model(ctx, horizon)
        x = rearrange(torch.as_tensor(series[-ctx:], dtype=torch.float32), "t -> 1 t 1")
        with torch.no_grad():
            samples_list = model.predict(x)
        return np.median(np.asarray(samples_list[0]), axis=0)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        return _per_channel_forecast(context_2d, horizon, self._predict_univariate)


class ParrotModel(ForecastModel):
    name = "parrot"

    def __init__(self, D: int = 30, tau: int = 1) -> None:
        from models.parrot import context_parroting_forecast

        self._D = D
        self._tau = tau
        self._context_parroting_forecast = context_parroting_forecast

    def _predict_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        _, _, forecast = self._context_parroting_forecast(
            series, D=self._D, tau=self._tau, forecast_total_length=horizon
        )
        return np.asarray(forecast)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        return _per_channel_forecast(context_2d, horizon, self._predict_univariate)


class SimplexModel(ForecastModel):
    name = "simplex"

    def __init__(self, m: int = 30, tau: int = 1, k: Optional[int] = None) -> None:
        from models.parrot import SimplexForecaster

        self._m = m
        self._tau = tau
        self._k = k
        self._SimplexForecaster = SimplexForecaster

    def _predict_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        model = self._SimplexForecaster(m=self._m, tau=self._tau, k=self._k)
        model.fit(series)
        return model.forecast(horizon)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        return _per_channel_forecast(context_2d, horizon, self._predict_univariate)


class ARIMAModel(ForecastModel):
    name = "arima"

    def __init__(self, auto_arima_kwargs: Optional[Dict[str, Any]] = None) -> None:
        from pmdarima import auto_arima

        self._auto_arima = auto_arima
        defaults = dict(
            error_action="ignore",
            trace=False,
            suppress_warnings=True,
            maxiter=5,
            seasonal=False,
            m=1,
            stepwise=True,
            start_p=0,
            start_q=0,
            start_P=0,
            start_Q=0,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            max_order=3,
        )
        if auto_arima_kwargs:
            defaults.update(auto_arima_kwargs)
        self._auto_arima_kwargs = defaults

    def _predict_univariate(self, series: np.ndarray, horizon: int) -> np.ndarray:
        model = self._auto_arima(series, **self._auto_arima_kwargs)
        return model.predict(horizon)

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        return _per_channel_forecast(context_2d, horizon, self._predict_univariate)


class PandaPatchTSTModel(ForecastModel):
    name = "panda_patchtst"

    def __init__(
        self,
        pretrain_path: str = "GilpinLab/panda-72M",
        device_map: str = "cpu",
        sliding_context: bool = True,
        limit_prediction_length: bool = False,
    ) -> None:
        import torch
        from panda.patchtst.pipeline import PatchTSTPipeline

        self._torch = torch
        self._pipeline = PatchTSTPipeline.from_pretrained(
            mode="predict",
            pretrain_path=pretrain_path,
            device_map=device_map,
        )
        self._sliding_context = sliding_context
        self._limit_prediction_length = limit_prediction_length

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        _validate_horizon(horizon)
        context_2d = _ensure_2d(context)
        context_tensor = self._torch.tensor(
            context_2d[None, :], dtype=self._torch.float32
        )
        pred = self._pipeline.predict(
            context_tensor,
            horizon,
            limit_prediction_length=self._limit_prediction_length,
            sliding_context=self._sliding_context,
        )
        pred_np = _to_numpy(pred)
        if pred_np.ndim == 3 and pred_np.shape[0] == 1:
            pred_np = pred_np[0]
        if pred_np.ndim == 1:
            pred_np = pred_np[:, None]
        return pred_np


MODEL_REGISTRY: Dict[str, Type[ForecastModel]] = {
    # DynaMixModel.name: DynaMixModel, # working
    # ChronosModel.name: ChronosModel,
    # ChronosBoltModel.name: ChronosBoltModel,
    # TimesFMModel.name: TimesFMModel, # not working
    TimeMoEModel.name: TimeMoEModel,
    # Moirai2Model.name: Moirai2Model, # working
    # ParrotModel.name: ParrotModel, # working
    # SimplexModel.name: SimplexModel, # working
    # ARIMAModel.name: ARIMAModel, # working
    # PandaPatchTSTModel.name: PandaPatchTSTModel,
}


def list_models() -> Iterable[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(name: str, **kwargs: Any) -> ForecastModel:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(list_models())}")
    return MODEL_REGISTRY[key](**kwargs)


def _get_sys_path() -> list[str]:
    import sys

    return sys.path
