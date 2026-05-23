import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge

from src.config import ModelConfig
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def get_sklearn_model(cfg: ModelConfig) -> BaseEstimator:
    
    arch = cfg.arch.lower()
    
    if arch == "ridge":
        alpha = cfg.params.get("alpha")
        logger.info(f"  [Model Factory] Instantiating Sklearn Ridge (alpha={alpha})")
        print(f"alpha: {alpha}")

        return Ridge(alpha=alpha)
        
    elif arch == "linear_regression":
        return LinearRegression()
    else:
        raise ValueError(f"Unknown sklearn architecture: {cfg.arch}")


def get_model(cfg: ModelConfig) -> Any:

    logger.info(f"[Model Factory] Request received: type={cfg.type} | arch={cfg.arch}")
    
    if cfg.type == "sklearn":
        return get_sklearn_model(cfg)
        
    elif cfg.type == "neuralforecast":
        raise NotImplementedError("NeuralForecast instantiation will be added soon!")
        
    else:
        raise ValueError(f"Unknown model type: {cfg.type}. Choose 'sklearn' or 'neuralforecast'.")
