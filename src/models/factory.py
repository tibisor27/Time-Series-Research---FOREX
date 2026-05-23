import logging
from sklearn.linear_model import Ridge, LinearRegression

from src.config import ModelConfig
from src.models.base import SklearnModel

logger = logging.getLogger(__name__)


def get_sklearn_model(cfg: ModelConfig) -> SklearnModel:
    """
    Simple Factory: citeste cfg.arch + cfg.params si returneaza
    un model nativ sklearn care respecta protocolul SklearnModel.

    Protocolul SklearnModel garanteaza ca modelul are:
        .fit(X: np.ndarray, y: np.ndarray)
        .predict(X: np.ndarray) -> np.ndarray
    """
    arch = cfg.arch.lower()

    if arch == "ridge":
        alpha = cfg.params.get("alpha", 1.0)
        logger.info(f"  [Factory] Ridge(alpha={alpha})")
        return Ridge(alpha=alpha)

    elif arch == "linear_regression":
        logger.info("  [Factory] LinearRegression()")
        return LinearRegression()

    else:
        raise ValueError(
            f"Unknown sklearn arch: '{cfg.arch}'. "
            f"Supported: 'ridge', 'linear_regression'"
        )


def get_model(cfg: ModelConfig) -> SklearnModel:
    """
    Universal Factory — punct unic de intrare pentru orice model.

    Primeste: ModelConfig (din YAML)
    Returneaza: SklearnModel (Protocol — garanteaza fit/predict)

    """
    logger.info(f"[Factory] type={cfg.type} | arch={cfg.arch}")

    if cfg.type == "sklearn":
        return get_sklearn_model(cfg)

    elif cfg.type == "neuralforecast":
        raise NotImplementedError(
            "NeuralForecast models vor fi adaugate dupa validarea baseline-urilor sklearn."
        )

    else:
        raise ValueError(
            f"Unknown model type: '{cfg.type}'. "
            f"Supported: 'sklearn', 'neuralforecast'"
        )

