"""Load CatBoost model metadata from the local model repository."""

from functools import lru_cache

from catboost import CatBoost

from app.core.config import Settings, get_settings


@lru_cache
def _load_model(model_path: str) -> CatBoost:
    """Load and cache the CatBoost model from disk."""
    model = CatBoost()
    model.load_model(model_path)
    return model


def get_feature_names(settings: Settings | None = None) -> list[str]:
    """
    Return trained feature names from the CatBoost model file.

    Raises:
        FileNotFoundError: If model.cbm is missing.
    """
    settings = settings or get_settings()
    model = _load_model(settings.resolved_model_cbm_path())
    return list(model.feature_names_)


def is_model_available(settings: Settings | None = None) -> bool:
    """Check whether the model file exists and can be loaded."""
    settings = settings or get_settings()
    try:
        get_feature_names(settings)
        return True
    except (FileNotFoundError, RuntimeError, OSError):
        return False
