"""Trained feature listing endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import FeaturesResponse
from app.services.model_metadata import get_feature_names

router = APIRouter(tags=["features"])


@router.get("/features", response_model=FeaturesResponse)
async def list_features() -> FeaturesResponse:
    """Return feature names from the local CatBoost model file."""
    try:
        names = get_feature_names()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model file not found. Place model.cbm at "
                "models/catboost_model/1/model.cbm (container: /models/catboost_model/1/model.cbm)."
            ),
        ) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FeaturesResponse(features=names)
