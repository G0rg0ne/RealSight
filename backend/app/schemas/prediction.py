"""Request and response schemas for inference endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Feature payload for a single regression prediction."""

    features: dict[str, Any] = Field(
        ...,
        description="Feature name to value mapping matching the trained model.",
    )


class PredictResponse(BaseModel):
    """Regression output from the CatBoost model."""

    prediction: float = Field(..., description="Predicted numeric value.")


class FeaturesResponse(BaseModel):
    """Trained feature names from the loaded CatBoost model."""

    features: list[str] = Field(..., description="Ordered feature names.")


class HealthResponse(BaseModel):
    """Health check status for the API and Triton backend."""

    status: str
    triton_live: bool
    model_ready: bool
