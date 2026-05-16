"""Health check endpoint."""

from fastapi import APIRouter, Depends

from app.schemas.prediction import HealthResponse
from app.services.model_metadata import is_model_available
from app.services.triton_client import TritonInferenceClient, get_triton_client

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    triton: TritonInferenceClient = Depends(get_triton_client),
) -> HealthResponse:
    """Return API, Triton, and model readiness status."""
    triton_live = triton.is_server_live()
    model_ready = triton_live and triton.is_model_ready() and is_model_available()

    status = "healthy" if triton_live and model_ready else "degraded"
    if not triton_live:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        triton_live=triton_live,
        model_ready=model_ready,
    )
