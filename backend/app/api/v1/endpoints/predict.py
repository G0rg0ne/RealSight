"""Prediction endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from tritonclient.utils import InferenceServerException

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.triton_client import TritonInferenceClient, get_triton_client

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    triton: TritonInferenceClient = Depends(get_triton_client),
) -> PredictResponse:
    """Run regression inference via Triton."""
    if not triton.is_server_live():
        raise HTTPException(status_code=503, detail="Triton inference server is unavailable")

    if not triton.is_model_ready():
        raise HTTPException(status_code=503, detail="Model is not ready on Triton")

    try:
        value = triton.predict(body.features)
    except InferenceServerException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PredictResponse(prediction=value)
