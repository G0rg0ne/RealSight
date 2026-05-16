"""API v1 router aggregation."""

from fastapi import APIRouter

from app.api.v1.endpoints import features, health, predict

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(features.router)
