"""HTTP client for NVIDIA Triton Inference Server."""

import json
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from app.core.config import Settings, get_settings


class TritonInferenceClient:
    """Thin wrapper around tritonclient HTTP inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = httpclient.InferenceServerClient(
            url=self._settings.triton_url,
            verbose=False,
        )

    def is_server_live(self) -> bool:
        """Return True if Triton responds to a liveness check."""
        try:
            return self._client.is_server_live()
        except (InferenceServerException, OSError):
            return False

    def is_model_ready(self) -> bool:
        """Return True if the configured model version is ready."""
        try:
            return self._client.is_model_ready(
                self._settings.model_name,
                self._settings.model_version,
            )
        except (InferenceServerException, OSError):
            return False

    def predict(self, features: dict[str, Any]) -> float:
        """
        Run inference for a single feature dict.

        Args:
            features: Feature name to value mapping.

        Returns:
            Regression prediction as a float.

        Raises:
            InferenceServerException: On Triton inference errors.
        """
        payload = json.dumps(features, ensure_ascii=False).encode("utf-8")
        inputs = [
            httpclient.InferInput("features", [1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(np.array([payload], dtype=object))

        outputs = [httpclient.InferRequestedOutput("prediction")]

        response = self._client.infer(
            model_name=self._settings.model_name,
            model_version=self._settings.model_version,
            inputs=inputs,
            outputs=outputs,
        )

        result = response.as_numpy("prediction")
        if result is None or len(result) == 0:
            raise InferenceServerException("Empty prediction output from Triton")

        return float(result.flatten()[0])


def get_triton_client() -> TritonInferenceClient:
    """Factory for dependency injection."""
    return TritonInferenceClient()
