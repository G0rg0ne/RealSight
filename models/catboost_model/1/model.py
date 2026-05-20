"""
Triton Python backend for CatBoost regression.

Expects a single BYTES/STRING tensor named 'features' containing a JSON object
with feature name to value mappings. Returns a single FP32 prediction.
"""

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """CatBoost model served via Triton Python backend."""

    def _resolve_model_path(self, args: dict) -> str:
        """Resolve model.cbm under the Triton model version directory."""
        model_dir = args["model_repository"]
        model_version = args["model_version"]
        path = os.path.join(model_dir, model_version, "model.cbm")
        if os.path.isfile(path):
            return path

        raise FileNotFoundError(
            "CatBoost model not found. Place model.cbm at "
            f"{path} (MODEL_REPOSITORY_PATH=/models, version 1)."
        )

    def initialize(self, args: dict) -> None:
        """Load the CatBoost model from model.cbm."""
        from catboost import CatBoost

        model_path = self._resolve_model_path(args)

        self.model = CatBoost()
        self.model.load_model(model_path)
        self.feature_names = list(self.model.feature_names_)

    def execute(self, requests: list) -> list:
        """Run inference for each request."""
        responses = []

        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "features")
                raw = tensor.as_numpy().flatten()[0]

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                features = json.loads(raw)
                row = [features.get(name) for name in self.feature_names]
                prediction = float(self.model.predict([row])[0])

                output = pb_utils.Tensor(
                    "prediction",
                    np.array([prediction], dtype=np.float32),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(exc))
                    )
                )

        return responses

    def finalize(self) -> None:
        """Cleanup hook (no-op)."""
        self.model = None
