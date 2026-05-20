FROM nvcr.io/nvidia/tritonserver:24.12-py3

# Pre-install Python backend deps (Triton does not always pip-install requirements.txt
# before model load; baking them into the image is reliable for CPU inference).
RUN pip3 install --no-cache-dir \
    "catboost>=1.2.0,<2.0.0" \
    "numpy>=1.26.0,<3.0.0"

COPY models/catboost_model /models/catboost_model

ENV MODEL_REPOSITORY_PATH=/models

# Self-contained startup for Docker Hub / Kubernetes (no Compose or manifest command required).
CMD ["sh", "-c", "exec tritonserver --model-repository=${MODEL_REPOSITORY_PATH} --log-verbose=1"]

