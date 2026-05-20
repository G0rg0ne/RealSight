# RealSight

Tools to explore French **DVF** (Demandes de Valeurs Foncières) property transaction data.

## DVF reader (Docker + uv)

Python dependencies are managed with [uv](https://docs.astral.sh/uv/). The Docker image includes Python 3.12, uv, Polars, and a CLI script to inspect `data/dvf.csv`.

The CSV is **not** copied into the image (it is multi‑GB). Mount it at runtime under `/data`.

### Build

```bash
docker compose build
```

Or without Compose:

```bash
docker build -t realsight-dvf .
```

### Run

Default: print file info, column names, and the first 10 rows:

```bash
docker compose run --rm dvf-reader
```

Custom options (passed to the script after the service name):

```bash
docker compose run --rm dvf-reader --head 5
docker compose run --rm dvf-reader --count
docker compose run --rm dvf-reader /data/dvf.csv --head 3 --count
```

Direct `docker run` (mount your `data` folder):

```bash
docker run --rm -v "%cd%/data:/data:ro" realsight-dvf --head 10
```

### Local development (without Docker)

```bash
uv lock
uv sync
uv run python forging_pipeline/read_dvf.py
```

Environment variable `DVF_CSV_PATH` overrides the default path (`/data/dvf.csv` in Docker, `data/dvf.csv` when passed as an argument locally).

### Script options

| Option | Description |
|--------|-------------|
| `csv_path` | Optional path to CSV |
| `--head N` | Rows to display (default: 10) |
| `--count` | Stream entire file and print row count |
| `--chunksize N` | Chunk size for `--count` (default: 100000) |

## CatBoost inference API (FastAPI + NVIDIA Triton)

Regression inference for a CatBoost model loaded from a local model repository. The API proxies requests to [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) (Python backend).

### Technology stack

| Component | Version / image |
|-----------|-----------------|
| Python | 3.12 |
| FastAPI | 0.115+ |
| Triton | Custom image from [`triton/Dockerfile`](triton/Dockerfile) (based on `nvcr.io/nvidia/tritonserver:24.12-py3` + CatBoost) |
| CatBoost | 1.2+ |

### Project structure (inference)

```
backend/
├── app/
│   ├── main.py
│   ├── api/v1/endpoints/   # health, predict, features
│   ├── core/config.py
│   ├── schemas/prediction.py
│   └── services/             # triton_client, model_metadata
├── Dockerfile
└── requirements.txt

models/
└── catboost_model/           # Triton model repository (baked into both images as /models/catboost_model)
    ├── config.pbtxt
    └── 1/
        ├── model.py          # Triton Python backend
        └── model.cbm         # trained CatBoost artifact (not in git; required at build time)

triton/
└── Dockerfile                # pre-installs catboost, CMD starts tritonserver

k8s/realsight/                # Kubernetes manifests (namespace, config, deployments, services)
argocd/
└── realsight-application.yaml  # Argo CD Application (syncs k8s/realsight)
```

### Setup

1. Copy your trained CatBoost model to **`models/catboost_model/1/model.cbm`** (single canonical path for Triton and FastAPI):

   ```bash
   cp /path/to/your/trained_model.cbm models/catboost_model/1/model.cbm
   ```

2. Copy environment template (optional for local runs):

   ```bash
   cp .env.example .env
   ```

3. Build and start Triton and the API (model files are **baked into the images** at build time; no runtime volume mounts for `model_repository` or `model.cbm`):

   ```bash
   docker compose up --build triton api
   ```

   - API: http://localhost:8888 (docs at `/docs`)
   - Triton HTTP: http://localhost:8000

   Build context is the repo root: both `triton/Dockerfile` and `backend/Dockerfile` copy `models/catboost_model` to `/models/catboost_model`. FastAPI and Triton both read **`/models/catboost_model/1/model.cbm`**.

### Self-contained images (Docker Hub / Kubernetes)

Both inference images define their own startup command in the Dockerfile. Kubernetes repeats the Triton command explicitly so production deploys do not depend on a rebuilt image tag alone:

| Image | Default process | Notes |
|-------|-----------------|-------|
| `g0rg0ne/realsight-triton` | `tritonserver --model-repository=${MODEL_REPOSITORY_PATH} --log-verbose=1` | `MODEL_REPOSITORY_PATH` defaults to `/models` in the image |
| `g0rg0ne/realsight-api` | `uvicorn app.main:app --host 0.0.0.0 --port 8888` | Runs as non-root user `app` (UID/GID 1000); set `TRITON_URL` to reach Triton (e.g. `realsight-triton:8000` in-cluster) |

**Direct run** (after `docker pull`):

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 g0rg0ne/realsight-triton:v1.0.0

docker run --rm -p 8888:8888 -e TRITON_URL=host.docker.internal:8000 g0rg0ne/realsight-api:v1.0.0
```

### Argo CD deployment

Plain Kubernetes manifests live under [`k8s/realsight/`](k8s/realsight/). An Argo CD `Application` in [`argocd/realsight-application.yaml`](argocd/realsight-application.yaml) syncs that path into the `realsight` namespace.

| Resource | Name | Notes |
|----------|------|-------|
| Namespace | `realsight` | Created by manifest and/or `CreateNamespace=true` sync option |
| ConfigMap | `realsight-config` | `TRITON_URL`, `MODEL_NAME`, `MODEL_VERSION`, `MODEL_REPOSITORY_PATH` |
| Deployment + Service | `realsight-triton` | Image `g0rg0ne/realsight-triton:v1.0.0`, ports 8000/8001/8002 |
| Deployment + Service | `realsight-api` | Image `g0rg0ne/realsight-api:v1.0.0`, ClusterIP port 8888, `runAsUser` 1000 |

The Triton Deployment runs `tritonserver --model-repository=${MODEL_REPOSITORY_PATH} --log-verbose=1` (from the ConfigMap). Change that `command`/`args` block only if you intentionally change Triton flags. Bump image tags in `k8s/realsight/triton.yaml` and `k8s/realsight/api.yaml` when releasing new versions.

**Register the app** (adjust `repoURL` / `targetRevision` if your fork or branch differs):

```bash
kubectl apply -f argocd/realsight-application.yaml
```

Or apply workloads directly without Argo CD:

```bash
kubectl apply -f k8s/realsight/
```

**Port-forward** (API is ClusterIP-only):

```bash
kubectl -n realsight port-forward svc/realsight-api 8888:8888
curl http://localhost:8888/api/v1/health
```

**Verify Triton** (optional):

```bash
kubectl -n realsight port-forward svc/realsight-triton 8000:8000
curl http://localhost:8000/v2/health/ready
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton:8000` | Triton HTTP endpoint (host:port, no scheme) |
| `MODEL_NAME` | `catboost_model` | Triton model name |
| `MODEL_VERSION` | `1` | Triton model version |
| `MODEL_REPOSITORY_PATH` | `/models` | Path to model repository inside container |
| `MODEL_CBM_PATH` | *(unset)* | Optional override; default is `/models/{MODEL_NAME}/{MODEL_VERSION}/model.cbm` |

See [.env.example](.env.example) for all variables.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | API, Triton liveness, and model readiness |
| `GET` | `/api/v1/features` | List trained feature names from `model.cbm` |
| `POST` | `/api/v1/predict` | Regression prediction |

**Predict** — request body:

```json
{
  "features": {
    "feature_a": 1.0,
    "feature_b": "Paris"
  }
}
```

Response:

```json
{
  "prediction": 425000.0
}
```

### Local development (API only)

Requires a running Triton instance and `model.cbm` on disk:

```bash
cd backend
pip install -r requirements.txt
export TRITON_URL=localhost:8000
export MODEL_REPOSITORY_PATH=../model_repository
uvicorn app.main:app --reload --port 8080
```

### Testing

```bash
curl http://localhost:8888/api/v1/health
curl http://localhost:8888/api/v1/features
curl -X POST http://localhost:8888/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"your_feature": 1.0}}'
```

## CI/CD (Docker Hub)

Pushing a semver git tag (`v*.*.*`, e.g. `v1.0.0`) triggers [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml), which builds and pushes the inference images to Docker Hub in parallel.

| Image | Dockerfile | Example tags for `v1.0.0` |
|-------|------------|----------------------------|
| [`g0rg0ne/realsight-api`](https://hub.docker.com/r/g0rg0ne/realsight-api) | `backend/Dockerfile` | `v1.0.0`, `latest` |
| [`g0rg0ne/realsight-triton`](https://hub.docker.com/r/g0rg0ne/realsight-triton) | `triton/Dockerfile` | `v1.0.0`, `latest` |

Build context is the repository root (both images require `models/catboost_model/1/model.cbm` at build time).

### GitHub secrets

Add these in the GitHub repository under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|--------|
| `DOCKERHUB_USERNAME` | `g0rg0ne` |
| `DOCKERHUB_TOKEN` | Docker Hub [access token](https://hub.docker.com/settings/security) (not your account password) |

### Release a version

Ensure `models/catboost_model/1/model.cbm` is present locally, commit, then tag and push:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Monitor the workflow under **Actions** on GitHub. After it succeeds, pull published images:

```bash
docker pull g0rg0ne/realsight-api:v1.0.0
docker pull g0rg0ne/realsight-triton:v1.0.0
```

## Recent changes

See [DEVELOPMENT.md](DEVELOPMENT.md).
