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

model_repository/
└── catboost_model/
    ├── config.pbtxt
    └── 1/
        ├── model.py          # Triton Python backend
        └── model.cbm         # optional; see models/ below

models/
└── model.cbm                 # recommended location for your trained model (not in git)

triton/
└── Dockerfile                # pre-installs catboost for the Python backend
```

### Setup

1. Copy your trained CatBoost model to **`models/model.cbm`** (recommended):

   ```bash
   cp /path/to/your/trained_model.cbm models/model.cbm
   ```

   Or place it at `model_repository/catboost_model/1/model.cbm`.

2. Copy environment template (optional for local runs):

   ```bash
   cp .env.example .env
   ```

3. Build and start Triton and the API (model files are **baked into the images** at build time; no runtime volume mounts for `model_repository` or `model.cbm`):

   ```bash
   docker compose up --build triton api
   ```

   - API: http://localhost:8080 (docs at `/docs`)
   - Triton HTTP: http://localhost:8000

   Build context is the repo root: `triton/Dockerfile` copies `model_repository/catboost_model` and `models/model.cbm` into `/models/`; `backend/Dockerfile` copies `models/model.cbm` for the features endpoint.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton:8000` | Triton HTTP endpoint (host:port, no scheme) |
| `MODEL_NAME` | `catboost_model` | Triton model name |
| `MODEL_VERSION` | `1` | Triton model version |
| `MODEL_REPOSITORY_PATH` | `/models` | Path to model repository inside container |
| `MODEL_CBM_PATH` | `/models/model.cbm` | Path to `model.cbm` inside container (copied from `./models/model.cbm` at image build time) |

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
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/features
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"your_feature": 1.0}}'
```

## CI/CD (Docker Hub)

Pushing a semver git tag (`v*.*.*`, e.g. `v1.0.0`) triggers [`.github/workflows/docker-publish.yml`](.github/workflows/docker-publish.yml), which builds and pushes the inference images to Docker Hub in parallel.

| Image | Dockerfile | Example tags for `v1.0.0` |
|-------|------------|----------------------------|
| [`g0rg0ne/realsight-api`](https://hub.docker.com/r/g0rg0ne/realsight-api) | `backend/Dockerfile` | `v1.0.0`, `latest` |
| [`g0rg0ne/realsight-triton`](https://hub.docker.com/r/g0rg0ne/realsight-triton) | `triton/Dockerfile` | `v1.0.0`, `latest` |

Build context is the repository root (both images require `models/model.cbm` and Triton model files at build time).

### GitHub secrets

Add these in the GitHub repository under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|--------|
| `DOCKERHUB_USERNAME` | `g0rg0ne` |
| `DOCKERHUB_TOKEN` | Docker Hub [access token](https://hub.docker.com/settings/security) (not your account password) |

### Release a version

Ensure `models/model.cbm` is present locally, commit, then tag and push:

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
