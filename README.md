# RealSight

Explore French **DVF** (Demandes de Valeurs Foncières) property transaction data and serve machine-learning predictions for real-estate analytics.

> **Under construction** — This project is actively being built. Layout, APIs, deployment, and documentation may change.

## About

RealSight aims to combine open DVF sales records with a production-style inference stack: a **FastAPI** service in front of **NVIDIA Triton Inference Server** running a **CatBoost** regression model. Workloads can be deployed to Kubernetes (manifests under `k8s/realsight/`) with optional GitOps via Argo CD.

The repository is evolving toward end-to-end workflows—from understanding transaction data to exposing predictions through a stable HTTP API—not all pieces are finished yet.

## Repository overview

| Area | Purpose |
|------|---------|
| `backend/` | FastAPI application |
| `triton/`, `models/` | Triton server image and model repository |
| `k8s/realsight/` | Kubernetes namespace, config, deployments, services |
| `argocd/` | Argo CD Application for cluster sync |
| `.github/workflows/` | Publish inference images on semver tags |

## Status

Features, endpoints, and deployment instructions are not finalized. Check git history or open issues for the latest direction.

## License

See [LICENSE](LICENSE).
