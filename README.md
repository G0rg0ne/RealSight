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

## Recent changes

See [DEVELOPMENT.md](DEVELOPMENT.md).
