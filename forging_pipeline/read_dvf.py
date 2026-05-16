"""Read and inspect DVF (Demandes de Valeurs Foncières) CSV data."""

from __future__ import annotations

from pathlib import Path

import polars as pl

DVF_CSV = Path("/data/dvf.csv")

def main() -> int:
    print(f"Reading {DVF_CSV}...")
    LOT_COLS = [f"lot{i}_numero" for i in range(1, 6)] 
    df =pl.read_csv(
        DVF_CSV,
        schema_overrides={col: pl.Utf8 for col in LOT_COLS},
    )
    print(df)

if __name__ == "__main__":
    main()
