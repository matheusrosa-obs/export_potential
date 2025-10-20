import polars as pl
from pathlib import Path
import streamlit as st


def get_paths() -> dict:
    """Resolve key project paths regardless of current working dir."""
    current = Path(__file__).resolve()
    # project root is the parent of 'app' directory containing this file
    project_root = current.parent.parent
    return {
        "root": project_root,
        "app": project_root / "app",
        "data_raw": project_root / "data" / "raw",
        "data_processed": project_root / "data" / "processed",
        "data_interim": project_root / "data" / "interim",
        "references": project_root / "references",
    }

def load_all_app_data() -> dict:
    """Lê todos os arquivos da pasta app/data e retorna um dicionário de DataFrames."""
    paths = get_paths()
    data_dir = paths["app"] / "data"
    dfs = {}
    for file in data_dir.iterdir():
        if file.suffix == ".parquet":
            dfs[file.stem] = pl.read_parquet(file)
        elif file.suffix in [".csv", ".tsv"]:
            dfs[file.stem] = pl.read_csv(file, separator="\t" if file.suffix == ".tsv" else ",")
        elif file.suffix in [".xlsx", ".xls"]:
            dfs[file.stem] = pl.read_excel(file)
        # Adicione outros formatos conforme necessário
    return dfs
