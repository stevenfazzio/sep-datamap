"""Shared paths and helpers for the SEP datamap pipeline."""

import os
import stat
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

# Pinned to a fixed quarterly edition so the corpus is reproducible and citable.
EDITION = "fall2026"
ARCHIVE_BASE = f"https://plato.stanford.edu/archives/{EDITION}/"
LIVE_BASE = "https://plato.stanford.edu/"

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw" / EDITION
RAW_ENTRIES = RAW / "entries"
ENTRIES_PARQUET = DATA / "entries.parquet"


def _match_or_default_mode(tmp_path: Path, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace preserves the source mode.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def write_bytes_atomic(data: bytes, output_path: Path) -> None:
    """Write so output_path only ever holds a complete file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _match_or_default_mode(Path(tmp), output_path)
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_npz_atomic(output_path: Path, check=None, **arrays) -> None:
    """np.savez to a temp file, optionally verify it with check(loaded), then swap in."""
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        if check is not None:
            check(np.load(tmp))
        _match_or_default_mode(Path(tmp), output_path)
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_parquet_safely(df, output_path: Path) -> None:
    """Write df to a temp file, verify rows and schema from the footer, then swap in."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        _match_or_default_mode(Path(tmp), output_path)
        meta = pq.read_metadata(tmp)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        # to_arrow_schema(): the Parquet schema's .names are leaf names, so list
        # columns would show up as "element" rather than their top-level name.
        missing = set(df.columns) - set(meta.schema.to_arrow_schema().names)
        assert not missing, f"columns missing on disk: {missing}"
        os.replace(tmp, output_path)
    except BaseException:
        os.unlink(tmp)
        raise
