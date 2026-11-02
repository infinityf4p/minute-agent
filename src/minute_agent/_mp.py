"""Worker initializer for process pools. Kept free of heavy imports so that the
thread limits are set before polars / numpy are imported in the worker."""

import os


def init_worker(threads: int) -> None:
    n = str(max(1, int(threads)))
    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = n
