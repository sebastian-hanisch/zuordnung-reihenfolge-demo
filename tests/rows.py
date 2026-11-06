"""Die 240 festen Instanzen der Vergleiche für test_claims.py, parallel gerechnet (ein Prozess je Kern, höchstens vier) und nur einmal je Testlauf.

Zeiten stehen nicht in den Zeilen; die Parallelität ändert nur die Rechenzeit."""

import multiprocessing
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import zrf_evaluation as E  # noqa: E402

_ROWS: list = []


def _workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _job(job):
    return E.row(*job)


def all_rows() -> list:
    """Alle 240 Zeilen in der Reihenfolge von `E.jobs()`; die größten Instanzen (12, 4) zuerst gestartet ist nicht nötig, `chunksize=1` mischt gut genug."""
    if not _ROWS:
        jobs = E.jobs()
        with ProcessPoolExecutor(max_workers=_workers(), mp_context=multiprocessing.get_context("spawn")) as ex:
            _ROWS.extend(ex.map(_job, jobs, chunksize=1))
    return _ROWS


if __name__ == "__main__":
    import json
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    t = time.time()
    rows = all_rows()
    json.dump(rows, open(sys.argv[1], "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(len(rows), round(time.time() - t, 1), "s")
