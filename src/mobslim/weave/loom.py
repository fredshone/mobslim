"""Looms: where strands run.

A :class:`Loom` is started once per run with the shared
:class:`~mobslim.weave.strand.StrandContext`, then takes
:class:`~mobslim.weave.strand.StrandJob` objects and returns futures of
:class:`~mobslim.weave.strand.StrandResult`. Results depend only on the
job, never on the loom or on completion order.

* :class:`SerialLoom`: runs each job when submitted, in this process (the
  reference, and what tests use).
* :class:`ProcessLoom`: a persistent process pool. The context is pickled
  once per run into a file that each worker loads once (cached by a token,
  as :class:`~mobslim.replanning.ProcessPoolRunner` does). The ledger's plan
  store and the iteration's travel times are written once per snapshot to a
  file in shared memory (``/dev/shm``, where available); a job carries only
  its strand's rows and innovations, and each worker reads the snapshot once.
  :class:`~mobslim.weave.Weave` first slims the store to the rows its
  strands use (:func:`~mobslim.weave.strand.slim_jobs`). Each worker routes
  on ``router_threads`` threads. Results come back with a sparse travel-time
  aggregate (only the cells used), and with the time spent reading the
  snapshot and job and the result's size and unpickling time in their
  metrics; ``snapshot_s`` adds up the time spent writing snapshots.

The pool's size is independent of the number of strands: jobs queue when
there are more strands than workers.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from typing import Optional, Protocol, runtime_checkable

from mobslim.weave.strand import (
    StrandContext,
    StrandJob,
    StrandResult,
    run_strand,
)


@runtime_checkable
class Loom(Protocol):
    def start(self, ctx: StrandContext) -> None: ...
    def submit(self, job: StrandJob) -> "Future[StrandResult]": ...
    def close(self) -> None: ...


class SerialLoom:
    """Runs every strand in this process, at submission."""

    workers = 0

    def __init__(self):
        self._ctx: Optional[StrandContext] = None

    def start(self, ctx: StrandContext) -> None:
        self._ctx = ctx

    def submit(self, job: StrandJob) -> "Future[StrandResult]":
        if self._ctx is None:
            raise RuntimeError("SerialLoom.submit before start.")
        future: Future = Future()
        try:
            future.set_result(run_strand(self._ctx, job))
        except BaseException as e:  # noqa: BLE001 - delivered through the future
            future.set_exception(e)
        return future

    def close(self) -> None:
        self._ctx = None


# per worker process: (token, context) of the current run, and the last
# snapshot read: (path, (store, travel_time))
_shared: Optional[tuple[str, StrandContext]] = None
_store: Optional[tuple[str, tuple]] = None


def _run_job(token: str, path: str, store_path: str, job: bytes) -> bytes:
    """Runs a pickled job; returns the pickled result (so that the main
    process can measure and time its transfer)."""
    global _shared, _store
    t0 = time.perf_counter()
    if _shared is None or _shared[0] != token:
        with open(path, "rb") as f:
            _shared = (token, pickle.load(f))
    t1 = time.perf_counter()
    if _store is None or _store[0] != store_path:
        _store = None  # free the previous store first
        with open(store_path, "rb") as f:
            _store = (store_path, pickle.load(f))
    t2 = time.perf_counter()
    job = pickle.loads(job)
    t3 = time.perf_counter()
    store, travel_time = _store[1]
    result = run_strand(_shared[1], replace(job, store=store, travel_time=travel_time),
                        cross_process=True)
    result.metrics.update(context_read_s=t1 - t0, snapshot_read_s=t2 - t1, job_unpickle_s=t3 - t2)
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle_result(future: Future, out: Future, submitted: float) -> None:
    """``out`` gets ``future``'s pickled result, unpickled."""
    try:
        data = future.result()
        t = time.perf_counter()
        result = pickle.loads(data)
        result.metrics.update(result_mb=len(data) / 1e6, result_unpickle_s=time.perf_counter() - t,
                              worker_wall_s=t - submitted)
        out.set_result(result)
    except BaseException as e:  # noqa: BLE001 - delivered through the future
        out.set_exception(e)


def _init_worker(threads: int = 1):
    from mobslim.routing.compiled import THREADS_ENV

    os.environ.setdefault(THREADS_ENV, str(threads))


def physical_cores() -> int:
    """Physical CPU cores this process may use (hyperthreads not counted),
    from Linux's CPU topology; else the logical CPU count."""
    try:
        cpus = os.sched_getaffinity(0)
    except AttributeError:
        cpus = range(os.cpu_count() or 1)
    cores = set()
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/core_cpus_list") as f:
                cores.add(f.read().strip())
        except OSError:
            return len(cpus) or 1
    return len(cores) or 1


def logical_cpus() -> int:
    """Logical CPUs this process may use (hyperthreads counted)."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


def default_workers(strands: int) -> int:
    """Workers for ``strands`` strands (strand 0 runs in the main process):
    one per other strand, at most one fewer than the physical cores."""
    return max(1, min(strands - 1, physical_cores() - 1))


def _shm_dir() -> Optional[str]:
    """Shared memory where available (files never touch the disk)."""
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class ProcessLoom:
    """Strands in ``workers`` worker processes (a persistent pool, created
    on first use and reused until :meth:`close`).

    The mobsim, scorer and scenario must be picklable and importable by the
    workers. With the default ``spawn`` start method, scripts need an
    ``if __name__ == "__main__":`` guard. Mind the core budget: ``workers``
    times the mobsim's own threads (``PSim``, ``DSim``) should not exceed
    the cores. ``router_threads``: routing threads per strand (default: the
    logical CPUs shared out between the workers and the main process,
    which runs a strand too: routing waits on memory, so hyperthreads
    nearly double it while the other strands simulate);
    ``$MOBSLIM_ROUTER_THREADS`` overrides it.
    """

    def __init__(self, workers: Optional[int] = None, start_method: str = "spawn",
                 router_threads: Optional[int] = None):
        self.workers = workers or os.cpu_count() or 1
        self.router_threads = router_threads or max(1, logical_cpus() // (self.workers + 1))
        self.start_method = start_method
        self._pool: Optional[ProcessPoolExecutor] = None
        self._path: Optional[str] = None
        self._token: Optional[str] = None
        self._runs = 0
        self._store_ref = None
        self._store_path: Optional[str] = None
        self._store_jobs: dict[str, list] = {}
        self.snapshot_s = 0.0  # seconds spent writing snapshots (main process)

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                             mp_context=get_context(self.start_method),
                                             initializer=_init_worker,
                                             initargs=(self.router_threads,))
        return self._pool

    def start(self, ctx: StrandContext) -> None:
        self._release_files()
        self._runs += 1
        self._token = f"{os.getpid()}:{id(self)}:{self._runs}"
        fd, self._path = tempfile.mkstemp(prefix="mobslim-weave-", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(ctx.without_plans(), f, protocol=pickle.HIGHEST_PROTOCOL)
        self._executor()

    def submit(self, job: StrandJob) -> "Future[StrandResult]":
        if self._path is None:
            raise RuntimeError("ProcessLoom.submit before start.")
        if job.store is None:
            raise ValueError("ProcessLoom needs jobs that carry the ledger store.")
        t = time.perf_counter()
        store_path = self._store_file(job.store, job.travel_time)
        self.snapshot_s += time.perf_counter() - t
        data = pickle.dumps(replace(job, store=None, travel_time=None),
                            protocol=pickle.HIGHEST_PROTOCOL)
        submitted = time.perf_counter()
        future = self._executor().submit(_run_job, self._token, self._path, store_path, data)
        self._store_jobs[store_path].append(future)
        out: Future = Future()
        future.add_done_callback(lambda f: _unpickle_result(f, out, submitted))
        return out

    def _store_file(self, store, travel_time) -> str:
        """The store and travel times, written once per snapshot; files of
        older snapshots are removed once their jobs are done."""
        ref = self._store_ref
        if ref is None or ref[0] is not store or ref[1] != len(store) or ref[2] is not travel_time:
            for path, futures in list(self._store_jobs.items()):
                if all(f.done() for f in futures):
                    _unlink(path)
                    del self._store_jobs[path]
            fd, path = tempfile.mkstemp(prefix="mobslim-store-", suffix=".pkl", dir=_shm_dir())
            with os.fdopen(fd, "wb") as f:
                pickle.dump((store, travel_time), f, protocol=pickle.HIGHEST_PROTOCOL)
            # references, not ids: an id can be reused once its object is freed
            self._store_ref, self._store_path = (store, len(store), travel_time), path
            self._store_jobs[path] = []
        return self._store_path

    def _release_files(self) -> None:
        if self._path is not None:
            _unlink(self._path)
            self._path = None
        for path in self._store_jobs:
            _unlink(path)
        self._store_jobs, self._store_ref, self._store_path = {}, None, None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self._release_files()

    def __del__(self):
        try:
            self._release_files()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
