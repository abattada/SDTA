"""GPU-aware scheduler for entry/batch.py.

Memory-driven launch admission:
    Every time a ready step wants a GPU, the scheduler queries the live free
    RAM of all visible GPUs (pynvml preferred, nvidia-smi subprocess fallback).
    It picks the GPU with the most free RAM, gated by ``min_free_mb``. There
    is **no static per-GPU job cap** — the only knobs are
    ``min_free_mb`` (don't try if free RAM below threshold), ``max_concurrent``
    (global hard cap across all GPUs, optional), and ``launch_delay_s``
    (sleep between launches so cudaMalloc shows up in the next query).

OOM retry is event-driven: a step that dies with a CUDA / cuDNN / cuBLAS
allocation or internal error gets requeued; promotion back to ready waits for
another job to finish (memory freed); the retry then goes through the normal
launch path with its own `launch_delay_s` spacing.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


OOM_MARKERS = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: CUDA out of memory",
    "cudaErrorMemoryAllocation",
    # cuBLAS / cuDNN allocation failures — happen when GPU is fragmented or
    # nearly full at process startup. PyTorch wraps these as
    # `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED ...` (no "cuBLAS"
    # prefix) so match the inner status string directly.
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    # Internal / not-initialized cuBLAS / cuDNN states — when a batch packs
    # GPUs hard (high jobs_per_gpu + short delay), these can surface mid-train
    # (e.g. inside loss.backward) instead of clean OOM. The cause is almost
    # always pressure (memory tight, driver context exhausted, concurrent
    # cuDNN handle reuse). Treat as OOM-like so they get the wait-for-finish
    # retry path instead of aborting the whole batch.
    "CUDNN_STATUS_INTERNAL_ERROR",
    "CUDNN_STATUS_EXECUTION_FAILED",
    "CUBLAS_STATUS_INTERNAL_ERROR",
    "CUBLAS_STATUS_NOT_INITIALIZED",
)


@dataclass
class GpuSnapshot:
    gpu_id: int
    free_mb: int
    total_mb: int


@dataclass
class RunningJob:
    chain_index: int
    step_index_in_chain: int
    gpu_id: int
    step: Any  # batch.Step — duck-typed to keep scheduler decoupled from batch.py
    process: subprocess.Popen[str]
    handle: Any
    started_at: float
    attempt: int


@dataclass
class _Pending:
    chain_index: int
    step_index_in_chain: int
    attempts: int = 0


@dataclass
class SchedulerConfig:
    gpu_ids: list[int]
    max_oom_retries: int = 2
    poll_interval_s: float = 1.0
    keep_going: bool = False
    # GPU is considered usable only if its current free RAM is at least this
    # many MiB. Acts as both admission threshold (don't try to launch on a
    # nearly-full card) and self-throttle (when no card qualifies, scheduler
    # sits waiting for some job to finish, then re-queries).
    min_free_mb: int = 1000
    # Hard cap on simultaneous active step subprocesses across all GPUs.
    # Memory admission alone usually self-limits, but this knob exists for CPU
    # / driver headroom when memory could allow more. None = no cap.
    max_concurrent: int | None = None
    # Seconds to sleep after each successful step launch, before considering
    # the next ready step. Protects the NVIDIA driver from concurrent cudaInit
    # / cuBLAS handle creation across many step subprocesses in the same burst,
    # and lets each just-launched process actually claim CUDA memory so the
    # next live query sees the updated free RAM.
    launch_delay_s: float = 0.0


@dataclass
class _OomDeferred:
    pending: _Pending


# pynvml preferred (fast, in-process). Falls back to subprocess nvidia-smi if
# the library isn't installed. Initialization is lazy so importing this module
# doesn't touch the driver.
_PYNVML_STATE: dict[str, Any] = {
    "tried_init": False,
    "available": False,
    "handles": {},
}


def _pynvml_init_if_possible() -> None:
    if _PYNVML_STATE["tried_init"]:
        return
    _PYNVML_STATE["tried_init"] = True
    try:
        import pynvml  # noqa: F401
        pynvml.nvmlInit()
        _PYNVML_STATE["available"] = True
        _PYNVML_STATE["module"] = pynvml
    except Exception as exc:
        _PYNVML_STATE["available"] = False
        _PYNVML_STATE["init_error"] = repr(exc)


def _pynvml_handle(gpu_id: int):
    handles = _PYNVML_STATE["handles"]
    if gpu_id not in handles:
        pynvml = _PYNVML_STATE["module"]
        handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    return handles[gpu_id]


def _query_via_pynvml(gpu_ids: Iterable[int]) -> dict[int, GpuSnapshot]:
    pynvml = _PYNVML_STATE["module"]
    out: dict[int, GpuSnapshot] = {}
    for gid in gpu_ids:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(_pynvml_handle(gid))
            out[gid] = GpuSnapshot(
                gpu_id=gid,
                free_mb=int(info.free // (1024 * 1024)),
                total_mb=int(info.total // (1024 * 1024)),
            )
        except Exception:
            continue
    return out


def query_gpu_free_mb(gpu_ids: Iterable[int]) -> dict[int, GpuSnapshot]:
    """Return {gpu_id: GpuSnapshot}. Prefers pynvml; falls back to nvidia-smi."""
    wanted = list(gpu_ids)
    if not wanted:
        return {}
    _pynvml_init_if_possible()
    if _PYNVML_STATE["available"]:
        return _query_via_pynvml(wanted)
    return _query_via_nvidia_smi(wanted)


def _query_via_nvidia_smi(gpu_ids: list[int]) -> dict[int, GpuSnapshot]:
    """Subprocess fallback when pynvml is unavailable. Slower (~50-150ms per call)."""
    wanted = list(gpu_ids)
    if not wanted:
        return {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={','.join(str(i) for i in wanted)}",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"nvidia-smi query failed: {exc}") from exc

    snapshots: dict[int, GpuSnapshot] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free_mb, total_mb = int(parts[0]), int(parts[1]), int(parts[2])
        snapshots[idx] = GpuSnapshot(gpu_id=idx, free_mb=free_mb, total_mb=total_mb)
    return snapshots


def looks_like_oom(log_path: Path, tail_bytes: int = 64 * 1024) -> bool:
    if not log_path.exists():
        return False
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in tail for marker in OOM_MARKERS)


class Scheduler:
    """Pull-based GPU-aware step scheduler.

    Steps come from a list of chains (each chain is a sequential pipeline).
    The next step of chain ``c`` becomes ready only after the previous step of
    ``c`` succeeds, so per-chain ordering is preserved automatically.
    """

    def __init__(
        self,
        chains: list[list[Any]],
        config: SchedulerConfig,
        start_step: Callable[[Any, int, int, int, int], RunningJob],
        on_finish: Callable[[RunningJob, int], None],
        on_oom_requeue: Callable[[Any, int], None],
        on_skip: Callable[[Any], None] | None = None,
    ) -> None:
        self.chains = chains
        self.config = config
        self._start_step = start_step
        self._on_finish = on_finish
        self._on_oom_requeue = on_oom_requeue
        self._on_skip = on_skip

        # running_per_gpu kept only as a tiebreaker hint, no longer caps capacity.
        self.running_per_gpu: dict[int, int] = {gid: 0 for gid in config.gpu_ids}
        self.active: dict[int, RunningJob] = {}
        self._next_job_key = 0
        self._chain_next: list[int] = [0] * len(chains)
        self._ready: list[_Pending] = [
            _Pending(chain_index=ci, step_index_in_chain=0)
            for ci, chain in enumerate(chains) if chain
        ]
        self._deferred: list[_OomDeferred] = []
        self.failures = 0
        self.done = 0
        self.total_steps = sum(len(c) for c in chains)
        # Credit-per-finish gate for OOM-deferred retries: each job finish (any
        # exit code) grants one credit; each deferred promotion consumes one.
        # Ensures a retried step always waits for some other job to free GPU
        # memory before contending again. No separate cooldown timer — the
        # retry's launch goes through `_launch_one` which already enforces
        # `launch_delay_s`, the single source of pacing.
        self._deferred_credits: int = 0

    def _pick_gpu(self) -> int | None:
        """Query live free RAM on every visible GPU and return the best fit.

        Returns None if no GPU has at least `min_free_mb` free — caller should
        wait for a job to finish (which frees memory) before retrying.
        """
        snaps = query_gpu_free_mb(self.config.gpu_ids)
        # Pick max free; on tie, fewer running jobs wins; final tiebreaker is gpu_id order.
        best_key: tuple[int, int, int] | None = None  # (-free, running, gpu_id)
        best_gid: int | None = None
        for gid in self.config.gpu_ids:
            snap = snaps.get(gid)
            free_mb = snap.free_mb if snap else 0
            if free_mb < self.config.min_free_mb:
                continue
            key = (-free_mb, self.running_per_gpu[gid], gid)
            if best_key is None or key < best_key:
                best_key = key
                best_gid = gid
        return best_gid

    def _maybe_promote_deferred(self) -> None:
        """Promote OOM-deferred items back to ready, **purely event-driven**.

        Single gate: `_deferred_credits > 0`. Each job finish (any exit code)
        grants one credit; each promotion consumes one. Memory is released only
        when something finishes, so this directly couples retry pacing to
        actual capacity opening up.

        The retry's actual relaunch goes through the normal `_launch_one` path,
        which already enforces `launch_delay_s` between launches — that's the
        single source of truth for "don't stack cudaInit too fast". No
        separate OOM cooldown timer.

        Deadlock break: if there are zero active jobs to free memory and zero
        outstanding credits, grant one credit so the head can at least try
        (better to retry-and-fail-permanently than wait forever).
        """
        if not self._deferred:
            return
        if not self.active and self._deferred_credits == 0:
            self._deferred_credits = 1
        while self._deferred and self._deferred_credits > 0:
            item = self._deferred.pop(0)
            self._ready.append(item.pending)
            self._deferred_credits -= 1

    def _drain_skips(self) -> None:
        """Consume ready entries whose step was pre-marked as already-done on disk.

        Loops until stable because skipping a step may reveal the next step in
        the same chain, which may itself be skip-marked.
        """
        while True:
            if not self._ready:
                return
            progressed = False
            remaining: list[_Pending] = []
            for pending in self._ready:
                step = self.chains[pending.chain_index][pending.step_index_in_chain]
                if pending.attempts == 0 and getattr(step, "skip_reason", None):
                    self.done += 1
                    if self._on_skip is not None:
                        self._on_skip(step)
                    ci = pending.chain_index
                    nxt = pending.step_index_in_chain + 1
                    self._chain_next[ci] = nxt
                    if nxt < len(self.chains[ci]):
                        remaining.append(_Pending(chain_index=ci, step_index_in_chain=nxt))
                    progressed = True
                else:
                    remaining.append(pending)
            self._ready = remaining
            if not progressed:
                return

    def _launch_one(self) -> bool:
        self._maybe_promote_deferred()
        self._drain_skips()
        if not self._ready:
            return False
        if self.config.max_concurrent is not None and len(self.active) >= self.config.max_concurrent:
            return False
        gpu_id = self._pick_gpu()
        if gpu_id is None:
            return False
        pending = self._ready.pop(0)
        step = self.chains[pending.chain_index][pending.step_index_in_chain]
        job = self._start_step(
            step,
            pending.chain_index,
            pending.step_index_in_chain,
            gpu_id,
            pending.attempts,
        )
        job.chain_index = pending.chain_index
        job.step_index_in_chain = pending.step_index_in_chain
        job.attempt = pending.attempts
        self.running_per_gpu[gpu_id] = self.running_per_gpu.get(gpu_id, 0) + 1
        self.active[self._next_job_key] = job
        self._next_job_key += 1
        if self.config.launch_delay_s > 0:
            # Sleep before the next launch so cudaInit / cuBLAS handle creation
            # doesn't pile up on the driver's critical section. Also ensures the
            # next live query sees this process's cudaMalloc reflected.
            time.sleep(self.config.launch_delay_s)
        return True

    def _drain_active(self) -> list[tuple[int, RunningJob, int]]:
        finished: list[tuple[int, RunningJob, int]] = []
        for key, job in list(self.active.items()):
            rc = job.process.poll()
            if rc is None:
                continue
            finished.append((key, job, rc))
        return finished

    def _handle_finished(self, key: int, job: RunningJob, return_code: int) -> bool:
        """Returns True if the loop should abort (failure + not keep_going)."""
        job.handle.close()
        self.active.pop(key, None)
        self.running_per_gpu[job.gpu_id] = max(0, self.running_per_gpu.get(job.gpu_id, 0) - 1)
        # Grant one promotion credit so an OOM-deferred step can retry now that
        # memory has been freed somewhere in the pool. (No snapshot cache to
        # invalidate — every _pick_gpu queries live.)
        self._deferred_credits += 1

        if return_code == 0:
            self.done += 1
            self._on_finish(job, return_code)
            ci = job.chain_index
            self._chain_next[ci] = job.step_index_in_chain + 1
            nxt = self._chain_next[ci]
            if nxt < len(self.chains[ci]):
                # DFS-prioritize chain progression: the next step of a chain that
                # just made progress jumps to the front of the ready queue, so the
                # freed slot likely picks up the same chain's next stage before
                # any fresh chain starts. With chain-serial trains this means the
                # first N chains complete (pretrain→trains→test) faster, instead
                # of all N chains' pretrains finishing first BFS-style.
                self._ready.insert(0, _Pending(chain_index=ci, step_index_in_chain=nxt))
            return False

        oom = looks_like_oom(job.step.log_path)
        if oom and job.attempt < self.config.max_oom_retries:
            new_attempt = job.attempt + 1
            self._on_oom_requeue(job, new_attempt)
            self._deferred.append(
                _OomDeferred(
                    pending=_Pending(
                        chain_index=job.chain_index,
                        step_index_in_chain=job.step_index_in_chain,
                        attempts=new_attempt,
                    ),
                )
            )
            return False

        self.done += 1
        self.failures += 1
        self._on_finish(job, return_code)
        return not self.config.keep_going

    def run(self) -> int:
        # Initial fill.
        while self._launch_one():
            pass

        try:
            while self.active or self._ready or self._deferred:
                aborted = False
                for key, job, rc in self._drain_active():
                    if self._handle_finished(key, job, rc):
                        aborted = True
                        break
                if aborted:
                    self._terminate_active()
                    return self.failures

                # Refill any freed slots (or freshly promoted deferred items).
                while self._launch_one():
                    pass

                if self.active or self._deferred:
                    time.sleep(self.config.poll_interval_s)
                elif self._ready:
                    # _ready non-empty but no GPU capacity: brief sleep to avoid busy loop.
                    time.sleep(self.config.poll_interval_s)
        finally:
            self._close_handles()

        return self.failures

    def _terminate_active(self) -> None:
        for job in self.active.values():
            if job.process.poll() is None:
                job.process.terminate()
        deadline = time.time() + 10.0
        for job in self.active.values():
            while job.process.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if job.process.poll() is None:
                job.process.kill()
        self._close_handles()

    def _close_handles(self) -> None:
        for job in self.active.values():
            try:
                job.handle.close()
            except Exception:
                pass


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


__all__ = [
    "GpuSnapshot",
    "RunningJob",
    "Scheduler",
    "SchedulerConfig",
    "format_command",
    "looks_like_oom",
    "query_gpu_free_mb",
]
