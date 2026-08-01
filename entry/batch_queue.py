"""Sequential / N-concurrent queue runner over multiple batch configs.

Wraps entry/batch.py. Each config invokes a fresh subprocess; the queue
maintains up to N concurrent at a time and starts the next from the list
when a slot frees up.

Example:
    # Run the 5 configs of one ablation cohort, one at a time, fail-fast:
    python entry/batch_queue.py \\
        SDTA/medium/ablation/no_tda/heavy \\
        SDTA/medium/ablation/no_tda/light \\
        SDTA/medium/ablation/no_tda/mid \\
        SDTA/medium/ablation/no_tda/pems_heavy \\
        SDTA/medium/ablation/no_tda/pems_light

    # Two cohorts at once, keep going on failure, propagate --min-free-mb:
    python entry/batch_queue.py --concurrent 2 --keep-going --min-free-mb 15000 \\
        SDTA/medium/ablation/no_tda/heavy SDTA/medium/ablation/no_diffusion/heavy

    # Read queue from a text file (one config per line; '#' is comment):
    python entry/batch_queue.py --list main_table.txt --concurrent 2
    # other shipped lists: capacity_sweep.txt, extended.txt

GPU pinning is each config's own responsibility (cuda_visible_devices in the
config); the queue doesn't try to coordinate GPU allocation across batches.
Concurrent batches that share GPUs should rely on min_free_mb + the
scheduler's snapshot cache to keep out of each other's way.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

# Reuse batch.py's config resolver so the same name conventions work
# ("SDTA/v9/lineage_token/heavy" vs full path).
sys.path.insert(0, str(THIS_DIR))
from batch import _resolve_config_path  # noqa: E402


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


CONFIG_ROOT = THIS_DIR / "batch_configs"


def _resolve_list_path(raw: str) -> Path:
    """Find a list file; mirrors batch.py config resolution so users can pass
    either a full path or a short name under entry/batch_configs/."""
    p = Path(raw)
    if p.exists():
        return p
    candidate = CONFIG_ROOT / raw
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"List file not found: {raw} "
        f"(tried literal path and {candidate.relative_to(PROJECT_ROOT)})"
    )


def _read_list_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _print_tail(log_path: Path, lines: int = 30) -> None:
    if not log_path.exists():
        return
    try:
        tail = log_path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return
    print("    --- last log lines ---")
    for line in tail:
        print(f"    {line}")
    print("    --- end ---")


def _terminate_tree(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Tear down a batch.py subprocess *and its whole process group*, then reap it.

    batch.py spawns the actual training subprocesses (``model.*.pretrain`` etc.),
    which spawn their own DataLoader workers. Signalling only ``proc`` leaves
    those grandchildren holding CUDA memory — reparented to init and still
    running. Because we launch batch.py with ``start_new_session=True`` it leads
    its own process group (pgid == pid), so we can signal the entire tree at once:
    SIGTERM, then SIGKILL after a grace period. Finally ``wait()`` reaps ``proc``
    so it does not linger as a ``<defunct>`` zombie.
    """
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        deadline = time.time() + grace
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue runner over multiple batch configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("configs", nargs="*", help="Config paths or names (resolved by batch.py).")
    parser.add_argument("--list", help="Read additional config names from a file (one per line; '#' = comment).")
    parser.add_argument("--concurrent", "-N", type=int, default=1,
                        help=(
                            "Number of batch.py subprocesses running simultaneously (queue-level). "
                            "Default 1 (strict sequential). Note: each batch.py has its own "
                            "`concurrent` knob for step-level cap inside that batch — set via "
                            "`concurrent` config key or batch.py's own --concurrent."
                        ))
    parser.add_argument("--keep-going", action="store_true",
                        help=(
                            "Don't stop starting new batches when one fails (still keep running "
                            "ones). Also forwarded to each inner batch.py so a single step failure "
                            "inside one batch doesn't abort that batch either."
                        ))
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between status polls. Default 2.")
    parser.add_argument("--config-delay", type=float, default=15.0,
                        help=(
                            "Seconds to wait after launching one batch.py (one config) before "
                            "launching the next. Default 15 (default-on). Protects against (a) "
                            "driver instability when multiple processes hit cudaInit / cuBLAS "
                            "handle creation simultaneously, and (b) two batches picking the same "
                            "idle-looking GPU because the first hasn't claimed CUDA memory yet. "
                            "Set 0 only if you accept those risks. No effect when --concurrent 1."
                        ))
    parser.add_argument("--delay", type=float, default=None,
                        help=(
                            "Seconds between step launches *within* each batch.py — forwarded "
                            "via batch.py --delay. None (default) lets batch.py use its own "
                            "default (15). Set explicitly to override every spawned batch.py."
                        ))
    # Pass-through flags forwarded to every batch.py invocation.
    parser.add_argument("--rerun", action="store_true",
                        help="Forwarded: force re-run, ignore existing markers.")
    parser.add_argument("--min-free-mb", type=int,
                        help="Forwarded: skip GPUs with free RAM below this many MiB.")
    parser.add_argument("--max-oom-retries", type=int,
                        help="Forwarded: override OOM retry cap.")
    parser.add_argument("--stages", help="Forwarded: comma-separated stages.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Forwarded: each batch.py prints commands and exits without launching.")
    args = parser.parse_args()

    configs = list(args.configs)
    if args.list:
        configs.extend(_read_list_file(_resolve_list_path(args.list)))
    if not configs:
        raise SystemExit("No configs provided. Use positional args or --list.")
    if args.concurrent < 1:
        raise SystemExit("--concurrent must be >= 1")

    # Resolve & validate all configs upfront; fail before launching anything.
    resolved: list[tuple[str, Path]] = []
    for cfg in configs:
        try:
            resolved.append((cfg, _resolve_config_path(cfg)))
        except SystemExit as exc:
            raise SystemExit(f"Config not found: {cfg}") from exc

    forward: list[str] = []
    if args.rerun:
        forward.append("--rerun")
    if args.keep_going:
        forward.append("--keep-going")
    if args.min_free_mb is not None:
        forward.extend(["--min-free-mb", str(args.min_free_mb)])
    if args.max_oom_retries is not None:
        forward.extend(["--max-oom-retries", str(args.max_oom_retries)])
    if args.stages:
        forward.extend(["--stages", args.stages])
    if args.dry_run:
        forward.append("--dry-run")
    if args.delay is not None:
        forward.extend(["--delay", str(args.delay)])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "outputs" / "batch_queue" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    total = len(resolved)
    print(f"Queue: {total} config(s), concurrent={args.concurrent}, keep_going={args.keep_going}")
    print(f"Log dir: {log_dir.relative_to(PROJECT_ROOT)}")
    if forward:
        print(f"Forwarded flags: {shlex.join(forward)}")
    for i, (name, path) in enumerate(resolved, 1):
        print(f"  {i:>3}. {name}  ({path.relative_to(PROJECT_ROOT)})")
    print()

    pending: list[tuple[str, Path]] = list(resolved)
    # cfg_name -> (Popen, log_handle, log_path, started_at)
    running: dict[str, tuple[subprocess.Popen, Any, Path, float]] = {}
    completed: list[str] = []
    failed: list[tuple[str, int]] = []
    stop_new = False

    def status_summary() -> str:
        done = len(completed) + len(failed)
        return (
            f"running {len(running)}, pending {len(pending)}, "
            f"done {len(completed)}, failed {len(failed)} | overall {done}/{total}"
        )

    def shutdown_running(reason: str) -> None:
        """Kill every still-running batch and its GPU jobs, then reap + close logs."""
        if running:
            print(f"[queue] {reason}: terminating {len(running)} running batch(es) and their GPU jobs")
        for name, (proc, handle, _log_path, _started) in list(running.items()):
            _terminate_tree(proc)
            try:
                handle.close()
            except Exception:
                pass
        running.clear()

    # Treat SIGTERM like Ctrl-C so `kill <queue-pid>` also tears the whole tree
    # down instead of orphaning batch.py + its training subprocesses on the GPUs.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt
    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Loop while work is in flight OR there is still launchable pending work.
        # Once fail-fast sets stop_new, pending is no longer launchable, so the
        # loop must exit as soon as the last running batch drains — otherwise it
        # spins forever on a non-empty pending list (the old `pending or running`
        # condition hung here with `running 0, pending N`).
        while running or (pending and not stop_new):
            # Launch as many as concurrent allows.
            while not stop_new and pending and len(running) < args.concurrent:
                name, path = pending.pop(0)
                log_path = log_dir / f"{_sanitize(name)}.log"
                # `-u` forces unbuffered stdout/stderr so the per-config log
                # streams in real time instead of sitting in batch.py's internal
                # buffer until the subprocess exits.
                cmd = [sys.executable, "-u", str(THIS_DIR / "batch.py"), name] + forward
                handle = log_path.open("w")
                handle.write(f"# config: {name}\n")
                handle.write(f"# resolved: {path}\n")
                handle.write(f"# command: {shlex.join(cmd)}\n")
                handle.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
                handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # Own process group so teardown can signal batch.py *and* the
                    # training subprocesses it spawns (plus their DataLoader
                    # workers) as one unit — see _terminate_tree.
                    start_new_session=True,
                )
                running[name] = (proc, handle, log_path, time.time())
                rel_log = log_path.relative_to(PROJECT_ROOT)
                print(f"[queue-start] {name} | log: {rel_log} | {status_summary()}")
                # Stagger: give the just-launched batch.py time to claim its CUDA
                # memory before the next one queries nvidia-smi. Only sleep if
                # there's still work waiting AND a slot would open.
                if args.config_delay > 0 and pending and len(running) < args.concurrent:
                    print(f"[queue-config-delay] sleeping {args.config_delay:g}s before next launch")
                    time.sleep(args.config_delay)

            time.sleep(args.poll_interval)

            for name, (proc, handle, log_path, started_at) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                elapsed = time.time() - started_at
                handle.close()
                del running[name]
                if rc == 0:
                    completed.append(name)
                    print(f"[queue-done] {name} ({elapsed:,.0f}s) | {status_summary()}")
                else:
                    failed.append((name, rc))
                    rel_log = log_path.relative_to(PROJECT_ROOT)
                    print(f"[queue-failed] {name} rc={rc} ({elapsed:,.0f}s) | log: {rel_log} | {status_summary()}")
                    _print_tail(log_path)
                    if not args.keep_going and not stop_new:
                        stop_new = True
                        print(
                            f"[queue] fail-fast: not starting new batches; "
                            f"waiting for {len(running)} still running to finish "
                            f"({len(pending)} skipped)"
                        )
    except KeyboardInterrupt:
        print("\n[queue] interrupted — shutting down")
        shutdown_running("interrupt")
        raise SystemExit(130)
    finally:
        # Guarantee no batch (and no GPU grandchild) is left running/zombied on
        # any exit path — normal, fail-fast, KeyboardInterrupt, or crash.
        shutdown_running("cleanup")
        signal.signal(signal.SIGTERM, previous_sigterm)

    print()
    skipped = total - len(completed) - len(failed)
    print(f"Queue done: {len(completed)} ok, {len(failed)} failed, {skipped} skipped (of {total})")
    if failed:
        print("Failed:")
        for name, rc in failed:
            print(f"  - {name} (rc={rc})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
