"""Background job manager for web-launched runs.

A :class:`JobManager` singleton owns a small process pool (default 2
concurrent processes, override with ``WEB_PROFILER_MAX_WORKERS``).  A job is
one or more *units*: sim/dse/thermal jobs have exactly one unit, a batch job
has up to 64.  A scheduler thread dispatches queued units into free process
slots; a reaper thread collects exit codes and advances the schedule.

Every state change is persisted to ``web_profiler/runtime/jobs/<id>.json``
and all child output is appended to ``<id>.log`` next to it.  Jobs that were
queued/running when the server last stopped are marked ``interrupted`` at
startup and are never re-run automatically.

Input validation lives in ``api_jobs``; the manager assumes sane input.
"""

import copy
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .commands import (build_sim_command, build_dse_command,
                       build_thermal_command, expand_sim_specs,
                       ensure_hw_config)
from .config import JOBS_DIR, PROJECT_ROOT

JOB_TYPES = ("sim", "batch", "dse", "thermal")
TERMINAL_STATUSES = ("done", "failed", "cancelled", "interrupted")
UNIT_TERMINAL = ("done", "failed", "cancelled")
MAX_BATCH_UNITS = 64
MAX_LOG_WINDOW = 1024 * 1024
SIGTERM_GRACE_S = 3.0


def max_workers() -> int:
    """Process-slot limit from ``WEB_PROFILER_MAX_WORKERS`` (default 2)."""
    try:
        return max(1, int(os.environ.get("WEB_PROFILER_MAX_WORKERS", "2")))
    except ValueError:
        return 2


class JobManager:
    """Thread-safe scheduler/runner/persister for simulation jobs."""

    def __init__(self, jobs_dir=None):
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict] = {}
        # (job_id, unit_idx) -> runtime-only state (never persisted)
        self._procs: Dict[tuple, subprocess.Popen] = {}
        self._unit_meta: Dict[tuple, Dict] = {}
        self._log_files: Dict[str, object] = {}
        self._cancelling: set = set()
        self._jobs_dir = Path(jobs_dir) if jobs_dir else JOBS_DIR
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._max_workers = max_workers()
        self._sched_event = threading.Event()
        self._load_persisted()
        threading.Thread(target=self._scheduler_loop, daemon=True,
                         name="job-scheduler").start()
        threading.Thread(target=self._reaper_loop, daemon=True,
                         name="job-reaper").start()

    # ------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------
    def submit_sim(self, params: Dict) -> Dict:
        cmd = build_sim_command(params)
        return self._new_job("sim", cmd["label"], cmd["params"], [cmd])

    def submit_batch(self, params: Dict) -> Dict:
        specs = expand_sim_specs(params)
        if len(specs) > MAX_BATCH_UNITS:
            raise ValueError(f"batch expands to {len(specs)} units "
                             f"(max {MAX_BATCH_UNITS})")
        cmds = [build_sim_command(s) for s in specs]
        label = f"batch sweep ({len(cmds)} units)"
        return self._new_job("batch", label, params, cmds)

    def submit_dse(self, params: Dict) -> Dict:
        cmd = build_dse_command(params)
        return self._new_job("dse", cmd["label"], params, [cmd])

    def submit_thermal(self, params: Dict) -> Dict:
        cmd = build_thermal_command(params)
        return self._new_job("thermal", cmd["label"], params, [cmd])

    def _submit_argv(self, argv: List[str], label: str = "raw",
                     job_type: str = "sim", cwd: Optional[str] = None) -> Dict:
        """Internal: submit a job from a raw argv (tests, internal reuse)."""
        argv = [str(a) for a in argv]
        spec = {"argv": argv, "label": label,
                "display": " ".join(shlex.quote(a) for a in argv),
                "cwd": cwd or str(PROJECT_ROOT), "expected_output": None}
        return self._new_job(job_type, label, {"argv": argv}, [spec])

    def _new_job(self, job_type: str, label: str, params: Dict,
                 unit_specs: List[Dict]) -> Dict:
        with self._lock:
            job_id = uuid.uuid4().hex[:8]
            while job_id in self._jobs or \
                    (self._jobs_dir / f"{job_id}.json").exists():
                job_id = uuid.uuid4().hex[:8]
            log_file = self._jobs_dir / f"{job_id}.log"
            try:
                log_path = log_file.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:  # custom jobs_dir outside the project
                log_path = str(log_file)
            units = []
            for i, spec in enumerate(unit_specs):
                units.append({
                    "idx": i,
                    "label": spec.get("label") or f"unit {i}",
                    "argv": list(spec["argv"]),
                    "display": spec.get("display") or
                    " ".join(shlex.quote(a) for a in spec["argv"]),
                    "expected_output": spec.get("expected_output"),
                    "status": "queued",
                    "returncode": None,
                    "pid": None,
                })
                self._unit_meta[(job_id, i)] = {
                    "cwd": spec.get("cwd") or str(PROJECT_ROOT),
                    "hw_config_args": spec.get("hw_config_args"),
                }
            job = {"id": job_id, "type": job_type, "label": label,
                   "status": "queued", "created_at": time.time(),
                   "started_at": None, "ended_at": None,
                   "params": params, "units": units,
                   "log_path": log_path, "error": None}
            self._jobs[job_id] = job
            self._persist_locked(job)
        self._sched_event.set()
        return self._copy(job)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def list_jobs(self, limit: Optional[int] = None) -> List[Dict]:
        """All jobs, newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(),
                          key=lambda j: j["created_at"], reverse=True)
            if limit:
                jobs = jobs[:limit]
            return [self._copy(j) for j in jobs]

    def get(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._copy(job) if job is not None else None

    @staticmethod
    def progress(job: Dict) -> Dict[str, int]:
        """Per-status unit counts, computed on demand."""
        out = {"total": len(job["units"]), "queued": 0, "running": 0,
               "done": 0, "failed": 0, "cancelled": 0}
        for u in job["units"]:
            out[u["status"]] = out.get(u["status"], 0) + 1
        return out

    def log_window(self, job_id: str, offset: int = 0,
                   length: int = 65536, tail: int = 0) -> Optional[Dict]:
        """A byte window of the job log (``tail`` reads from the end)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            log_path = job["log_path"]
        path = Path(log_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / log_path
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        length = max(0, min(int(length), MAX_LOG_WINDOW))
        offset = max(0, int(offset))
        if tail:
            offset = max(0, size - length)
        offset = min(offset, size)
        raw = b""
        if length and size:
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    raw = f.read(length)
            except OSError:
                raw = b""
        return {"total_size": size, "offset": offset, "length": len(raw),
                "eof": offset + len(raw) >= size,
                "text": raw.decode("utf-8", errors="replace")}

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------
    def cancel(self, job_id: str) -> Optional[Dict]:
        """SIGTERM (then SIGKILL after 3 s) running units; drop queued ones.

        Returns the job, ``None`` if unknown; raises ``ValueError`` when the
        job is not queued/running.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in ("queued", "running"):
                raise ValueError(f"job is {job['status']}")
            self._cancelling.add(job_id)
            procs = []
            for unit in job["units"]:
                key = (job_id, unit["idx"])
                if unit["status"] == "queued":
                    unit["status"] = "cancelled"
                elif unit["status"] == "running" and key in self._procs:
                    procs.append(self._procs[key])
            self._maybe_finish_locked(job)
            self._persist_locked(job)
        # Signal outside the lock: whole process groups, parallel grace.
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
        deadline = time.time() + SIGTERM_GRACE_S
        for proc in procs:
            try:
                proc.wait(timeout=max(0.05, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # unkillable (D-state); the reaper keeps polling it
        self._reap()
        with self._lock:
            return self._copy(self._jobs[job_id])

    def delete(self, job_id: str) -> Optional[Dict]:
        """Remove a terminal job plus its json/log files.

        Returns the removed job, ``None`` if unknown; raises ``ValueError``
        when the job is not in a terminal status.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in TERMINAL_STATUSES:
                raise ValueError(f"job is {job['status']}")
            self._jobs.pop(job_id)
            for key in [k for k in self._unit_meta if k[0] == job_id]:
                self._unit_meta.pop(key, None)
        for suffix in (".json", ".log"):
            try:
                (self._jobs_dir / f"{job_id}{suffix}").unlink()
            except OSError:
                pass
        return job

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------
    def _scheduler_loop(self):
        while True:
            self._sched_event.wait(timeout=1.0)
            self._sched_event.clear()
            try:
                self._dispatch()
            except Exception:
                traceback.print_exc()

    def _dispatch(self):
        with self._lock:
            free = self._max_workers - len(self._procs)
            if free <= 0:
                return
            for job in sorted(self._jobs.values(),
                              key=lambda j: j["created_at"]):
                if free <= 0:
                    break
                if job["status"] not in ("queued", "running"):
                    continue
                if job["id"] in self._cancelling:
                    continue
                for unit in job["units"]:
                    if free <= 0:
                        break
                    if unit["status"] != "queued":
                        continue
                    if self._start_unit_locked(job, unit):
                        free -= 1

    def _start_unit_locked(self, job: Dict, unit: Dict) -> bool:
        key = (job["id"], unit["idx"])
        meta = self._unit_meta.get(key) or {}
        hw = meta.get("hw_config_args")
        if hw:
            try:
                ensure_hw_config(**hw)
            except OSError as e:
                unit["status"] = "failed"
                job["error"] = f"hw_config setup failed: {e}"
                self._maybe_finish_locked(job)
                self._persist_locked(job)
                return False
        try:
            logf = self._log_files.get(job["id"])
            if logf is None:
                logf = open(self._jobs_dir / f"{job['id']}.log", "ab")
                self._log_files[job["id"]] = logf
            logf.write((f"\n===== unit {unit['idx']}: {unit['label']} "
                        f"=====\n$ {unit['display']}\n\n").encode())
            logf.flush()
            proc = subprocess.Popen(
                unit["argv"], cwd=meta.get("cwd") or str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL, stdout=logf,
                stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            unit["status"] = "failed"
            job["error"] = f"launch failed: {e}"
            try:
                logf.write(f"[launch failed] {e}\n".encode())
                logf.flush()
            except (OSError, UnboundLocalError):
                pass
            self._maybe_finish_locked(job)
            self._persist_locked(job)
            return False
        unit["status"] = "running"
        unit["pid"] = proc.pid
        self._procs[key] = proc
        if job["status"] == "queued":
            job["status"] = "running"
            job["started_at"] = time.time()
        self._persist_locked(job)
        return True

    # ------------------------------------------------------------------
    # reaping
    # ------------------------------------------------------------------
    def _reaper_loop(self):
        while True:
            time.sleep(0.2)
            try:
                self._reap()
            except Exception:
                traceback.print_exc()

    def _reap(self):
        with self._lock:
            reaped = False
            for key, proc in list(self._procs.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                self._finalize_unit_locked(key, rc)
                reaped = True
        if reaped:
            self._sched_event.set()  # freed slots -> dispatch more units

    def _finalize_unit_locked(self, key: tuple, rc: int):
        job_id, idx = key
        self._procs.pop(key, None)
        job = self._jobs.get(job_id)
        if job is None:
            return
        unit = job["units"][idx]
        if unit["status"] in UNIT_TERMINAL:
            return
        unit["returncode"] = rc
        if job_id in self._cancelling:
            unit["status"] = "cancelled"
        else:
            unit["status"] = "done" if rc == 0 else "failed"
        self._maybe_finish_locked(job)
        self._persist_locked(job)

    def _maybe_finish_locked(self, job: Dict):
        if any(u["status"] not in UNIT_TERMINAL for u in job["units"]):
            return
        job["ended_at"] = time.time()
        if job["id"] in self._cancelling:
            job["status"] = "cancelled"
            self._cancelling.discard(job["id"])
        elif all(u["returncode"] == 0 for u in job["units"]):
            job["status"] = "done"
        else:
            job["status"] = "failed"
        logf = self._log_files.pop(job["id"], None)
        if logf is not None:
            try:
                logf.close()
            except OSError:
                pass
        self._sched_event.set()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load_persisted(self):
        for path in sorted(self._jobs_dir.glob("*.json")):
            try:
                job = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or "id" not in job \
                    or "units" not in job:
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "interrupted"
                job["ended_at"] = time.time()
                job["error"] = "server stopped while job was active"
                for u in job["units"]:
                    if u.get("status") in ("queued", "running"):
                        u["status"] = "cancelled"
                        u["pid"] = None
                self._persist_locked(job)
            self._jobs[job["id"]] = job

    def _persist_locked(self, job: Dict):
        path = self._jobs_dir / f"{job['id']}.json"
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(job, indent=2))
            tmp.replace(path)
        except OSError:
            pass

    @staticmethod
    def _copy(job: Dict) -> Dict:
        return copy.deepcopy(job)


_MANAGER: Optional[JobManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> JobManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER
