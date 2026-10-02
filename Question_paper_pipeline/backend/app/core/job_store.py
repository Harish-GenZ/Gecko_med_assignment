"""
In-memory job store.  For production this would be Redis/DB.
"""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.core.config import JOBS_DIR
from app.models.schemas import JobStatus


_lock: threading.Lock = threading.Lock()
_jobs: Dict[str, dict] = {}


def _init_from_disk() -> None:
    """Preload existing jobs from disk on server startup."""
    if not JOBS_DIR.exists():
        return
    for p_file in JOBS_DIR.glob("*_p*.*"):
        try:
            p_file.unlink(missing_ok=True)
        except Exception:
            pass
    for json_file in JOBS_DIR.glob("*.json"):
        if "_p" in json_file.stem:
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            jid = data.get("job_id")
            if jid:
                if data.get("status") in (JobStatus.QUEUED, JobStatus.PROCESSING, "queued", "processing"):
                    data["status"] = JobStatus.FAILED
                    data["error"] = "Server reloaded while processing. Please re-upload the document."
                _jobs[jid] = data
        except Exception:
            pass


_init_from_disk()


def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    state = {
        "job_id":       job_id,
        "status":       JobStatus.QUEUED,
        "filename":     filename,
        "progress":     0,
        "current_step": "Queued",
        "error":        None,
        "pages_info":   [],
        "result":       None,
        "created_at":   datetime.utcnow().isoformat(),
    }
    with _lock:
        _jobs[job_id] = state
    _persist(job_id, state)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        state = _jobs.get(job_id)
        if state is not None:
            job_data = dict(state)
        else:
            job_data = None

    if job_data is None:
        # Disk fallback
        path = JOBS_DIR / f"{job_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                jid = data.get("job_id")
                if jid:
                    if data.get("status") in (JobStatus.QUEUED, JobStatus.PROCESSING, "queued", "processing"):
                        data["status"] = JobStatus.FAILED
                        data["error"] = "Server reloaded while processing. Please re-upload the document."
                    with _lock:
                        _jobs[jid] = data
                    job_data = dict(data)
            except Exception:
                pass

    if job_data is not None:
        # Collect dynamic worker progress files if processing
        workers = []
        try:
            for p_file in JOBS_DIR.glob(f"{job_id}_p*.json"):
                try:
                    with open(p_file, "r", encoding="utf-8") as f:
                        w_info = json.load(f)
                    if "worker_pid" in w_info:
                        workers.append(w_info)
                except Exception:
                    pass
        except Exception:
            pass

        if workers:
            workers.sort(key=lambda w: w.get("page_number", 0))
            job_data["workers_info"] = workers
        else:
            job_data["workers_info"] = None

    return job_data


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id not in _jobs:
            path = JOBS_DIR / f"{job_id}.json"
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        _jobs[job_id] = json.load(f)
                except Exception:
                    pass
        if job_id not in _jobs:
            return
        _jobs[job_id].update(kwargs)
        state = dict(_jobs[job_id])
    _persist(job_id, state)


def _persist(job_id: str, state: dict) -> None:
    """Write job state to disk (JSON) for durability."""
    path = JOBS_DIR / f"{job_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass  # Non-critical
