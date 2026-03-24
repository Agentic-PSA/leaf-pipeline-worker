from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import Settings
from runner import WorkerControl, run_pipeline

app = FastAPI(title="LEAF Pipeline Worker API", version="1.0.0")
settings = Settings()
security = HTTPBearer(auto_error=False)


class StartJobRequest(BaseModel):
    from_bos: int = Field(..., ge=0)
    to_bos: int = Field(..., ge=0)
    workers: int | None = Field(default=None, ge=1, le=200)
    force: bool = False


class ResumeJobRequest(BaseModel):
    workers: int | None = Field(default=None, ge=1, le=200)


@dataclass
class JobHandle:
    control: WorkerControl
    task: asyncio.Task
    requested_at: str
    requested_workers: int | None = None
    job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkerRuntime:
    def __init__(self, base_settings: Settings):
        self.base_settings = base_settings
        self.handles: dict[str, JobHandle] = {}
        self._lock = asyncio.Lock()

    def _serialize_handle(self, handle: JobHandle) -> dict[str, Any]:
        task_error: str | None = None
        if handle.task.done() and not handle.task.cancelled():
            try:
                exc = handle.task.exception()
            except Exception as runtime_exc:  # pragma: no cover
                exc = runtime_exc
            if exc:
                task_error = str(exc)

        return {
            "job_id": handle.job_id or handle.control.state.get("job_id"),
            "requested_at": handle.requested_at,
            "requested_workers": handle.requested_workers,
            "running": not handle.task.done(),
            "done": handle.task.done(),
            "cancelled": handle.task.cancelled(),
            "pause_requested": handle.control.pause_requested,
            "stop_requested": handle.control.stop_requested,
            "task_error": task_error,
            "state": dict(handle.control.state),
            "log_count": len(handle.control.logs),
            "metadata": handle.metadata,
        }

    async def _register_new_handle(
        self,
        control: WorkerControl,
        task: asyncio.Task,
        requested_workers: int | None,
        metadata: dict[str, Any],
    ) -> JobHandle:
        handle = JobHandle(
            control=control,
            task=task,
            requested_at=datetime.now(timezone.utc).isoformat(),
            requested_workers=requested_workers,
            metadata=metadata,
        )
        temp_key = f"pending:{id(handle)}"
        self.handles[temp_key] = handle

        await asyncio.wait_for(control.started_event.wait(), timeout=20)
        job_id = control.state.get("job_id")
        if not job_id:
            if task.done():
                try:
                    exc = task.exception()
                except Exception as runtime_exc:  # pragma: no cover
                    exc = runtime_exc
                raise RuntimeError(str(exc) if exc else "Worker exited before returning job_id")
            raise RuntimeError("Worker did not expose job_id")

        handle.job_id = job_id
        self.handles[job_id] = handle
        self.handles.pop(temp_key, None)
        return handle

    async def start_job(self, payload: StartJobRequest) -> dict[str, Any]:
        async with self._lock:
            control = WorkerControl(max_logs=self.base_settings.worker_api_log_limit)
            task = asyncio.create_task(
                run_pipeline(
                    self.base_settings,
                    None,
                    payload.from_bos,
                    payload.to_bos,
                    payload.force,
                    control=control,
                    workers_override=payload.workers,
                )
            )
            handle = await self._register_new_handle(
                control,
                task,
                payload.workers,
                {
                    "action": "start",
                    "from_bos": payload.from_bos,
                    "to_bos": payload.to_bos,
                    "force": payload.force,
                },
            )
            return self._serialize_handle(handle)

    async def resume_job(self, job_id: str, workers: int | None = None) -> dict[str, Any]:
        async with self._lock:
            existing = self.handles.get(job_id)
            if existing and not existing.task.done():
                existing.control.request_resume()
                existing.control.log("info", "Resume requested via FastAPI")
                return self._serialize_handle(existing)

            control = WorkerControl(max_logs=self.base_settings.worker_api_log_limit)
            task = asyncio.create_task(
                run_pipeline(
                    self.base_settings,
                    job_id,
                    None,
                    None,
                    False,
                    control=control,
                    workers_override=workers,
                )
            )
            handle = JobHandle(
                control=control,
                task=task,
                requested_at=datetime.now(timezone.utc).isoformat(),
                requested_workers=workers,
                job_id=job_id,
                metadata={"action": "resume"},
            )
            self.handles[job_id] = handle
            await asyncio.wait_for(control.started_event.wait(), timeout=20)
            return self._serialize_handle(handle)

    async def pause_job(self, job_id: str) -> dict[str, Any]:
        handle = self.handles.get(job_id)
        if not handle:
            raise KeyError(job_id)
        handle.control.request_pause()
        handle.control.log("warning", "Pause requested via FastAPI")
        return self._serialize_handle(handle)

    async def stop_job(self, job_id: str) -> dict[str, Any]:
        handle = self.handles.get(job_id)
        if not handle:
            raise KeyError(job_id)
        handle.control.request_stop()
        handle.control.log("warning", "Stop requested via FastAPI")
        return self._serialize_handle(handle)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for key, handle in self.handles.items():
            if key.startswith("pending:"):
                continue
            jobs.append(self._serialize_handle(handle))
        jobs.sort(key=lambda item: item["requested_at"], reverse=True)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        handle = self.handles.get(job_id)
        if not handle:
            return None
        return self._serialize_handle(handle)

    def get_logs(self, job_id: str, limit: int) -> list[dict[str, str]]:
        handle = self.handles.get(job_id)
        if not handle:
            raise KeyError(job_id)
        return list(handle.control.logs)[-limit:]


runtime = WorkerRuntime(settings)


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    x_api_key: str | None = Header(default=None),
) -> None:
    expected = settings.worker_api_token
    if not expected:
        return

    candidate = x_api_key or (credentials.credentials if credentials else None)
    if candidate != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


@app.get("/health")
async def health(_: None = Depends(require_auth)) -> dict[str, Any]:
    return {
        "ok": True,
        "service": "leaf-pipeline-worker",
        "active_jobs": sum(1 for job in runtime.list_jobs() if job["running"]),
        "known_jobs": len(runtime.list_jobs()),
    }


@app.get("/jobs")
async def list_jobs(_: None = Depends(require_auth)) -> dict[str, Any]:
    return {"jobs": runtime.list_jobs()}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    job = runtime.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found in worker runtime")
    return job


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 200, _: None = Depends(require_auth)) -> dict[str, Any]:
    if limit < 1 or limit > 2000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be between 1 and 2000")
    try:
        logs = runtime.get_logs(job_id, limit)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found in worker runtime") from exc
    return {"job_id": job_id, "logs": logs}


@app.post("/jobs/start", status_code=status.HTTP_202_ACCEPTED)
async def start_job(payload: StartJobRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    if payload.to_bos < payload.from_bos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="to_bos must be >= from_bos")
    try:
        return await runtime.start_job(payload)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def resume_job(job_id: str, payload: ResumeJobRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    try:
        return await runtime.resume_job(job_id, payload.workers)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/pause", status_code=status.HTTP_202_ACCEPTED)
async def pause_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    try:
        return await runtime.pause_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found in worker runtime") from exc


@app.post("/jobs/{job_id}/stop", status_code=status.HTTP_202_ACCEPTED)
async def stop_job(job_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    try:
        return await runtime.stop_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found in worker runtime") from exc
