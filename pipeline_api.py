"""Client for LEAF pipeline edge function API (replaces direct Supabase access)."""

from __future__ import annotations

from typing import Any

import httpx


class PipelineAPIClient:
    """Communicates with the leaf-pipeline edge function for job management."""

    def __init__(self, api_url: str, api_key: str, client: httpx.AsyncClient):
        self.url = api_url.rstrip("/")
        self.api_key = api_key
        self.client = client

    async def _call(self, action: str, **kwargs: Any) -> dict:
        payload = {"action": action, **kwargs}
        r = await self.client.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "apikey": self.api_key,
            },
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    # ── Job management ───────────────────────────────────────

    async def create_job(
        self, from_bos: int, to_bos: int, force: bool = False
    ) -> dict:
        return await self._call(
            "worker_create_job",
            numer_bos_from=from_bos,
            numer_bos_to=to_bos,
            force=force,
        )

    async def get_job(self, job_id: str) -> dict | None:
        data = await self._call("worker_get_job", job_id=job_id)
        if data.get("job_missing"):
            return None
        return data.get("job")

    async def read_job_status(self, job_id: str) -> str | None:
        data = await self._call("worker_read_status", job_id=job_id)
        if data.get("job_missing"):
            return None
        return data.get("status")

    async def update_progress(self, job_id: str, updates: dict) -> None:
        await self._call("worker_update_progress", job_id=job_id, updates=updates)

    async def append_error(self, job_id: str, error: dict) -> None:
        await self._call("worker_append_error", job_id=job_id, error=error)

    # ── Prompts ──────────────────────────────────────────────

    async def get_prompts(self) -> dict:
        """Returns {prompts: {case: {system, user}, law: ..., solution: ..., model}}."""
        data = await self._call("worker_get_prompts")
        return data.get("prompts", {})

    # ── Matching (delegated to edge function) ────────────────

    async def match_batch(self, leaf_ids: list[int]) -> dict[str, int]:
        """Delegate matching to edge function which has full alias map in Supabase."""
        if not leaf_ids:
            return {"matched": 0, "partial": 0, "unmatched": 0, "total": 0}
        data = await self._call("worker_match_batch", leaf_ids=leaf_ids)
        return data
