"""Client for Legal DB (PostgREST API)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from config import Settings


class LegalDBClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.base = settings.legal_db_base.rstrip("/")
        self.client = client

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    async def get(self, path: str) -> list[dict]:
        r = await self.client.get(f"{self.base}{path}", headers=self._headers())
        r.raise_for_status()
        return r.json()

    async def get_with_count(self, path: str) -> tuple[list[dict], int]:
        r = await self.client.get(
            f"{self.base}{path}",
            headers=self._headers({"Prefer": "count=exact"}),
        )
        r.raise_for_status()
        cr = r.headers.get("content-range", "")
        count = int(cr.split("/")[1]) if "/" in cr else 0
        return r.json(), count

    async def insert(self, table: str, rows: list[dict]) -> Any:
        r = await self.client.post(
            f"{self.base}/{table}",
            headers=self._headers({"Prefer": "return=representation"}),
            content=json.dumps(rows),
        )
        r.raise_for_status()
        text = r.text.strip()
        return json.loads(text) if text else None

    async def insert_leafs(self, leafs: list[dict]) -> Any:
        """Insert leaf records (no upsert — unique constraint removed)."""
        r = await self.client.post(
            f"{self.base}/orzeczenie_leaf",
            headers=self._headers({"Prefer": "return=representation"}),
            content=json.dumps(leafs),
        )
        r.raise_for_status()
        text = r.text.strip()
        return json.loads(text) if text else None

    async def delete_leafs_for_judgment(self, orz_id: int) -> None:
        """Delete all existing leafs (and their article rels) for a judgment."""
        leafs = await self.get(f"/orzeczenie_leaf?orzeczenie_id=eq.{orz_id}&select=id")
        for lf in leafs:
            await self.delete("leaf_artykul_rel", f"leaf_id=eq.{lf['id']}")
        await self.delete("orzeczenie_leaf", f"orzeczenie_id=eq.{orz_id}")

    async def update(self, table: str, filter_: str, data: dict) -> Any:
        r = await self.client.patch(
            f"{self.base}/{table}?{filter_}",
            headers=self._headers({"Prefer": "return=representation"}),
            content=json.dumps(data),
        )
        r.raise_for_status()
        text = r.text.strip()
        return json.loads(text) if text else None

    async def delete(self, table: str, filter_: str) -> bool:
        r = await self.client.delete(
            f"{self.base}/{table}?{filter_}",
            headers=self._headers(),
        )
        return r.is_success

    # ── High-level helpers ───────────────────────────────────

    async def get_judgments_in_range(
        self, from_bos: int, to_bos: int, after_bos: int | None = None
    ) -> list[dict]:
        start = after_bos if after_bos is not None else from_bos
        if start > to_bos:
            return []

        pending: list[dict] = []
        CHUNK = 200

        for chunk_from in range(start, to_bos + 1, CHUNK):
            if len(pending) >= 50:
                break
            chunk_to = min(chunk_from + CHUNK - 1, to_bos)
            bos_vals = ",".join(str(b) for b in range(chunk_from, chunk_to + 1))

            all_in_chunk = await self.get(
                f"/orzeczenia?numer_bos=in.({bos_vals})&select=id,numer_bos,sygnatura"
            )
            if not all_in_chunk:
                continue

            ids = ",".join(str(o["id"]) for o in all_in_chunk)
            existing = await self.get(
                f"/orzeczenie_leaf?orzeczenie_id=in.({ids})&select=orzeczenie_id"
            )
            leafed = {l["orzeczenie_id"] for l in existing}

            pending.extend(
                sorted(
                    [o for o in all_in_chunk if o["id"] not in leafed],
                    key=lambda o: int(o["numer_bos"]),
                )
            )

        return pending[:50]

    async def count_judgments_in_range(self, from_bos: int, to_bos: int) -> int:
        total = 0
        CHUNK = 200
        for chunk_from in range(from_bos, to_bos + 1, CHUNK):
            chunk_to = min(chunk_from + CHUNK - 1, to_bos)
            bos_vals = ",".join(str(b) for b in range(chunk_from, chunk_to + 1))
            _, count = await self.get_with_count(
                f"/orzeczenia?numer_bos=in.({bos_vals})&select=id&limit=1"
            )
            total += count
        return total

    async def get_judgment_content(self, id_: int) -> str | None:
        rows = await self.get(f"/orzeczenia?id=eq.{id_}&select=tresc_orzeczenia")
        return rows[0]["tresc_orzeczenia"] if rows else None

    async def get_leafs_by_judgment(self, orz_id: int) -> list[dict]:
        return await self.get(
            f"/orzeczenie_leaf?orzeczenie_id=eq.{orz_id}"
            f"&select=id,typ,status,tresc,embed_model&order=id.asc"
        )

    async def get_leaf_article_rels(self, leaf_id: int) -> list[dict]:
        return await self.get(
            f"/leaf_artykul_rel?leaf_id=eq.{leaf_id}&select=id,artykul_text_fallback"
        )

    async def update_leaf_embedding(
        self, leaf_id: int, embedding: list[float], model: str
    ) -> None:
        vec_str = f"[{','.join(str(v) for v in embedding)}]"
        await self.update(
            "orzeczenie_leaf",
            f"id=eq.{leaf_id}",
            {
                "embedding": vec_str,
                "status": "embedded",
                "embed_model": model,
                "updated_at": "now()",
            },
        )

    async def mark_leaf_error(self, leaf_id: int, msg: str) -> None:
        await self.update(
            "orzeczenie_leaf",
            f"id=eq.{leaf_id}",
            {"status": "error", "error_message": msg[:500], "updated_at": "now()"},
        )

    async def get_pipeline_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for status in ("pending", "generated", "embedded", "error"):
            _, c = await self.get_with_count(
                f"/orzeczenie_leaf?status=eq.{status}&select=id&limit=1"
            )
            counts[status] = c
        return counts
