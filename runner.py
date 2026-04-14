#!/usr/bin/env python3
"""Core LEAF pipeline runner with optional in-process control hooks."""

from __future__ import annotations

import asyncio
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from rich.console import Console

from config import Settings
from legal_db import LegalDBClient
from llm import (
    CreditsExhaustedError,
    Prompts,
    RateLimitError,
    generate_embeddings,
    generate_leafs,
)
from pipeline_api import PipelineAPIClient
import re


def map_court_type_id(rodzaj_organu: str | None) -> int | None:
    """Map rodzaj_organu string to court_type_id (1-7).

    Uses fuzzy / contains matching to handle trailing spaces, typos
    (e.g. 'Konsytytucyjny'), and extra suffixes in the source data.
    """
    if not rodzaj_organu:
        return None
    # Normalize: collapse whitespace, strip, lowercase
    r = " ".join(rodzaj_organu.split()).lower()
    if not r:
        return None
    # 1 – Sąd Najwyższy
    if r.startswith("sąd najwyższy") or r.startswith("sad najwyzszy"):
        return 1
    # 2 – Naczelny Sąd Administracyjny
    if "naczelny" in r and "administracyjny" in r:
        return 2
    # 3 – Sąd powszechny (Rejonowy, Okręgowy, Apelacyjny)
    if re.match(r"^s[aą]d (rejonowy|okr[eę]gowy|apelacyjny|powszechny)", r):
        return 3
    # 4 – Wojewódzki Sąd Administracyjny
    if ("wojew" in r) and ("administracyjny" in r):
        return 4
    if r.startswith("wsa"):
        return 4
    # 5 – Trybunał Konstytucyjny (handles typos like konsytytucyjny)
    if ("trybuna" in r) and ("konstytu" in r):
        return 5
    if r.strip() in ("tk",):
        return 5
    # 6 – Krajowa Izba Odwoławcza
    if ("krajowa" in r) and ("odwo" in r):
        return 6
    if r.startswith("kio"):
        return 6
    # 7 – Trybunał Sprawiedliwości
    if ("trybuna" in r) and ("sprawiedliwo" in r):
        return 7
    if "tsue" in r:
        return 7
    return None

console = Console()


@dataclass
class WorkerControl:
    """Local runtime control for CLI signals and FastAPI actions."""

    max_logs: int = 500
    stop_requested: bool = False
    pause_requested: bool = False
    logs: deque[dict[str, str]] = field(default_factory=deque)
    state: dict[str, Any] = field(default_factory=dict)
    started_event: asyncio.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.started_event = asyncio.Event()

    def log(self, level: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if len(self.logs) >= self.max_logs:
            self.logs.popleft()
        self.logs.append(entry)
        self.state["last_log_at"] = entry["timestamp"]
        self.state["last_message"] = message

        style = {
            "debug": "dim",
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }.get(level, "white")
        console.print(f"[{style}]{message}[/]")

    def request_pause(self) -> None:
        self.pause_requested = True
        self.stop_requested = False

    def request_resume(self) -> None:
        self.pause_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True
        self.pause_requested = False


def _sync_state(control: WorkerControl, **kwargs: Any) -> None:
    control.state.update(kwargs)


def prompts_from_api(data: dict) -> Prompts:
    return Prompts(
        case_system=data.get("case", {}).get("system", ""),
        case_user=data.get("case", {}).get("user", ""),
        law_system=data.get("law", {}).get("system", ""),
        law_user=data.get("law", {}).get("user", ""),
        solution_system=data.get("solution", {}).get("system", ""),
        solution_user=data.get("solution", {}).get("user", ""),
        model=data.get("model", "gpt-4o-mini"),
    )


# ── Pause helper ─────────────────────────────────────────────────


async def _wait_if_paused(
    api: PipelineAPIClient,
    job_id: str,
    control: WorkerControl,
) -> None:
    if not control.pause_requested:
        return

    announced = False
    while control.pause_requested and not control.stop_requested:
        if not announced:
            try:
                await api.update_progress(job_id, {"status": "paused", "phase": "paused"})
            except Exception:
                pass
            _sync_state(control, status="paused")
            announced = True

        await asyncio.sleep(2)


# ── Single judgment processor ────────────────────────────────────


async def process_one(
    db: LegalDBClient,
    http: httpx.AsyncClient,
    judgment: dict,
    settings: Settings,
    prompts: Prompts,
    api: PipelineAPIClient,
    worker_idx: int,
    control: WorkerControl,
) -> dict[str, Any]:
    bos = int(judgment["numer_bos"])
    court_type_id = map_court_type_id(judgment.get("rodzaj_organu"))
    start = time.monotonic()

    try:
        existing = await db.get_leafs_by_judgment(judgment["id"])
        existing_types = {leaf["typ"] for leaf in existing}
        existing_ids = [leaf["id"] for leaf in existing if leaf.get("id")]

        rows: list[dict] = []
        all_law_p: list[dict] = []
        gen_ms = 0

        if not {"case", "law", "solution"}.issubset(existing_types):
            gen_start = time.monotonic()
            content = await db.get_judgment_content(judgment["id"])
            if not content or len(content.strip()) < 50:
                return {"ok": True, "numer_bos": bos, "embedded": 0, "skipped": True}

            leafs = await generate_leafs(http, content, settings.openai_api_key, prompts)

            # Log per-type extraction errors for retry
            if leafs.errors:
                extraction_errors = [
                    {
                        "job_id": control.state.get("job_id"),
                        "numer_bos": bos,
                        "orzeczenie_id": judgment["id"],
                        "typ": typ,
                        "error_message": msg,
                    }
                    for typ, msg in leafs.errors.items()
                ]
                try:
                    await api.log_extraction_errors(extraction_errors)
                except Exception as log_exc:
                    control.log("warning", f"worker[{worker_idx}] BOS={bos} failed to log extraction errors: {log_exc}")
                for typ, msg in leafs.errors.items():
                    control.log("warning", f"worker[{worker_idx}] BOS={bos} type={typ} error (logged for retry): {msg[:100]}")

            for typ in ("case", "solution"):
                valid = [text.strip() for text in getattr(leafs, typ) if len(text.strip()) > 10]
                if valid and typ not in existing_types:
                    for leaf_text in valid:
                        rows.append(
                            {
                                "orzeczenie_id": judgment["id"],
                                "typ": typ,
                                "tresc": leaf_text,
                                "status": "generated",
                                "llm_model": prompts.model,
                                "court_type_id": court_type_id,
                            }
                        )

            valid_law = [law for law in leafs.law if len(law.tresc.strip()) > 10]
            if valid_law and "law" not in existing_types:
                for law in valid_law:
                    rows.append(
                        {
                            "orzeczenie_id": judgment["id"],
                            "typ": "law",
                            "tresc": law.tresc.strip(),
                            "status": "generated",
                            "llm_model": prompts.model,
                            "court_type_id": court_type_id,
                        }
                    )
                    all_law_p.extend(law.przepisy)

            if rows:
                await db.insert_leafs(rows)
                existing = await db.get_leafs_by_judgment(judgment["id"])
                existing_ids = [leaf["id"] for leaf in existing if leaf.get("id")]

            law_leafs = [leaf for leaf in existing if leaf["typ"] == "law"]
            if law_leafs and all_law_p:
                first_law = law_leafs[0]
                existing_rels = await db.get_leaf_article_rels(first_law["id"])
                if not existing_rels:
                    rels = [
                        {
                            "leaf_id": first_law["id"],
                            "artykul_text_fallback": przepisy["oznaczenie"],
                            "match_status": "unmatched",
                            "scoring": min(10, max(0, round(przepisy["scoring"]))),
                        }
                        for przepisy in all_law_p
                    ]
                    try:
                        await db.insert("leaf_artykul_rel", rels)
                    except Exception as exc:
                        control.log("error", f"worker[{worker_idx}] relation insert error: {exc}")

            gen_ms = int((time.monotonic() - gen_start) * 1000)

        embed_start = time.monotonic()
        embedded = 0
        if existing_ids:
            ids_str = ",".join(str(x) for x in existing_ids)
            to_embed = await db.get(f"/orzeczenie_leaf?id=in.({ids_str})&status=eq.generated&select=id,tresc")
            if to_embed:
                embs = await generate_embeddings(
                    http,
                    [leaf["tresc"] for leaf in to_embed],
                    settings.openai_api_key,
                    settings.embedding_model,
                )
                for idx, leaf in enumerate(to_embed):
                    emb = embs[idx] if idx < len(embs) else []
                    if emb and len(emb) == settings.embedding_dim:
                        try:
                            await db.update_leaf_embedding(leaf["id"], emb, settings.embedding_model)
                            embedded += 1
                        except Exception as exc:
                            await db.mark_leaf_error(leaf["id"], str(exc))
                    else:
                        await db.mark_leaf_error(leaf["id"], "Embedding failed")
        embed_ms = int((time.monotonic() - embed_start) * 1000)

        # NOTE: Matching is NOT done here — it runs as a separate process after all leafs are generated.

        total_ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": True,
            "numer_bos": bos,
            "embedded": embedded,
            "gen_ms": gen_ms,
            "embed_ms": embed_ms,
            "total_ms": total_ms,
            "leafs": len(existing),
        }

    except RateLimitError:
        return {"ok": False, "numer_bos": bos, "embedded": 0, "error": "RATE_LIMIT"}
    except CreditsExhaustedError:
        return {"ok": False, "numer_bos": bos, "embedded": 0, "error": "CREDITS_EXHAUSTED"}
    except Exception as exc:
        control.log("error", f"worker[{worker_idx}] BOS={bos} error: {exc}")
        return {"ok": False, "numer_bos": bos, "embedded": 0, "error": str(exc)[:200]}


# ── Main pipeline loop ───────────────────────────────────────────


async def run_pipeline(
    settings: Settings,
    job_id: str | None,
    from_bos: int | None,
    to_bos: int | None,
    force: bool = False,
    control: WorkerControl | None = None,
    workers_override: int | None = None,
) -> dict[str, Any]:
    control = control or WorkerControl()
    if workers_override is not None:
        settings = settings.model_copy(update={"workers": workers_override})

    job: dict[str, Any] | None = None

    try:
        async with httpx.AsyncClient(timeout=120, http2=True) as http:
            db = LegalDBClient(settings, http)
            api = PipelineAPIClient(settings.pipeline_api_url, settings.pipeline_api_key, http)

            prompts_data = await api.get_prompts()
            prompts = prompts_from_api(prompts_data)
            control.log("success", f"✓ Loaded prompts via API, model: {prompts.model}")

            if job_id:
                job = await api.get_job(job_id)
                if not job:
                    control.log("error", f"Job {job_id} not found")
                    if not control.started_event.is_set():
                        control.started_event.set()
                    return {"ok": False, "error": "job_not_found", "job_id": job_id}
                control.log(
                    "info",
                    f"Resuming job {job_id[:8]}… BOS {job['numer_bos_from']}–{job['numer_bos_to']}",
                )
            elif from_bos is not None and to_bos is not None:
                if force:
                    control.log(
                        "warning",
                        f"FORCE mode: deleting existing leafs in BOS {from_bos}–{to_bos}…",
                    )
                    chunk = 200
                    for chunk_from in range(from_bos, to_bos + 1, chunk):
                        chunk_to = min(chunk_from + chunk - 1, to_bos)
                        bos_vals = ",".join(str(bos) for bos in range(chunk_from, chunk_to + 1))
                        judgments = await db.get(f"/orzeczenia?numer_bos=in.({bos_vals})&select=id")
                        if not judgments:
                            continue
                        for judgment in judgments:
                            leafs = await db.get(
                                f"/orzeczenie_leaf?orzeczenie_id=eq.{judgment['id']}&select=id"
                            )
                            for leaf in leafs:
                                await db.delete("leaf_artykul_rel", f"leaf_id=eq.{leaf['id']}")
                            await db.delete("orzeczenie_leaf", f"orzeczenie_id=eq.{judgment['id']}")

                result = await api.create_job(from_bos, to_bos, force)
                job = result.get("job")
                if not job:
                    control.log("error", f"Failed to create job: {result}")
                    if not control.started_event.is_set():
                        control.started_event.set()
                    return {"ok": False, "error": "create_job_failed", "details": result}
                job_id = job["id"]
                control.log("success", f"✓ Job created via API: {job_id[:8]}")
            else:
                control.log("error", "Specify --job-id or --from/--to")
                if not control.started_event.is_set():
                    control.started_event.set()
                return {"ok": False, "error": "missing_arguments"}

            _sync_state(
                control,
                job_id=job_id,
                status=job.get("status"),
                phase=job.get("phase"),
                numer_bos_from=job.get("numer_bos_from"),
                numer_bos_to=job.get("numer_bos_to"),
                current_numer_bos=job.get("current_numer_bos") or job.get("numer_bos_from"),
                processed_count=job.get("processed_count", 0),
                embedded_count=job.get("embedded_count", 0),
                error_count=job.get("error_count", 0),
                workers=settings.workers,
            )
            if not control.started_event.is_set():
                control.started_event.set()

            session_processed = 0
            session_embedded = 0
            session_errors = int(job.get("error_count", 0) or 0)
            session_start = time.monotonic()
            workers = settings.workers

            await api.update_progress(job_id, {"status": "running", "phase": "generate"})
            _sync_state(control, status="running", phase="generate")

            while not control.stop_requested:
                await _wait_if_paused(api, job_id, control)
                if control.stop_requested:
                    break

                current_bos = job.get("current_numer_bos") or job["numer_bos_from"]
                judgments = await db.get_judgments_in_range(job["numer_bos_from"], job["numer_bos_to"], current_bos)

                if not judgments:
                    await api.update_progress(
                        job_id,
                        {
                            "status": "completed",
                            "phase": "done",
                            "error_count": session_errors,
                        },
                    )
                    control.log("success", "✅ Job completed — no more judgments")
                    _sync_state(control, status="completed", phase="done", error_count=session_errors)
                    break

                batch = judgments[:workers]
                batch_start = time.monotonic()

                async def _run(judgment: dict, idx: int) -> dict[str, Any]:
                    await asyncio.sleep(idx * settings.stagger_ms / 1000)
                    return await process_one(db, http, judgment, settings, prompts, api, idx, control)

                tasks = [_run(judgment, idx) for idx, judgment in enumerate(batch)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                batch_ms = int((time.monotonic() - batch_start) * 1000)
                processed = 0
                embedded = 0
                last_bos = current_bos
                rate_limited = False

                for result in results:
                    if isinstance(result, Exception):
                        session_errors += 1
                        control.log("error", f"Unhandled worker exception: {result}")
                        continue

                    if result["ok"]:
                        processed += 1
                        embedded += result.get("embedded", 0)
                        bos = result["numer_bos"]
                        if not result.get("skipped"):
                            control.log(
                                "success",
                                (
                                    f"✓ BOS={bos} | {result.get('gen_ms', 0)}ms gen / "
                                    f"{result.get('embed_ms', 0)}ms emb"
                                ),
                            )
                    else:
                        err = result.get("error", "?")
                        if err in ("RATE_LIMIT", "CREDITS_EXHAUSTED"):
                            rate_limited = True
                        else:
                            session_errors += 1
                            control.log("error", f"✗ BOS={result['numer_bos']}: {err}")
                            await api.append_error(job_id, {"numer_bos": result["numer_bos"], "message": err})

                    if result["numer_bos"] >= last_bos:
                        last_bos = result["numer_bos"] + 1

                session_processed += processed
                session_embedded += embedded
                elapsed = time.monotonic() - session_start
                throughput = session_processed / (elapsed / 3600) if elapsed > 0 else 0

                job["current_numer_bos"] = last_bos
                job["processed_count"] = job.get("processed_count", 0) + processed
                job["embedded_count"] = job.get("embedded_count", 0) + embedded
                job["error_count"] = session_errors

                remaining = await db.get_judgments_in_range(job["numer_bos_from"], job["numer_bos_to"], last_bos)
                new_status = "completed" if not remaining else ("paused" if rate_limited else "running")
                new_phase = "done" if not remaining else "generate"

                await api.update_progress(
                    job_id,
                    {
                        "current_numer_bos": last_bos,
                        "processed_count": job["processed_count"],
                        "embedded_count": job["embedded_count"],
                        "error_count": session_errors,
                        "status": new_status,
                        "phase": new_phase,
                    },
                )
                _sync_state(
                    control,
                    current_numer_bos=last_bos,
                    processed_count=job["processed_count"],
                    embedded_count=job["embedded_count"],
                    error_count=session_errors,
                    status=new_status,
                    phase=new_phase,
                    last_batch_ms=batch_ms,
                    throughput_per_hour=round(throughput, 2),
                )

                control.log(
                    "info",
                    (
                        f"Batch: {processed}/{len(batch)} ok, {embedded} embedded "
                        f"| {batch_ms}ms | Session: {session_processed} ({throughput:.0f}/h) | Errors: {session_errors}"
                    ),
                )

                if new_status == "completed":
                    control.log("success", "✅ Job completed!")
                    break

                if rate_limited:
                    wait = 30
                    control.log("warning", f"⏸ Rate limited — waiting {wait}s…")
                    for _ in range(wait):
                        if control.stop_requested:
                            break
                        await asyncio.sleep(1)
                    if control.stop_requested:
                        break
                    await api.update_progress(job_id, {"status": "running", "error_count": session_errors})
                    _sync_state(control, status="running", error_count=session_errors)
                    continue

                await asyncio.sleep(settings.batch_delay_ms / 1000)

            if control.stop_requested:
                await api.update_progress(job_id, {"status": "paused", "phase": "paused", "error_count": session_errors})
                control.log("warning", "⏸ Job paused (graceful stop)")
                _sync_state(control, status="paused", phase="paused", error_count=session_errors)

            elapsed = time.monotonic() - session_start
            summary = {
                "ok": True,
                "job_id": job_id,
                "processed": session_processed,
                "embedded": session_embedded,
                "errors": session_errors,
                "elapsed_seconds": round(elapsed, 2),
                "throughput_per_hour": round(session_processed / (elapsed / 3600), 2) if elapsed > 0 else 0,
            }
            _sync_state(control, summary=summary)

            control.log("info", "Summary:")
            control.log("info", f"  Processed: {session_processed}")
            control.log("info", f"  Embedded: {session_embedded}")
            control.log("info", f"  Errors: {session_errors}")
            control.log("info", f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
            if session_processed > 0:
                control.log("info", f"  Avg: {elapsed / session_processed * 1000:.0f}ms/judgment")
                control.log("info", f"  Throughput:{session_processed / (elapsed / 3600):.0f}/h")

            return summary

    except Exception as exc:
        tb = traceback.format_exc()
        control.log("error", f"Fatal worker error: {exc}\n{tb}")
        _sync_state(control, status="failed", fatal_error=f"{exc}\n{tb}")
        if job_id:
            try:
                async with httpx.AsyncClient(timeout=30, http2=True) as http:
                    api = PipelineAPIClient(settings.pipeline_api_url, settings.pipeline_api_key, http)
                    await api.update_progress(job_id, {"status": "failed", "phase": "error"})
                    await api.append_error(job_id, {"message": f"FATAL: {str(exc)[:300]}"})
            except Exception as api_exc:
                control.log("error", f"Failed to report fatal error via API: {api_exc}")
        if not control.started_event.is_set():
            control.started_event.set()
        raise