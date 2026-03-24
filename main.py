#!/usr/bin/env python3
"""
LEAF Pipeline Worker — standalone Python app.

Usage:
  python main.py --job-id <UUID>                    # Resume existing job
  python main.py --from 100000 --to 105000          # Create & run new job
  python main.py --from 100000 --to 105000 --workers 15
  python main.py --from 100000 --to 105000 --force  # Regenerate existing leafs

Signals:
  CTRL+C / SIGTERM → graceful stop after current batch
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from typing import Any

import httpx
from rich.console import Console

from config import Settings
from legal_db import LegalDBClient
from llm import (
    CreditsExhaustedError,
    GeneratedLeafs,
    Prompts,
    RateLimitError,
    generate_embeddings,
    generate_leafs,
)
from matching import match_judgment_refs
from pipeline_api import PipelineAPIClient

console = Console()

# ── Graceful shutdown ─────────────────────────────────────────────

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True
    console.print("\n[yellow]⏸  Graceful stop requested — finishing current batch…[/]")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Prompts conversion ───────────────────────────────────────────


def prompts_from_api(data: dict) -> Prompts:
    """Convert API prompts response to internal Prompts dataclass."""
    p = Prompts()
    for typ in ("case", "law", "solution"):
        if typ in data:
            setattr(p, f"{typ}_system", data[typ].get("system", getattr(p, f"{typ}_system")))
            setattr(p, f"{typ}_user", data[typ].get("user", getattr(p, f"{typ}_user")))
    if data.get("model"):
        p.model = data["model"]
    return p


# ── Single judgment processor ─────────────────────────────────────


async def process_one(
    db: LegalDBClient,
    http: httpx.AsyncClient,
    judgment: dict,
    settings: Settings,
    prompts: Prompts,
    alias_map: dict[str, int],
    api: PipelineAPIClient,
    worker_idx: int,
) -> dict[str, Any]:
    bos = int(judgment["numer_bos"])
    start = time.monotonic()

    try:
        existing = await db.get_leafs_by_judgment(judgment["id"])
        existing_types = {l["typ"] for l in existing}
        existing_ids = [l["id"] for l in existing if l.get("id")]

        rows: list[dict] = []
        all_law_p: list[dict] = []
        gen_ms = 0

        if not {"case", "law", "solution"}.issubset(existing_types):
            gen_start = time.monotonic()
            content = await db.get_judgment_content(judgment["id"])
            if not content or len(content.strip()) < 50:
                return {"ok": True, "numer_bos": bos, "embedded": 0, "matched": 0, "skipped": True}

            leafs = await generate_leafs(http, content, settings.gemini_api_key, prompts)

            for typ in ("case", "solution"):
                valid = [t.strip() for t in getattr(leafs, typ) if len(t.strip()) > 10]
                if valid and typ not in existing_types:
                    for leaf_text in valid:
                        rows.append({
                            "orzeczenie_id": judgment["id"],
                            "typ": typ,
                            "tresc": leaf_text,
                            "status": "generated",
                            "llm_model": prompts.model,
                        })

            valid_law = [l for l in leafs.law if len(l.tresc.strip()) > 10]
            if valid_law and "law" not in existing_types:
                for l in valid_law:
                    rows.append({
                        "orzeczenie_id": judgment["id"],
                        "typ": "law",
                        "tresc": l.tresc.strip(),
                        "status": "generated",
                        "llm_model": prompts.model,
                    })
                    all_law_p.extend(l.przepisy)

            if rows:
                await db.insert_leafs(rows)
                existing = await db.get_leafs_by_judgment(judgment["id"])
                existing_ids = [l["id"] for l in existing if l.get("id")]

            law_leafs = [l for l in existing if l["typ"] == "law"]
            if law_leafs and all_law_p:
                first_law = law_leafs[0]
                ex_rels = await db.get_leaf_article_rels(first_law["id"])
                if not ex_rels:
                    rels = [
                        {
                            "leaf_id": first_law["id"],
                            "artykul_text_fallback": p["oznaczenie"],
                            "match_status": "unmatched",
                            "scoring": min(10, max(0, round(p["scoring"]))),
                        }
                        for p in all_law_p
                    ]
                    try:
                        await db.insert("leaf_artykul_rel", rels)
                    except Exception as e:
                        console.print(f"[red]  w{worker_idx} rel error: {e}[/]")

            gen_ms = int((time.monotonic() - gen_start) * 1000)

        # Embed
        embed_start = time.monotonic()
        embedded = 0
        if existing_ids:
            ids_str = ",".join(str(x) for x in existing_ids)
            to_embed = await db.get(
                f"/orzeczenie_leaf?id=in.({ids_str})&status=eq.generated&select=id,tresc"
            )
            if to_embed:
                embs = await generate_embeddings(
                    http,
                    [l["tresc"] for l in to_embed],
                    settings.openai_api_key,
                    settings.embedding_model,
                )
                for i, leaf in enumerate(to_embed):
                    emb = embs[i] if i < len(embs) else []
                    if emb and len(emb) == settings.embedding_dim:
                        try:
                            await db.update_leaf_embedding(leaf["id"], emb, settings.embedding_model)
                            embedded += 1
                        except Exception as e:
                            await db.mark_leaf_error(leaf["id"], str(e))
                    else:
                        await db.mark_leaf_error(leaf["id"], "Embedding failed")
        embed_ms = int((time.monotonic() - embed_start) * 1000)

        # Match
        match_start = time.monotonic()
        match_r = await match_judgment_refs(db, existing_ids, alias_map, api)
        match_ms = int((time.monotonic() - match_start) * 1000)

        total_ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": True, "numer_bos": bos, "embedded": embedded,
            "matched": match_r["matched"],
            "gen_ms": gen_ms, "embed_ms": embed_ms, "match_ms": match_ms,
            "total_ms": total_ms, "leafs": len(existing),
        }

    except RateLimitError:
        return {"ok": False, "numer_bos": bos, "embedded": 0, "matched": 0, "error": "RATE_LIMIT"}
    except CreditsExhaustedError:
        return {"ok": False, "numer_bos": bos, "embedded": 0, "matched": 0, "error": "CREDITS_EXHAUSTED"}
    except Exception as e:
        return {"ok": False, "numer_bos": bos, "embedded": 0, "matched": 0, "error": str(e)[:300]}


# ── Main loop ─────────────────────────────────────────────────────


async def run_pipeline(settings: Settings, job_id: str | None, from_bos: int | None, to_bos: int | None, force: bool):
    global _stop

    async with httpx.AsyncClient(timeout=120, http2=True) as http:
        db = LegalDBClient(settings, http)
        api = PipelineAPIClient(settings.pipeline_api_url, settings.pipeline_api_key, http)

        # Load prompts via API
        prompts_data = await api.get_prompts()
        prompts = prompts_from_api(prompts_data)
        console.print(f"[green]✓ Loaded prompts via API, model: {prompts.model}[/]")

        # Load alias map via API
        alias_map = await api.get_aliases()
        console.print(f"[green]✓ Loaded {len(alias_map)} aliases via API[/]")

        # Create or load job
        if job_id:
            job = await api.get_job(job_id)
            if not job:
                console.print(f"[red]Job {job_id} not found[/]")
                return
            console.print(f"[cyan]Resuming job {job_id[:8]}… BOS {job['numer_bos_from']}–{job['numer_bos_to']}[/]")
        elif from_bos is not None and to_bos is not None:
            if force:
                console.print(f"[yellow]FORCE mode: deleting existing leafs in BOS {from_bos}–{to_bos}…[/]")
                CHUNK = 200
                for cf in range(from_bos, to_bos + 1, CHUNK):
                    ct = min(cf + CHUNK - 1, to_bos)
                    bos_vals = ",".join(str(b) for b in range(cf, ct + 1))
                    orz = await db.get(f"/orzeczenia?numer_bos=in.({bos_vals})&select=id")
                    if not orz:
                        continue
                    for o in orz:
                        leafs = await db.get(f"/orzeczenie_leaf?orzeczenie_id=eq.{o['id']}&select=id")
                        for lf in leafs:
                            await db.delete("leaf_artykul_rel", f"leaf_id=eq.{lf['id']}")
                        await db.delete("orzeczenie_leaf", f"orzeczenie_id=eq.{o['id']}")

            result = await api.create_job(from_bos, to_bos, force)
            job = result.get("job")
            if not job:
                console.print(f"[red]Failed to create job: {result}[/]")
                return
            job_id = job["id"]
            console.print(f"[green]✓ Job created via API: {job_id[:8]}[/]")
        else:
            console.print("[red]Specify --job-id or --from/--to[/]")
            return

        # Processing loop
        session_processed = 0
        session_embedded = 0
        session_errors = 0
        session_start = time.monotonic()
        workers = settings.workers

        await api.update_progress(job_id, {"status": "running", "phase": "generate"})

        while not _stop:
            # Check if UI signaled pause
            db_status = await api.read_job_status(job_id)
            if db_status == "paused":
                console.print("[yellow]⏸ Job paused via UI — waiting for resume signal…[/]")
                while not _stop:
                    await asyncio.sleep(5)
                    db_status = await api.read_job_status(job_id)
                    if db_status != "paused":
                        console.print(f"[green]▶ Job resumed (status: {db_status})[/]")
                        break
                if _stop:
                    break

            current_bos = job.get("current_numer_bos", job["numer_bos_from"])
            judgments = await db.get_judgments_in_range(
                job["numer_bos_from"], job["numer_bos_to"], current_bos
            )

            if not judgments:
                await api.update_progress(job_id, {"status": "completed", "phase": "done"})
                console.print("[green]✅ Job completed — no more judgments[/]")
                break

            batch = judgments[:workers]
            batch_start = time.monotonic()

            # Staggered parallel execution
            async def _run(j, idx):
                await asyncio.sleep(idx * settings.stagger_ms / 1000)
                return await process_one(db, http, j, settings, prompts, alias_map, api, idx)

            tasks = [_run(j, i) for i, j in enumerate(batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_ms = int((time.monotonic() - batch_start) * 1000)
            processed = 0
            embedded = 0
            matched = 0
            last_bos = current_bos
            rate_limited = False

            for r in results:
                if isinstance(r, Exception):
                    session_errors += 1
                    console.print(f"[red]  Exception: {r}[/]")
                    continue
                if r["ok"]:
                    processed += 1
                    embedded += r.get("embedded", 0)
                    matched += r.get("matched", 0)
                    bos = r["numer_bos"]
                    if not r.get("skipped"):
                        console.print(
                            f"  [green]✓[/] BOS={bos} "
                            f"[dim]{r.get('gen_ms',0)}ms gen / {r.get('embed_ms',0)}ms emb / {r.get('match_ms',0)}ms match[/]"
                        )
                else:
                    err = r.get("error", "?")
                    if err in ("RATE_LIMIT", "CREDITS_EXHAUSTED"):
                        rate_limited = True
                    else:
                        session_errors += 1
                    console.print(f"  [red]✗[/] BOS={r['numer_bos']}: {err}")
                    await api.append_error(job_id, {"numer_bos": r["numer_bos"], "message": err})

                if r["numer_bos"] >= last_bos:
                    last_bos = r["numer_bos"] + 1

            session_processed += processed
            session_embedded += embedded
            elapsed = time.monotonic() - session_start
            throughput = session_processed / (elapsed / 3600) if elapsed > 0 else 0

            # Update job via API
            job["current_numer_bos"] = last_bos
            job["processed_count"] = job.get("processed_count", 0) + processed
            job["embedded_count"] = job.get("embedded_count", 0) + embedded

            remaining = await db.get_judgments_in_range(job["numer_bos_from"], job["numer_bos_to"], last_bos)
            new_status = "completed" if not remaining else ("paused" if rate_limited else "running")
            new_phase = "done" if not remaining else "generate"

            await api.update_progress(job_id, {
                "current_numer_bos": last_bos,
                "processed_count": job["processed_count"],
                "embedded_count": job["embedded_count"],
                "status": new_status,
                "phase": new_phase,
            })

            console.print(
                f"[cyan]Batch: {processed}/{len(batch)} ok, {embedded}E, {matched}M "
                f"| {batch_ms}ms | Session: {session_processed} ({throughput:.0f}/h) | Errors: {session_errors}[/]"
            )

            if new_status == "completed":
                console.print("[green]✅ Job completed![/]")
                break

            if rate_limited:
                wait = 30
                console.print(f"[yellow]⏸ Rate limited — waiting {wait}s…[/]")
                for _ in range(wait):
                    if _stop:
                        break
                    await asyncio.sleep(1)
                if _stop:
                    break
                await api.update_progress(job_id, {"status": "running"})
                continue

            await asyncio.sleep(settings.batch_delay_ms / 1000)

        if _stop:
            await api.update_progress(job_id, {"status": "paused"})
            console.print("[yellow]⏸ Job paused (graceful stop)[/]")

        # Summary
        elapsed = time.monotonic() - session_start
        console.print(f"\n[bold]Summary:[/]")
        console.print(f"  Processed: {session_processed}")
        console.print(f"  Embedded:  {session_embedded}")
        console.print(f"  Errors:    {session_errors}")
        console.print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
        if session_processed > 0:
            console.print(f"  Avg:       {elapsed/session_processed*1000:.0f}ms/judgment")
            console.print(f"  Throughput:{session_processed/(elapsed/3600):.0f}/h")


def main():
    parser = argparse.ArgumentParser(description="LEAF Pipeline Worker")
    parser.add_argument("--job-id", help="Resume existing job by UUID")
    parser.add_argument("--from", dest="from_bos", type=int, help="Start BOS number")
    parser.add_argument("--to", dest="to_bos", type=int, help="End BOS number")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default from .env)")
    parser.add_argument("--force", action="store_true", help="Delete existing leafs before processing")
    args = parser.parse_args()

    settings = Settings()
    if args.workers:
        settings.workers = args.workers

    console.print(f"[bold cyan]LEAF Pipeline Worker[/] — {settings.workers} workers, model: {settings.llm_model}")

    asyncio.run(run_pipeline(settings, args.job_id, args.from_bos, args.to_bos, args.force))


if __name__ == "__main__":
    main()
