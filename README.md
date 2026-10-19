# LEAF Pipeline Worker

Standalone Python app for generating, embedding and matching legal judgment leafs (case/law/solution).

Replaces the Edge Function-based pipeline with a long-running process — no timeouts, configurable concurrency.

## Architecture

```
main.py          — CLI entrypoint + FastAPI server mode
runner.py        — core processing loop + local runtime control
worker_api.py    — FastAPI control API for start/pause/resume/stop/logs
config.py        — Settings from .env (pydantic-settings)
legal_db.py      — PostgREST client for Legal DB (data read/write)
llm.py           — Google Gemini API + OpenAI embeddings
matching.py      — Article reference parsing & matching
pipeline_api.py  — API client for job management (edge function)
```

### Communication flow

```
Worker API/UI ──► FastAPI worker runtime     — start/pause/resume/stop/logs
Worker        ──► Legal DB (PostgREST)       — leafs, embeddings, matching
Worker        ──► Edge Function API          — job CRUD, prompts, aliases, progress
```

Worker job control does NOT access the platform database directly. It goes through
`leaf-pipeline` API actions (`worker_*`). Legal result data is still written only to your PG / Legal DB.

## Quick Start

```bash
# 1. Copy and fill config
cp .env.example .env
# edit .env with your keys

# 2. Install locally (optional)
pip install -r requirements.txt

# 3a. Run one job from CLI
docker compose run leaf-pipeline --from 100000 --to 105000 --workers 10

# 3b. Run worker control API as a service
docker compose up -d leaf-pipeline-api
```

## CLI usage

```bash
# New job
python main.py --from 100000 --to 105000

# Resume existing job
python main.py --job-id <UUID>

# Force regenerate (delete existing leafs first)
python main.py --from 100000 --to 105000 --force

# Custom worker count
python main.py --from 100000 --to 105000 --workers 20
```

## FastAPI control API

Optional auth: set `WORKER_API_TOKEN` and send it as either:
- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

### Endpoints

```bash
GET    /health
GET    /jobs
GET    /jobs/{job_id}
GET    /jobs/{job_id}/logs?limit=200
POST   /jobs/start
POST   /jobs/{job_id}/resume
POST   /jobs/{job_id}/pause
POST   /jobs/{job_id}/stop
```

### Example calls

```bash
curl -X POST http://localhost:8080/jobs/start \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -d '{"from_bos":1000000,"to_bos":1100000,"workers":20}'

curl http://localhost:8080/jobs
curl http://localhost:8080/jobs/<JOB_ID>/logs?limit=100
curl -X POST http://localhost:8080/jobs/<JOB_ID>/pause
curl -X POST http://localhost:8080/jobs/<JOB_ID>/resume -H 'Content-Type: application/json' -d '{}'
curl -X POST http://localhost:8080/jobs/<JOB_ID>/stop
```

## Pipeline per judgment

1. **Generate** — 3 LLM calls (case/law/solution) via Google Gemini API
2. **Embed** — OpenAI text-embedding-3-small (1536 dim)
3. **Match** — link law references to `articles` table via alias map

## Monitoring

- job progress is persisted through the mediated pipeline API
- runtime logs are available from FastAPI `/jobs/{job_id}/logs`
- SuperAdmin can now be wired to this worker API instead of assuming manual CLI starts

## Graceful stop

Press `Ctrl+C` or send `SIGTERM` — the worker finishes current batch, reports progress, and exits.
