#!/usr/bin/env python3
"""LEAF Pipeline Worker entrypoint.

Modes:
  python main.py --job-id <UUID>
  python main.py --from 100000 --to 105000 --workers 15
  python main.py api --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import signal

import uvicorn

from config import Settings
from runner import WorkerControl, console, run_pipeline

control = WorkerControl()


def _handle_signal(signum, frame):
    control.request_stop()
    control.log("warning", "⏸ Graceful stop requested — finishing current batch…")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEAF Pipeline Worker")
    subparsers = parser.add_subparsers(dest="command")

    api_parser = subparsers.add_parser("api", help="Run FastAPI control server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    api_parser.add_argument("--port", type=int, default=8080, help="Bind port")

    parser.add_argument("--job-id", help="Resume existing job by UUID")
    parser.add_argument("--from", dest="from_bos", type=int, help="Start BOS number")
    parser.add_argument("--to", dest="to_bos", type=int, help="End BOS number")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default from env)")
    parser.add_argument("--force", action="store_true", help="Delete existing leafs before processing")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    if args.command == "api":
        console.print("[bold cyan]LEAF Pipeline Worker API[/]")
        uvicorn.run("worker_api:app", host=args.host, port=args.port, reload=False)
        return

    console.print(
        f"[bold cyan]LEAF Pipeline Worker[/] — {args.workers or settings.workers} workers, model: {settings.llm_model}"
    )

    asyncio.run(
        run_pipeline(
            settings,
            args.job_id,
            args.from_bos,
            args.to_bos,
            args.force,
            control=control,
            workers_override=args.workers,
        )
    )


if __name__ == "__main__":
    main()
