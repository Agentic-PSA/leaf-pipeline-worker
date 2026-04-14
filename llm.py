"""LLM calls (OpenAI API) and OpenAI embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class RateLimitError(Exception):
    pass


class CreditsExhaustedError(Exception):
    pass


# ── Tool schemas ─────────────────────────────────────────────────

SIMPLE_LEAF_TOOL = {
    "type": "function",
    "function": {
        "name": "save_leafs",
        "description": "Zapisz wyodrębnione leafy",
        "parameters": {
            "type": "object",
            "properties": {
                "leafs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista wyodrębnionych leafów (1-5 sztuk)",
                }
            },
            "required": ["leafs"],
            "additionalProperties": False,
        },
    },
}

LAW_LEAF_TOOL = {
    "type": "function",
    "function": {
        "name": "save_law_leafs",
        "description": "Zapisz wyodrębnione leafy prawne z oceną przepisów",
        "parameters": {
            "type": "object",
            "properties": {
                "leafs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tresc": {"type": "string"},
                            "przepisy": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "oznaczenie": {"type": "string"},
                                        "scoring": {"type": "number"},
                                    },
                                    "required": ["oznaczenie", "scoring"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["tresc", "przepisy"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["leafs"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class Prompts:
    case_system: str = "Wyodrębnij stan faktyczny z orzeczenia. Zwróć 1-5 leafów, każdy 1-3 zdania."
    case_user: str = "Wyodrębnij stan faktyczny:\n\n{{TEXT}}"
    law_system: str = "Wyodrębnij podstawy prawne z orzeczenia z oceną przepisów (scoring 0-10)."
    law_user: str = "Wyodrębnij podstawy prawne:\n\n{{TEXT}}"
    solution_system: str = "Wyodrębnij rozstrzygnięcie z orzeczenia."
    solution_user: str = "Wyodrębnij rozstrzygnięcie:\n\n{{TEXT}}"
    model: str = "gpt-4o-mini"


@dataclass
class LawLeaf:
    tresc: str
    przepisy: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratedLeafs:
    case: list[str] = field(default_factory=list)
    law: list[LawLeaf] = field(default_factory=list)
    solution: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # typ -> error message


# ── LLM call with retry ──────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def call_llm(
    client: httpx.AsyncClient,
    text: str,
    typ: str,
    api_key: str,
    prompts: Prompts,
) -> dict:
    is_law = typ == "law"
    tool = LAW_LEAF_TOOL if is_law else SIMPLE_LEAF_TOOL
    tool_name = "save_law_leafs" if is_law else "save_leafs"

    sys_prompt = getattr(prompts, f"{typ}_system")
    usr_template = getattr(prompts, f"{typ}_user")

    if "{{TEXT}}" in usr_template:
        user_content = usr_template.replace("{{TEXT}}", text[:15000])
    else:
        user_content = usr_template + "\n\n" + text[:15000]

    resp = await client.post(
        OPENAI_CHAT_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": prompts.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        },
        timeout=90,
    )

    if resp.status_code == 429:
        raise RateLimitError("RATE_LIMIT")
    if resp.status_code == 402:
        raise CreditsExhaustedError("CREDITS_EXHAUSTED")
    resp.raise_for_status()

    data = resp.json()
    tc = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    if not tc:
        raise ValueError(f"No tool call in LLM response for type={typ}")

    return json.loads(tc[0]["function"]["arguments"])


async def generate_leafs(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    prompts: Prompts,
) -> GeneratedLeafs:
    import asyncio

    result = GeneratedLeafs()

    # Process each type independently — failures don't block others
    for typ in ("case", "law", "solution"):
        try:
            r = await call_llm(client, text, typ, api_key, prompts)
            if typ == "law":
                result.law = [
                    LawLeaf(
                        tresc=l.get("tresc", ""),
                        przepisy=[
                            {
                                "oznaczenie": p.get("oznaczenie", ""),
                                "scoring": min(10, max(0, int(p.get("scoring", 0)))),
                            }
                            for p in l.get("przepisy", [])
                        ],
                    )
                    for l in r.get("leafs", [])
                ]
            else:
                setattr(result, typ, r.get("leafs", []))
        except (RateLimitError, CreditsExhaustedError):
            raise  # These should propagate immediately
        except Exception as exc:
            result.errors[typ] = str(exc)[:300]

        await asyncio.sleep(0.15)

    return result


# ── Embeddings ────────────────────────────────────────────────────

EMBED_BATCH = 50


async def generate_embeddings(
    client: httpx.AsyncClient,
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    import asyncio

    all_embs: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        if i > 0:
            await asyncio.sleep(0.2)

        resp = await client.post(
            OPENAI_EMBED_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={"model": model, "input": batch},
            timeout=60,
        )

        if not resp.is_success:
            all_embs.extend([[] for _ in batch])
            continue

        data = resp.json()
        embs = sorted(data.get("data", []), key=lambda d: d["index"])
        all_embs.extend([e["embedding"] for e in embs])

    return all_embs