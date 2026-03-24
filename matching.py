"""Article reference matching logic — ported from TypeScript."""

from __future__ import annotations

import json
import re
from typing import Any

from legal_db import LegalDBClient


# ── Reference detection ──────────────────────────────────────────


def is_judgment_reference(text: str) -> bool:
    t = text.strip().lower()
    if re.search(r"sygn\.?\s*akt", t, re.I):
        return True
    if re.match(r"^(wyrok|postanowienie|uchwał[aę]|orzeczenie|decyzja)\s", t, re.I):
        return True
    if re.search(r"\b(SN|NSA|WSA|SA|SO|SR|TK|TSUE|ETPCz)\b", t, re.I) and re.search(
        r"\d{4}\s*r\.?", t, re.I
    ):
        return True
    return False


def extract_sygnatura(text: str) -> str | None:
    m = re.search(r"sygn\.?\s*akt\s+([A-Z0-9][A-Za-z0-9\s/\-–\.]+)", text.strip(), re.I)
    if m:
        return re.sub(r"[.,;:\s]+$", "", m.group(1).strip()).strip()
    return None


# ── Article parsing ──────────────────────────────────────────────


def parse_article_reference(text: str) -> dict[str, str] | None:
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"<sup>(\d+)</sup>", r"^\1", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\)\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    prefix_m = re.match(r"^(.+?)\s+art\.?\s+(\d+.*)$", t, re.I)
    if prefix_m and not re.match(r"^art\.?$", prefix_m.group(1), re.I):
        act_prefix = prefix_m.group(1).strip()
        art_part = prefix_m.group(2).strip()
        if not re.match(r"^(§|ust\.?|pkt\.?|lit\.?)\s", act_prefix, re.I):
            escaped = re.escape(act_prefix)
            art_part = re.sub(rf"\s+{escaped}\s*$", "", art_part, flags=re.I).strip()
            t = f"art. {art_part} {act_prefix}"

    i_art = re.search(r"\s+i\s+art\.?\s+\d", t, re.I)
    if i_art:
        first_part = t[: i_art.start()].strip()
        second_part = re.sub(r"^\s+i\s+art\.?\s+", "art. ", t[i_art.start():], flags=re.I).strip()
        second_alias = re.sub(
            r"(?:^|\s+)(?:§|ust\.?|pkt\.?|lit\.?)\s*\d+[a-z]?(?:\s+i\s+\d+[a-z]?)*",
            "",
            re.sub(r"^art\.?\s*\d+[a-z]?(?:\^\d+)?", "", second_part).strip(),
            flags=re.I,
        ).strip()
        first_alias = re.sub(
            r"(?:^|\s+)(?:§|ust\.?|pkt\.?|lit\.?)\s*\d+[a-z]?(?:\s+i\s+\d+[a-z]?)*",
            "",
            re.sub(r"^art\.?\s*\d+[a-z]?(?:\^\d+)?", "", first_part).strip(),
            flags=re.I,
        ).strip()
        if first_alias and len(first_alias) >= 2:
            t = first_part
        elif second_alias and len(second_alias) >= 2:
            t = first_part + " " + second_alias
        else:
            t = first_part

    t = re.sub(r"\s+oraz\s+art\.?\s+\d+.*$", "", t, flags=re.I).strip()
    t = re.sub(r"^(art\.?\s*\d+[a-z]?(?:\^\d+)?)\s*[-–—]\s*\d+[a-z]?\s", r"\1 ", t, flags=re.I)

    art_m = re.match(r"^art\.?\s*(\d+[a-z]?(?:\^\d+)?)(.*)$", t, re.I)
    par_m = re.match(r"^§\s*(\d+[a-z]?(?:\^\d+)?)(.*)$", t, re.I)

    if art_m:
        number = art_m.group(1).strip()
        rest = (art_m.group(2) or "").strip()
    elif par_m:
        number = par_m.group(1).strip()
        rest = (par_m.group(2) or "").strip()
    else:
        return None

    number_parts = [number]
    prev_rest = ""
    while rest != prev_rest:
        prev_rest = rest

        m = re.match(
            r"^§\s*(\d+[a-z]?(?:\s*[-–—]\s*§?\s*\d+[a-z]?)?)(\s+i\s+§?\s*\d+[a-z]?(?:\s*[-–—]\s*\d+[a-z]?)?)?(.*)$",
            rest, re.I,
        )
        if m:
            val = f"§ {re.sub(r'\\s*[-–—]\\s*§?\\s*', '-', m.group(1).strip())}"
            if m.group(2):
                val += " " + re.sub(r"\s+", " ", m.group(2).strip())
            number_parts.append(val)
            rest = (m.group(3) or "").strip()
            continue

        m = re.match(r"^ust\.?\s*(\d+[a-z]?(?:\s*[-–—]\s*\d+[a-z]?)?)(.*)$", rest, re.I)
        if m:
            val = f"ust. {re.sub(r'\\s*[-–—]\\s*', '-', m.group(1).strip())}"
            rest = (m.group(2) or "").strip()
            i_conn = re.match(r"^i\s+(\d+[a-z]?(?:\s*[-–—]\s*\d+[a-z]?)?)\s*(.*)$", rest, re.I)
            if i_conn:
                after = (i_conn.group(2) or "").strip()
                if not re.match(r"^(ust\.?|§|pkt\.?|lit\.?|art\.?)\s", after, re.I):
                    val += f" i {i_conn.group(1).strip()}"
                    rest = after
            number_parts.append(val)
            continue

        m = re.match(
            r"^pkt\.?\s*(\d+[a-z]?(?:\s+i\s+\d+[a-z]?)*(?:\s*[-–—]\s*\d+[a-z]?)?)(.*)$",
            rest, re.I,
        )
        if m:
            number_parts.append(f"pkt {re.sub(r'\\s+', ' ', m.group(1).strip())}")
            rest = (m.group(2) or "").strip()
            continue

        m = re.match(r"^lit\.?\s*([a-z](?:\s+i\s+[a-z])*)(.*)$", rest, re.I)
        if m:
            number_parts.append(f"lit. {re.sub(r'\\s+', ' ', m.group(1).strip())}")
            rest = (m.group(2) or "").strip()
            continue

        m = re.match(
            r"^i\s+(?:ust\.?|pkt\.?|§|lit\.?)\s*\d+[a-z]?(?:\s+(?:ust\.?|pkt\.?|§|lit\.?)\s*\d+[a-z]?)*\s*(.*)$",
            rest, re.I,
        )
        if m:
            rest = (m.group(1) or "").strip()

        rest = re.sub(
            r"^w\s+zw\.?\s+z\s+(?:art\.?|§)\s*\d+[a-z]?(?:\s+(?:§|ust\.?|pkt\.?|lit\.?)\s*\d+[a-z]?(?:\s+i\s+\d+[a-z]?)*)*\s*",
            "", rest, flags=re.I,
        ).strip()
        break

    rest = re.sub(r"^[-–—]\s*", "", rest).strip()
    rest = re.sub(
        r"^i\s+\d+[a-z]?(?:\s+(?:ust\.?|§|pkt\.?|lit\.?)\s*\d+[a-z]?(?:\s+i\s+\d+[a-z]?)*)*\s*",
        "", rest, flags=re.I,
    ).strip()
    rest = re.sub(
        r"^i\s+art\.?\s+\d+[a-z]?(?:\s+(?:ust\.?|§|pkt\.?|lit\.?)\s*\d+[a-z]?)*\s*",
        "", rest, flags=re.I,
    ).strip()
    rest = re.sub(
        r"^(ustawy?|rozporządzeni[ae])\s+z\s+dnia\s+\d{1,2}\s+[^-–—]+\s+[-–—]\s*",
        "", rest, flags=re.I,
    ).strip()
    rest = re.sub(r"^ustawy?\s*[-–—]\s*", "", rest, flags=re.I).strip()
    rest = re.sub(r"^ustawy?\s+", "", rest, flags=re.I).strip()
    rest = re.sub(r"^rozporządzeni[ae]\s+", "", rest, flags=re.I).strip()
    rest = re.sub(r"\.$", "", rest).strip()

    if not rest or len(rest) < 2:
        return None

    return {"number": " ".join(number_parts).strip(), "actAlias": rest}


def build_article_number_candidates(number: str) -> list[str]:
    n = re.sub(r"\s+", " ", number).strip()
    candidates = [n]

    wo_i = re.sub(r"\s+i\s+(?:§\s*)?\d+[a-z]?(?:\s*[-–—]\s*\d+[a-z]?)?$", "", n, flags=re.I).strip()
    if wo_i and wo_i not in candidates:
        candidates.append(wo_i)

    base = wo_i or n
    wo_lit = re.sub(r"\s+lit\.\s+[a-z](?:\s+i\s*[a-z])?$", "", base, flags=re.I).strip()
    if wo_lit and wo_lit not in candidates:
        candidates.append(wo_lit)

    wo_pkt = re.sub(
        r"\s+pkt\s+\d+[a-z]?(?:\s+i\s*\d+[a-z]?)*(?:\s*[-–—]\s*\d+[a-z]?)?$",
        "", wo_lit, flags=re.I,
    ).strip()
    if wo_pkt and wo_pkt not in candidates:
        candidates.append(wo_pkt)

    wo_ust = re.sub(
        r"\s+ust\.\s*\d+[a-z]?(?:\s+i\s+\d+[a-z]?)?(?:\s*[-–—]\s*\d+[a-z]?)?$",
        "", wo_pkt, flags=re.I,
    ).strip()
    if wo_ust and wo_ust not in candidates:
        candidates.append(wo_ust)

    wo_par = re.sub(
        r"\s+§\s*\d+[a-z]?(?:\s+i\s+§?\s*\d+[a-z]?)?(?:\s*[-–—]\s*\d+[a-z]?)?$",
        "", wo_ust, flags=re.I,
    ).strip()
    if wo_par and wo_par not in candidates:
        candidates.append(wo_par)

    return [c for c in candidates if c]


# ── Genitive → Nominative map ────────────────────────────────────

GENITIVE_MAP: dict[str, str] = {
    "ordynacji podatkowej": "Ordynacja podatkowa",
    "prawa budowlanego": "Prawo budowlane",
    "prawa bankowego": "Prawo bankowe",
    "prawa zamówień publicznych": "Prawo zamówień publicznych",
    "prawa energetycznego": "Prawo energetyczne",
    "prawa wodnego": "Prawo wodne",
    "prawa ochrony środowiska": "Prawo ochrony środowiska",
    "prawa pocztowego": "Prawo pocztowe",
    "prawa celnego": "Prawo celne",
    "prawa farmaceutycznego": "Prawo farmaceutyczne",
    "prawa geologicznego i górniczego": "Prawo geologiczne i górnicze",
    "prawa lotniczego": "Prawo lotnicze",
    "prawa o ustroju sądów powszechnych": "Prawo o ustroju sądów powszechnych",
    "prawa o adwokaturze": "Prawo o adwokaturze",
    "prawa o notariacie": "Prawo o notariacie",
    "prawa telekomunikacyjnego": "Prawo telekomunikacyjne",
    "prawa przewozowego": "Prawo przewozowe",
    "prawa spółdzielczego": "Prawo spółdzielcze",
    "prawa wekslowego": "Prawo wekslowe",
    "prawa restrukturyzacyjnego": "Prawo restrukturyzacyjne",
    "prawa upadłościowego": "Prawo upadłościowe",
    "kodeksu karnego": "Kodeks karny",
    "kodeksu cywilnego": "Kodeks cywilny",
    "kodeksu postępowania karnego": "Kodeks postępowania karnego",
    "kodeksu postępowania cywilnego": "Kodeks postępowania cywilnego",
    "kodeksu postępowania administracyjnego": "Kodeks postępowania administracyjnego",
    "kodeksu pracy": "Kodeks pracy",
    "kodeksu rodzinnego i opiekuńczego": "Kodeks rodzinny i opiekuńczy",
    "kodeksu wykroczeń": "Kodeks wykroczeń",
    "kodeksu spółek handlowych": "Kodeks spółek handlowych",
    "kodeksu karnego skarbowego": "Kodeks karny skarbowy",
    "kodeksu morskiego": "Kodeks morski",
    "prawa o stowarzyszeniach": "Prawo o stowarzyszeniach",
}


async def fuzzy_find_act(db: LegalDBClient, alias: str) -> dict | None:
    from urllib.parse import quote

    cleaned = re.sub(r"\.$", "", alias)
    cleaned = re.sub(r"^ustawy?\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^z\s+dnia\s+\d{1,2}\s+\w+\s+\d{4}\s*r?\.?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(o|z dnia|z)\s+", "", cleaned, flags=re.I).strip()

    if len(cleaned) < 3:
        return None

    nominative = GENITIVE_MAP.get(cleaned.lower())
    searches = []
    if nominative:
        searches.append(nominative)
    searches.append(cleaned)

    for term in searches:
        try:
            rows = await db.get(
                f"/acts?short_title=ilike.*{quote(term)}*&select=id,title,short_title&limit=5"
            )
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                rows.sort(key=lambda r: len(r.get("short_title") or "z" * 999))
                return rows[0]
        except Exception:
            pass

        try:
            rows = await db.get(
                f"/acts?title=ilike.*{quote(term)}*&select=id,title,short_title&limit=5"
            )
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                rows.sort(key=lambda r: len(r.get("title") or "z" * 999))
                return rows[0]
        except Exception:
            pass

    return None


async def match_judgment_refs(
    db: LegalDBClient,
    leaf_ids: list[int],
    alias_map: dict[str, int],
    api: Any,  # PipelineAPIClient
) -> dict[str, int]:
    """Match unmatched article refs for given leaf IDs."""
    if not leaf_ids:
        return {"matched": 0, "partial": 0, "unmatched": 0}

    rels: list[dict] = []
    for i in range(0, len(leaf_ids), 200):
        batch = leaf_ids[i : i + 200]
        ids_str = ",".join(str(x) for x in batch)
        rows = await db.get(
            f"/leaf_artykul_rel?leaf_id=in.({ids_str})&match_status=in.(unmatched,partial)&select=id,leaf_id,artykul_id,artykul_text_fallback"
        )
        rels.extend(rows)

    matched = partial = unmatched = 0

    for rel in rels:
        fallback = rel.get("artykul_text_fallback", "")

        if is_judgment_reference(fallback):
            syg = extract_sygnatura(fallback)
            if syg:
                try:
                    exact = await db.get(
                        f"/orzeczenia?sygnatura=eq.{_q(syg)}&select=id,numer_bos,sygnatura&limit=1"
                    )
                    if exact:
                        await db.update("leaf_artykul_rel", f"id=eq.{rel['id']}", {
                            "match_status": "exact",
                            "role": json.dumps({"ref_type": "judgment", "orzeczenie_ref_id": exact[0]["id"]}),
                        })
                        matched += 1
                        continue
                except Exception:
                    pass
            unmatched += 1
            continue

        parsed = parse_article_reference(fallback)
        if not parsed:
            try:
                await db.update("leaf_artykul_rel", f"id=eq.{rel['id']}", {"role": "unparseable"})
            except Exception:
                pass
            unmatched += 1
            continue

        alias_lower = parsed["actAlias"].lower()
        act_id = _resolve_alias(alias_lower, alias_map)

        if not act_id:
            act = await fuzzy_find_act(db, parsed["actAlias"])
            if act:
                act_id = act["id"]
                alias_map[alias_lower] = act_id
                # Save new alias via API (no direct Supabase access)
                await api.save_alias(
                    parsed["actAlias"],
                    act_id,
                    act.get("short_title") or act.get("title"),
                )

        if not act_id:
            unmatched += 1
            continue

        candidates = build_article_number_candidates(parsed["number"])
        matched_art = None
        for cand in candidates:
            from urllib.parse import quote

            arts = await db.get(
                f"/articles?act_id=eq.{act_id}&number=eq.{quote(cand)}&select=id,number&limit=1"
            )
            if arts:
                matched_art = arts[0]
                break

        if matched_art:
            try:
                await db.update("leaf_artykul_rel", f"id=eq.{rel['id']}", {
                    "artykul_id": matched_art["id"],
                    "match_status": "exact",
                    "role": json.dumps({"act_id": act_id, "matched_number": matched_art["number"]}),
                })
                matched += 1
            except Exception:
                matched += 1
        else:
            try:
                await db.update("leaf_artykul_rel", f"id=eq.{rel['id']}", {
                    "match_status": "partial",
                    "role": json.dumps({"act_id": act_id, "parsed_number": parsed["number"]}),
                })
            except Exception:
                pass
            partial += 1

    return {"matched": matched, "partial": partial, "unmatched": unmatched}


def _resolve_alias(alias_lower: str, alias_map: dict[str, int]) -> int | None:
    a = alias_map.get(alias_lower)
    if a:
        return a
    a = alias_map.get(alias_lower.rstrip("."))
    if a:
        return a
    a = alias_map.get(alias_lower + ".")
    if a:
        return a
    collapsed = re.sub(r"\.\s+i\s+", ".", alias_lower)
    collapsed = re.sub(r"\.\s+", ".", collapsed)
    collapsed = re.sub(r"([a-z])$", r"\1.", collapsed)
    a = alias_map.get(collapsed)
    if a:
        return a
    a = alias_map.get(collapsed.rstrip("."))
    return a


def _q(s: str) -> str:
    from urllib.parse import quote
    return quote(s)
