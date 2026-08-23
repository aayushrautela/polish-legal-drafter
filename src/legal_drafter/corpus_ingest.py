from __future__ import annotations

"""Fetch a Polish statute from the official Sejm ELI API and emit corpus chunks
that match the schema of the recovered ``sejm_eli`` chunks.

The output mirrors ``cleaned_rag_chunks.jsonl`` exactly (``source_ref``,
``source_id``, ``chunk_id``, ``article_number``, ``display_address``,
``text_hash`` …) so the new act can be indexed alongside the existing corpus by
any consumer of :mod:`legal_drafter.rag` / :mod:`legal_drafter.retrieval`.

Source of truth: Sejm ELI API (https://api.sejm.gov.pl/eli), the official
primary publisher of Dziennik Ustaw / Monitor Polski.
"""

import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable

_ELI_API = "https://api.sejm.gov.pl/eli"
_USER_AGENT = "polish-legal-drafter/0.1 (+academic research)"


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def _collect_articles(struct: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return ``(article_name, article_title)`` tuples from the act structure."""
    out: list[tuple[str, str]] = []

    def walk(node: dict[str, Any]) -> None:
        if node.get("type") == "arti":
            out.append((str(node.get("name", "")), str(node.get("title", ""))))
        for child in node.get("children", []) or []:
            walk(child)

    for part in struct:
        walk(part)
    return out


def fetch_sejm_act(
    *,
    publisher: str,
    year: int,
    position: int,
    title: str,
    display_address: str,
    base_url: str = _ELI_API,
) -> list[dict[str, Any]]:
    """Fetch every article of a Sejm act and return corpus records."""
    base = f"{base_url}/acts/{publisher}/{year}/{position}"
    struct = _http_json(f"{base}/struct")
    articles = _collect_articles(struct)
    if not articles:
        raise ValueError(f"No articles found in structure for {publisher} {year} {position}")

    prefix = f"sejm_eli:sejm_eli_{publisher.lower()}_{year}_{position}"
    source_id_prefix = f"sejm_eli_{publisher.lower()}_{year}_{position}"
    eli_url = f"https://eli.gov.pl/eli/{publisher}/{year}/{position}/ogl"

    records: list[dict[str, Any]] = []
    for name, _title in articles:
        fragment_url = f"{base}/text.html/art={name}"
        try:
            raw = _http_text(fragment_url)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"Failed to fetch article {name}: {exc}") from exc
        text = _strip_html(raw)
        if not text:
            continue
        source_ref = f"{prefix}_art_{name}"
        source_id = f"{source_id_prefix}_art_{name}"
        match = re.match(r"\d+", name)
        article_number = int(match.group()) if match else None
        text_hash = hashlib.md5(f"{source_ref}|{text}".encode("utf-8")).hexdigest()[:16]
        records.append(
            {
                "source_ref": source_ref,
                "origin": "sejm_eli",
                "source_id": source_id,
                "chunk_id": source_id,
                "item_id": None,
                "observation_id": None,
                "source_type": "statute",
                "publisher": "Sejm ELI",
                "title": title,
                "display_address": display_address,
                "article_number": article_number,
                "top_category": None,
                "risk_or_safe": None,
                "doc_types": [],
                "source_url": eli_url,
                "text": text,
                "text_hash": text_hash,
            }
        )
    return records


def write_corpus(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count
