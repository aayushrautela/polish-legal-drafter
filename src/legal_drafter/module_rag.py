"""Module-aware RAG retriever for full-document contract generation.

Extends BM25 retrieval with module/variant/authority filtering
for the UMOWA_NAJMU full-document pipeline.

Usage:
    from legal_drafter.module_rag import ModuleRAG
    rag = ModuleRAG()
    results = rag.retrieve(module_ids=["M06_deposit"], query="kaucja zwrot", top_k=5)
"""

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "module_source_catalog.jsonl"

POLISH_STOPWORDS = {
    "i", "w", "z", "na", "do", "nie", "się", "to", "jest", "że", "o", "od",
    "co", "jak", "ale", "za", "po", "tak", "już", "lub", "tylko", "jest",
    "też", "ten", "ta", "te", "ich", "tym", "był", "by", "być", "może",
    "go", "je", "jej", "jego", "są", "będzie", "które", "który", "która",
    "a", "dla", "przez", "przy", "między", "pod", "nad", "przed", "bez",
    "oraz", "natomiast", "jednak", "więc", "czy", "gdy", "jeśli",
    "ani", "albo", "czyli", "tego", "tym", "tej", "tych",
}

POLISH_SYNONYMS = {
    "czynsz": ["opłata", "należność", "stawkа", "opłaty"],
    "kaucja": ["zabezpieczenie", "depozyt"],
    "wypowiedzenie": ["rozwiązanie", "zakończenie", "wypowiedzenie umowy"],
    "eksmisja": ["opuszczenie", "usunięcie", "wyrzucenie"],
    "naprawa": ["remont", "uszczelka", "usterka", "wada"],
    "najemca": ["lokator", "wynajmujący"],
    "wynajmujący": ["właściciel", "najemca"],
    "protokół": ["protokół zdawczo-odbiorczy", "protokołu"],
    "podwyżka": ["zwiększenie", "wzrost"],
    "umowa": ["kontrakt", "porozumienie"],
}


@dataclass
class ModuleSourceRow:
    source_id: str
    parent_chunk_id: str
    original_source_id: str
    module_ids: list[str]
    slot_ids: list[str]
    lease_variants_allowed: list[str]
    lease_variants_forbidden: list[str]
    source_role: str
    authority_level: str
    article_display: str
    text: str
    legal_regime: str
    risk_tags: list[str]
    norm_ids: list[str]
    keywords: list[str]


@dataclass
class RetrievalResult:
    source_id: str
    module_ids: list[str]
    source_role: str
    authority_level: str
    article_display: str
    text: str
    risk_tags: list[str]
    norm_ids: list[str]
    score: float
    match_reasons: list[str]


def normalize(text: str) -> str:
    """Lowercase and strip Polish diacritics."""
    text = text.lower()
    replacements = {"ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z"}
    for pl, lat in replacements.items():
        text = text.replace(pl, lat)
    return text


def tokenize(text: str) -> list[str]:
    """Split into normalized tokens, filtering stopwords."""
    norm = normalize(text)
    tokens = re.findall(r"\w+", norm)
    return [t for t in tokens if t not in POLISH_STOPWORDS and len(t) > 1]


def expand_query(tokens: list[str]) -> list[str]:
    """Add synonym tokens."""
    expanded = list(tokens)
    norm_synonyms = {normalize(k): [normalize(v) for v in vals] for k, vals in POLISH_SYNONYMS.items()}
    for token in tokens:
        for key, syns in norm_synonyms.items():
            if token == key:
                expanded.extend(syns)
            elif token in syns:
                expanded.append(key)
    return list(set(expanded))


class ModuleRAG:
    """Module-aware retriever for UMOWA_NAJMU full-document RAG."""

    def __init__(self, catalog_path: Path | str | None = None, bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.catalog_path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG
        self.k1 = bm25_k1
        self.b = bm25_b
        self.rows: list[ModuleSourceRow] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._df: Counter = Counter()
        self._n_docs: int = 0
        self._loaded = False

    def load(self):
        """Load catalog from JSONL file."""
        if self._loaded:
            return
        rows = []
        with open(self.catalog_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rows.append(ModuleSourceRow(**data))
        self.rows = rows
        self._build_index()
        self._loaded = True

    def _build_index(self):
        """Build BM25 index over catalog rows."""
        self._doc_tokens = []
        self._doc_lens = []
        self._df = Counter()
        for row in self.rows:
            searchable = f"{row.text} {row.article_display} {' '.join(row.keywords)} {' '.join(row.norm_ids)} {' '.join(row.risk_tags)}"
            tokens = tokenize(searchable)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            unique = set(tokens)
            for t in unique:
                self._df[t] += 1
        self._n_docs = len(self.rows)
        self._avg_dl = sum(self._doc_lens) / max(self._n_docs, 1)

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = self._doc_lens[doc_idx]
        tf = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            term_freq = tf[qt]
            doc_freq = self._df.get(qt, 0)
            idf = math.log((self._n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            tf_norm = (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_dl, 1)))
            score += idf * tf_norm
        return score

    def retrieve(
        self,
        query: str = "",
        module_ids: list[str] | None = None,
        lease_variant: str | None = None,
        source_roles: list[str] | None = None,
        authority_levels: list[str] | None = None,
        exclude_roles: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve sources with module/variant/authority filtering + BM25 ranking.

        Args:
            query: Free-text query for BM25 scoring (can be empty for filter-only).
            module_ids: Filter to rows matching ANY of these module IDs.
            lease_variant: Filter to rows allowing this lease variant.
            source_roles: Filter to rows with these roles (e.g. ["positive_rule"]).
            authority_levels: Filter to rows with these authority levels.
            exclude_roles: Exclude rows with these roles.
            top_k: Max results to return.
            min_score: Minimum BM25 score threshold (0 = no threshold).

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        self.load()

        # Expand query tokens
        query_tokens = tokenize(query) if query else []
        expanded = expand_query(query_tokens) if query_tokens else []

        candidates = []
        for i, row in enumerate(self.rows):
            # Module filter
            if module_ids and not any(m in row.module_ids for m in module_ids):
                continue
            # Lease variant filter
            if lease_variant:
                if row.lease_variants_forbidden and lease_variant in row.lease_variants_forbidden:
                    continue
                if row.lease_variants_allowed and lease_variant not in row.lease_variants_allowed:
                    continue
            # Source role filter
            if source_roles and row.source_role not in source_roles:
                continue
            # Authority filter
            if authority_levels and row.authority_level not in authority_levels:
                continue
            # Exclude roles
            if exclude_roles and row.source_role in exclude_roles:
                continue

            # BM25 score
            score = self._bm25_score(expanded, i) if expanded else 1.0

            # Authority boost
            auth_boost = {
                "primary_statute": 1.3,
                "negative_precedent": 1.25,  # boost curated negatives
                "secondary_guidance": 1.0,
                "court_precedent": 0.9,
                "other": 0.7,
            }.get(row.authority_level, 0.7)
            score *= auth_boost

            # Source role boost
            role_boost = {"positive_rule": 1.2, "negative_clause": 1.1, "background": 0.8}.get(row.source_role, 1.0)
            score *= role_boost

            # Curated source boost
            if row.source_id.startswith("curated_"):
                score *= 1.5

            if score >= min_score:
                match_reasons = []
                if module_ids and any(m in row.module_ids for m in module_ids):
                    match_reasons.append("module_match")
                if lease_variant:
                    match_reasons.append("variant_match")
                if source_roles and row.source_role in source_roles:
                    match_reasons.append("role_match")
                if expanded:
                    matched_tokens = set(expanded) & set(self._doc_tokens[i])
                    if matched_tokens:
                        match_reasons.append(f"bm25_tokens:{','.join(sorted(matched_tokens)[:5])}")

                # Curated sources get floor score when module-matched
                if row.source_id.startswith("curated_") and "module_match" in match_reasons:
                    score = max(score, 0.5)

                candidates.append(RetrievalResult(
                    source_id=row.source_id,
                    module_ids=row.module_ids,
                    source_role=row.source_role,
                    authority_level=row.authority_level,
                    article_display=row.article_display,
                    text=row.text,
                    risk_tags=row.risk_tags,
                    norm_ids=row.norm_ids,
                    score=round(score, 4),
                    match_reasons=match_reasons,
                ))

        candidates.sort(key=lambda r: r.score, reverse=True)
        result = candidates[:top_k]

        # Ensure curated sources are included when module-matched
        if module_ids:
            curated_in_result = {r.source_id for r in result if r.source_id.startswith("curated_")}
            for row in self.rows:
                if not row.source_id.startswith("curated_"):
                    continue
                if row.source_id in curated_in_result:
                    continue
                if not any(m in row.module_ids for m in module_ids):
                    continue
                if exclude_roles and row.source_role in exclude_roles:
                    continue
                if lease_variant and row.lease_variants_forbidden and lease_variant in row.lease_variants_forbidden:
                    continue
                match_reasons = ["module_match", "curated_guaranteed"]
                if lease_variant:
                    match_reasons.append("variant_match")
                result.append(RetrievalResult(
                    source_id=row.source_id,
                    module_ids=row.module_ids,
                    source_role=row.source_role,
                    authority_level=row.authority_level,
                    article_display=row.article_display,
                    text=row.text,
                    risk_tags=row.risk_tags,
                    norm_ids=row.norm_ids,
                    score=0.5,
                    match_reasons=match_reasons,
                ))

        return result

    def retrieve_for_module(
        self,
        module_id: str,
        query: str = "",
        lease_variant: str = "ordinary_residential_indefinite",
        include_negatives: bool = True,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Convenience: retrieve sources for a single module with sensible defaults."""
        roles = ["positive_rule"]
        if include_negatives:
            roles.append("negative_clause")
        return self.retrieve(
            query=query,
            module_ids=[module_id],
            lease_variant=lease_variant,
            source_roles=roles,
            top_k=top_k,
        )

    def retrieve_for_modules(
        self,
        module_ids: list[str],
        query: str = "",
        lease_variant: str = "ordinary_residential_indefinite",
        include_negatives: bool = True,
        top_k: int = 15,
    ) -> list[RetrievalResult]:
        """Retrieve sources for multiple modules."""
        roles = ["positive_rule"]
        if include_negatives:
            roles.append("negative_clause")
        return self.retrieve(
            query=query,
            module_ids=module_ids,
            lease_variant=lease_variant,
            source_roles=roles,
            top_k=top_k,
        )

    def get_module_summary(self) -> dict:
        """Return summary stats of loaded catalog."""
        self.load()
        module_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        authority_counts: dict[str, int] = {}
        for row in self.rows:
            role_counts[row.source_role] = role_counts.get(row.source_role, 0) + 1
            authority_counts[row.authority_level] = authority_counts.get(row.authority_level, 0) + 1
            for mid in row.module_ids:
                module_counts[mid] = module_counts.get(mid, 0) + 1
        return {
            "total_rows": len(self.rows),
            "module_coverage": dict(sorted(module_counts.items())),
            "role_distribution": role_counts,
            "authority_distribution": authority_counts,
        }


def format_retrieval_context(results: list[RetrievalResult], max_chars: int = 4000) -> str:
    """Format retrieval results into a context string for LLM prompt."""
    parts = []
    total = 0
    for r in results:
        header = f"[{r.source_id}] {r.article_display or r.source_id} (moduły: {', '.join(r.module_ids)})"
        snippet = r.text[:500]
        entry = f"{header}\n{snippet}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)
