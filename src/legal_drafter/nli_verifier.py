from __future__ import annotations

import re
from typing import Any

from .source_semantics import map_final_verdict, source_aware_verdict

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

NEGATION_PATTERNS = [
    r"nie\s+(może|moze|stanowi|oznacza|jest|będzie|bedzie|zatrzymuje|zatrzyma|nalicza|przewiduje|stosuje)",
    r"bez\s+jednostronnego",
    r"zgodnie\s+z\s+powszechnie\s+obowiązującymi\s+przepisami",
]
SAFE_PATTERNS = [
    r"właściw\w*\s+zgodnie\s+z\s+.*przepis",
    r"niewykorzystan\w*\s+częś\w*\s+.*zwrot|zwraca\w*\s+.*niewykorzystan",
    r"rzeczywist\w*\s+koszt",
    r"milczen\w*.*nie\s+(stanowi|oznacza)|brak\s+(odpowiedzi|sprzeciwu).*nie\s+(stanowi|oznacza)",
    r"odsetk\w*\s+za\s+opóźnienie|wezwanie\s+do\s+zapłaty",
    r"ważn\w*\s+przyczyn|istotn\w*\s+narus",
]
UNSAFE_PATTERNS = [
    r"wyłącznie\s+sąd\w*\s+właściw\w*\s+dla\s+siedzib",
    r"zatrzyma\w*\s+cał\w*\s+zaliczk",
    r"kara\w*\s+umown\w*[^.]{0,120}30%",
    r"brak\s+(sprzeciwu|odpowiedzi)[^.]{0,120}oznacza\w*\s+akcept",
    r"milczen\w*[^.]{0,120}akcept",
    r"kara\w*\s+umown\w*[^.]{0,160}opóźn\w*[^.]{0,80}zapłat|kara\w*\s+umown\w*[^.]{0,160}opóźn\w*[^.]{0,80}wynagrodz",
    r"dowoln\w*\s+powod[^.]{0,160}cał\w*\s+wynagrodz",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("text") or "") for item in evidence)


def keyword_support_score(claim_text: str, evidence: list[dict[str, Any]]) -> float:
    claim_tokens = {token for token in re.findall(r"[a-ząćęłńóśźż0-9]+", claim_text.lower()) if len(token) > 3}
    if not claim_tokens:
        return 0.0
    evidence_tokens = {token for token in re.findall(r"[a-ząćęłńóśźż0-9]+", evidence_text(evidence).lower()) if len(token) > 3}
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def heuristic_nli_verdict(claim: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    claim_text = normalize(str(claim.get("claim_text") or ""))
    evidence_blob = normalize(evidence_text(evidence))
    support_score = keyword_support_score(claim_text, evidence)
    unsafe = matches_any(UNSAFE_PATTERNS, claim_text)
    safe = matches_any(SAFE_PATTERNS, claim_text) or matches_any(NEGATION_PATTERNS, claim_text)
    evidence_mentions_risk = any(word in evidence_blob for word in ["niedozwol", "kara", "sąd", "sad", "odsetk", "wypowied", "zaliczk", "akcept"])

    if unsafe and evidence_mentions_risk:
        verdict = "contradicted"
        confidence = max(0.72, min(0.95, 0.72 + support_score))
    elif safe and support_score >= 0.08:
        verdict = "supported"
        confidence = max(0.62, min(0.9, 0.62 + support_score))
    elif support_score >= 0.22:
        verdict = "supported"
        confidence = min(0.82, 0.5 + support_score)
    else:
        verdict = "insufficient"
        confidence = max(0.35, min(0.65, 0.35 + support_score))

    return {
        "claim_id": claim.get("claim_id"),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "support_score": round(support_score, 4),
        "best_evidence_source_id": evidence[0].get("source_id") if evidence else None,
        "method": "heuristic_baseline",
    }


class TransformerNLIVerifier:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, *, device: str | None = None, max_length: int = 512, batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}

    def pair_scores(self, claim_text: str, premises: list[str]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(premises), self.batch_size):
                batch = premises[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    [claim_text] * len(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                probs = self.torch.softmax(self.model(**encoded).logits.detach().float().cpu(), dim=-1)
                for row in probs.tolist():
                    values = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
                    for idx, score in enumerate(row):
                        label = self.id2label.get(idx, str(idx)).lower()
                        if "entail" in label:
                            values["entailment"] = float(score)
                        elif "contrad" in label:
                            values["contradiction"] = float(score)
                        elif "neutral" in label:
                            values["neutral"] = float(score)
                    rows.append(values)
        return rows

    def verify(self, claim: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        claim_text = str(claim.get("claim_text") or "")
        premises = [str(item.get("text") or "") for item in evidence]
        scores = self.pair_scores(claim_text, premises) if premises else []
        if not scores:
            return {
                "claim_id": claim.get("claim_id"),
                "verdict": "insufficient",
                "confidence": 0.0,
                "best_evidence_source_id": None,
                "method": self.model_name,
                "pair_scores": [],
            }
        best_entailment = max(enumerate(scores), key=lambda item: item[1].get("entailment", 0.0))
        best_contradiction = max(enumerate(scores), key=lambda item: item[1].get("contradiction", 0.0))
        entailment_score = best_entailment[1].get("entailment", 0.0)
        contradiction_score = best_contradiction[1].get("contradiction", 0.0)
        if contradiction_score >= 0.55 and contradiction_score >= entailment_score + 0.08:
            verdict = "contradicted"
            best_index = best_contradiction[0]
            confidence = contradiction_score
        elif entailment_score >= 0.55:
            verdict = "supported"
            best_index = best_entailment[0]
            confidence = entailment_score
        else:
            verdict = "insufficient"
            best_index = best_entailment[0] if entailment_score >= contradiction_score else best_contradiction[0]
            confidence = max(entailment_score, contradiction_score)
        return {
            "claim_id": claim.get("claim_id"),
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "best_evidence_source_id": evidence[best_index].get("source_id") if evidence else None,
            "method": self.model_name,
            "pair_scores": [
                {
                    "source_id": evidence[index].get("source_id") if index < len(evidence) else None,
                    "entailment": round(values.get("entailment", 0.0), 4),
                    "neutral": round(values.get("neutral", 0.0), 4),
                    "contradiction": round(values.get("contradiction", 0.0), 4),
                }
                for index, values in enumerate(scores)
            ],
        }


def verify_claim(claim: dict[str, Any], evidence: list[dict[str, Any]], *, verifier: TransformerNLIVerifier | None = None) -> dict[str, Any]:
    if verifier:
        nli_verdict = verifier.verify(claim, evidence)
    else:
        nli_verdict = heuristic_nli_verdict(claim, evidence)
    return source_aware_verdict(claim, evidence, nli_verdict)


def map_expected_verdict(verdict: str) -> str:
    return map_final_verdict(verdict)
