from __future__ import annotations

import re
from typing import Any

from .rag import as_text, preview, search_rule_context, source_id

PROHIBITED_RULES = [
    {
        "id": "no_contractual_penalty_for_payment_delay",
        "pattern": re.compile(r"(kar\w*\s+umown\w*[^.]{0,220}((za|z\s+tytułu|w\s+przypadku)\s+[^.]{0,90}(opóźn|zwłok)[^.]{0,90}(płatno|zapłat|wynagrodzen|należno)|opóźn\w*[^.]{0,140}(płatno|zapłat|wynagrodzen|należno)|zwłok\w*[^.]{0,140}(płatno|zapłat|wynagrodzen|należno))|(opóźn|zwłok)[^.]{0,180}(zapłat|płatno|wynagrodzen|należno)[^.]{0,180}kar\w*\s+umown\w*)", re.IGNORECASE),
        "source_pattern": re.compile(r"kara\w*\s+umown\w*|art\.\s*483|art\.\s*484|odsetk", re.IGNORECASE),
        "prohibited": "Nie konstruuj kary umownej za samo opóźnienie w zapłacie świadczenia pieniężnego.",
        "allowed": "Dla opóźnienia w zapłacie użyj odsetek za opóźnienie, terminu płatności, wezwania do zapłaty i ewentualnie odpowiedzialności odszkodowawczej na zasadach ogólnych.",
        "safe_alternative": "Zapisz procedurę zapłaty i odsetek; karę umowną zostaw wyłącznie dla jasno opisanych obowiązków niepieniężnych.",
    },
    {
        "id": "no_grossly_excessive_or_one_sided_consumer_penalty",
        "pattern": re.compile(r"(konsument|osob\w*\s+fizyczn|zaliczk|rezygnac|odstąp|odstap|30%|cał\w*\s+zaliczk|rażąco\s+wygórowan|kara\w*\s+umown\w*[^.]{0,120}(rezygnac|odstąp|odstap|zaliczk))", re.IGNORECASE),
        "source_pattern": re.compile(r"niedozwolon|uokik|rażąco|wygórowan|kara|zaliczk|odstąp|odstap", re.IGNORECASE),
        "prohibited": "Nie wpisuj jednostronnej, automatycznej albo rażąco wygórowanej kary/utraty zaliczki wobec konsumenta.",
        "allowed": "Stosuj proporcjonalne rozliczenie wykonanych prac, rzeczywistych kosztów i zwrot niewykorzystanej części świadczenia.",
        "safe_alternative": "Zapisz rozliczenie proporcjonalne oraz możliwość miarkowania/oceny kary, zamiast automatycznego zatrzymania całości.",
    },
    {
        "id": "no_silence_as_acceptance_for_material_changes",
        "pattern": re.compile(r"(milczen|brak\s+sprzeciwu|brak\s+odpowiedzi|jednostron[^.]{0,120}(zmienić|zmienic|zmian)|3\s+dni[^.]{0,120}(akcept|sprzeciw|odpowiedzi)|zmian\w*[^.]{0,120}(cena|zakres|termin|miejsce))", re.IGNORECASE),
        "source_pattern": re.compile(r"niedozwolon|uokik|jednostron|brak\s+odpowiedzi|sprzeciw|zmian", re.IGNORECASE),
        "prohibited": "Nie uznawaj milczenia lub braku sprzeciwu za zgodę na istotną zmianę ceny, terminu, miejsca lub zakresu usługi.",
        "allowed": "Dopuszczaj zmianę istotnych warunków tylko z ważnej przyczyny, po jasnym powiadomieniu i przy wyraźnej akceptacji albo prawie rozwiązania umowy.",
        "safe_alternative": "Zapisz mechanizm powiadomienia, wyraźnej zgody i prawa odstąpienia/rozwiązania bez sankcji przy braku akceptacji.",
    },
    {
        "id": "no_one_sided_consumer_forum_clause",
        "pattern": re.compile(r"(?=.*(konsument|osob\w*\s+fizyczn))(?=.*(sąd|sad|spór|spor|siedzib))", re.IGNORECASE),
        "source_pattern": re.compile(r"sąd\w*\s+właściw|sad\w*\s+wlasciw|siedzib|niedozwolon|uokik", re.IGNORECASE),
        "prohibited": "Nie narzucaj konsumentowi wyłącznej właściwości sądu według siedziby przedsiębiorcy.",
        "allowed": "Zapisz właściwość sądu zgodnie z bezwzględnie obowiązującymi przepisami albo bez jednostronnego ograniczania praw konsumenta.",
        "safe_alternative": "Użyj neutralnej klauzuli: spory rozstrzyga sąd właściwy według przepisów prawa.",
    },
    {
        "id": "no_one_sided_immediate_termination_without_settlement",
        "pattern": re.compile(r"(natychmiast|dowoln\w*\s+powod|bez\s+okresu|bez\s+wypowiedzenia|bez\s+podania\s+przyczyn|zachowuj\w*\s+całość|niewykonan)", re.IGNORECASE),
        "source_pattern": re.compile(r"niedozwolon|uokik|rozwiąz|wypowied|bez\s+ważn|świadczen\w*\s+niespełnion", re.IGNORECASE),
        "prohibited": "Nie dawaj jednej stronie prawa rozwiązania umowy z dowolnego powodu bez rozliczenia świadczeń niewykonanych.",
        "allowed": "Natychmiastowe rozwiązanie ogranicz do ważnej przyczyny lub istotnego naruszenia i dodaj rozliczenie proporcjonalne.",
        "safe_alternative": "Zapisz ważne przyczyny, wezwanie do usunięcia naruszeń gdy właściwe, oraz zwrot/rozliczenie części niewykonanej.",
    },
    {
        "id": "no_full_payment_when_access_or_service_blocked",
        "pattern": re.compile(r"(blokad|zablok|odebran|wstrzym|brak\s+dostępu|brak\s+dostepu|pełn\w*\s+wynagrodzen|pełn\w*\s+czynsz|niewykonan)", re.IGNORECASE),
        "source_pattern": re.compile(r"art\.\s*353|art\.\s*58|nieważn|sprzeczn|ekwiwalent|blokad|pełn\w*\s+wynagrodzen|niewykonan", re.IGNORECASE),
        "prohibited": "Nie utrzymuj pełnego wynagrodzenia za okres, w którym druga strona jest pozbawiona uzgodnionego dostępu albo świadczenie nie jest wykonywane.",
        "allowed": "Stosuj proporcjonalne rozliczenie, przywrócenie dostępu, zwrot za niewykonaną część, odsetki albo zabezpieczenia zgodne z prawem.",
        "safe_alternative": "Zapisz obniżenie/rozliczenie wynagrodzenia za okres braku świadczenia i tryb wezwania do zapłaty.",
    },
]

SECTION_HINTS = {
    "5": ["no_contractual_penalty_for_payment_delay", "no_grossly_excessive_or_one_sided_consumer_penalty"],
    "6": ["no_silence_as_acceptance_for_material_changes", "no_one_sided_immediate_termination_without_settlement", "no_full_payment_when_access_or_service_blocked"],
    "7": ["no_one_sided_consumer_forum_clause", "no_silence_as_acceptance_for_material_changes"],
}

CONSTRAINT_VIOLATION_RULES = {
    "no_contractual_penalty_for_payment_delay": {
        "violation": re.compile(r"kara\w*\s+umown\w*[^.]{0,220}(opóźn|zwłok|płatno|zapłat|wynagrodzen|należno)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(przewiduje|nalicza|stosuje|zastrzega)|kara\w*\s+umown\w*[^.]{0,120}nie\s+dotyczy|zamiast\s+kary[^.]{0,80}odsetk)", re.IGNORECASE),
    },
    "no_grossly_excessive_or_one_sided_consumer_penalty": {
        "violation": re.compile(r"(30\s*%|zatrzym\w*[^.]{0,120}cał\w*|cał\w*[^.]{0,80}zaliczk|bezzwrotn\w*[^.]{0,80}zaliczk|rażąco\s+wygórowan\w*\s+kar)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|podlega|jest)|proporcjonaln\w*\s+rozlicz|rzeczywist\w*\s+koszt|zwrot\w*[^.]{0,80}niewykorzystan)", re.IGNORECASE),
    },
    "no_silence_as_acceptance_for_material_changes": {
        "violation": re.compile(r"(brak\s+(sprzeciwu|odpowiedzi)|milczen\w*|3\s+dni)[^.]{0,160}(oznacz|traktowan|uznan|równoznaczn|akcept)", re.IGNORECASE),
        "safe": re.compile(r"((brak\s+(sprzeciwu|odpowiedzi)|milczen\w*)[^.]{0,120}nie\s+(oznacza|stanowi|jest|będzie|może|moze)|nie\s+(może|moze|mogą|moga|oznacza|stanowi|jest|będzie)[^.]{0,120}(akcept|zgod))", re.IGNORECASE),
    },
    "no_one_sided_consumer_forum_clause": {
        "violation": re.compile(r"(wyłączn\w*[^.]{0,80})?sąd\w*\s+właściw\w*[^.]{0,160}(siedzib\w*\s+(wykonawc|przedsiębiorc|zleceniobiorc)|wykonawc\w*\s+siedzib)", re.IGNORECASE),
        "safe": re.compile(r"(według\s+przepis|zgodnie\s+z\s+przepisami|nie\s+narzuca|nie\s+ogranicza|bez\s+jednostronnego)", re.IGNORECASE),
    },
    "no_one_sided_immediate_termination_without_settlement": {
        "violation": re.compile(r"((natychmiast|bez\s+okresu|bez\s+wypowiedzenia)[^.]{0,160}(dowoln\w*\s+powod|bez\s+podania\s+przyczyn|jakiejkolwiek\s+przyczyn)|zachow\w*[^.]{0,80}(całość|pełn\w*)\s+wynagrodzen[^.]{0,160}niewykonan)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|moze|uprawnia|pozwala)|ważn\w*\s+przyczyn|istotn\w*\s+narus|proporcjonaln\w*\s+rozlicz|zwrot\w*[^.]{0,100}niewykonan)", re.IGNORECASE),
    },
    "no_full_payment_when_access_or_service_blocked": {
        "violation": re.compile(r"(pełn\w*\s+(wynagrodzen|czynsz)[^.]{0,220}(blokad|odebran|wstrzym|zablok|brak\s+dostęp|niewykonan)|zachow\w*[^.]{0,80}(całość|pełn\w*)\s+wynagrodzen[^.]{0,160}niewykonan)", re.IGNORECASE),
        "safe": re.compile(r"(nie\s+(może|moze|zachowuje|przysługuje)|proporcjonaln\w*\s+rozlicz|obniż|zwrot\w*[^.]{0,100}niewykonan|niewykorzystan\w*\s+częś)", re.IGNORECASE),
    },
}


def _source_text(doc: dict[str, Any]) -> str:
    return " ".join([
        as_text(doc.get("title")),
        as_text(doc.get("display_address")),
        as_text(doc.get("number")),
        as_text(doc.get("industry")),
        as_text(doc.get("text_preview") or doc.get("text")),
    ])


def derive_legal_constraints(
    *,
    facts: str,
    section_number: str,
    section_title: str,
    rag_results: list[tuple[float, dict[str, Any]]],
    rag_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = "\n".join([facts, section_title])
    constraints = []
    preferred = set(SECTION_HINTS.get(section_number, []))
    for rule in PROHIBITED_RULES:
        if preferred and rule["id"] not in preferred and not rule["pattern"].search(context):
            continue
        if not rule["pattern"].search(context):
            continue
        rule_results = search_rule_context(rag_index, rule["id"], top_k=4) if rag_index else []
        combined_results = rule_results + rag_results
        supporting_sources = []
        seen_sources = set()
        for score, doc in combined_results:
            sid = source_id(doc)
            if sid in seen_sources:
                continue
            text = _source_text(doc)
            if rule["source_pattern"].search(text):
                supporting_sources.append({
                    "source_id": sid,
                    "score": round(score, 4),
                    "source_type": doc.get("source_type"),
                    "label": doc.get("display_address") or doc.get("number") or doc.get("case_number"),
                    "basis_preview": preview(text, 260),
                })
                seen_sources.add(sid)
        if supporting_sources:
            constraints.append({
                "id": rule["id"],
                "prohibited": rule["prohibited"],
                "allowed": rule["allowed"],
                "safe_alternative": rule["safe_alternative"],
                "source_ids": [item["source_id"] for item in supporting_sources[:4]],
                "supporting_sources": supporting_sources[:4],
            })
    return {
        "section_number": section_number,
        "section_title": section_title,
        "constraints": constraints,
    }


def merge_legal_constraints(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    section_numbers = []
    section_titles = []
    for item in items:
        if not item:
            continue
        if item.get("section_number"):
            section_numbers.append(str(item.get("section_number")))
        if item.get("section_title"):
            section_titles.append(str(item.get("section_title")))
        for constraint in item.get("constraints", []):
            cid = constraint.get("id")
            if not cid:
                continue
            if cid not in merged:
                merged[cid] = dict(constraint)
                merged[cid]["source_ids"] = []
                merged[cid]["supporting_sources"] = []
            for sid in constraint.get("source_ids", []):
                if sid not in merged[cid]["source_ids"]:
                    merged[cid]["source_ids"].append(sid)
            for source in constraint.get("supporting_sources", []):
                if source.get("source_id") not in {item.get("source_id") for item in merged[cid]["supporting_sources"]}:
                    merged[cid]["supporting_sources"].append(source)
            merged[cid]["source_ids"] = merged[cid]["source_ids"][:4]
            merged[cid]["supporting_sources"] = merged[cid]["supporting_sources"][:4]
    return {
        "section_number": ",".join(section_numbers),
        "section_title": " / ".join(section_titles),
        "constraints": list(merged.values()),
    }


def format_legal_constraints(constraints: dict[str, Any]) -> str:
    items = constraints.get("constraints", [])
    if not items:
        return "Brak dodatkowych ograniczeń wykrytych dla tej sekcji poza zwykłymi wymaganiami i źródłami RAG."
    lines = ["WYKRYTE OGRANICZENIA PRAWNE DLA TEJ SEKCJI:"]
    for index, item in enumerate(items, start=1):
        source_ids = ", ".join(item.get("source_ids", []))
        lines.append(
            f"{index}. ID={item['id']}\n"
            f"   ZAKAZ: {item['prohibited']}\n"
            f"   DOZWOLONE: {item['allowed']}\n"
            f"   BEZPIECZNA WERSJA: {item['safe_alternative']}\n"
            f"   ŹRÓDŁA: {source_ids}"
        )
    return "\n".join(lines)


def constraint_violation(constraint_id: str, text: str) -> bool:
    rule = CONSTRAINT_VIOLATION_RULES.get(constraint_id)
    if not rule:
        return False
    if rule["safe"].search(text):
        return False
    return bool(rule["violation"].search(text))


def constraints_from_rag_debug(rag_debug: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = []
    for key, value in rag_debug.items():
        if not isinstance(key, str) or not key.endswith("_constraints"):
            continue
        if isinstance(value, dict):
            constraints.append(value)
    return constraints
