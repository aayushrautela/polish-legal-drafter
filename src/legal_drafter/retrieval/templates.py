"""Contract templates for the A-RAG ``get_template`` tool.

``get_template_real`` serves the REAL, sourced templates from the
``templates`` Qdrant collection (built from the CC-BY-4.0
``Anyone5559/legal-templates-multilingual`` dataset on Modal): each is a
full, professional Polish template/guide with the correct section
structure, standard clause language and real statutory references. The
agentic generator uses it as the authoritative structural AND substantive
skeleton, adapting the concrete facts to the scenario and to the other REAL
sources it retrieved, and preserving the CC-BY-4.0 attribution.

``TEMPLATES`` is the small synthetic SCAFFOLD (``{{PLACEHOLDERS}}``) kept
ONLY as a fallback when the real collection is unavailable. It is grounded
in standard Polish contract structure (Kodeks cywilny arts.
535/627/659/750/876/98, ustawa o prawach konsumenta, ustawa o
przeciwdziałaniu nieuczciwym praktykom rynkowym) per public legal-template
guidance (poradnikprzedsiebiorcy.pl, gofin.pl, infor.pl, rankomat.pl).
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "umowa_zlecenia": """\
UMOWA ZLECENIA

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} (zwanym dalej "Zlecającym")
a
{{STRONA_B}} (zwanym dalej "Przyjmującym zlecenie").

§ 1. Przedmiot zlecenia
Przyjmujący zlecenie zobowiązuje się do {{OPIS_CZYNNOSCI}} na rzecz Zlecającego.

§ 2. Obowiązki Przyjmującego zlecenie
1. Wykonać zlecenie starannie i w terminie.
2. Informować Zlecającego o przebiegu wykonywania zlecenia.

§ 3. Wynagrodzenie
Za wykonanie zlecenia Przyjmujący otrzymuje wynagrodzenie w kwocie {{KWOTA}} zł, płatne {{TERMIN_PLATNOSCI}}.

§ 4. Termin wykonania
Zlecenie należy wykonać do dnia {{TERMIN_WYKONANIA}}.

§ 5. Odpowiedzialność
Przyjmujący odpowiada za szkodę za nienależyte wykonanie zlecenia, chyba że nienależyte wykonanie było następstwem okoliczności, za które odpowiedzialności nie ponosi.

§ 6. Wypowiedzenie
Każda ze stron może wypowiedzieć umowę ze skutkiem natychmiastowym w przypadku niewykonania lub nienależytego wykonania istotnych obowiązków.

§ 7. Postanowienia końcowe
Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.
ZLECAJĄCY: ________________  PRZYJMUJĄCY ZLECENIE: ________________""",

    "umowa_o_dzielo": """\
UMOWA O DZIEŁO

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} (zwanym dalej "Zamawiającym")
a
{{STRONA_B}} (zwanym dalej "Wykonawcą").

§ 1. Przedmiot umowy
Wykonawca zobowiązuje się do wykonania dzieła: {{OPIS_DZIELA}}, a Zamawiający do zapłaty wynagrodzenia.

§ 2. Termin wykonania i odbiór
1. Dzieło należy wykonać do dnia {{TERMIN_WYKONANIA}}.
2. Odbiór następuje po zgłoszeniu gotowości; Zamawiający ma {{TERMIN_NA_POPRAWKI}} na zgłoszenie wad.

§ 3. Wynagrodzenie
Wynagrodzenie w kwocie {{KWOTA}} zł, ustalone {{RYCZALT_LUB_KOSZTORYS}}, płatne {{TERMIN_PLATNOSCI}}.

§ 4. Prawa autorskie
{{POSTANOWIENIA_PRAW_AUTORSKICH}}.

§ 5. Kary umowne
Za opóźnienie w wykonaniu dzieła Wykonawca zapłaci karę umowną w wysokości {{WYSOKOSC_KARY}}.

§ 6. Postanowienia końcowe
Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.
ZAMAWIAJĄCY: ________________  WYKONAWCA: ________________""",

    "umowa_najmu": """\
UMOWA NAJMU LOKALU MIESZKALNEGO

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} (zwaną dalej "Wynajmującym")
a
{{STRONA_B}} (zwaną dalej "Najemcą").

§ 1. Oświadczenia stron
1. Wynajmujący oświadcza, że jest właścicielem lokalu {{OPIS_LOKALU}} (księga wieczysta {{KS_WIECZYSTA}}).
2. Najemca oświadcza, że lokal będzie wykorzystywany wyłącznie na cele mieszkaniowe.

§ 2. Przedmiot umowy
Wynajmujący oddaje Najemcy lokal wraz z wyposażeniem, a Najemca zobowiązuje się płacić czynsz.

§ 3. Czynsz i opłaty eksploatacyjne
1. Czynsz miesięczny {{KWOTA_CZYNSZU}} zł, płatny {{TERMIN_PLATNOSCI}}.
2. Najemca pokrywa opłaty za {{OPLATY_MEDIA}}.

§ 4. Prawa i obowiązki stron
1. Najemca nie może oddać lokalu w podnajem bez zgody Wynajmującego.
2. Zmiany w lokalu wymagają zgody pisemnej Wynajmującego.

§ 5. Czas trwania umowy
Umowa zawarta na czas {{OKRES}}; wypowiedzenie {{ZASADY_WYPOWIEDZENIA}}.

§ 6. Kaucja
Najemca wpłaca kaucję w kwocie {{KWOTA_KAUCJI}} zł, zwracaną po rozliczeniu.

§ 7. Wydanie i zwrot lokalu
Protokół zdawczo-odbiorczy stanowi Załącznik nr 1.

§ 8. Postanowienia końcowe
Umowę sporządzono w dwóch egzemplarzach.
WYNAJMUJĄCY: ________________  NAJEMCA: ________________
Załączniki: 1. Protokół zdawczo-odbiorczy.""",

    "umowa_sprzedazy": """\
UMOWA SPRZEDAŻY

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} (zwaną dalej "Sprzedającym")
a
{{STRONA_B}} (zwaną dalej "Kupującym").

§ 1. Przedmiot sprzedaży
Sprzedający sprzedaje, a Kupujący kupuje: {{DOKLADNY_OPIS_RZECZY}} (nr {{NUMERY_IDENTYFIKACYJNE}}).

§ 2. Stan prawny i fizyczny
Przedmiot jest wolny od obciążeń; Sprzedający oświadcza, że rzecz jest sprawna.

§ 3. Cena
Cena sprzedaży wynosi {{KWOTA}} zł (słownie: {{KWOTA_SLOWNIE}}).

§ 4. Zapłata i wydanie
Zapłata {{SPOSOB_I_TERMIN_ZAPLATY}}; wydanie rzeczy {{TERMIN_WYDANIA}}.

§ 5. Rękojmia
Sprzedający odpowiada z tytułu rękojmi za wady fizyczne i prawne na zasadach Kodeksu cywilnego.

§ 6. Zmiany umowy
Wszelkie zmiany wymagają formy pisemnej pod rygorem nieważności.

§ 7. Postanowienia końcowe
Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.
SPRZEDAJĄCY: ________________  KUPUJĄCY: ________________""",

    "kaucja_zaliczka": """\
OŚWIADCZENIE O KAUCJI / ZALICZCE

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} a {{STRONA_B}}.

§ 1. Przedmiot transakcji
Kaucja / zaliczka dotyczy {{OPIS_TRANSAKCJI}}.

§ 2. Kwota
Strony ustalają kaucję / zaliczkę w kwocie {{KWOTA}} zł.

§ 3. Zwrot i rozliczenie
1. Zwrot następuje {{ZASADY_ZWROTU}}.
2. Zaliczka zalicza się na poczet ceny przy zawarciu umowy głównej.

§ 4. Skutki niezawarcia umowy głównej
{{SKUTKI_BRAKU_UMOWY_GLOWNEJ}}.

STRONA A: ________________  STRONA B: ________________""",

    "odstapienie_konsumenta": """\
OŚWIADCZENIE O ODSTĄPIENIU OD UMOWY

{{MIEJSCOWOSC}}, {{DATA}}

Dane konsumenta: {{DANE_KONSUMENTA}}
Dane przedsiębiorcy: {{DANE_PRZEDSIEBIORCY}}

Oświadczam, że zgodnie z art. 27 ustawy o prawach konsumenta odstępuję od umowy zawartej w dniu {{DATA_ZAWARCIA_UMOWY}}, przedmiot: {{PRZEDMIOT_UMOWY}}.

Żądam zwrotu wszystkich poniesionych płatności, w tym kosztów dostawy, w terminie ustawowym.

............................................
(czytelny podpis konsumenta)""",

    "wezwanie_do_zaplaty": """\
WEZWANIE DO ZAPŁATY

{{MIEJSCOWOSC}}, {{DATA}}

Wzywam {{DANE_DLUZNIKA}} do zapłaty kwoty {{KWOTA}} zł z tytułu {{PODSTAWA_ZOBOWIAZANIA}} ({{NR_FAKTURY_UMOWY}}) w terminie {{TERMIN}} od dnia otrzymania wezwania.

W przypadku braku zapłaty naliczę odsetki ustawowe za opóźnienie w transakcjach handlowych / odsetki ustawowe za opóźnienie w spełnieniu świadczenia pieniężnego i skieruję sprawę na drogę postępowania sądowego / przed sąd polubowny.

To wezwanie jest ostateczne.

............................................
(czytelny podpis wierzyciela)""",

    "reklamacja_konsumenta": """\
REKLAMACJA TOWARU

{{MIEJSCOWOSC}}, {{DATA}}

Dane konsumenta: {{DANE_KONSUMENTA}}
Dane sprzedawcy: {{DANE_SPRZEDAWCY}}

Przedmiot reklamacji: {{OPIS_TOWARU}}, nabyty w dniu {{DATA_ZAKUPU}} (nr paragonu/faktury {{NR_DOKUMENTU}}).

Wada: {{OPIS_WADY}}.

Na podstawie przepisów o rękojmi za wady rzeczy sprzedanej żądam: {{WYBOR_ROZWIAZANIA - naprawa / wymiana / obniżenie ceny / odstąpienie od umowy}}.

Oczekuję rozpatrzenia reklamacji w ustawowym terminie.

............................................
(czytelny podpis konsumenta)""",

    "klauzula_niedozwolona": """\
OCENA KLAUZULI UMOWNEJ POD KĄTEM NIEDOZWOLONYCH POSTANOWIEŃ

§ 1. Strony / kontekst
{{STRONY_I_KONTEKST}}.

§ 2. Przedmiotowa klauzula
Treść klauzuli: "{{TRESC_KLAUZULI}}".

§ 3. Ocena
1. Czy klauzula kształtuje prawa i obowiązki konsumenta w sposób sprzeczny z dobrymi obyczajami i rażąco naruszający jego interesy (art. 385[1] Kodeksu cywilnego)?
2. Odniesienie do wzorców umów uznanych za niedozwolone (załącznik do ustawy o przeciwdziałaniu nieuczciwym praktykom rynkowym / rejestr UOKiK).

§ 4. Propozycja klauzuli zgodnej
{{PROPOZYCJA_KLAUZULI}}.

............................................
(podpis)""",

    "kara_umowna": """\
ZAST RZEŻENIE KARY UMOWNEJ

zawarta w dniu {{DATA}} r. w {{MIEJSCOWOSC}} pomiędzy:
{{STRONA_A}} a {{STRONA_B}}.

§ 1. Zobowiązanie główne
Strony łączy zobowiązanie {{OPIS_ZOBOWIAZANIA_GLOWNEGO}}.

§ 2. Zastrzeżenie kary umownej
Za niewykonanie lub nienależyte wykonanie zobowiązania Strona zapłaci karę umowną w wysokości {{WYSOKOSC_KARY}} w przypadku {{PRZESLANKI}}.

§ 3. Sposób naliczania
{{SPOSOB_NALICZANIA}}.

§ 4. Ograniczenia
Zastrzeżenie kary umownej nie może być sprzeczne z art. 483-485 Kodeksu cywilnego (m.in. niedopuszczalność kary za opóźnienie w zapłacie ceny).

§ 5. Postanowienia końcowe
STRONA A: ________________  STRONA B: ________________""",

    "pelnomocnictwo": """\
PEŁNOMOCNICTWO

{{MIEJSCOWOSC}}, {{DATA}}

Ja, niżej podpisany(a) {{DANE_MOCODAWCY}}, ustanawiam pełnomocnikiem {{DANE_PELNOMOCNIKA}} do dokonywania w moim imieniu następujących czynności: {{ZAKRES_UMOCOWANIA}}.

Pełnomocnictwo jest ważne do dnia {{TERMIN_WAZNOSCI}}.

............................................
(czytelny podpis mocodawcy)
{{POŚWIADCZENIE_NOTARIALNE}}""",

    "poreczenie": """\
OŚWIADCZENIE PORĘCZYCIELA

{{MIEJSCOWOSC}}, {{DATA}}

Wierzyciel: {{DANE_WIERZYCIELA}}
Dłużnik główny: {{DANE_DLUZNIKA}}

Oświadczam, że poręczam za zapłatę zobowiązania {{OPIS_ZOBOWIAZANIA_GLOWNEGO}} do kwoty {{SUMA_PORECZENIA}} zł.

§ 1. Zakres odpowiedzialności
Odpowiadam solidarnie z dłużnikiem do wysokości sumy poręczenia.

§ 2. Termin
Poręczenie obowiązuje do dnia {{TERMIN}}.

............................................
(czytelny podpis poręczyciela)""",

    "other": """\
DOKUMENT PRAWNO-ROZPOZNAWCZY

{{MIEJSCOWOSC}}, {{DATA}}

Strony: {{STRONA_A}} / {{STRONA_B}}.

§ 1. Przedmiot / stan faktyczny
{{OPIS_STANU_FAKTYCZNEGO}}.

§ 2. Podstawa prawna
{{PODSTAWA_PRAWNA}}.

§ 3. Rozstrzygnięcie / żądania
{{ROZSTRZYGNIECIE_LUB_ZADANIA}}.

§ 4. Podpisy
STRONA A: ________________  STRONA B: ________________""",
}


def get_template_text(doc_type: str) -> str:
    """Return the structural template for ``doc_type`` (falls back to 'other')."""
    return TEMPLATES.get(doc_type) or TEMPLATES["other"]


# Mapping from this project's ``doc_type`` slugs onto a Polish natural-language
# query used to retrieve the best matching real template from the ``templates``
# Qdrant collection (built from the CC-BY-4.0 legal-templates-multilingual set).
# For any doc_type not listed here, the slug is used verbatim (underscores -> spaces).
DOC_TYPE_QUERY: dict[str, str] = {
    "umowa_najmu": "umowa najmu lokalu mieszkalnego",
    "umowa_o_dzielo": "umowa o dzieło przeniesienie praw autorskich",
    "umowa_zlecenia": "umowa zlecenia wynagrodzenie",
    "umowa_sprzedazy": "umowa sprzedaży rzeczy ruchomej",
    "kaucja_zaliczka": "oświadczenie o kaucji lub zaliczce",
    "odstapienie_konsumenta": "odstąpienie od umowy przez konsumenta",
    "wezwanie_do_zaplaty": "wezwanie do zapłaty dłużnika",
    "reklamacja_konsumenta": "reklamacja towaru przez konsumenta",
    "klauzula_niedozwolona": "ocena klauzuli umownej pod kątem niedozwolonych postanowień",
    "kara_umowna": "zastrzeżenie kary umownej",
    "pelnomocnictwo": "pełnomocnictwo do czynności",
    "poreczenie": "oświadczenie poręczyciela poręczenie",
}

# Ordered fields that make up the real template body (per the dataset schema).
TEMPLATE_CONTENT_FIELDS = [
    "what_is",
    "when_needed",
    "key_elements",
    "how_to_fill",
    "legal_requirements",
    "common_mistakes",
]


def build_template_text(row: dict) -> str:
    """Concatenate the substantive guidance fields of a dataset row into one body."""
    parts = []
    for field in TEMPLATE_CONTENT_FIELDS:
        value = row.get(field)
        if value and str(value).strip():
            parts.append(str(value).strip())
    return "\n\n".join(parts)


def format_template_payload(payload: dict) -> str:
    """Render a retrieved templates-collection payload as a usable template string."""
    title = payload.get("title") or ""
    body = payload.get("text") or ""
    source_url = payload.get("source_url") or ""
    out: list[str] = []
    if title:
        out.append(f"# {title}")
    if body:
        out.append(body)
    if source_url:
        out.append(f"\nŹródło: {source_url} (licencja CC-BY-4.0)")
    return "\n".join(out).strip()


def get_template_real(store, doc_type: str) -> str | None:
    """Retrieve the real structured template for ``doc_type`` from a Qdrant store.

    Returns the formatted template string, or ``None`` if no match is found
    (caller should fall back to the synthetic scaffold).
    """
    dt = (doc_type or "other").strip().lower() or "other"
    if dt == "other":
        return None
    query = DOC_TYPE_QUERY.get(dt) or dt.replace("_", " ")
    try:
        hits = store.recall(query, top_k=1, candidate_k=10)
    except Exception:
        return None
    if not hits:
        return None
    return format_template_payload(hits[0][1])


def template_result(doc_type: str, text: str) -> list[dict]:
    """Uniform tool result shape for both local and remote ``get_template``."""
    return [
        {
            "doc_type": (doc_type or "other").strip().lower() or "other",
            "kind": "structural_template",
            "text": text,
        }
    ]
