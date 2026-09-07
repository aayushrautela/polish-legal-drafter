"""Contract templates for the A-RAG ``get_template`` tool.

``TEMPLATES`` is a hand-curated collection of 27 real Polish contract
templates — full professional prose with correct section structure,
standard clause language and real statutory references (Kodeks cywilny,
Kodeks pracy, ustawa o prawach konsumenta, ustawa o prawie autorskim,
ustawa o zwalczaniu nieuczciwej konkurencji, Prawo bankowe, etc.).

Each template is a complete, ready-to-use contract skeleton with:
- proper party identification blocks,
- numbered paragraphs (§) with substantive clauses,
- blank lines (________________) for filling in concrete facts,
- statutory references (art. XX KC, art. YY KP, etc.),
- standard boilerplate (confidentiality, penalties, termination, governing law).

The ``other`` entry is the only synthetic scaffold (``{{PLACEHOLDERS}}``)
kept as a generic fallback for unrecognized doc_types.

Serving is ``TEMPLATES``-only: ``get_template_text`` resolves alias, then
exact key, then fuzzy match, else the ``other`` scaffold. (An earlier
iteration served a CC-BY templates vector collection via ``get_template_real``;
that path is superseded — no live caller remains — and its helpers stay in
this module only so the archived Modal build script still imports.)

Sources: poradnikprzedsiebiorcy.pl, bezpodatku.pl, porady.pl, zus.info.pl,
fakturtax.pl, spolkajawnablog.pl, rcponline.pl, nieruchomosci-online.pl,
dobrykalkulator.pl, pracuj.pl, and standard Polish legal practice.
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "umowa_zlecenia": """\
UMOWA ZLECENIA nr ____/____

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

ZLECENIODAWCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
________________________________________________________________________________
PESEL / NIP: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

ZLECENIOBIORCĄ:
________________________________________________________________________________
(imię i nazwisko), zamieszkałym w ______________________________________________
PESEL: ____________________________, NIP (jeśli dotyczy): ______________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot umowy

1. Zleceniodawca zleca, a Zleceniobiorca zobowiązuje się do starannego wykonania następujących czynności:
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
(dalej jako „Zlecenie").

2. Zleceniobiorca oświadcza, że posiada niezbędną wiedzę, kwalifikacje i doświadczenie do należytego wykonania Zlecenia.

3. Zleceniobiorca zobowiązuje się wykonać Zlecenie osobiście, bez powierzania go osobie trzeciej, chyba że Zleceniodawca wyrazi na to pisemną zgodę (art. 738 KC).

§ 2. Czas trwania umowy

Niniejsza umowa zostaje zawarta na czas:
- określony od dnia ______________ do dnia ______________, /
- nieokreślony, począwszy od dnia ______________.

§ 3. Wynagrodzenie

1. Za należyte wykonanie Zlecenia Zleceniobiorcy przysługuje wynagrodzenie w wysokości:
- ______________ zł (słownie: ________________________________ złotych) brutto za każdą godzinę faktycznie wykonanego Zlecenia, /
- ______________ zł (słownie: ________________________________ złotych) brutto (wynagrodzenie ryczałtowe).

2. Wynagrodzenie płatne jest przelewem na rachunek bankowy Zleceniobiorcy nr ______________
w terminie ______ dni od dnia przedłożenia rachunku przez Zleceniobiorcę.

3. Strony zgodnie postanawiają, że wynagrodzenie obejmuje / nie obejmuje zwrotu wydatków poniesionych przez Zleceniobiorcę w związku z wykonaniem Zlecenia.

§ 4. Ewidencja czasu pracy

1. Zleceniobiorca zobowiązuje się do prowadzenia ewidencji liczby godzin poświęconych na wykonanie Zlecenia i przedstawiania jej Zleceniodawcy w formie miesięcznego zestawienia do 5. dnia każdego miesiąca za miesiąc poprzedni.

2. Zatwierdzona przez Zleceniodawcę ewidencja liczby godzin stanowi podstawę do naliczenia wynagrodzenia, o którym mowa w § 3 ust. 1.

§ 5. Obowiązki Zleceniobiorcy

1. Zleceniobiorca zobowiązuje się wykonać Zlecenie z należytą starannością, zgodnie z obowiązującymi przepisami prawa, postanowieniami niniejszej Umowy oraz wskazówkami Zleceniodawcy, o ile nie są one sprzeczne z prawem lub Umową (art. 740 KC).

2. Zleceniobiorca zobowiązuje się do bieżącego informowania Zleceniodawcy o postępach w realizacji Zlecenia oraz o wszelkich okolicznościach mogących mieć wpływ na jego wykonanie.

3. Po wykonaniu Zlecenia lub po rozwiązaniu Umowy, Zleceniobiorca zobowiązuje się złożyć Zleceniodawcy sprawozdanie z wykonanych czynności oraz wydać wszystko, co przy wykonywaniu Zlecenia dla niego uzyskał (art. 740 KC).

4. Zleceniobiorca zobowiązuje się do zachowania w tajemnicy wszelkich informacji uzyskanych w związku z wykonywaniem Zlecenia.

§ 6. Obowiązki Zleceniodawcy

1. Zleceniodawca zobowiązuje się do współdziałania ze Zleceniobiorcą w zakresie niezbędnym do prawidłowego wykonania Zlecenia, w szczególności do udzielania potrzebnych informacji i materiałów.

2. Zleceniodawca zobowiązuje się do zwrotu Zleceniobiorcy uzasadnionych wydatków, które ten poczynił w celu należytego wykonania Zlecenia, po ich uprzednim uzgodnieniu i udokumentowaniu (art. 742 KC).

3. Zleceniodawca zobowiązuje się do zapłaty wynagrodzenia w terminie i na zasadach określonych w § 3 Umowy.

§ 7. Przeniesienie praw autorskich (opcjonalnie)

1. W przypadku, gdy w wyniku wykonania Zlecenia powstanie utwór w rozumieniu ustawy z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych, Zleceniobiorca przenosi na Zleceniodawcę autorskie prawa majątkowe do utworu na następujących polach eksploatacji: ________________________________________________________________________________

2. Przeniesienie praw następuje z chwilą przyjęcia utworu przez Zleceniodawcę i zapłaty całości wynagrodzenia.

§ 8. Klauzula poufności

1. Zleceniobiorca zobowiązuje się do zachowania w tajemnicy wszelkich informacji dotyczących Zleceniodawcy, jego działalności, klientów, kontrahentów oraz danych technicznych i handlowych, do których uzyska dostęp w związku z wykonywaniem Zlecenia.

2. Obowiązek poufności obowiązuje przez okres trwania Umowy oraz przez 2 (dwa) lata po jej rozwiązaniu lub wygaśnięciu.

3. Naruszenie klauzuli poufności uprawnia Zleceniodawcę do żądania od Zleceniobiorcy zapłaty kary umownej w wysokości ______________ zł (słownie: ________________________________ złotych) za każde naruszenie, bez uszczerbku dla prawa Zleceniodawcy do żądania odszkodowania przenoszącego wysokość kary na zasadach ogólnych.

§ 9. Kary umowne

1. W przypadku nieterminowego wykonania Zlecenia z winy Zleceniobiorcy, Zleceniodawca ma prawo naliczyć karę umowną w wysokości ______________ zł (słownie: ________________________________ złotych) za każdy dzień opóźnienia, nie więcej niż ______________ zł (łącznie).

2. W przypadku niewykonania lub nienależytego wykonania Zlecenia z winy Zleceniobiorcy, Zleceniodawca ma prawo odstąpić od Umowy i żądać odszkodowania na zasadach ogólnych.

§ 10. Wypowiedzenie umowy

1. Każda ze stron może wypowiedzieć Umowę w każdym czasie z zachowaniem 14-dniowego okresu wypowiedzenia (art. 746 KC).

2. Wypowiedzenie Umowy wymaga formy pisemnej pod rygorem nieważności.

3. W przypadku rażącego naruszenia postanowień Umowy przez jedną ze stron, druga strona może wypowiedzieć Umowę ze skutkiem natychmiastowym.

§ 11. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego, w szczególności art. 734-751 KC.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Zleceniodawcy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach, po jednym dla każdej ze stron.


ZLECENIODAWCA: ____________________________    ZLECENIOBIORCA: ____________________________""",

    "umowa_o_dzielo": """\
UMOWA O DZIEŁO nr ____/____

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

ZAMAWIAJĄCYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
________________________________________________________________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

WYKONAWCĄ:
________________________________________________________________________________
(imię i nazwisko), zamieszkałym w ______________________________________________
PESEL: ____________________________, NIP (jeśli dotyczy): ______________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot umowy

1. Zamawiający zleca, a Wykonawca zobowiązuje się do wykonania dzieła polegającego na:
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
(dalej jako „Dzieło").

2. Dzieło zostanie wykonane zgodnie ze specyfikacją techniczną stanowiącą Załącznik nr 1 do niniejszej Umowy.

3. Wykonawca oświadcza, że posiada niezbędne kwalifikacje, doświadczenie i zasoby do należytego wykonania Dzieła.

4. Wykonawca zobowiązuje się wykonać Dzieło z należytą starannością i zgodnie z zasadami wiedzy technicznej (art. 627 KC).

§ 2. Materiały i narzędzia

1. Materiały niezbędne do wykonania Dzieła zapewnia: Wykonawca / Zamawiający / Strony łącznie.

2. Wykonawca oświadcza, że materiały dostarczone przez Zamawiającego są odpowiednie do wykonania Dzieła.

3. Ryzyko przypadkowej utraty lub zniszczenia Dzieła do momentu odbioru ponosi: Wykonawca / Zamawiający (art. 633-634 KC).

§ 3. Termin wykonania i odbiór

1. Wykonawca zobowiązuje się wykonać Dzieło i zgłosić je do odbioru do dnia ________________ r.

2. O gotowości Dzieła do odbioru Wykonawca poinformuje Zamawiającego pisemnie lub drogą elektroniczną.

3. Zamawiający dokona odbioru Dzieła w terminie 14 dni od daty poinformowania o jego gotowości.

4. Odbiór Dzieła nastąpi na podstawie protokołu odbioru podpisanego przez obie Strony (Załącznik nr 2).

5. W przypadku stwierdzenia wad w Dziele, Zamawiający ma prawo odmówić odbioru i wezwać Wykonawcę do ich usunięcia w wyznaczonym terminie (art. 636-637 KC).

6. Zamawiający zgłosi ewentualne wady w terminie 14 dni od przekazania Dzieła; Wykonawca usunie je w terminie 14 dni.

§ 4. Wynagrodzenie

1. Za wykonanie Dzieła Wykonawcy przysługuje wynagrodzenie w kwocie ______________ zł brutto (słownie: ________________________________ złotych).

2. Wynagrodzenie płatne jest przelewem na rachunek bankowy Wykonawcy nr ______________
w terminie 14 dni od dnia podpisania protokołu odbioru i wystawienia rachunku przez Wykonawcę.

3. Do wynagrodzenia stosuje się koszty uzyskania przychodu w wysokości 20% (dzieło niebędące utworem) / 50% (dzieło będące utworem z prawami autorskimi).

4. Podwyższenie wynagrodzenia wymaga pisemnego aneksu pod rygorem nieważności (art. 632 § 2 KC).

§ 5. Prawa autorskie

1. Z chwilą zapłaty pełnego wynagrodzenia Wykonawca przenosi na Zamawiającego autorskie prawa majątkowe do Dzieła na następujących polach eksploatacji:
________________________________________________________________________________
________________________________________________________________________________
(art. 41, 50 ustawy o prawie autorskim i prawach pokrewnych).

2. Przeniesienie praw nie jest ograniczone terytorialnie ani czasowo.

3. Wykonawca zachowuje autorskie prawa osobiste do Dzieła (art. 16 pr.aut.).

4. Wynagrodzenie z § 4 obejmuje wynagrodzenie za przeniesienie praw autorskich.

§ 6. Rękojmia i gwarancja

1. Wykonawca ponosi odpowiedzialność za wady Dzieła na zasadach Kodeksu cywilnego (art. 638 KC w zw. z rękojmią za wady — art. 556 KC).

2. Okres rękojmi wynosi 12 miesięcy od dnia odbioru Dzieła.

3. Wykonawca udziela gwarancji na Dzieło na okres ______ miesięcy od dnia odbioru, w zakresie: ________________________________________________________________________________

§ 7. Kary umowne

1. W przypadku nieterminowego wykonania Dzieła z winy Wykonawcy, Zamawiający ma prawo naliczyć karę umowną w wysokości ______________ zł (słownie: ________________________________ złotych) za każdy dzień zwłoki, nie więcej niż ______________ zł (łącznie).

2. W przypadku naruszenia przez Wykonawcę postanowień § 5 ust. 4, Zamawiający ma prawo dochodzić odszkodowania na zasadach ogólnych.

3. W przypadku niewykonania lub nienależytego wykonania Dzieła z winy Wykonawcy, Zamawiający ma prawo odstąpić od Umowy i żądać odszkodowania.

§ 8. Odstąpienie od umowy

1. Zamawiający może odstąpić od Umowy w przypadku niewykonania Dzieła w terminie z § 3 ust. 1 (art. 635 KC).

2. Zamawiający może odstąpić od Umowy w przypadku, gdy Wykonawca wykonuje Dzieło z wadami lub odmawia usunięcia wad (art. 640 KC).

3. Odstąpienie wymaga formy pisemnej.

§ 9. Klauzula poufności

1. Wykonawca zobowiązuje się do zachowania w tajemnicy wszelkich informacji dotyczących Zamawiającego, jego działalności, klientów oraz danych technicznych i handlowych, do których uzyska dostęp w związku z wykonaniem Dzieła.

2. Obowiązek poufności obowiązuje przez okres trwania Umowy oraz przez 2 (dwa) lata po jej rozwiązaniu lub wygaśnięciu.

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego, w szczególności art. 627-646 KC.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Zamawiającego.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach, po jednym dla każdej ze Stron.

Załączniki: 1. Specyfikacja techniczna Dzieła, 2. Protokół odbioru.


ZAMAWIAJĄCY: ____________________________    WYKONAWCA: ____________________________""",

    "umowa_najmu": """\
UMOWA NAJMU LOKALU MIESZKALNEGO

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

WYNAJMUJĄCYM:
________________________________________________________________________________
(imię i nazwisko), zamieszkałym w ______________________________________________
PESEL: ____________________________, seria i nr dowodu tożsamości: ______________

a

NAJEMCĄ:
________________________________________________________________________________
(imię i nazwisko), zamieszkałym w ______________________________________________
PESEL: ____________________________, seria i nr dowodu tożsamości: ______________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Oświadczenia Wynajmującego

1. Wynajmujący oświadcza, że jest właścicielem stanowiącej odrębną nieruchomość lokalu mieszkalnego nr ______________, znajdującego się w budynku wielolokalowym położonym w ________________________________, przy ul. ________________________________ (zwanego dalej „Lokalem mieszkalnym"), dla którego Sąd Rejonowy ________________________________, Wydział Ksiąg Wieczystych prowadzi księgę wieczystą o numerze ________________.

2. Lokal mieszkalny znajduje się na ______ kondygnacji w budynku, składa się z ______ pomieszczeń: ________________________________, jego powierzchnia wynosi ______ m².

3. Lokal mieszkalny wolny jest od obciążeń na rzecz osób trzecich, które mogłyby uniemożliwić lub utrudnić Najemcy wykonywanie uprawnień wynikających z Umowy.

4. Wykaz elementów wyposażenia Lokalu mieszkalnego stanowi Załącznik nr 1 do Umowy.

§ 2. Oświadczenia Najemcy

1. Najemca oświadcza, że Lokal mieszkalny obejrzał i nie wnosi zastrzeżeń co do stanu technicznego Lokalu mieszkalnego ani Wyposażenia.

2. Najemca oświadcza, że Lokal mieszkalny będzie wykorzystywany wyłącznie w celu zaspokajania potrzeb mieszkaniowych Najemcy oraz następujących osób:
________________________________________________________________________________
(łącznie ______ osób).

§ 3. Przedmiot umowy

Wynajmujący oddaje Najemcy Lokal mieszkalny wraz z Wyposażeniem w celu zaspokajania potrzeb mieszkaniowych Najemców, zaś Najemcy zobowiązują się płacić Wynajmującemu umówiony czynsz (art. 659 § 1 KC).

§ 4. Czynsz najmu oraz opłaty eksploatacyjne

1. Z tytułu najmu Najemcy zobowiązani są solidarnie do zapłaty na rzecz Wynajmującego czynszu w kwocie ______________ zł (słownie: ________________________________ złotych) miesięcznie.

2. Czynsz najmu płatny jest z góry do ______ dnia każdego miesiąca kalendarzowego przelewem na rachunek bankowy Wynajmującego o numerze: ________________________________.

3. Niezależnie od czynszu Najemcy pokrywają opłaty za: energię elektryczną, gaz, wodę, ogrzewanie, wywóz odpadów, zarządcę budynku, fundusz remontowy, według wskazań liczników / ryczałtu.

4. Czynsz może być waloryzowany raz w roku o wskaźnik inflacji ogłaszany przez GUS, nie więcej niż o 10% rocznie.

§ 5. Sposób korzystania z Lokalu i Wyposażenia

1. Najemcy uprawnieni są do używania Lokalu i Wyposażenia wyłącznie w celu określonym w § 3 Umowy, w szczególności w Lokalu mieszkalnym nie mogą prowadzić działalności gospodarczej.

2. Najemcom zabrania się:
   a) palenia oraz umożliwiania innym osobom palenia wyrobów tytoniowych (w tym e-papierosów) w Lokalu mieszkalnym,
   b) utrzymywania w Lokalu mieszkalnym jakichkolwiek zwierząt,
   c) używania Lokalu mieszkalnego w sposób sprzeczny z zasadami współżycia społecznego,
   d) używania elementów Wyposażenia w sposób sprzeczny z ich przeznaczeniem.

3. Najemcy nie mogą oddać Lokalu w podnajem ani do bezpłatnego używania bez pisemnej zgody Wynajmującego (art. 668 KC).

4. Zmiany adaptacyjne w Lokalu wymagają pisemnej zgody Wynajmującego.

§ 6. Prawa i obowiązki stron

1. Drobne nakłady związane ze zwykłym użytkowaniem Lokalu ponosi Najemca (art. 681 KC).

2. Nakłady na utrzymanie Lokalu w stanie nadającym się do używania ponosi Wynajmujący.

3. Wynajmujący zobowiązuje się do zapewnienia sprawnego działania instalacji i urządzeń związanych z budynkiem.

4. Najemca zobowiązuje się do utrzymywania wynajętego Lokalu we właściwym stanie technicznym i dokonywania drobnych napraw związanych z bieżącym użytkowaniem.

§ 7. Kaucja

1. W dniu zawarcia Umowy Najemcy wpłacili Wynajmującemu gotówką / przelewem kaucję w kwocie ______________ zł (słownie: ________________________________ złotych), równą ______-miesięcznemu czynszowi (art. 6 ustawy o ochronie praw lokatorów).

2. Kaucja zabezpiecza roszczenia Wynajmującego z tytułu niewykonania lub nienależytego wykonania zobowiązań Najemcy, w szczególności z tytułu niezapłaconego czynszu i uszkodzeń Lokalu.

3. Zwrot kaucji nastąpi w terminie 1 miesiąca od dnia opróżnienia Lokalu przez Najemcę, po potrąceniu należności Wynajmującego (art. 6 ust. 4 ustawy o ochronie praw lokatorów — termin ustawowy, nie może być wydłużony umową). Kaucja nie może przekraczać dwunastokrotności miesięcznego czynszu (sześciokrotności przy najmie okazjonalnym).

§ 8. Okres obowiązywania Umowy i wypowiedzenie

1. Umowa zostaje zawarta na czas nieoznaczony / od dnia ________________ do dnia ________________.

2. Najemca może wypowiedzieć Umowę z zachowaniem 3-miesięcznego okresu wypowiedzenia ze skutkiem na koniec miesiąca kalendarzowego (art. 688 KC).

3. Wynajmujący uprawniony jest do wypowiedzenia Umowy z zachowaniem 3-miesięcznego okresu wypowiedzenia w przypadkach:
   a) Najemca używa Lokalu w sposób sprzeczny z umową lub niezgodnie z jego przeznaczeniem,
   b) Najemca jest w zwłoce z zapłatą czynszu lub innych opłat za używanie Lokalu mieszkalnego co najmniej za trzy pełne okresy płatności pomimo uprzedzenia,
   c) Najemca dopuszcza się rażącego naruszenia porządku domowego.

4. Wypowiedzenie wymaga formy pisemnej.

§ 9. Zwrot Lokalu

1. W terminie 7 dni od dnia ustania najmu Najemcy zwrócą Wynajmującemu Lokal mieszkalny wraz z Wyposażeniem w stanie niepogorszonym, z wyłączeniem normalnego zużycia.

2. Zwrot Lokalu zostanie potwierdzony protokołem zdawczo-odbiorczym.

3. W przypadku opóźnienia w zwrocie Lokalu, Najemca zapłaci Wynajmującemu karę umowną w wysokości ______________ zł za każdy dzień opóźnienia.

§ 10. Postanowienia końcowe

1. Wszelkie zmiany Umowy wymagają zachowania formy pisemnej pod rygorem nieważności.

2. W sprawach nieuregulowanych niniejszą umową zastosowanie mają przepisy Kodeksu cywilnego (art. 659-692 KC) oraz ustawy z dnia 21 czerwca 2001 r. o ochronie praw lokatorów.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla miejsca położenia Lokalu mieszkalnego.

4. Umowa została sporządzona w dwóch jednobrzmiących egzemplarzach, po jednym dla każdej ze Stron.

Załączniki: 1. Wykaz elementów wyposażenia Lokalu mieszkalnego, 2. Wzór protokołu zdawczo-odbiorczego.


WYNAJMUJĄCY: ____________________________    NAJEMCY: ____________________________""",

    "umowa_sprzedazy": """\
UMOWA SPRZEDAŻY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

SPRZEDAJĄCYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
________________________________________________________________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

KUPUJĄCYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
________________________________________________________________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot sprzedaży

1. Sprzedający sprzedaje, a Kupujący kupuje:
________________________________________________________________________________
(dokładny opis przedmiotu, w tym marka, model, numer seryjny/VIN, rok produkcji, cechy identyfikujące)
________________________________________________________________________________
zwanym dalej „Przedmiotem" (art. 535 KC).

2. Sprzedający oświadcza, że Przedmiot stanowi jego wyłączną własność, jest wolny od wad prawnych i obciążeń na rzecz osób trzecich oraz nie toczy się wobec niego postępowanie egzekucyjne.

3. Kupujący oświadcza, że zbadał stan Przedmiotu i nie wnosi zastrzeżeń co do jego stanu fizycznego i prawnego.

§ 2. Cena i sposób zapłaty

1. Cena sprzedaży Przedmiotu wynosi ______________ zł (słownie: ________________________________ złotych) brutto / netto.

2. Cena obejmuje / nie obejmuje podatek VAT. Stawka VAT: ______%.

3. Zapłata ceny nastąpi:
   a) w całości w dniu zawarcia Umowy przelewem na rachunek Sprzedającego nr ________________________________, /
   b) w częściach: zaliczka ______________ zł w dniu zawarcia Umowy, pozostała kwota ______________ zł w terminie ________________, /
   c) gotówką w dniu zawarcia Umowy za pokwitowaniem.

4. Sprzedający potwierdza otrzymanie ceny w całości / zaliczki.

§ 3. Wydanie Przedmiotu

1. Wydanie Przedmiotu nastąpi w dniu ________________ r. w miejscu: ________________________________.

2. Wydanie Przedmiotu nastąpi wraz z następującymi dokumentami:
________________________________________________________________________________
(akt własności, dowód rejestracyjny, karta gwarancyjna, instrukcje, faktura, itp.)

3. Przejście własności Przedmiotu oraz ryzyka przypadkowej utraty lub uszkodzenia nastąpi z chwilą wydania Przedmiotu Kupującemu (art. 548-5481 KC).

4. Koszty związane z wydaniem i transportem Przedmiotu ponosi: Sprzedający / Kupujący / Strony po połowie.

§ 4. Rękojmia

1. Sprzedający odpowiada z tytułu rękojmi za wady fizyczne i prawne Przedmiotu na zasadach art. 556-576 KC.

2. W przypadku stwierdzenia wad, Kupujący ma prawo:
   a) żądać wymiany Przedmiotu na wolny od wad lub usunięcia wady,
   b) żądać obniżenia ceny,
   c) odstąpić od Umowy, jeśli wada jest istotna.

3. Sprzedający zobowiązuje się do usunięcia wad lub wymiany Przedmiotu w terminie 14 dni od zgłoszenia wady.

4. Uprawnienia Kupującego będącego konsumentem nie mogą być ograniczone ani wyłączone (art. 10 ustawy o prawach konsumenta).

§ 5. Gwarancja

1. Sprzedający udziela gwarancji na Przedmiot na okres ______ miesięcy od dnia wydania.

2. Zakres gwarancji obejmuje:
________________________________________________________________________________
(naprawa, wymiana, zwrot kosztów)

3. Gwarancja nie obejmuje wad wynikłych z niewłaściwego użytkowania, uszkodzeń mechanicznych lub działania siły wyższej.

4. Reklamacje w ramach gwarancji należy zgłaszać na adres: ________________________________.

§ 6. Kary umowne

1. W przypadku niewykonania lub nienależytego wykonania zobowiązań z Umowy, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł (słownie: ________________________________ złotych).

2. Strona poszkodowana ma prawo dochodzić odszkodowania przenoszącego wysokość kary na zasadach ogólnych (art. 484 KC).

§ 7. Odstąpienie od umowy

1. Kupujący będący konsumentem ma prawo odstąpić od Umowy w terminie 14 dni od dnia wydania Przedmiotu bez podania przyczyny (art. 27 ustawy o prawach konsumenta).

2. W przypadku odstąpienia, Kupujący zwróci Przedmiot, a Sprzedający zwróci cenę w terminie 14 dni.

3. Koszty zwrotu Przedmiotu ponosi: Kupujący / Sprzedający.

§ 8. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego, w szczególności art. 535-581 KC.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Sprzedającego / miejsca zawarcia Umowy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach, po jednym dla każdej ze Stron.


SPRZEDAJĄCY: ____________________________    KUPUJĄCY: ____________________________""",

    "kaucja_zaliczka": """\
OŚWIADCZENIE O KAUCJI / ZALICZCE / ZADATKU
(zgodnie z art. 394 KC; zaliczka — art. 410 §2 KC)

________________, dnia ________________ r.

Strony:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy) — Wpłacający / Dłużnik
zamieszkały/z siedzibą w ________________________________
NIP / PESEL: ____________________________

________________________________________________________________________________
(imię i nazwisko / nazwa firmy) — Odbiorca / Wierzyciel
zamieszkały/z siedzibą w ________________________________
NIP / PESEL: ____________________________

§ 1. Tytuł wpłaty i podstawa

Strony zgodnie kwalifikują wpłatę otrzymaną przez Odbiorcę jako:
- kaucję zabezpieczającą roszczenia Odbiorcy z tytułu: ________________________________________________________________________________
- zaliczkę na poczet ceny / wynagrodzenia z tytułu umowy ________________ z dnia ________________,
- zadatek w rozumieniu art. 394 KC z tytułu umowy ________________ z dnia ________________.

§ 2. Kwota i sposób wpłaty

1. Kwota wpłaty wynosi ______________ zł (słownie: ________________________________ złotych).

2. Wpłata została dokonana:
- przelewem na rachunek Odbiorcy nr ________________ w dniu ________________,
- gotówką za pokwitowaniem w dniu ________________.

3. Odbiorca potwierdza otrzymanie kwoty wymienionej w ust. 1.

§ 3. Cel i zaliczenie

1. Kaucja zabezpiecza roszczenia Odbiorcy z tytułu:
________________________________________________________________________________
(niewykonanie lub nienależite wykonanie zobowiązania, odszkodowanie, kary umowne, itp.)

2. Zaliczka podlega zaliczeniu na poczet ceny / wynagrodzenia przy wykonaniu umowy głównej.

3. Zadatek:
- przy wykonaniu umowy — zaliczany na poczet świadczenia,
- przy niewykonaniu umowy przez Wpłacającego — Odbiorca może od umowy odstąpić i zachować zadatek,
- przy niewykonaniu umowy przez Odbiorcę — Wpłacający może od umowy odstąpić i żądać zwrotu dwukrotności zadatku (art. 394 KC).

4. Zaliczka podlega zwrotowi w całości przy niewykonaniu umowy, bez skutku z art. 394 KC.

§ 4. Zwrot i rozliczenie

1. Zwrot kaucji nastąpi w terminie 14 dni od:
- wykonania zobowiązania zabezpieczonego kaucją,
- rozwiązania umowy,
- upływu okresu, na jaki kaucja została ustanowiona.

2. Odbiorca może potrącić z kaucji należności z tytułu:
________________________________________________________________________________
po uprzednim wezwaniu Wpłacającego do zapłaty w terminie 14 dni.

3. Zwrot kaucji pomniejszonej o potrącone należności nastąpi przelewem na rachunek Wpłacającego nr ________________ w terminie 7 dni od rozliczenia.

§ 5. Zabezpieczenie wykonania umowy głównej

1. Kaucja / zadatek zabezpiecza także zobowiązanie Odbiorcy do zawarcia umowy głównej w terminie ________________.

2. W przypadku niewykonania tego zobowiązania przez Odbiorcę, Wpłacający ma prawo żądać zwrotu dwukrotności kaucji / zadatku.

§ 6. Postanowienia końcowe

1. Pokwitowanie stanowi dowód wpłaty kwoty.

2. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 394 KC — zadatek; art. 410 §2 KC — zaliczka jako świadczenie nienależne).

3. Wszelkie zmiany niniejszego oświadczenia wymagają formy pisemnej.

4. Oświadczenie sporządzono w dwóch jednobrzmiących egzemplarzach.


Wpłacający/Dłużnik: ____________________________    Odbiorca/Wierzyciel: ____________________________

Potwierdzam otrzymanie kwoty ______________ zł: ____________________________
(data i podpis Odbiorcy)""",

    "odstapienie_konsumenta": """\
OŚWIADCZENIE O ODSTĄPIENIU OD UMOWY ZAWARTEJ NA ODLEGŁOŚĆ / POZA LOKALEM PRZEDSIĘBIORSTWA
(art. 27 ustawy z dnia 30 maja 2014 r. o prawach konsumenta, Dz.U. 2024 poz. 1796)

________________, dnia ________________ r.

Dane konsumenta:
________________________________________________________________________________
(imię i nazwisko), zamieszkały w ______________________________________________
PESEL: ____________________________, telefon: ____________________________, e-mail: ____________________________

Dane przedsiębiorcy:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________

Działając na podstawie art. 27 ustawy z dnia 30 maja 2014 r. o prawach konsumenta (Dz.U. 2024 poz. 1796), oświadczam, że odstępuję od umowy:
- umowa o dzieło / umowa zlecenia / umowa sprzedaży / umowa o świadczenie usług (nieprawidłowe wykreślić),
- zawartej w dniu ________________ r.,
- sposób zawarcia: poza lokalem przedsiębiorstwa / na odległość (nieprawidłowe wykreślić),
- przedmiot umowy: ______________________________________________________________
(nazwa towaru/usługi, numer zamówienia/klienta: ____________________________)
- cena umowy: ______________ zł.

§ 1. Żądanie zwrotu płatności

Żądam zwrotu wszystkich dokonanych płatności, w tym kosztów dostarczenia towaru (z wyjątkiem dodatkowych kosztów wybranych przez konsumenta — art. 32 ust. 2 u.p.k.), w terminie 14 dni od otrzymania niniejszego oświadczenia, na rachunek bankowy nr:
________________________________________________________________________________
lub w sposób: przelew na konto / przekaz pocztowy / gotówką przy odbiorze.

§ 2. Zwrot towaru / rezygnacja z usługi

1. Towar zostanie zwrócony / został zwrócony w terminie 14 dni od dnia odstąpienia od umowy (art. 34 u.p.k.).

2. Przyjmuję do wiadomości, że ponoszę bezpośrednie koszty zwrotu towaru, chyba że przedsiębiorca zgodził się je ponieść lub nie poinformował konsumenta o obowiązku ich poniesienia (art. 34 ust. 2 u.p.k.).

3. W przypadku umowy o świadczenie usług: oświadczam, że usługa nie została zaczęta / została rozpoczęta przed upływem terminu do odstąpienia.

§ 3. Pouczenie o terminach

1. Termin 14 dni na odstąpienie od umowy liczy się:
   a) dla umowy sprzedaży — od dnia objęcia towaru w posiadanie,
   b) dla umowy o świadczenie usług — od dnia zawarcia umowy,
   c) dla umowy o dzieło — od dnia zawarcia umowy.
   Wyjątek: 30 dni — jeżeli umowa została zawarta poza lokalem przedsiębiorstwa podczas nieumówionej wizyty przedsiębiorcy w miejscu zamieszkania/pobytu konsumenta albo podczas wycieczki (art. 27 ust. 2 u.p.k.).

2. Do zachowania terminu wystarczy wysłanie oświadczenia przed jego upływem (art. 30 u.p.k.).

3. Przedsiębiorca ma obowiązek zwrotu płatności w terminie 14 dni od otrzymania oświadczenia o odstąpieniu.

§ 4. Załączniki

- kopia dowodu zakupu (paragon/faktura),
- potwierdzenie zamówienia / umowa,
- dokumentacja fotograficzna (w przypadku zwrotu towaru),
- potwierdzenie zwrotu towaru (po dokonaniu).

............................................
(czytelny podpis konsumenta)

....................................................................................
(data)""",

    "wezwanie_do_zaplaty": """\
WEZWANIE DO ZAPŁATY — OSTATECZNE PRZEDSĄDOWE
(art. 455 KC, ustawa z dnia 8 marca 2013 r. o przeciwdziałaniu nadmiernym opóźnieniom)

________________, dnia ________________ r.

Nadawca (Wierzyciel):
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkały/z siedzibą w ________________________
NIP: ____________________________, telefon: ____________________________

Adresat (Dłużnik):
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkały/z siedzibą w ________________________
NIP: ____________________________, telefon: ____________________________

§ 1. Podstawa zobowiązania

Dotyczy: zapłaty kwoty ______________ zł (słownie: ________________________________ złotych) z tytułu:
- umowa nr ________________ z dnia ________________,
- faktura nr ________________ z dnia ________________,
- termin płatności: ________________,
- dzień zapłaty: ________________ (zaległość od ______ dni).

§ 2. Kwota i odsetki

1. Kwota główna: ______________ zł.

2. Odsetki ustawowe za opóźnienie: ______________ zł (liczone od dnia ________________ do dnia ________________, stawka: ______%).

3. Odsetki za opóźnienie w transakcjach handlowych (art. 4 ustawy o przeciwdziałaniu nadmiernym opóźnieniom): ______________ zł.

4. Koszty odsetkowe: ______________ zł.

5. Łączna kwota do zapłaty: ______________ zł.

§ 3. Wezwanie do zapłaty

Wzywam do zapłaty łącznej kwoty ______________ zł wraz z odsetkami ustawowymi za opóźnienie / za opóźnienie w transakcjach handlowych (art. 455 KC; ustawa z dnia 8 marca 2013 r. o przeciwdziałaniu nadmiernym opóźnieniom) w terminie 14 dni od dnia otrzymania niniejszego wezwania, przelewem na rachunek nr:
________________________________________________________________________________
(tytuł przelewu: „Zapłata zaległości — faktura nr ________________").

§ 4. Konsekwencje braku zapłaty

1. W przypadku braku zapłaty w terminie 14 dni, naliczę dalsze odsetki ustawowe za opóźnienie.

2. Sprawa zostanie skierowana na drogę postępowania sądowego z wnioskiem o zasądzenie:
   a) kwoty głównej wraz z odsetkami,
   b) kosztów procesu, w tym kosztów zastępstwa procesowego,
   c) kosztów egzekucyjnych.

3. W przypadku braku zapłaty, złożę wniosek o wszczęcie postępowania egzekucyjnego.

§ 5. Przedawnienie — pouczenie

Niniejsze wezwanie przedsądowe nie przerywa biegu przedawnienia (art. 123 KC). Bieg przedawnienia przerywa dopiero wniesienie pozwu, zawezwanie do próby ugodowej (art. 185 KPC) lub inna czynność przed sądem/organem. Zaleca się dochodzenie roszczenia przed upływem terminu przedawnienia (co do zasady: 6 lat, dla roszczeń związanych z działalnością gospodarczą i świadczeń okresowych — 3 lata, art. 118 KC).

§ 6. Załączniki

- kopia umowy nr ________________ z dnia ________________,
- kopia faktury nr ________________ z dnia ________________,
- kopia upomnienia o zapłacie (jeśli było kierowane),
- potwierdzenie doręczenia niniejszego wezwania.

............................................
(czytelny podpis wierzyciela / pieczęć)

....................................................................................
(data)""",

    "reklamacja_konsumenta": """\
REKLAMACJA Z TYTUŁU NIEZGODNOŚCI TOWARU Z UMOWĄ / RĘKOJMI
(art. 43a-43g ustawy o prawach konsumenta / art. 556-576 KC)

________________, dnia ________________ r.

Dane konsumenta:
________________________________________________________________________________
(imię i nazwisko), zamieszkały w ______________________________________________
PESEL: ____________________________, telefon: ____________________________, e-mail: ____________________________

Dane sprzedawcy:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________

§ 1. Przedmiot reklamacji

Dotyczy: towaru ______________________________________________________________
(nazwa, model, numer seryjny, kolor, rozmiar, inne cechy identyfikujące)
nabytego w dniu ________________ r. za kwotę ______________ zł,
dowód zakupu: paragon / faktura nr ________________ z dnia ________________,
miejsce zakupu: ________________________________.

§ 2. Opis niezgodności / wady

Opis niezgodności / wady: ______________________________________________________
________________________________________________________________________________
________________________________________________________________________________
(wady fizyczne: pęknięcie, wada elektroniczna, niesprawność, niezgodność z opisem, itp.)

Data ujawnienia wady: ________________ r.

Towar jest niezgodny z umową w rozumieniu art. 43a ustawy z dnia 30 maja 2014 r. o prawach konsumenta (Dz.U. 2024 poz. 1796) / posiada wadę fizyczną w rozumieniu art. 556 KC.

§ 3. Żądanie

Na podstawie art. 43d u.p.k. / art. 560 KC żądam (nieprawidłowe wykreślić):
- naprawy towaru,
- wymiany towaru na wolny od wad,
- obniżenia ceny o ______________ zł (słownie: ________________________________ złotych),
- odstąpienia od umowy i zwrotu ceny,
- usunięcia wady.

Uzasadnienie: _________________________________________________________________
________________________________________________________________________________

§ 4. Termin rozpatrzenia

Wzywam do ustosunkowania się w terminie 14 dni od otrzymania niniejszej reklamacji; brak odpowiedzi w tym terminie poczytuje się za uznanie roszczenia (art. 43d ust. 8 u.p.k. / art. 561 § 5 KC w zw. z art. 7a u.p.k. dla konsumenta).

§ 5. Sposób załatwienia

Oczekuję rozpatrzenia reklamacji i poinformowania o sposobie załatwienia na adres:
________________________________________________________________________________
lub e-mail: ____________________________.

§ 6. Załączniki

- kopia dowodu zakupu (paragon/faktura),
- zdjęcia wady / uszkodzenia,
- kopia gwarancji (jeśli dotyczy),
- protokół naprawy (jeśli dotyczy).

............................................
(czytelny podpis konsumenta)

....................................................................................
(data)""",

    "klauzula_niedozwolona": """\
OCENA KLAUZULI UMOWNEJ POD KĄTEM NIEDOZWOLONYCH POSTANOWIEŃ
(art. 3851–3853 Kodeksu cywilnego, ustawa z dnia 23 sierpnia 2007 r. o przeciwdziałaniu nieuczciwym praktykom rynkowym)

________________, dnia ________________ r.

Zleceniodawca / Klient:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkały/z siedzibą w ________________________
NIP / PESEL: ____________________________

Data oceny: ________________ r. w ________________________________

Przedmiot oceny: umowa ________________________________________________________
(zlecenie, dzieło, dostawy, świadczenie usług, sprzedaż, inne)
z dnia ________________ r., pomiędzy:
- Zleceniodawcą / Przedsiębiorcą: ______________________________________________
- Klientem / Konsumentem: ______________________________________________________

§ 1. Badana klauzula

1. Treść badanej klauzuli:
„________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________"

2. Lokalizacja klauzula w umowie: paragraf ________________, punkt ________________,
strona ________________, zapisana: na początku / w środku / na końcu umowy.

3. Klauzula ma charakter: umowny (zawarta w indywidualnej umowie) / wzorzec umowny (zawarta w regulaminie/wzorcu umowy).

§ 2. Wzorzec kontroli (art. 3851 § 1 KC)

1. Postanowienie umowne podlega kontroli, jeżeli:
- nie zostało uzgodnione indywidualnie z konsumentem (art. 3851 § 3 KC),
- jest sprzeczne z dobrymi obyczajami,
- rażąco narusza interesy konsumenta.

2. Ciężar dowodu indywidualnego uzgodnienia klauzuli spoczywa na przedsiębiorcy (art. 3851 § 4 KC).

3. Postanowienie zawarte w formie pisemnej nie dowodzi jego indywidualnego uzgodnienia (art. 3851 § 4 KC).

4. Przedsiębiorca nie może powoływać się na klauzulę niedozwoloną wobec konsumenta.

§ 3. Ocena w świetle katalogu z art. 3853 KC i rejestru UOKiK

1. Katalog postanowień niedozwolonych (art. 3853 KC):
   a) pkt 1: wyłączenie lub ograniczenie odpowiedzialności za szkody na osobie,
   b) pkt 2: wyłączenie lub ograniczenie odpowiedzialności za wadliwe wykonanie umowy,
   c) pkt 3: prawo do odstąpienia od umowy bez podania przyczyny,
   d) pkt 4: prawo do rozwiązania umowy bez zachowania terminu wypowiedzenia,
   e) pkt 5: prawo do jednostronnej zmiany umowy bez ważnej przyczyny,
   f) pkt 6: prawo do jednostronnej zmiany ceny,
   g) pkt 7: prawo do jednostronnej zmiany zakresu świadczenia,
   h) pkt 8: prawo do zatrzymania pieniędzy wpłaconych przez konsumenta,
   i) pkt 9: obowiązek zapłaty kary umownej w wygórowanej wysokości,
   j) pkt 10: przeniesienie ciężaru dowodu na konsumenta,
   k) pkt 11: ograniczenie prawa do złożenia zastrzeżeń do jakości towaru,
   l) pkt 12: ograniczenie prawa do złożenia reklamacji,
   m) pkt 13: wyłączenie prawa do odstąpienia od umowy,
   n) pkt 14: obowiązek zapłaty wynagrodzenia w razie odstąpienia od umowy,
   o) pkt 15: obowiązek zapłaty wynagrodzenia w razie rozwiązania umowy,
   p) pkt 16: prawo do przejścia zobowiązania na konsumenta,
   q) pkt 17: prawo do przejścia praw na osobę trzecią bez zgody konsumenta,
   r) pkt 18: wyłączenie odpowiedzialności za okoliczności siły wyższej,
   s) pkt 19: wyłączenie prawa do odsetek,
   t) pkt 20: wyłączenie prawa do kosztów sądowych,
   u) pkt 21: obowiązek wyboru sądu polskiego,
   v) pkt 22: obowiązek wyboru prawa polskiego,
   w) pkt 23: inne postanowienia wymienione w art. 3853 KC.

2. Odniesienie do rejestru klauzul niedozwolonych UOKiK:
- nr wpisu w rejestrze: ________________,
- data wpisu: ________________,
- podstawa prawna wpisu: ________________,
- z treści wpisu wynika, że: ____________________________________________________

3. Odniesienie do decyzji Prezesa UOKiK i orzeczeń sądowych:
- sygnatura akt: ________________,
- data: ________________,
- treść: ________________________________________________________________________

§ 4. Ocena klauzuly

1. Klauzula jest sprzeczna z dobrymi obyczajami: tak / nie.
   Uzasadnienie: ________________________________________________________________

2. Klauzula rażąco narusza interesy konsumenta: tak / nie.
   Uzasadnienie: ________________________________________________________________

3. Klauzula nie została uzgodniona indywidualnie z konsumentem: tak / nie.
   Uzasadnienie: ________________________________________________________________

4. Klauzula odpowiada wzorcowi z art. 3853 KC: tak / nie.
   Wzorzec: pkt ________________.

5. Klauzula znajduje się w rejestrze klauzul niedozwolonych UOKiK: tak / nie.

6. Wniosek końcowy: klauzula jest / nie jest niedozwolona.

§ 5. Skutki prawne uznania klauzuli za niedozwoloną

1. Postanowienie niedozwolone nie wiąże konsumenta (art. 3851 § 1 KC).

2. Umowa w pozostałym zakresie wiąże strony (art. 3851 § 2 KC).

3. Sąd dokonuje kontroli klauzuli z urzędu (art. 3851 § 1 KC).

4. Możliwość kontroli:
   a) incydentalnej przez sąd powszechny,
   b) abstrakcyjnej przed SOKiK (Sąd Ochrony Konkurencji i Konsumentów).

5. Konsument może dochodzić roszczeń z tytułu niedozwolonej klauzuli w postępowaniu sądowym lub przed UOKiK.

6. Przedsiębiorca podlega karze pieniężnej nakazanej przez Prezesa UOKiK do kwoty 10% obrotu rocznego.

§ 6. Rekomendacja — klauzula zgodna

1. Proponowane brzmienie klauzuli zgodnej z prawem:
„________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________"

2. Uzasadnienie zgodności:
- klauzula została uzgodniona indywidualnie z konsumentem,
- klauzula jest przejrzysta i zrozumiała,
- klauzula nie jest sprzeczna z dobrymi obyczajami,
- klauzula nie rażąco narusza interesów konsumenta,
- klauzula nie odpowiada żadnemu ze wzorców z art. 3853 KC,
- klauzula nie znajduje się w rejestrze klauzul niedozwolonych UOKiK.

§ 7. Wnioski i zalecenia

1. Klauzula podlega usunięciu z umowy / zmianie na wariant zgodny z prawem.

2. Przedsiębiorca powinien poinformować konsumentów o zmianie klauzuli.

3. Przedsiębory powinien udostępnić zaktualizowany wzorzec umowny.

4. W przypadku stwierdzenia naruszenia, należy zgłosić sprawę do UOKiK lub SOKiK.

............................................
(podpis oceniającego / radca prawny / adwokat)

Podstawa: art. 3851–3853 KC, ustawa z dnia 23 sierpnia 2007 r. o przeciwdziałaniu nieuczciwym praktykom rynkowym, rejestr klauzul niedozwolonych UOKiK, dyrektywa 93/13/EWG.""",

    "kara_umowna": """\
ZASTRZEŻENIE KARY UMOWNEJ
(zgodnie z art. 483-484 Kodeksu cywilnego)

zawarte w dniu ________________ r. w ________________________________ pomiędzy:

STRONĄ A:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

STRONĄ B:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

jako postanowienie umowy głównej nr ________________ z dnia ________________ r.
(umowa o dzieło / umowa zlecenia / umowa dostawy / umowa o pracę / inna: _______________)

§ 1. Zobowiązanie główne

1. Strony łączy zobowiązanie: __________________________________________________
________________________________________________________________________________
(świadczenie niepieniężne — art. 483 § 1 KC).

2. Strona zobowiązana do wykonania świadczenia: Strona A / Strona B.

3. Termin wykonania zobowiązania: ________________________________________________

§ 2. Zastrzeżenie kary umownej

1. Za niewykonanie lub nienależite wykonanie zobowiązania określonego w § 1, Strona zobowiązana zapłaci karę umowną w wysokości: ________________________________________________________________________________
(art. 483 KC).

2. Kara umowna należy się za:
- niewykonanie zobowiązania w terminie,
- nienależite wykonanie zobowiązania,
- opóźnienie w wykonaniu zobowiązania,
- naruszenie zakazu konkurencji,
- naruszenie klauzuli poufności,
- inne przesłanki: ________________________________________________________________

3. Kara umowna należy się niezależnie od wysokości szkody, chyba że zastrzeżono inaczej.

§ 3. Sposób naliczania i limit

1. Sposób naliczania kary umownej:
- kwota stała: ______________ zł,
- stawka dzienna: ______________ zł za każdy dzień opóźnienia,
- procentowo: ______% wartości niewykonanego zobowiązania.

2. Łączny limit kar umownych: ______________ zł / brak limitu.

3. Kara umowna należy się za każdy dzień opóźnienia / za każde naruszenie / jednorazowo.

4. Kara umowna nie obejmuje odsetek za opóźnienie w spełnieniu świadczenia pieniężnego (art. 483 § 1 zd. 2 KC w zw. z art. 481 KC).

§ 4. Zbieg kary umownej i odszkodowania

1. Dochodzenie odszkodowania przenoszącego karę umowną jest dopuszczalne / niedopuszczalne na zasadach art. 484 § 1 KC.

2. W przypadku dochodzenia odszkodowania przenoszącego karę, kwota kary umownej zostanie odliczona od wysokości odszkodowania.

3. Strona uprawniona może dochodzić odszkodowania uzupełniającego przekraczającego wysokość kary umownej.

§ 5. Miarkowanie kary umownej

1. Na zasadach art. 484 § 2 KC, dłużnik może żądać miarkowania kary rażąco wygórowanej.

2. Sąd dokonuje miarkowania kary z uwzględnieniem:
- wysokości szkody,
- wartości zobowiązania,
- charakteru naruszenia,
- sytuacji majątkowej stron.

3. Kara umowna nie może być rażąco wygórowana w stosunku do wartości zobowiązania.

§ 6. Forma zastrzeżenia

1. Zastrzeżenie kary umownej wymaga formy pisemnej pod rygorem nieważności co do kary (art. 483 KC a contrario).

2. Zmiana wysokości kary umownej wymaga formy pisemnej.

§ 7. Płatność kary umownej

1. Kara umowna płatna jest w terminie 14 dni od wezwania do zapłaty.

2. Płatność kary umownej następuje przelewem na rachunek strony uprawnionej nr ________________.

3. W przypadku opóźnienia w płatności kary umownej, należą się odsetki ustawowe za opóźnienie (art. 481 KC).

§ 8. Postanowienia końcowe

1. Zastrzeżenie kary umownej stanowi integralną część umowy głównej.

2. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 483-484 KC).

3. Wszelkie zmiany zastrzeżenia kary umownej wymagają formy pisemnej.


STRONA A: ____________________________    STRONA B: ____________________________""",

    "pelnomocnictwo": """\
PEŁNOMOCNICTWO
(art. 98-106 Kodeksu cywilnego)

________________, dnia ________________ r.

Ja, niżej podpisany(a):
________________________________________________________________________________
(imię i nazwisko), PESEL: ____________________________, seria i nr dowodu tożsamości: ____________________________, zamieszkały(a) w ______________________________________________

działając na podstawie art. 98-99 KC, ustanawiam pełnomocnikiem:

________________________________________________________________________________
(imię i nazwisko), PESEL: ____________________________, seria i nr dowodu tożsamości: ____________________________, zamieszkały(a) w ______________________________________________

do dokonywania w moim imieniu następujących czynności:
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

Pełnomocnictwo ma charakter ogólny / rodzajowy / szczególny (niepotrzebne skreślić) w rozumieniu art. 98 KC.

§ 1. Zakres umocowania

Pełnomocnik jest umocowany do dokonywania czynności wymienionych w preambule przed:
- urzędami administracji publicznej, sądami, prokuraturą,
- bankami i instytucjami finansowymi,
- podmiotami trzecimi we wszelkich sprawach związanych z powierzonym zakresem,
- odbierania korespondencji i dokumentów.

§ 2. Okres ważności

Pełnomocnictwo jest ważne:
- do dnia ________________ r. / do odwołania / do czasu wykonania czynności,
- z chwilą śmierci Mocodawcy lub Pełnomocnika pełnomocnictwo wygasa (art. 101 § 2 KC).

§ 3. Substytucja

Pełnomocnik może / nie może ustanawiać dalszych pełnomocników (substytucja — art. 106 KC).

§ 4. Odwołanie

Mocodawca może w każdym czasie odwołać pełnomocnictwo w formie pisemnej (art. 101 § 1 KC).

§ 5. Obowiązki Pełnomocnika

1. Pełnomocnik zobowiązuje się do wykonywania umocowania z należytą starannością.

2. Pełnomocnik zobowiązuje się do składania Mocodawcy sprawozdań z działania.

3. Pełnomocnik zobowiązuje się niezwłocznie po zakończeniu sprawy wydać Mocodawcy udokumentowanie poniesionych wydatków.

§ 6. Wynagrodzenie

1. Pełnomocnictwo jest odpłatne / nieodpłatne.

2. W przypadku pełnomocnictwa odpłatnego, Mocodawca zobowiązuje się zapłacić Pełnomocnikowi wynagrodzenie w wysokości ______________ zł (słownie: ________________________________ złotych) oraz zwrócić udokumentowane wydatki.

§ 7. Postanowienia końcowe

1. Pełnomocnictwo sporządzone zostało w dwóch jednobrzmiących egzemplarzach.

2. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 98-106 KC).


MOCODAWCA: ____________________________    PEŁNOMOCNIK: ____________________________

....................................................................................
(data)

....................................................................................
(data)""",

    "poreczenie": """\
OŚWIADCZENIE PORĘCZYCIELA — PORĘCZENIE ZA ZOBOWIĄZANIE DŁUŻNIKA
(art. 876-887 Kodeksu cywilnego)

________________, dnia ________________ r.

Strony:

WIERZYCIEL:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkały/z siedzibą w ________________________
NIP / PESEL: ____________________________

DŁUŻNIK GŁÓWNY:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkały/z siedzibą w ________________________
NIP / PESEL: ____________________________

PORĘCZYCIEL:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkały/z siedzibą w ________________________
NIP / PESEL: ____________________________

§ 1. Przedmiot poręczenia

Oświadczam, że poręczam za zobowiązanie Dłużnika Głównego wynikające z:
- umowa nr ________________ z dnia ________________,
- faktura nr ________________ z dnia ________________,
- inna podstawa: ________________________________________________________________

w zakresie zapłaty kwoty ______________ zł (słownie: ________________________________ złotych).

§ 2. Zakres i charakter odpowiedzialności

1. Odpowiadam jako Poręczyciel solidarnie z Dłużnikiem Głównym do wysokości sumy poręczenia ______________ zł (art. 876 § 1 KC).

2. Poręczenie obejmuje:
- kwotę główną: ______________ zł,
- odsetki ustawowe za opóźnienie: tak / nie,
- koszty dochodzenia należności: tak / nie,
- koszty egzekucyjne: tak / nie.

3. Odpowiedzialność Poręczyciela jest solidarna z Dłużnikiem Głównym — Wierzyciel może dochodzić roszczenia od Poręczyciela bez uprzedniego dochodzenia od Dłużnika Głównego (art. 876 § 1 KC).

§ 3. Forma oświadczenia

Oświadczenie Poręczyciela wymaga formy pisemnej pod rygorem nieważności (art. 876 § 2 KC).

§ 4. Termin poręczenia

1. Poręczenie obowiązuje do dnia ________________ r. / do czasu wykonania zobowiązania przez Dłużnika Głównego.

2. Roszczenie przeciwko Poręczycielowi przedawnia się z upływem przedawnienia roszczenia Dłużnika Głównego (art. 881 KC).

§ 5. Zarzuty Poręczyciela

Poręczyciel może podnosić wszelkie zarzuty przysługujące Dłużnikowi Głównemu, w tym zarzut nieważności zobowiązania, zarzut przedawnienia, zarzut nienależytego wykonania (art. 883 KC).

§ 6. Obowiązki Wierzyciela

1. Wierzyciel zawiadomi Poręczyciela o opóźnieniu Dłużnika Głównego w terminie 14 dni od powstania zaległości.

2. Wierzyciel przekaże Poręczycielowi dokumentację niezbędną do ustalenia zakresu zobowiązania.

§ 7. Prawa Poręczyciela po wykonaniu poręczenia

1. W przypadku zapłaty przez Poręczyciela, przechodzą na niego roszczenia Wierzyciela przysługujące wobec Dłużnika Głównego w zakresie zapłaconej kwoty (art. 882 KC).

2. Poręczyciel ma prawo dochodzić od Dłużnika Głównego zwrotu zapłaconej kwoty wraz z odsetkami.

§ 8. Postanowienia końcowe

1. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 876-887 KC).

2. Wszelkie zmiany niniejszego oświadczenia wymagają formy pisemnej.

3. Oświadczenie sporządzone zostało w dwóch jednobrzmiących egzemplarzach.


WIERZYCIEL: ____________________________    PORĘCZYCIEL: ____________________________

....................................................................................
(data)

....................................................................................
(data)""",

    "umowa_darowizny": """\
UMOWA DAROWIZNY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

DARCZYŃCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

OBDAROWANYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot darowizny

1. Darczyńca oświadcza, że daruje Obdarowanemu:
________________________________________________________________________________
(dokładny opis przedmiotu darowizny: rzecz ruchoma / nieruchomość / prawo majątkowe / kwota pieniędzy)
________________________________________________________________________________
o wartości ______________ zł (słownie: ________________________________ złotych) (art. 888 KC).

2. Darczyńca oświadcza, że przedmiot darowizny stanowi jego wyłączną własność i jest wolny od obciążeń i roszczeń osób trzecich.

3. Obdarowany oświadcza, że darowiznę przyjmuje.

§ 2. Wydanie przedmiotu darowizny

1. Wydanie przedmiotu darowizny nastąpi w dniu ________________ r. w miejscu: ________________________________________________________________________________

2. Wydanie przedmiotu nastąpi wraz z następującymi dokumentami:
________________________________________________________________________________
(akt własności, dokumentacja techniczna, klucze, itp.)

3. Koszty związane z wydaniem i transportem przedmiotu ponosi: Darczyńca / Obdarowany / Strony po połowie.

§ 3. Forma umowy

1. Umowa darowizny nieruchomości wymaga formy aktu notarialnego (art. 890 KC w zw. z art. 158 KC).

2. Darowizna ruchomości co do zasady nie wymaga formy szczególnej, jednak Strony postanawiają o sporządzeniu umowy w formie pisemnej dla celów dowodowych.

3. W przypadku darowizny w formie aktu notarialnego, koszty notarialne ponosi: Darczyńca / Obdarowany / Strony po połowie.

§ 4. Polecenie / warunek / termin

1. Darowizna jest bezwarunkowa / warunkowa — warunek: ________________________________________________________________________________

2. Darowizna obowiązuje z chwilą zawarcia umowy / z chwilą wydania przedmiotu / z chwilą zaistnienia warunku.

3. W przypadku darowizny z poleceniem, Obdarowany zobowiązuje się do:
________________________________________________________________________________

§ 5. Odwołanie darowizny

1. Darczyńca może odwołać darowiznę w przypadku:
   a) niewdzięczności Obdarowanego (art. 896 KC),
   b) niezaspokojenia potrzeb życiowych Darczyńcy (art. 897 KC),
   c) niewykonania polecenia lub warunku (art. 898 KC).

2. Odwołanie darowizny wymaga formy pisemnej.

3. W przypadku odwołania darowizny, Obdarowany zobowiązany jest zwrócić przedmiot darowizny.

§ 6. Koszty i podatki

1. Koszty zawarcia umowy ponosi: Darczyńca / Obdarowany / Strony po połowie.

2. Podatek od spadków i darowizn ponosi: Darczyńca / Obdarowany / Strony po połowie.

3. Zwolnienie z podatku od spadków i darowizn dla najbliższej rodziny na zasadach ustawy o podatku od spadków i darowizn.

§ 7. Oświadczenia stron

1. Darczyńca oświadcza, że czyni darowiznę dobrowolnie, bez nacisku i pod wpływem błędu.

2. Obdarowany oświadcza, że przyjmuje darowiznę świadomie i dobrowolnie.

3. Strony oświadczają, że zapoznały się z treścią Umowy i nie wnoszą zastrzeżeń.

§ 8. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 888-902 KC).

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla miejsca zamieszkania Darczyńcy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach, po jednym dla każdej ze Stron.


DARCZYŃCA: ____________________________    OBDAROWANY: ____________________________""",

    "umowa_dzierzawy": """\
UMOWA DZIERŻAWY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

WYDZIERŻAWIAJĄCYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

DZIERŻAWCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot dzierżawy

1. Wydzierżawiający oddaje Dzierżawcy do używania i pobierania pożytków nieruchomości położonej w ________________________________, przy ul. ________________________________, zapisanej w księdze wieczystej nr ________________, o powierzchni ______ m² (zwaną dalej „Przedmiotem dzierżawy") (art. 693 KC).

2. Przedmiot dzierżawy obejmuje:
- grunt o powierzchni ______ m²,
- budynki i urządzenia: ________________________________,
- wyposażenie: ________________________________ (Załącznik nr 1).

3. Wydzierżawiający oświadcza, że Przedmiot dzierżawy jest wolny od obciążeń na rzecz osób trzecich, które mogłyby uniemożliwić lub utrudnić Dzierżawcy korzystanie z Przedmiotu.

§ 2. Czas trwania umowy

1. Umowa zawarta zostaje na czas określony od dnia ________________ do dnia ________________.

2. Umowa ulega przedłużeniu na czas nieokreślony, jeśli któraś ze stron nie wypowie jej na 3 miesiące przed upływem terminu.

3. Wydzierżawiający może rozwiązać umowę przed upływem terminu w przypadku:
   a) Dzierżawca używa Przedmiotu w sposób sprzeczny z umową lub z przeznaczeniem,
   b) Dzierżawca jest w zwłoce z czynszem co najmniej za dwa pełne okresy płatności,
   c) Dzierżawca oddał Przedmiot w podnajem bez zgody.

§ 3. Czynsz dzierżawny

1. Czynsz dzierżawny miesięczny wynosi ______________ zł (słownie: ________________________________ złotych) netto / brutto.

2. Czynsz płatny jest z góry do ______ dnia każdego miesiąca kalendarzowego przelewem na rachunek nr ________________________________.

3. Czynsz obejmuje / nie obejmuje podatku VAT. Stawka VAT: ______%.

4. Czynsz podlega waloryzacji raz w roku o wskaźnik inflacji ogłaszany przez GUS, nie więcej niż o 10% rocznie.

5. Niezależnie od czynszu Dzierżawca pokrywa opłaty za: ________________________________
(media, podatki, opłaty administracyjne)

§ 4. Obowiązki Dzierżawcy

1. Używać Przedmiotu zgodnie z przeznaczeniem i zasadami prawidłowej gospodarki (art. 696 KC).

2. Dokonywać napraw i ponosić nakłady konieczne do utrzymania Przedmiotu w stanie niepogorszonym.

3. Dokonywać ulepszeń za zgodą Wydzierżawiającego.

4. Nie oddawać Przedmiotu w podnajem ani do bezpłatnego używania bez pisemnej zgody Wydzierżawiającego.

5. Ubezpieczyć Przedmiot na własny koszt w zakresie ________________________________.

6. Utrzymywać porządek w lokalu i dbać o stan techniczny urządzeń.

§ 5. Obowiązki Wydzierżawiającego

1. Wydać Przedmiot w stanie przydatnym do umówionego użytku i utrzymywać go w tym stanie.

2. Pokrywać nakłady na utrzymanie Przedmiotu w stanie nadającym się do używania.

3. Dostarczyć Dzierżawcy dokumentację niezbędną do korzystania z Przedmiotu.

§ 6. Wydanie i zwrot Przedmiotu

1. Wydanie i zwrot Przedmiotu dzierżawy nastąpi protokołem zdawczo-odbiorczym.

2. Protokół zdawczo-odbiorczy będzie zawierał opis stanu Przedmiotu, stan liczników, wykaz wyposażenia.

3. Dzierżawca zwróci Przedmiot w stanie niepogorszonym, z wyłączeniem normalnego zużycia (art. 705 KC).

§ 7. Kaucja

1. Dzierżawca wpłaca kaucję w kwocie ______________ zł (słownie: ________________________________ złotych) zabezpieczającą roszczenia Wydzierżawiającego.

2. Kaucja podlega zwrotowi w terminie 14 dni od zwrotu Przedmiotu, po potrąceniu należności.

§ 8. Kary umowne

1. W przypadku nieterminowego płacenia czynszu, Dzierżawca zapłaci karę umowną w wysokości ______________ zł za każdy dzień opóźnienia.

2. W przypadku niewykonania lub nienależytego wykonania zobowiązań, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł.

§ 9. Rozwiązanie umowy

1. Każda ze stron może rozwiązać umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

2. Rozwiązanie umowy wymaga formy pisemnej.

3. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 701-704 KC).

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 693-709 KC).

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla miejsca położenia Przedmiotu dzierżawy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Wykaz wyposażenia Przedmiotu dzierżawy, 2. Protokół zdawczo-odbiorczy.


WYDZIERŻAWIAJĄCY: ____________________________    DZIERŻAWCA: ____________________________""",

    "umowa_dostawy": """\
UMOWA DOSTAWY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

DOSTAWCĄ:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

ODBIORCĄ:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot dostawy

1. Dostawca zobowiązuje się dostarczać Odbiorcy sukcesywnie towar: ________________________________________________________________________________
(nazwa, opis, specyfikacja techniczna, parametry)
o jakości zgodnej z: normami / specyfikacją techniczną / wzorcem (Załącznik nr 1) (art. 605 KC).

2. Towar musi być nowy, nieużywany, wolny od wad fizycznych i prawnych.

3. Łączna ilość towaru wynosi: ________________________________ (w jednostkach: szt., kg, m², itp.).

§ 2. Terminy i harmonogram dostaw

1. Dostawy realizowane będą w następującym harmonogramie:
- partia nr 1: ______ szt. do dnia ________________,
- partia nr 2: ______ szt. do dnia ________________,
- partia nr 3: ______ szt. do dnia ________________.

2. Każda partia dostawy na podstawie zamówienia Odbiorcy złożonego w terminie 14 dni przed planowaną datą dostawy.

3. Dostawca zobowiązany jest do powiadomienia Odbiorcy o terminie dostawy co najmniej 3 dni przed planowaną datą dostawy.

§ 3. Cena i warunki płatności

1. Cena jednostkowa towaru wynosi ______________ zł netto / brutto za 1 szt. / kg / m².

2. Łączna wartość umowy szacunkowo wynosi ______________ zł netto / brutto.

3. Cena obejmuje / nie obejmuje podatku VAT. Stawka VAT: ______%.

4. Cena obejmuje / nie obejmuje koszty transportu, ubezpieczenia i wydania towaru.

5. Płatność za każdą partię towaru w terminie 14 dni od dnia dostawy i wystawienia faktury.

6. Waloryzacja cen: raz w roku o wskaźnik inflacji GUS, nie więcej niż 5% rocznie.

§ 4. Warunki dostawy i odbiór

1. Dostawa towaru nastąpi:
- DDP (Incoterms 2020) — do miejsca przeznaczenia: ________________________________,
- EXW (Incoterms 2020) — z magazynu Dostawcy,
- inny sposób: ________________________________________________________________.

2. Odbiór ilościowy i jakościowy towaru nastąpi w momencie dostawy.

3. Protokół dostawy podpisany przez obie Strony stanowi potwierdzenie przyjęcia towaru.

4. Ryzyko przypadkowej utraty lub uszkodzenia towaru przechodzi na Odbiorcę z chwilą wydania towaru.

§ 5. Reklamacje i wady

1. Odbiorca zgłasza wady ilościowe w terminie 7 dni od dostawy.

2. Odbiorca zgłasza wady jakościowe w terminie 14 dni od dostawy.

3. W przypadku stwierdzenia wad, Odbiorca ma prawo:
   a) żądać wymiany towaru na wolny od wad,
   b) żądać usunięcia wady,
   c) żądać obniżenia ceny,
   d) odstąpić od umowy (jeśli wada jest istotna).

4. Dostawca zobowiązuje się do rozpatrzenia reklamacji w terminie 14 dni od jej otrzymania.

5. Brak odpowiedzi w terminie 14 dni oznacza uznanie reklamacji.

§ 6. Kary umowne

1. Za opóźnienie w dostawie towaru Dostawca zapłaci karę umowną w wysokości ______________ zł za każdy dzień zwłoki, nie więcej niż ______________ zł (łącznie).

2. Za opóźnienie w usunięciu wad Dostawca zapłaci karę umowną w wysokości ______________ zł za każdy dzień zwłoki.

3. W przypadku niewykonania lub nienależytego wykonania zobowiązań, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł.

§ 7. Okres obowiązywania i rozwiązanie umowy

1. Umowa obowiązuje od dnia ________________ do dnia ________________.

2. Każda ze stron może rozwiązać umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

3. Rozwiązanie umowy wymaga formy pisemnej.

4. W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego (art. 605-612 KC).

§ 8. Postanowienia końcowe

1. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

2. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Dostawcy.

3. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Specyfikacja techniczna towaru, 2. Harmonogram dostaw.


DOSTAWCA: ____________________________    ODBIORCA: ____________________________""",

    "umowa_franczyzy": """\
UMOWA FRANCZYZY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

FRANCZYZODAWCĄ:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

FRANCZYZOBIORCĄ:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot franczyzy

1. Franczyzodawca udziela Franczyzobiorcy prawa prowadzenia działalności gospodarczej w zakresie: ________________________________________________________________________________
(pokój usług, branża, asortyment)
pod znakiem towarowym: ________________________________,
w lokalu położonym w: ________________________________.

2. Franczyzodawca udziela Franczyzobiorcy prawa do korzystania z:
- znaku towarowego: ________________________________,
- know-how i tajemnicy przedsiębiorstwa,
- podręcznika operacyjnego,
- systemu informatycznego,
- materiałów reklamowych i marketingowych.

3. Prawa udzielone na podstawie niniejszej umowy są niezbywalne i nie mogą być przenoszone na osoby trzecie bez pisemnej zgody Franczyzodawcy.

§ 2. Wyłączność terytorialna

1. Franczyzobiorca otrzymuje wyłączność / brak wyłączności na terytorium: ________________________________________________________________________________

2. Franczyzodawca zobowiązuje się nie udzielać franczyzy innym osobom na terytorium objętym wyłącznością.

3. Franczyzobiorca zobowiązuje się nie prowadzić działalności konkurencyjnej na terytorium wyłączności.

§ 3. Opłaty franczyzowe

1. Opłata wstępna (franchise fee): ______________ zł (słownie: ________________________________ złotych) płatna w dniu zawarcia umowy / w terminie 14 dni od zawarcia umowy.

2. Opłata bieżąca (royalty):
- ______% miesięcznego obrotu Franczyzobiorcy,
- płatna do ______ dnia każdego miesiąca za miesiąc poprzedni,
- wysokość minimalna: ______________ zł miesięcznie.

3. Opłata marketingowa / reklamowa:
- ______% miesięcznego obrotu Franczyzobiorcy,
- płatna do ______ dnia każdego miesiąca,
- przeznaczona na działalność promocyjną systemu franczyzowego.

§ 4. Obowiązki Franczyzodawcy

1. Przekazać Franczyzobiorcy know-how i tajemnicę przedsiębiorstwa.

2. Przekazać podręcznik operacyjny i dokumentację niezbędną do prowadzenia działalności.

3. Zapewnić szkolenie wstępne Franczyzobiorcy i personelu w zakresie: ________________________________________________________________________________

4. Zapewnić wsparcie marketingowe i reklamowe w zakresie: ________________________________________________________________________________

5. Zapewnić dostawy towarów / materiałów w zakresie: ________________________________________________________________________________

6. Prowadzić działalność promocyjną systemu franczyzowego na poziomie ogólnokrajowym.

7. Przekazywać Franczyzobiorcy aktualizacje know-how i podręcznika operacyjnego.

§ 5. Obowiązki Franczyzobiorcy

1. Prowadzić działalność zgodnie ze standardami sieci franczyzowej i z podręcznikiem operacyjnym.

2. Utrzymywać lokal i wyposażenie w stanie odpowiadającym standardom sieci.

3. Korzystać ze znaku towarowego wyłącznie w sposób określony w umowie.

4. Zachować w tajemnicy know-how i tajemnicę przedsiębiorstwa.

5. Nie prowadzić działalności konkurencyjnej w trakcie trwania umowy i przez ______ miesięcy po jej rozwiązaniu.

6. Przestrzegać zasad polityki cenowej ustalonych przez Franczyzodawcę.

7. Przekazywać Franczyzodawcy raporty sprzedażowe w terminie ______ dnia każdego miesiąca.

§ 6. Czas trwania i rozwiązanie umowy

1. Umowa zostaje zawarta na czas: określony od dnia ________________ do dnia ________________ / nieokreślony.

2. Wypowiedzenie umowy wymaga formy pisemnej i zachowania 3-miesięcznego okresu wypowiedzenia.

3. Każda ze stron może rozwiązać umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

§ 7. Zakaz konkurencji po rozwiązaniu

1. Franczyzobiorca zobowiązuje się nie prowadzić działalności konkurencyjnej na terytorium: ________________________________________________________________________________

2. Zakaz konkurencji obowiązuje przez okres ______ miesięcy od rozwiązania umowy.

3. W przypadku naruszenia zakazu konkurencji, Franczyzobiorca zapłaci karę umowną w wysokości ______________ zł.

§ 8. Kary umowne

1. W przypadku niewykonania lub nienależytego wykonania zobowiązań, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł.

2. W przypadku naruszenia klauzuli poufności, Franczyzobiorca zapłaci karę umowną w wysokości ______________ zł.

3. W przypadku opóźnienia w płatności opłat franczyzowych, Franczyzobiorca zapłaci odsetki w wysokości ______% za każdy dzień opóźnienia.

§ 9. Postanowienia końcowe

1. Know-how stanowi tajemnicę przedsiębiorstwa w rozumieniu art. 11 ust. 2 ustawy o zwalczaniu nieuczciwej konkurencji.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Franczyzodawcy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Podręcznik operacyjny, 2. Specyfikacja znaku towarowego, 3. Opis know-how.


FRANCZYZODAWCA: ____________________________    FRANCZYZOBIORCA: ____________________________""",

    "umowa_faktoringu": """\
UMOWA FAKTORINGU

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

FAKTOREM:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

FAKTORANTEM:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot faktoringu

1. Faktorant przekazuje na własność Faktora wierzytelności istniejące i przyszłe z tytułu:
- dostaw towarów / świadczenia usług / umów,
- wobec dłużników należących do kategorii: ________________________________________________________________________________

2. Wierzytelności przekazywane są wraz ze wszystkimi prawami i roszczeniami z nimi związanymi.

3. Łączna wartość nominalna wierzytelności objętych umową wynosi szacunkowo ______________ zł.

4. Przekazanie wierzytelności następuje poprzez przekazanie dokumentów potwierdzających wierzytelności (faktury, umowy, listy przewodnie).

§ 2. Rodzaj faktoringu

1. Faktoring pełny (bez regresu) — Faktor przejmuje ryzyko kredytowe i nie może żądać od Faktoranta zwrotu wypłaconego finansowania w przypadku niewypłacalności dłużnika.

2. Faktoring niepełny (z regresem) — Faktor może żądać od Faktoranta zwrotu wypłaconego finansowania w przypadku niewypłacalności dłużnika po upływie ______ dni od terminu zapłaty.

3. Faktoring mieszany — warunki szczegółowe określa Załącznik nr 1.

4. Faktoring zwykły / dyskretny (niepowiadamiany) — dłużnik zostanie / nie zostanie powiadomiony o cesji wierzytelności.

§ 3. Finansowanie

1. Faktor wypłaca Faktorantowi zaliczkę w wysokości ______% wartości brutto przekazanych wierzytelności.

2. Zaliczka wypłacana jest w terminie ______ dni od przedstawienia dokumentów potwierdzających wierzytelności.

3. Pozostała część wartości wierzytelności pomniejszona o wynagrodzenie Faktora i ewentualne potrącenia wypłacana jest Faktorantowi po otrzymaniu zapłaty od dłużnika.

4. Limit finansowania wynosi ______________ zł.

§ 4. Wynagrodzenie Faktora

1. Prowizja faktoringowa wynosi ______% wartości brutto wierzytelności.

2. Odsetki / dyskonto liczone są od wypłaconej zaliczki według stawki: WIBOR ______ + ______% marży.

3. Opłata za prowadzenie rachunku: ______________ zł miesięcznie.

4. Opłata za każdą przekazaną wierzytelność: ______________ zł.

§ 5. Przekazanie wierzytelności i zawiadomienie dłużnika

1. Przekazanie wierzytelności następuje w formie cesji na podstawie umowy (art. 509-518 KC).

2. Faktorant przekazuje Faktorowi dokumenty potwierdzające wierzytelności (faktury, umowy, listy przewodnie).

3. Zawiadomienie dłużnika o cesji wierzytelności:
- zostanie doręczone przez Faktora w terminie ______ dni od cesji,
- zostanie doręczone przez Faktoranta w terminie ______ dni od cesji,
- nie zostanie doręczone (faktoring dyskretny).

4. Dłużnik spełnia świadczenie do rąk Faktora ze skutkiem zwalniającym (art. 510 KC).

§ 6. Odpowiedzialność Faktoranta

1. Faktorant odpowiada za istnienie i wymagalność przekazanych wierzytelności (art. 516 KC analogia).

2. Faktorant oświadcza, że przekazane wierzytelności istnieją, są wymagalne i nie są obciążone prawami osób trzecich.

3. Faktorant zobowiązuje się do zwrotu otrzymanego finansowania w przypadku stwierdzenia, że przekazana wierzytelność nie istnieje lub nie jest wymagalna.

4. Faktorant zobowiązuje się do niezawierania Faktora w transakcjach z dłużnikami.

§ 7. Czas trwania i rozwiązanie umowy

1. Umowa obowiązuje od dnia ________________ do dnia ________________ / na czas nieokreślony.

2. Wypowiedzenie umowy wymaga formy pisemnej i zachowania ______-dniowego okresu wypowiedzenia.

3. Każda ze stron może rozwiązać umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

4. W przypadku rozwiązania umowy, Faktorant zobowiązany jest do wykupu nieuregulowanych wierzytelności lub zapewnić ich regulowanie w terminie ______ dni.

§ 8. Poufność i tajemnica przedsiębiorstwa

1. Strony zobowiązują się do zachowania w tajemnicy wszelkich informacji uzyskanych w związku z umową.

2. Obowiązek poufności obowiązuje przez okres trwania umowy oraz przez 2 lata po jej rozwiązaniu.

3. Naruszenie klauzuli poufności uprawnia stronę poszkodowaną do żądania kary umownej w wysokości ______________ zł.

§ 9. Kary umowne

1. W przypadku niewykonania lub nienależytego wykonania zobowiązań, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł.

2. W przypadku opóźnienia w płatności, dłużnik zapłaci odsetki w wysokości ______% za każdy dzień opóźnienia.

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 509-518 KC) oraz ustawy o zwalczaniu nieuczciwej konkurencji.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Faktora.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Warunki szczególne faktoringu, 2. Formularz przekazania wierzytelności.


FAKTOR: ____________________________    FAKTORANT: ____________________________""",

    "umowa_pozyczki": """\
UMOWA POŻYCZKI

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

POŻYCZKODAWCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

POŻYCZKOBIORCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot pożyczki

1. Pożyczkodawca udziela Pożyczkobiorcy pożyczki w kwocie ______________ zł (słownie: ________________________________ złotych) (art. 720 KC).

2. Pożyczka jest przeznaczona na: ________________________________________________________________________________

3. Pożyczkodawca oświadcza, że kwota pożyczki stanowi jego własne środki.

§ 2. Przekazanie środków

1. Przekazanie środków nastąpi:
- przelewem na rachunek Pożyczkobiorcy nr ________________________________ w dniu ________________,
- gotówką w dniu ________________ za pokwitowaniem.

2. Pożyczkodawca potwierdza przekazanie środków w dniu ________________.

§ 3. Zwrot pożyczki

1. Pożyczkobiorca zwróci pożyczkę w terminie do dnia ________________ r.

2. Zwrot pożyczki nastąpi:
- przelewem na rachunek Pożyczkodawcy nr ________________________________,
- gotówką za pokwitowaniem.

3. Pożyczkobiorca może dokonać wcześniejszego zwrotu pożyczki w całości lub w częściach z zachowaniem 30-dniowego terminu wypowiedzenia.

§ 4. Odsetki

1. Pożyczka jest odpłatna — odsetki wynoszą ______% w skali roku (art. 720 § 1 KC).

2. Odsetki są kapitałowe / stałe / zmienne (nieprawidłowe wykreślić).

3. Odsetki płatne są: miesięcznie / kwartalnie / na końcu okresu kredytowania.

4. Odsetki maksymalne nie mogą przekroczyć stawki ustawowej na podstawie art. 359 KC.

5. Odsetki za opóźnienie wynoszą ______% w skali roku (art. 481 KC).

§ 5. Zabezpieczenie zwrotu pożyczki

1. Zwrot pożyczki zabezpieczony jest:
- wekslem in blanco z klauzulą „bez protestu" na kwotę ______________ zł,
- poręczeniem osoby trzeciej: ________________________________,
- zastawem na: ________________________________,
- innym sposobem: ________________________________,
- pożyczka bez zabezpieczenia.

2. Koszty zabezpieczenia ponosi: Pożyczkodawca / Pożyczkobiorca / Strony po połowie.

§ 6. Wypowiedzenie i wcześniejszy zwrot

1. Pożyczka na czas nieoznaczony — wypowiedzenie z zachowaniem 6-tygodniowego terminu wypowiedzenia (art. 723 KC).

2. Pożyczka na czas określony — wypowiedzenie tylko w przypadku rażącego naruszenia postanowień umowy.

3. Pożyczkodawca może wypowiedzieć umowę w przypadku:
   a) opóźnienia w płatności odsetek lub raty pożyczki o więcej niż 30 dni,
   b) pogorszenia sytuacji majątkowej Pożyczkobiorcy,
   c) naruszenia postanowień umowy przez Pożyczkobiorcę.

§ 7. Kary umowne

1. W przypadku opóźnienia w zwrocie pożyczki Pożyczkobiorca zapłaci karę umowną w wysokości ______________ zł za każdy dzień opóźnienia.

2. W przypadku niewykonania lub nienależytego wykonania zobowiązań, strona niewykonująca zapłaci karę umowną w wysokości ______________ zł.

§ 8. Postanowienia końcowe

1. Pożyczka, której wartość przekracza 1000 zł, wymaga zachowania formy dokumentowej (art. 720 § 2 KC) — np. pisemnej, e-mail, SMS utrwalający treść — dla celów dowodowych.

2. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 720-724 KC).

3. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

4. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Pożyczkodawcy.

5. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Grafik spłat (jeśli dotyczy), 2. Weksel in blanko (jeśli dotyczy).


POŻYCZKODAWCA: ____________________________    POŻYCZKOBIORCA: ____________________________""",

    "umowa_o_prace": """\
UMOWA O PRACĘ

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

PRACODAWCĄ:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, REGON: ______________________________________
reprezentowanym przez: ________________________________________________________________________________

a

PRACOWNIKIEM:
________________________________________________________________________________
(imię i nazwisko), zamieszkałym w ______________________________________________
PESEL: ____________________________, seria i nr dowodu tożsamości: ______________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Rodzaj umowy i stanowisko

1. Pracodawca zatrudnia Pracownika na podstawie umowy o pracę:
- na okres próbny: od dnia ________________ do dnia ________________ (art. 25 KP),
- na czas określony: od dnia ________________ do dnia ________________ (art. 25 KP),
- na czas nieokreślony: od dnia ________________ (art. 25 KP).

2. Stanowisko: ______________________________________________________________
________________________________________________________________________________

3. Miejsce wykonywania pracy: __________________________________________________

4. Wymiar czasu pracy: ________________________________________________________
(pełny etat / niepełny etat, dobowa norma: ______ h, tygodniowa norma: ______ h)

§ 2. Okres zatrudnienia

1. Okres zatrudnienia rozpoczyna się w dniu ________________.

2. Umowa na okres próbny nie może trwać dłużej niż 3 miesiące.

3. Umowa na czas określony może być zawarta na okres nie dłuższy niż 33 miesiące, a łącznie nie więcej niż 3 umowy na czas określony.

§ 3. Wynagrodzenie

1. Wynagrodzenie zasadnicze Pracownika wynosi ______________ zł brutto miesięcznie (słownie: ________________________________ złotych) (art. 86 KP).

2. Wynagrodzenie płatne jest do ______ dnia każdego miesiąca za miesiąc bieżący.

3. Wynagrodzenie zasadnicze obejmuje wynagrodzenie za czas pracy oraz inne składniki:
- dodatek za staż pracy: ______________ zł,
- dodatek za pracę w warunkach szkodliwych: ______________ zł,
- premia uznaniowa: ______________ zł,
- inne: ________________________________________________________________________

4. Wynagrodzenie za urlop wypoczynkowy i czas niezdolności do pracy na zasadach KP.

§ 4. Czas pracy i urlop

1. Czas pracy wynosi 8 godzin na dobę i średnio 40 godzin w tygodniu (art. 129 KP).

2. Rozpoczęcie i zakończenie pracy: ______________________________________________

3. Przerwy w pracy: ____________________________________________________________

4. Urlop wypoczynkowy w wymiarze: ______ dni roboczych (art. 154 KP).

5. Pracownik ma prawo do urlopu wypoczynkowego w wymiarze:
- 20 dni — przy stażu pracy do 10 lat,
- 26 dni — przy stażu pracy powyżej 10 lat.

§ 5. Obowiązki Pracownika

1. Wykonywać pracę sumiennie i starannie na określonym stanowisku (art. 100 KP).

2. Stosować się do poleceń przełożonych w zakresie wykonywania pracy.

3. Przestrzegać porządku pracy i regulaminu pracy.

4. Dbać o dobro firmy i zachować w tajemnicy informacje poufne.

5. Przestrzegać przepisów BHP i przeciwpożarowych.

6. Nie konkurować z Pracodawcą bez jego zgody (art. 101§1 KP).

§ 6. Obowiązki Pracodawcy

1. Zapewnić Pracownikowi warunki pracy zgodne z przepisami BHP (art. 207 KP).

2. Przydzielać pracę zgodnie z umową o pracę.

3. Wypłacać wynagrodzenie w terminie i w wysokości określonej w umowie.

4. Przestrzegać przepisów Kodeksu pracy.

5. Zapewnić Pracownikowi dostęp do danych osobowych i dokumentacji pracy.

§ 7. Zakaz konkurencji i klauzula poufności

1. Pracownik zobowiązuje się do zachowania w tajemnicy informacji poufnych Pracodawcy (art. 101§1 KP).

2. Zakaz konkurencji po ustaniu stosunku pracy może być ustalony w odrębnej umowie.

3. W przypadku naruszenia zakazu konkurencji, Pracownik zapłaci karę umowną w wysokości ______________ zł.

§ 8. Rozwiązanie umowy i wypowiedzenie

1. Umowa o pracę może być rozwiązana na podstawie:
- porozumienia stron (art. 30 KP),
- wypowiedzenia przez jedną ze stron (art. 31-52 KP),
- ze skutkiem natychmiastowym bez wypowiedzenia (art. 55 KP),
- śmierci Pracownika lub Pracodawcy.

2. Okresy wypowiedzenia:
- umowa na okres próbny: 3 dni robocze / 1 tydzień / 2 tygodnie,
- 6 miesięcy stażu pracy: 2 tygodnie,
- powyżej 6 miesięcy: 1 miesiąc,
- powyżej 3 lat: 3 miesiące.

3. Wypowiedzenie umowy wymaga formy pisemnej.

§ 9. Postanowienia końcowe

1. Pracownik oświadcza, że zapoznał się z regulaminem pracy i przepisami BHP.

2. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu pracy.

3. Zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Regulamin pracy, 2. Klauzula poufności (jeśli dotyczy), 3. Oświadczenie o zapoznaniu się z BHP.


PRACODAWCA: ____________________________    PRACOWNIK: ____________________________""",

    "umowa_licencyjna": """\
UMOWA LICENCYJNA

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

LICENCJODAWCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

LICENCJOBIORCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot licencji

1. Licencjodawca — uprawniony z tytułu autorskich praw majątkowych do utworu: ________________________________________________________________________________
(tytuł utworu, rodzaj utworu: tekst, grafika, oprogramowanie, baza danych, muzyka, film, inne)
udziela Licencjobiorcy licencji na korzystanie z Utworu.

2. Licencjodawca oświadcza, że jest wyłącznym właścicielem autorskich praw majątkowych do Utworu lub uzyskał zgodę właściciela na udzielenie licencji.

3. Licencjodawca oświadcza, że Utwór nie narusza praw osób trzecich.

§ 2. Zakres licencji

1. Rodzaj licencji:
- wyłączna (Licencjodawca nie może udzielać licencji innym osobom) (art. 67 pr.aut.),
- niewyłączna (Licencjodawca może udzielać licencji innym osobom).

2. Pola eksploatacji (art. 50 pr.aut.):
- utrwalanie i zwielokrotnianie Utworu,
- wprowadzanie do obrotu,
- najem lub dzierżawa egzemplarzy,
- publiczne wykonanie, wywieszenie lub odtwarzanie,
- nadawanie i reemitowanie,
- publiczne udostępnianie Utworu w internecie,
- modyfikacja i przerabianie Utworu,
- inne pola: ____________________________________________________________________

3. Terytorium: ________________________________________________________________
(Polska / świat / Europa / kraje: _____________________________________________)

4. Czas trwania licencji:
- na czas określony: od dnia ________________ do dnia ________________,
- na czas nieokreślony (do odwołania),
- na czas trwania praw autorskich.

§ 3. Wynagrodzenie

1. Wynagrodzenie za udzielenie licencji wynosi:
- kwotę ryczałtową: ______________ zł (słownie: ________________________________ złotych),
- stawkę procentową: ______% od przychodu ze sprzedaży Utworu,
- stawkę miesięczną: ______________ zł miesięcznie,
- inne: ________________________________________________________________________

2. Wynagrodzenie płatne jest:
- jednorazowo w dniu zawarcia umowy,
- miesięcznie do ______ dnia każdego miesiąca,
- kwartalnie do ______ dnia każdego kwartału,
- w inny sposób: ________________________________________________________________

3. Wynagrodzenie płatne przelewem na rachunek Licencjodawcy nr ________________.

§ 4. Prawa i obowiązki Licencjobiorcy

1. Korzystać z Utworu zgodnie z przeznaczeniem i zakresem licencji.

2. Oznaczać autorstwo Utworu zgodnie z art. 16 pr.aut. (prawa osobiste).

3. Nie dokonywać zmian w Utworze bez zgody Licencjodawchy (chyba że zmiany są niezbędne do korzystania z Utworu).

4. Nie udzielać sublicencji bez pisemnej zgody Licencjodawchy (art. 67 ust. 3 pr.aut.).

5. Płacić wynagrodzenie w terminie i w wysokości określonej w umowie.

6. Zapewnić ochronę Utworu przed nieuprawnionym dostępem osób trzecich.

§ 5. Sublicencje i przeniesienie licencji

1. Licencjobiorca może / nie może udzielać sublicencji osobom trzecim (art. 67 ust. 3 pr.aut.).

2. W przypadku udzielenia sublicencji, Licencjobiorca zobowiązany jest do poinformowania Licencjodawchy.

3. Licencjobiorca nie może przenosić licencji na osoby trzecie bez pisemnej zgody Licencjodawchy.

§ 6. Zapewnienia Licencjodawchy

1. Licencjodawca zapewnia, że przysługują mu autorskie prawa majątkowe do Utworu.

2. Licencjodawca zapewnia, że Utwór nie narusza praw osób trzecich.

3. Licencjodawca zapewnia, że Utwór jest oryginalny i nie został skopiowany z innych utworów.

4. W przypadku naruszenia praw osób trzecich, Licencjodawca zobowiązuje się do zwrotu wynagrodzenia i pokrycia szkód.

§ 7. Kary umowne

1. W przypadku naruszenia postanowień umowy przez Licencjobiorcę, Licencjodawca może żądać kary umownej w wysokości ______________ zł.

2. W przypadku opóźnienia w płatności wynagrodzenia, Licencjobiorca zapłaci odsetki w wysokości ______% za każdy dzień opóźnienia.

3. W przypadku udzielenia sublicencji bez zgody, Licencjobiorca zapłaci karę umowną w wysokości ______________ zł.

§ 8. Rozwiązanie i wypowiedzenie

1. Każda ze stron może wypowiedzieć umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

2. Wypowiedzenie umowy wymaga formy pisemnej.

3. W przypadku rozwiązania umowy, Licencjobiorca zobowiązany jest do zaprzestania korzystania z Utworu w terminie 14 dni.

§ 9. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy ustawy z dnia 4 lutego 1994 r. o prawie autorskim i prawach pokrewnych.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Licencjodawchy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.


LICENCJODAWCA: ____________________________    LICENCJOBIORCA: ____________________________""",

    "umowa_ramowa": """\
UMOWA RAMOWA O WSPÓŁPRACY

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

STRONĄ A:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

STRONĄ B:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot umowy ramowej

1. Strony określają ogólne zasady przyszłej współpracy w zakresie: ________________________________________________________________________________
(opis współpracy: dostawy, usługi, produkcja, itp.)

2. Umowa ramowa nie rodzi obowiązku złożenia zamówienia przez żadną ze stron (art. 3531 KC — swoboda umów).

3. Wszelkie zamówienia wykonawcze będą składane i realizowane na zasadach określonych w niniejszej Umowie Ramowej i w odrębnych zamówieniach.

4. Umowa ramowa nie wyłącza możliwości zawierania umów z podmiotami trzecimi.

§ 2. Zamówienia wykonawcze

1. Poszczególne dostawy lub usługi realizowane będą na podstawie odrębnych zamówień / zleceń wykonawczych.

2. Każde zamówienie wykonawcze będzie zawierać:
- przedmiot zamówienia,
- ilość,
- cenę jednostkową i łączną,
- termin wykonania,
- miejsce dostawy / wykonania,
- inne istotne warunki.

3. Zamówienie wykonawcze składane będzie pisemnie (e-mail, faks, papier).

4. Wykonawca zamówienia potwierdzi jego przyjęcie w terminie 3 dni roboczych.

§ 3. Ceny i warunki płatności

1. Ceny ustalane będą dla każdego zamówienia wykonawcze odrębnie, na podstawie cennika stanowiącego Załącznik nr 1 do Umowy Ramowej.

2. Ceny wyrażone będą w PLN netto / brutto.

3. Płatność za zamówienia wykonawcze nastąpi w terminie 14 dni od dostawy / wykonania i wystawienia faktury.

4. Płatność przelewem na rachunek Wykonawcy nr ________________.

§ 4. Jakość i odbiór

1. Towary / usługi dostarczane na podstawie zamówień wykonawczych muszą być zgodne ze specyfikacją techniczną i wymaganiami jakościowymi określonymi w zamówieniu.

2. Odbiór ilościowy i jakościowy nastąpi w momencie dostawy / wykonania.

3. Reklamacje ilościowe zgłaszane w terminie 7 dni od dostawy / wykonania.

4. Reklamacje jakościowe zgłaszane w terminie 14 dni od dostawy / wykonania.

5. Brak reklamacji w terminie oznacza uznanie dostawy / wykonania za zgodne z umową.

§ 5. Odpowiedzialność

1. Za niewykonanie lub nienależite wykonanie zamówienia wykonawczej, Wykonawca odpowiada na zasadach ogólnych (art. 471 KC).

2. W przypadku opóźnienia w wykonaniu zamówienia, Wykonawca zapłaci karę umowną w wysokości ______________ zł za każdy dzień zwłoki.

3. W przypadku wadliwej dostawy / wykonania, Wykonawca zobowiązuje się do naprawy / wymiany / usunięcia wad w terminie 14 dni.

§ 6. Poufność i tajemnica przedsiębiorstwa

1. Strony zobowiązują się do zachowania w tajemnicy wszelkich informacji uzyskanych w związku z Umową Ramową i zamówieniami wykonawczymi.

2. Obowiązek poufności obowiązuje przez okres trwania Umowy Ramowej oraz przez 3 lata po jej rozwiązaniu.

3. Naruszenie klauzuli poufności uprawnia stronę poszkodowaną do żądania kary umownej w wysokości ______________ zł.

§ 7. Czas trwania umowy

1. Umowa Ramowa zostaje zawarta na czas:
- określony od dnia ________________ do dnia ________________,
- nieokreślony (do odwołania z zachowaniem 3-miesięcznego okresu wypowiedzenia).

2. Umowa ulega przedłużeniu na kolejny okres, jeśli żadna ze stron nie wypowie jej na 3 miesiące przed upływem terminu.

§ 8. Rozwiązanie umowy

1. Każda ze stron może rozwiązać Umowę Ramową w przypadku:
- rażącego naruszenia postanowień przez drugą stronę,
- upadłości lub likwidacji drugiej strony,
- utraty zdolności do wykonywania zobowiązań.

2. Rozwiązanie Umowy Ramowej wymaga formy pisemnej.

3. Rozwiązanie Umowy Ramowej nie wpływa na zamówienia wykonawcze złożone przed rozwiązaniem.

§ 9. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Strony A.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Cennik, 2. Wzór zamówienia wykonawczego.


STRONA A: ____________________________    STRONA B: ____________________________""",

    "umowa_konsygnacji": """\
UMOWA KOMISU (tzw. konsygnacja handlowa)

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

KOMITENTEM:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

KOMISANTEM:
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot komisu

1. Komitent powierza Komisantowi do sprzedaży w lokalu położonym w: ________________________________________________________________________________
ruchomości (towary): ________________________________________________________________________
(nazwa, asortyment, ilość, cechy identyfikujące — art. 765 KC dotyczy wyłącznie rzeczy ruchomych, w zakresie działalności przedsiębiorstwa Komisanta) (art. 765 KC — umowa komisu).

2. Komisant działa we własnym imieniu, lecz na rachunek Komitenta (art. 765 KC).

3. Komisant zobowiązuje się do sprzedaży towarów zgodnie z zasadami współżycia społecznego i przepisami prawa.

§ 2. Miejsce i warunki sprzedaży

1. Sprzedaż towarów odbywa się w lokalu Komisanta położonym w: ________________________________________________________________________________

2. Sprzedaż może być prowadzona również przez internet / zamówienia telefoniczne.

3. Komisant zobowiązuje się do:
- prowadzenia sprzedaży w sposób profesjonalny,
- dbania o dobre imię marki i produktów Komitenta,
- przechowywania towarów w odpowiednich warunkach,
- informowania Komitenta o przebiegu sprzedaży.

§ 3. Cena i rozliczenie

1. Cena minimalna towarów: ______________ zł / za szt. / kg / m².

2. Komisant może obniżyć cenę towarów za zgodą Komitenta.

3. Prowizja Komisanta:
- wynosi ______% ceny sprzedaży towarów,
- płatna w terminie 14 dni od zakończenia miesiąca rozliczeniowego.

4. Rozliczenie sprzedaży następuje raportem miesięcznym składanym przez Komisanta w terminie 7 dni od zakończenia miesiąca.

5. Komisant przekaże Komitentowi środki ze sprzedaży pomniejszone o prowizję przelewem na rachunek Komitenta nr ________________.

§ 4. Własność i ryzyko

1. Własność towaru pozostaje przy Komiticie do chwili sprzedaży osobie trzeciej (art. 765 KC).

2. Ryzyko przypadkowej utraty lub uszkodzenia towaru do momentu sprzedaży ponosi: Komisant / Komitent.

3. Komisant zobowiązuje się do ubezpieczenia towarów na kwotę ______________ zł.

§ 5. Termin i zwrot niesprzedanego towaru

1. Towar niesprzedany do dnia ________________ podlega zwrotowi na koszt: Komisanta / Komitenta.

2. Zwrot towaru nastąpi w terminie 14 dni od zakończenia umowy lub wcześniejszego wezwania Komitenta.

3. Towar zostanie zwrócony w stanie niepogorszonym, z wyłączeniem normalnego zużycia.

§ 6. Obowiązki Komisanta

1. Przechowywać towar z należytą starannością w odpowiednich warunkach.

2. Ubezpieczyć towar od wszelkich ryzyk na kwotę nie mniejszą niż wartość towarów.

3. Prowadzić ewidencję towarów i sprzedaży.

4. Informować Komitenta o:
- stanie magazynowym towarów,
- przebiegu sprzedaży,
- zgłoszeniach i reklamacjach klientów,
- zmianach rynkowych mogących wpłynąć na sprzedaż.

5. Nie udostępniać osobom trzecim informacji o relacjach handlowych z Komitentem.

§ 7. Odpowiedzialność Komisanta

1. Komisant odpowiada na zasadach art. 771-773 KC.

2. Komisant ponosi odpowiedzialność za:
- utratę lub uszkodzenie towarów,
- niezgodność sprzedaży z umową,
- naruszenie praw osób trzecich.

3. Odpowiedzialność del credere (solidarna odpowiedzialność za zapłatę przez klientów): obowiązuje / nie obowiązuje.

4. W przypadku naruszenia postanowień umowy, Komisant zapłaci karę umowną w wysokości ______________ zł.

§ 8. Gwarancja i reklamacje

1. Komisant zobowiązuje się do przyjmowania reklamacji od klientów w imieniu Komitenta.

2. Komisant przekaże Komitentowi dokumentację reklamacyjną w terminie 7 dni od otrzymania reklamacji.

3. Koszty gwarancji i reklamacji ponosi: Komitent / Komisant.

§ 9. Czas trwania i rozwiązanie umowy

1. Umowa zostaje zawarta na czas:
- określony od dnia ________________ do dnia ________________,
- nieokreślony (do odwołania z zachowaniem 3-miesięcznego okresu wypowiedzenia).

2. Każda ze stron może rozwiązać umowę w przypadku rażącego naruszenia postanowień przez drugą stronę.

3. Rozwiązanie umowy wymaga formy pisemnej.

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 765-773 KC).

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Komitenta.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.


KOMITENT: ____________________________    KOMISANT: ____________________________""",

    "umowa_ubezpieczenia": """\
UMOWA UBEZPIECZENIA

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

UBEZPIECZYCIELEM:
________________________________________________________________________________
(nazwa towarzystwa ubezpieczeniowego), z siedzibą w ________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

UBEZPIECZAJĄCYM:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

na rzecz UBEZPIECZONEGO:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot i zakres ubezpieczenia

1. Przedmiot ubezpieczenia: ______________________________________________________
(majątek / życie / zdrowie / odpowiedzialność cywilna / podróż / inne)

2. Zakres ryzyk objętych ubezpieczeniem:
________________________________________________________________________________
(pożar, powódź, kradzież, wypadek, choroba, odpowiedzialność cywilna, itp.)

3. Ubezpieczyciel zobowiązuje się do wypłaty odszkodowania w przypadku zajścia zdarzenia objętego ubezpieczeniem, zgodnie z warunkami ogólnymi ubezpieczenia (OWU) stanowiącymi Załącznik nr 1.

4. Ubezpieczenie obejmuje / nie obejmuje: ________________________________________________________________________________

§ 2. Suma ubezpieczenia i suma gwarancyjna

1. Suma ubezpieczenia wynosi ______________ zł (słownie: ________________________________ złotych).

2. Suma gwarancyjna (maksymalna wysokość odszkodowania) wynosi ______________ zł.

3. Suma ubezpieczenia jest różna od sumy gwarancyjnej / równa sumie gwarancyjnej.

§ 3. Składka ubezpieczeniowa

1. Składka ubezpieczeniowa wynosi ______________ zł (słownie: ________________________________ złotych).

2. Składka płatna jest:
- jednorazowo w dniu zawarcia umowy,
- w ratach miesięcznych / kwartalnych / półrocznych / rocznych po ______________ zł.

3. Składka płatna przelewem na rachunek Ubezpieczyciela nr ________________.

4. Składka ulega waloryzacji zgodnie z OWU.

§ 4. Okres ubezpieczenia

1. Okres ubezpieczenia rozpoczyna się w dniu ________________ godz. ______ i trwa do dnia ________________ godz. ______.

2. Ubezpieczenie obowiązuje na terenie: ________________________________________________________________________________
(Polska / Europa / świat)

3. Ubezpieczenie obowiązuje 24 godziny na dobę / w godzinach: ______ — ______.

§ 5. Obowiązki Ubezpieczającego i Ubezpieczonego

1. Ubezpieczający zobowiązuje się do:
- dokładnej deklaracji ryzyka przed zawarciem umowy (art. 815 KC),
- płacenia składek w terminie,
- powiadamiania Ubezpieczyciela o zmianach mogących wpłynąć na ryzyko,
- przestrzegania obowiązków prewencyjnych zgodnie z OWU.

2. Ubezpieczony zobowiązuje się do:
- przestrzegania przepisów BHP i przeciwpożarowych,
- stosowania się do zaleceń Ubezpieczyciela,
- nieudostępniania osobom trzecim informacji o ubezpieczeniu.

3. Obowiązki prewencyjne: ________________________________________________________________________________
(instalacja alarmu, gaśnice, apteczka, itp.)

§ 6. Zgłoszenie szkody i odszkodowanie

1. Ubezpieczony zobowiązuje się do zgłoszenia szkody Ubezpieczycielowi w terminie 7 dni od dnia jej powstania (lub w innym terminie określonym w OWU).

2. Zgłoszenie szkody powinno zawierać:
- opis zdarzenia i jego przyczyn,
- wysokość szkody,
- dokumentację fotograficzną,
- dokumenty potwierdzające szkodę (faktury, rachunki, zaświadczenia).

3. Ubezpieczyciel zobowiązuje się do ustalenia wysokości odszkodowania w terminie 30 dni od zgłoszenia szkody (art. 817 KC).

4. Wypłata odszkodowania nastąpi w terminie 14 dni od ustalenia jej wysokości.

5. Odszkodowanie wypłacane jest przelewem na rachunek Ubezpieczonego nr ________________.

§ 7. Wyłączenia odpowiedzialności

1. Ubezpieczyciel nie wypłaca odszkodowania w przypadku:
- celowego spowodowania szkody przez Ubezpieczonego,
- szkody powstałej w wyniku działań wojennych,
- szkody powstałej w wyniku promieniowania jądrowego,
- szkody powstałej w wyniku alkoholu lub narkotyków,
- innych wyłaćzeń określonych w OWU.

2. Pełny katalog wyłaćzeń określony jest w OWU stanowiącym Załącznik nr 1.

§ 8. Rozwiązanie i odstąpienie

1. Ubezpieczający ma prawo odstąpić od umowy w terminie 30 dni od jej zawarcia (art. 812 KC).

2. Ubezpieczyciel może wypowiedzieć umowę w przypadku:
- niezapłacenia składki w terminie,
- zatajeniu informacji o ryzyku,
- rażącego naruszenia obowiązków prewencyjnych.

3. Rozwiązanie umowy wymaga formy pisemnej.

4. W przypadku rozwiązania umowy, Ubezpieczyciel zwraca niewykorzystaną część składki.

§ 9. Postanowienia końcowe

1. Warunki Ogólne Ubezpieczenia (OWU) nr ________________ stanowią integralną część umowy.

2. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 805-834 KC) oraz ustawy o działalności ubezpieczeniowej.

3. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

4. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Ubezpieczyciela.

5. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Warunki Ogólne Ubezpieczenia (OWU), 2. Formularz zgłoszenia szkody.


UBEZPIECZYCIEL: ____________________________    UBEZPIECZAJĄCY: ____________________________""",

    "umowa_o_zachowanie_poufnosci": """\
UMOWA O ZACHOWANIE POUFNOŚCI (NDA — Non-Disclosure Agreement)

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

STRONĄ UJAWNIAJĄCĄ:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

STRONĄ OTRZYMUJĄCĄ:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot i cel umowy

1. Strony niniejszą umową regulują zasady ochrony informacji poufnych, które Strona Ujawniająca przekażę Stronie Otrzymującej w związku z: ________________________________________________________________________________
(cele: negocjacje, współpraca, analiza, testowanie, inne)

2. Informacje Poufne mogą być przekazywane ustnie, pisemnie, w formie elektronicznej lub w jakiejkolwiek innej formie.

3. Strona Otrzymująca zobowiązuje się do ochrony Informacji Poufnych z należytą starannością, nie mniejszą niż stosuje do własnych informacji poufnych.

§ 2. Definicja Informacji Poufnych

1. Informacje Poufne oznaczają wszelkie informacje techniczne, handlowe, finansowe, biznesowe, prawne, know-how, patenty, wzorce, oprogramowanie, dane, raporty, analizy, plany, projekty, specyfikacje, prototypy, próbki, przekazane przez Stronę Ujawniającą Stronie Otrzymującej, niezależnie od formy przekazu.

2. Informacje Poufne oznaczone są przez Stronę Ujawniającą jako „Poufne" lub „Confidential" lub w podobny sposób.

3. Informacje przekazywane ustnie uznaje się za Poufne, jeżeli Strona Ujawniająca potwierdzi ich poufność pisemnie w terminie 14 dni.

§ 3. Zakres obowiązku poufności

1. Strona Otrzymująca zobowiązuje się:
- zachować Informacje Poufne w ścisłej tajemnicy,
- nie ujawniać Informacji Poufnych osobom trzecim bez uprzedniej pisemnej zgody Strony Ujawniającej,
- wykorzystywać Informacje Poufne wyłącznie w celu określonym w § 1 ust. 1,
- ograniczyć dostęp do Informacji Poufnych wyłącznie do osób, które muszą mieć dostęp do tych informacji w celu realizacji umowy,
- nie kopiować Informacji Poufnych bez zgody Strony Ujawniającej,
- chronić Informacje Poufne z co najmniej taką samą starannością, jaką chroni własne informacje poufne, nie mniej jednak z należytą starannością.

2. Strona Otrzymująca ponosi odpowiedzialność za działania i zaniechania osób, którym ujawniła Informacje Poufne.

§ 4. Wyłączenia z obowiązku poufności

1. Obowiązek poufności nie dotyczy informacji, które:
- były publicznie dostępne w momencie ujawnienia lub stały się publicznie dostępne bez naruszenia umowy,
- były już znane Stronie Otrzymującej przed ich ujawnieniem, co Strona Otrzymująca może udowodnić,
- zostały otrzymane od osoby trzeciej, która nie była zobowiązana do zachowania poufności,
- zostały niezależnie opracowane przez Stronę Otrzymującą bez wykorzystania Informacji Poufnych,
- muszą być ujawnione na podstawie przepisów prawa, decyzji sądu lub organu administracji publicznej.

2. W przypadku ujawnienia informacji na podstawie przepisów prawa, Strona Otrzymująca zobowiązana jest do niezwłocznego powiadomienia Strony Ujawniającej.

§ 5. Okres poufności

1. Obowiązek poufności obowiązuje przez okres trwania niniejszej Umowy oraz przez 5 (pięć) lat po jej rozwiązaniu lub wygaśnięciu.

2. Po upływie okresu poufności, Strona Otrzymująca zwróci lub zniszczy wszystkie nośniki Informacji Poufnych na żądanie Strony Ujawniającej.

§ 6. Zwrot i zniszczenie materiałów

1. Na żądanie Strony Ujawniającej, nie później niż w terminie 14 dni od rozwiązania umowy, Strona Otrzymująca zwróci lub zniszczy wszystkie dokumenty, nośniki i materiały zawierające Informacje Poufne.

2. Strona Otrzymująca potwierdzi na piśmie dokonanie zwrotu lub zniszczenia materiałów.

3. Strona Otrzymująca może zachować jedną kopię Informacji Poufnych wyłącznie w celach dowodowych i archiwalnych.

§ 7. Kary umowne

1. W przypadku naruszenia postanowień niniejszej Umowy przez Stronę Otrzymującą, Strona Ujawniająca ma prawo żądać zapłaty kary umownej w wysokości ______________ zł (słownie: ________________________________ złotych) za każde naruszenie, niezależnie od prawa do żądania odszkodowania przenoszącego wysokość kary na zasadach ogólnych (art. 484 KC).

2. Strona Otrzymująca ponosi odpowiedzialność za szkody wyrządzone Stronie Ujawniającej naruszeniem poufności.

§ 8. Postanowienia końcowe

1. Informacje Poufne stanowią tajemnicę przedsiębiorstwa w rozumieniu art. 11 ust. 2 ustawy z dnia 21 czerwca 1993 r. o zwalczaniu nieuczciwej konkurencji.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach nieuregulowanych mają zastosowanie przepisy Kodeksu cywilnego oraz ustawy o zwalczaniu nieuczciwej konkurencji.

4. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Strony Ujawniającej.

5. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.


STRONA UJAWNIAJĄCA: ____________________________    STRONA OTRZYMUJĄCA: ____________________________""",

    "umowa_przechowania": """\
UMOWA PRZECHOWANIA

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

PRZECHOWAWCĄ:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

a

SKŁADAJĄCYM:
________________________________________________________________________________
(nazwa firmy / imię i nazwisko), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot przechowania

1. Składający oddaje, a Przechowawca przyjmuje na przechowanie:
________________________________________________________________________________
(dokładny opis rzeczy: nazwa, ilość, waga, wymiary, kolor, cechy identyfikujące, stan)
o wartości szacunkowej ______________ zł (art. 835 KC).

2. Składający oświadcza, że rzecz stanowi jego własność i jest wolna od obciążeń na rzecz osób trzecich.

3. Rzecz zostaje przekazana Przechowawcy w dniu ________________ w stanie: ________________________________________________________________________________

§ 2. Miejsce i sposób przechowania

1. Przechowanie odbywa się w: ________________________________________________________________________________
(adres miejsca przechowania)

2. Warunki przechowania:
- temperatura: ________________,
- wilgotność: ________________,
- zabezpieczenia: ________________________________________________________________,
- inne: ________________________________________________________________________

3. Przechowawca nie może bez zgody Składającego oddać rzeczy na przechowanie osobie trzeciej (art. 839 KC).

4. Przechowawca zobowiązuje się do zapewnienia ochrony rzeczy przed kradzieżą, zniszczeniem i zawaleniem.

§ 3. Czas przechowania

1. Przechowanie trwa od dnia ________________ do dnia ________________ / na czas nieoznaczony.

2. Wydanie rzeczy nastąpi na każde żądanie Składającego (art. 844 KC).

3. W przypadku przechowania na czas nieokreślony, wydanie nastąpi po uprzednim powiadomieniu z zachowaniem 7-dniowego terminu.

§ 4. Wynagrodzenie i zwrot wydatków

1. Wynagrodzenie za przechowanie:
- wynosi ______________ zł (słownie: ________________________________ złotych) miesięcznie / za cały okres,
- płatne przelewem na rachunek Przechowawcy nr ________________,
- termin płatności: do ______ dnia każdego miesiąca / w dniu odbioru rzeczy.

2. Składający zobowiązuje się do zwrotu Przechowawcy wydatków koniecznych poniesionych w związku z przechowaniem (art. 842 KC).

3. Przechowawca nie ma prawa do zatrzymania rzeczy za wynagrodzenie po upływie 6 miesięcy od terminu wydania (chyba że rzecz jest przechowywana w zakładzie przechowalniczym).

§ 5. Obowiązki Przechowawcy

1. Przechowywać rzecz z należytą starannością (art. 840 KC).

2. Nie używać rzeczy bez zgody Składającego.

3. Informować Składającego o zagrożeniach dla rzeczy i podjąć działania w celu jej ochrony.

4. Nie przenosić rzeczy do innego miejsca przechowania bez zgody Składającego.

5. Prowadzić ewidencję przechowywanej rzeczy.

§ 6. Zwrot rzeczy

1. Przechowawca wyda rzecz Składającemu / osobie wskazanej w dniu ________________.

2. Zwrot rzeczy nastąpi w miejscu przechowania / w miejscu wskazanym przez Składającego.

3. Rzecz zostanie zwrócona w stanie niepogorszonym, z wyłączeniem normalnego zużycia.

4. Przechowawca wyda rzecz wraz z pożytkami uzyskanymi z rzeczy (art. 844 KC).

5. W przypadku opóźnienia w wydaniu rzeczy, Przechowawca zapłaci karę umowną w wysokości ______________ zł za każdy dzień opóźnienia.

§ 7. Odpowiedzialność Przechowawcy

1. Przechowawca odpowiada za szkodę wynikłą z utraty, uszkodzenia lub zniszczenia rzeczy na zasadach art. 471 KC.

2. Ciężar dowodu tego, że szkoda powstała z przyczyn niezależnych od Przechowawcy, spoczywa na Przechowawcy (art. 840 KC analogia).

3. W przypadku utraty lub uszkodzenia rzeczy, Przechowawca zapłaci Składającemu odszkodowanie w wysokości ______________ zł (wartość rzeczy) oraz ewentualne szkody na zasadach ogólnych.

§ 8. Ubezpieczenie rzeczy

1. Składający może ubezpieczyć rzecz na własny koszt.

2. Przechowawca zobowiązuje się do współpracy w zakresie niezbędnym do ubezpieczenia rzeczy.

3. W przypadku szkody, odszkodowanie z ubezpieczenia zostanie przekazane Składającemu.

§ 9. Zatrzymanie rzeczy (opcjonalnie — prawo zastawu)

1. Strony mogą zastrzec umowne prawo zastawu na rzeczy do zabezpieczenia roszczeń z przechowania (art. 306 KC — zastaw umowny). Ustawowe prawo zastawu z art. 852 KC dotyczy wyłącznie przechowania hotelowego (art. 850–852 KC), nie ogólnego przechowania (art. 835–845 KC).

2. Umowny zastaw — jeżeli zastrzeżony — obejmuje wynagrodzenie za przechowanie, wydatki konieczne i odszkodowanie za szkody.

3. Do ustanowienia zastawu umownego wymagana jest odrębna umowa i — dla skuteczności wobec osób trzecich — wydanie rzeczy lub wpis do rejestru zastawów (art. 307 KC).

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 835–845 KC — przechowanie).

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Przechowawcy.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.


PRZECHOWAWCA: ____________________________    SKŁADAJĄCY: ____________________________""",

    "umowa_kredytu": """\
UMOWA KREDYTU

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

BANKIEM / KREDODAWCĄ:
________________________________________________________________________________
(nazwa banku / instytucji finansowej), z siedzibą w ________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

KREDYTOBIORCĄ:
________________________________________________________________________________
(imię i nazwisko / nazwa firmy), zamieszkałym/z siedzibą w ________________________
NIP / PESEL: ____________________________, reprezentowanym przez: ________________
________________________________________________________________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Rodzaj, kwota i cel kredytu

1. Bank udziela Kredytobiorcy kredytu w kwocie ______________ zł (słownie: ________________________________ złotych) (art. 69 ust. 1 Prawa bankowego).

2. Rodzaj kredytu:
- kredyt gotówkowy,
- kredyt hipoteczny,
- kredyt konsumencki,
- kredyt obrotowy,
- kredyt inwestycyjny,
- inny: ________________________________________________________________________

3. Cel kredytu: __________________________________________________________________
(zakup nieruchomości, konsumpcja, inwestycja, refinansowanie, obrót, itp.)

4. Kredyt udzielony został na okres od dnia ________________ do dnia ________________.

§ 2. Oprocentowanie

1. Oprocentowanie kredytu:
- stałe: ______% w skali roku,
- zmienne: WIBOR 3M/6M + ______% marży w skali roku.

2. Oprocentowanie zmienia się automatycznie wraz ze zmianą stawki referencyjnej.

3. Prowizja przygotowawcza: ______________ zł (______% kwoty kredytu).

4. Odsetki maksymalne nie mogą przekroczyć stawki ustawowej na podstawie art. 359 KC.

5. Odsetki za opóźnienie wynoszą ______% w skali roku (art. 481 KC).

§ 3. Warunki uruchomienia i wypłata

1. Kredyt zostanie wypłacony po spełnieniu następujących warunków:
- zabezpieczenia spłaty kredytu (hipoteka, poręczenie, weksel, zastaw),
- dokumenty: __________________________________________________________________,
- ubezpieczenie przedmiotu zabezpieczenia,
- inne: ________________________________________________________________________

2. Wypłata kredytu nastąpi przelewem na rachunek Kredytobiorcy nr ________________.

3. Kredyt wypłacony zostanie w ciągu 14 dni od spełnienia warunków.

§ 4. Spłata kredytu

1. Spłata kredytu następuje w ratach: równych / malejących miesięcznych.

2. Wysokość raty miesięcznej wynosi ______________ zł.

3. Termin płatności raty: do ______ dnia każdego miesiąca.

4. Harmonogram spłat stanowi Załącznik nr 1 do Umowy.

5. Kolejność zaliczenia wpłat:
- koszty egzekucyjne,
- odsetki za opóźnienie,
- odsetki bieżące,
- kapitał kredytu.

6. Kredytobiorca może dokonać wcześniejszej spłaty kredytu w całości lub częściach z zachowaniem 30-dniowego terminu wypowiedzenia.

7. W przypadku wcześniejszej spłaty, Bank może naliczyć opłatę w wysokości ______% wcześniejszej kwoty spłaty.

§ 5. Zabezpieczenia spłaty kredytu

1. Spłata kredytu zabezpieczona jest:
- hipoteką na nieruchomości: __________________________________________________
(zapis w księdze wieczystej nr ________________, Sąd Rejonowy ________________),
- poręczeniem osoby trzeciej: __________________________________________________,
- wekslem in blanko z klauzulą „nie za przewartościowanie",
- zastawem na: ______________________________________________________________,
- ubezpieczeniem na wypadek śmierci / utraty zdolności do pracy,
- przelewem wynagrodzenia na rachunek Kredytobiorcy w Banku.

2. Koszty zabezpieczenia ponosi: Kredytobiorca / Strony po połowie.

3. Bank ma prawo zażadać dodatkowego zabezpieczenia w przypadku pogorszenia sytuacji majątkowej Kredytobiorcy.

§ 6. Obowiązki Kredytobiorcy

1. Kredytobiorca zobowiązuje się do:
- spłacania rat kredytu w terminie,
- informowania Banku o zmianie danych osobowych i sytuacji majątkowej,
- niezaciągania innych zobowiązań powyżej ______________ zł bez zgody Banku,
- informowania Banku o zdarzeniach mogących wpłynąć na zdolność spłaty,
- ubezpieczenia przedmiotu zabezpieczenia,
- utrzymywania przedmiotu zabezpieczenia w dobrym stanie.

2. Kredytobiorca zobowiązuje się do nieudostępniania osobom trzecim informacji o warunkach kredytu.

§ 7. Prawa Banku

1. Bank ma prawo do:
- kontroli wykorzystania kredytu zgodnie z celem,
- kontroli stanu przedmiotu zabezpieczenia,
- zażądania wcześniejszej spłaty kredytu w przypadku naruszenia postanowień umowy,
- wypowiedzenia umowy w przypadku rażącego naruszenia postanowień przez Kredytobiorcę.

2. Bank ma prawo do zmiany oprocentowania w przypadku zmiany stawki referencyjnej.

§ 8. Wypowiedzenie umowy

1. Bank może wypowiedzieć umowę na zasadach art. 75 Prawa bankowego w przypadku:
- opóźnienia w spłacie raty o więcej niż 30 dni,
- rażącego naruszenia postanowień umowy,
- utraty zdolności kredytowej / pogorszenia sytuacji majątkowej Kredytobiorcy zagrażającego spłacie,
- pogorszenia stanu zabezpieczenia kredytu.

2. Wypowiedzenie umowy wymaga formy pisemnej.

3. W przypadku wypowiedzenia umowy, Kredytobiorca zobowiązany jest do niezwłocznej spłaty całego zadłużenia wraz z odsetkami.

§ 9. Ubezpieczenie

1. Kredytobiorca zobowiązuje się do ubezpieczenia przedmiotu zabezpieczenia na kwotę nie mniejszą niż ______________ zł od ryzyk: ________________________________________________________________________________

2. Polisa ubezpieczeniowa zostanie cedowana na Bank.

3. Koszty ubezpieczenia ponosi Kredytobiorca.

§ 10. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Prawa bankowego oraz Kodeksu cywilnego (art. 720-724 KC).

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Banku.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Harmonogram spłat, 2. Oświadczenie o zabezpieczeniu, 3. Regulamin kredytowania.


BANK/KREDODAWCA: ____________________________    KREDYTOBIORCA: ____________________________""",

    "umowa_leasingu": """\
UMOWA LEASINGU

zawarta w dniu ________________ r. w ________________________________ pomiędzy:

FINANSUJĄCYM (LEASINGODAWCĄ):
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________
________________________________________________________________________________

a

KORZYSTAJĄCYM (LEASINGOBIORCĄ):
________________________________________________________________________________
(nazwa firmy), z siedzibą w ____________________________________________________
NIP: ____________________________, reprezentowanym przez: ________________________

zwanymi dalej łącznie „Stronami", a każdą z osobna „Stroną".

§ 1. Przedmiot leasingu

1. Finansujący oddaje Korzystającemu do używania przedmiotu leasingu:
- pojazd mechaniczny: marka ________________, model ________________, numer VIN ________________, numer rejestracyjny ________________, rok produkcji ________________, pojemność silnika ________________ cm³,
- maszyna / urządzenie: ________________________________________________________,
- nieruchomość: ______________________________________________________________,
(dalej jako „Przedmiot leasingu") (art. 7091 KC).

2. Korzystający oświadcza, że zapoznał się z technicznym i prawnym stanem Przedmiotu leasingu i nie wnosi zastrzeżeń.

3. Odbiór Przedmiotu leasingu nastąpi protokołem zdawczo-odbiorczym podpisanym przez obie Strony.

§ 2. Okres leasingu

1. Przedmiot leasingu zostaje oddany do używania na okres od dnia ________________ do dnia ________________.

2. Przedłużenie okresu leasingu wymaga pisemnego aneksu.

3. Korzystający może zwrócić Przedmiot leasingu przed upływem okresu na zasadach art. 70913-70915 KC.

§ 3. Opłaty leasingowe

1. Opłata wstępna (czynsz inicjalny) wynosi ______________ zł (słownie: ________________________________ złotych), płatna w dniu zawarcia umowy / w terminie 14 dni od zawarcia umowy.

2. Raty leasingowe miesięczne wynoszą ______________ zł (łącznie ______ rat), płatne do ______ dnia każdego miesiąca.

3. Raty płatne są przelewem na rachunek Finansującego nr ________________.

4. Opłata końcowa (wykup) wynosi ______________ zł / ______% wartości Przedmiotu leasingu.

5. Wartość całkowita leasingu (opłata wstępna + raty + opłata końcowa) wynosi ______________ zł.

6. Opłaty podlegają waloryzacji zgodnie z wskaźnikiem inflacji GUS, nie więcej niż o 5% rocznie.

§ 4. Obowiązki Korzystającego

1. Używać Przedmiotu leasingu zgodnie z przeznaczeniem i zasadami współżycia społecznego.

2. Dokonywać napraw i ponosić wszystkie koszty związane z eksploatacją Przedmiotu leasingu (paliwo, serwis, części zamienne, przeglądy, opłaty drogowe).

3. Ubezpieczyć Przedmiot leasingu w zakresie: OC, AC, NNW, assistance, inny: ________________________________________________________________________________

4. Nie dokonywać zmian, przebudowań ani ulepszeń Przedmiotu leasingu bez pisemnej zgody Finansującego.

5. Przechowywać Przedmiot leasingu w bezpiecznym miejscu i chronić przed kradzieżą i zniszczeniem.

6. Nie oddawać Przedmiotu leasingu w podnajem ani do bezpłatnego używania bez zgody Finansującego.

7. Umożliwiać Finansującemu dostęp do Przedmiotu leasingu w celu kontroli stanu i przeznaczenia.

§ 5. Własność i ryzyko

1. Własność Przedmiotu leasingu pozostaje przy Finansującym do przeniesienia na Korzystającego po wykupie (art. 7092 KC).

2. Ryzyko przypadkowej utraty, zniszczenia lub uszkodzenia Przedmiotu leasingu obciąża Korzystającego od chwili wydania Przedmiotu (art. 7098 KC).

3. Korzystający zobowiązuje się do niezwłocznego powiadomienia Finansującego o każdym zdarzeniu mogącym wpłynąć na stan Przedmiotu leasingu.

§ 6. Ubezpieczenie

1. Korzystający zobowiązuje się do ubezpieczenia Przedmiotu leasingu na kwotę nie mniejszą niż wartość rynkowa Przedmiotu od ryzyk: ________________________________________________________________________________

2. Polisa ubezpieczeniowa zostaje cedowana na Finansującego jako beneficjenta.

3. W przypadku szkody, odszkodowanie z polisy zostanie przekazane Finansującemu.

4. Korzystający ponosi wszystkie koszty ubezpieczenia.

§ 7. Naprawy i przeglądy

1. Wszystkie naprawy i przeglądy techniczne Przedmiotu leasingu dokonywane są na koszt Korzystającego.

2. Naprawy gwarancyjne dokonywane są w autoryzowanych stacjach serwisowych.

3. Dokumentację napraw i przeglądów Korzystający przechowuje i udostępnia Finansującemu na żądanie.

§ 8. Zwrot Przedmiotu leasingu

1. Po zakończeniu okresu leasingu, Korzystający zwróci Przedmiot leasingu Finansującemu w stanie niepogorszonym, z wyłączeniem normalnego zużycia.

2. Zwrot nastąpi w miejscu wskazanym przez Finansującego.

3. Korzystający może dokonać wykupu Przedmiotu leasingu po zakończeniu okresu leasingu za opłatą określoną w § 3 ust. 4.

4. Przeniesienie własności nastąpi po zapłacie opłaty końcowej.

§ 9. Rozwiązanie umowy i skutki

1. Każda ze stron może rozwiązać umowę na zasadach art. 70913-70915 KC.

2. Finansujący może rozwiązać umowę w przypadku:
- opóźnienia w płatności raty o więcej niż 30 dni,
- rażącego naruszenia postanowień umowy przez Korzystającego,
- utraty zdolności finansowej / pogorszenia sytuacji majątkowej Korzystającego zagrażającego spłacie,
- zniszczenia lub utraty Przedmiotu leasingu.

3. W przypadku rozwiązania umowy, Korzystający zapłaci Finansującemu:
- wszystkie zaległe raty i odsetki,
- opłatę za rozwiązanie umowy w wysokości ______________ zł,
- odszkodowanie za ewentualne szkody na Przedmiocie leasingu.

4. Rozliczenie nastąpi na zasadach art. 70914 KC.

§ 10. Kary umowne

1. W przypadku opóźnienia w płatności raty, Korzystający zapłaci karę umowną w wysokości ______________ zł za każdy dzień opóźnienia.

2. W przypadku naruszenia postanowień umowy, Korzystający zapłaci karę umowną w wysokości ______________ zł.

§ 11. Postanowienia końcowe

1. W sprawach nieuregulowanych niniejszą umową mają zastosowanie przepisy Kodeksu cywilnego (art. 7091-70918 KC) oraz ustawy o zwalczaniu nieuczciwej konkurencji.

2. Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.

3. W sprawach spornych właściwy jest sąd powszechny właściwy dla siedziby Finansującego.

4. Umowę sporządzono w dwóch jednobrzmiących egzemplarzach.

Załączniki: 1. Protokół zdawczo-odbiorczy, 2. Harmonogram spłat, 3. Regulamin leasingu.


FINANSUJĄCY/LEASINGODAWCA: ____________________________    KORZYSTAJĄCY/LEASINGOBIORCA: ____________________________""",

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


# Expanded aliases: maps terms the model might infer to the right template.
# Legal domain knowledge that generic embeddings (BGE-M3) don't capture:
#   - art. 750 KC: zlecenie provisions apply to service contracts
#   - art. 750 KC: also applies to agency contracts
#   - etc.
ALIASES: dict[str, str] = {
    # umowa zlecenia (art. 734-751 KC) + art. 750 KC (services apply zlecenie rules)
    "umowa_zlecenia": "umowa_zlecenia",
    "umowa_o_swiadczenie_uslug": "umowa_zlecenia",
    "umowa_uslug": "umowa_zlecenia",
    "umowa_o_uslugi": "umowa_zlecenia",
    "kontrakt_na_uslugi": "umowa_zlecenia",
    "umowa_agencyjna": "umowa_zlecenia",
    "uslugi": "umowa_zlecenia",
    "zlecenie": "umowa_zlecenia",
    "zlecaj": "umowa_zlecenia",
    "swiadczenie_uslug": "umowa_zlecenia",
    
    # umowa o dzieło (art. 627-646 KC)
    "umowa_o_dzielo": "umowa_o_dzielo",
    "umowa_dzielo": "umowa_o_dzielo",
    "dzielo": "umowa_o_dzielo",
    "utwor": "umowa_o_dzielo",
    "prawa_autorskie": "umowa_o_dzielo",
    
    # umowa o pracę (art. 22-185 KP)
    "umowa_o_prace": "umowa_o_prace",
    "umowa_pracy": "umowa_o_prace",
    "kontrakt_pracy": "umowa_o_prace",
    "praca": "umowa_o_prace",
    "zatrudnienie": "umowa_o_prace",
    "pracownik": "umowa_o_prace",
    
    # umowa najmu
    "umowa_najmu": "umowa_najmu",
    "najem": "umowa_najmu",
    "wynajem": "umowa_najmu",
    "wynajmu": "umowa_najmu",
    
    # umowa sprzedaży
    "umowa_sprzedazy": "umowa_sprzedazy",
    "sprzedaz": "umowa_sprzedazy",
    "kupno": "umowa_sprzedazy",
    "kupno_sprzedaz": "umowa_sprzedazy",
    
    # umowa darowizny
    "umowa_darowizny": "umowa_darowizny",
    "darowizna": "umowa_darowizny",
    "daruj": "umowa_darowizny",
    
    # umowa dzierżawy
    "umowa_dzierzawy": "umowa_dzierzawy",
    "dzierzawa": "umowa_dzierzawy",
    
    # umowa dostawy
    "umowa_dostawy": "umowa_dostawy",
    "dostawa": "umowa_dostawy",
    "dostawca": "umowa_dostawy",
    
    # umowa franczyzy
    "umowa_franczyzy": "umowa_franczyzy",
    "franczyza": "umowa_franczyzy",
    
    # umowa faktoringu
    "umowa_faktoringu": "umowa_faktoringu",
    "faktoring": "umowa_faktoringu",
    
    # umowa pożyczki
    "umowa_pozyczki": "umowa_pozyczki",
    "pozyczka": "umowa_pozyczki",
    "pozyczkodawca": "umowa_pozyczki",
    
    # umowa o pracę (already covered above)
    
    # umowa licencyjna
    "umowa_licencyjna": "umowa_licencyjna",
    "licencja": "umowa_licencyjna",
    
    # umowa ramowa
    "umowa_ramowa": "umowa_ramowa",
    "ramowa": "umowa_ramowa",
    
    # umowa konsygnacji
    "umowa_konsygnacji": "umowa_konsygnacji",
    "konsygnacja": "umowa_konsygnacji",
    
    # umowa ubezpieczenia
    "umowa_ubezpieczenia": "umowa_ubezpieczenia",
    "ubezpieczenie": "umowa_ubezpieczenia",
    "ubezpieczajacy": "umowa_ubezpieczenia",
    "polisa": "umowa_ubezpieczenia",
    
    # umowa o zachowanie poufności (NDA)
    "umowa_o_zachowanie_poufnosci": "umowa_o_zachowanie_poufnosci",
    "nda": "umowa_o_zachowanie_poufnosci",
    "poufnosc": "umowa_o_zachowanie_poufnosci",
    "tajemnica": "umowa_o_zachowanie_poufnosci",
    
    # umowa przechowania
    "umowa_przechowania": "umowa_przechowania",
    "przechowanie": "umowa_przechowania",
    
    # umowa kredytu
    "umowa_kredytu": "umowa_kredytu",
    "kredyt": "umowa_kredytu",
    "kredytobiorca": "umowa_kredytu",
    
    # umowa leasingu
    "umowa_leasingu": "umowa_leasingu",
    "leasing": "umowa_leasingu",
    
    # kaucja/zaliczka
    "kaucja_zaliczka": "kaucja_zaliczka",
    "kaucja": "kaucja_zaliczka",
    "zaliczka": "kaucja_zaliczka",
    
    # odstąpienie konsumenta
    "odstapienie_konsumenta": "odstapienie_konsumenta",
    "odstapienie": "odstapienie_konsumenta",
    
    # wezwanie do zapłaty
    "wezwanie_do_zaplaty": "wezwanie_do_zaplaty",
    "wezwanie": "wezwanie_do_zaplaty",
    
    # reklamacja konsumenta
    "reklamacja_konsumenta": "reklamacja_konsumenta",
    "reklamacja": "reklamacja_konsumenta",
    
    # pełnomocnictwo
    "pelnomocnictwo": "pelnomocnictwo",
    "pelnomocnik": "pelnomocnictwo",
    
    # poręczenie
    "poreczenie": "poreczenie",
    "poręczyciel": "poreczenie",
}


def get_template_text(doc_type: str) -> str:
    """Return the structural template for ``doc_type``.
    
    Uses:
    1. Expanded alias map (legal domain knowledge)
    2. Exact key match
    3. Fuzzy match (catches Polish declension like "dziele" -> "dzielo")
    4. Final fallback to 'other' template
    """
    from difflib import get_close_matches
    
    dt = (doc_type or "other").strip().lower().replace(" ", "_").replace("-", "_")
    
    # Check aliases first (catches legal domain terms like art. 750 KC)
    if dt in ALIASES:
        return TEMPLATES[ALIASES[dt]]
    
    # Exact key match
    if dt in TEMPLATES:
        return TEMPLATES[dt]
    
    # Fuzzy match against template keys and aliases
    all_keys = list(TEMPLATES.keys()) + [k for k in ALIASES.keys() if k not in TEMPLATES]
    matches = get_close_matches(dt, all_keys, n=1, cutoff=0.6)
    if matches:
        key = matches[0]
        if key in ALIASES:
            return TEMPLATES[ALIASES[key]]
        return TEMPLATES[key]
    
    # Fallback
    return TEMPLATES["other"]


# Mapping from this project's ``doc_type`` slugs onto a Polish natural-language
# query used to retrieve the best matching real template from the ``templates``
# Qdrant collection (built from the CC-BY-4.0 legal-templates-multilingual set).
# For any doc_type not listed here, the slug is used verbatim (underscores -> spaces).
DOC_TYPE_QUERY: dict[str, str] = {
    "umowa_najmu": "umowa najmu lokalu mieszkalnego",
    "umowa_o_dzielo": "umowa o dzieło przeniesienie praw autorskich",
    "umowa_zlecenia": "umowa zlecenia wynagrodzenie",
    "umowa_sprzedazy": "umowa sprzedaży rzeczy ruchomej",
    "umowa_darowizny": "umowa darowizny nieruchomości",
    "umowa_dzierzawy": "umowa dzierżawy nieruchomości",
    "umowa_dostawy": "umowa dostawy towarów",
    "umowa_franczyzy": "umowa franczyzy sieci handlowej",
    "umowa_faktoringu": "umowa faktoringu wierzytelności",
    "umowa_pozyczki": "umowa pożyczki pieniężnej",
    "umowa_o_prace": "umowa o pracę stanowisko",
    "umowa_licencyjna": "umowa licencyjna prawa autorskie",
    "umowa_ramowa": "umowa ramowa współpraca",
    "umowa_konsygnacji": "umowa konsygnacji sprzedaż",
    "umowa_ubezpieczenia": "umowa ubezpieczenia majątkowego",
    "umowa_o_zachowanie_poufnosci": "umowa o zachowanie poufności NDA",
    "umowa_przechowania": "umowa przechowania rzeczy",
    "umowa_kredytu": "umowa kredytu bankowego",
    "umowa_leasingu": "umowa leasingu pojazdu",
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


def get_template_real(store, doc_type: str, query: str | None = None, top_k: int = 1) -> str | None:
    """Retrieve the real structured template for ``doc_type`` from a Qdrant store.

    ``query`` is optional free-text detail appended to the base doc_type query
    (e.g. "for IT services with non-compete") to disambiguate between variants.
    ``top_k`` controls how many candidates to consider (default 1).

    Returns the formatted template string, or ``None`` if no match is found
    (caller should fall back to the synthetic scaffold).
    """
    dt = (doc_type or "other").strip().lower() or "other"
    if dt == "other":
        return None
    base_query = DOC_TYPE_QUERY.get(dt) or dt.replace("_", " ")
    full_query = f"{base_query} {query}".strip() if query and query.strip() else base_query
    try:
        hits = store.recall(full_query, top_k=top_k, candidate_k=max(10, top_k * 5))
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
