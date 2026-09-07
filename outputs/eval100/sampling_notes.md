# eval100 sampling notes (2026-09-03)

## Composition

- Total questions: **100**
- Holdout (ckpt250, genuinely unseen): **42** (from `holdout_ids_true42.json`)
- External (newly authored, ext_* id namespace): **58** (from `external_58_raw.json` -> `external_58.jsonl`)
- Distribution: exactly **4 questions per doc_type × 25 types**.

## External 58 distribution (by backfill vs holdout)

| doc_type | holdout | external | total |
|---|---|---|---|
| kaucja_zaliczka | 2 | 2 | 4 |
| odstapienie_konsumenta | 2 | 2 | 4 |
| pelnomocnictwo | 3 | 1 | 4 |
| poreczenie | 0 | 4 | 4 |
| reklamacja_konsumenta | 2 | 2 | 4 |
| umowa_darowizny | 2 | 2 | 4 |
| umowa_dostawy | 1 | 3 | 4 |
| umowa_dzierzawy | 3 | 1 | 4 |
| umowa_faktoringu | 2 | 2 | 4 |
| umowa_franczyzy | 1 | 3 | 4 |
| umowa_konsygnacji | 1 | 3 | 4 |
| umowa_kredytu | 1 | 3 | 4 |
| umowa_leasingu | 1 | 3 | 4 |
| umowa_licencyjna | 1 | 3 | 4 |
| umowa_najmu | 1 | 3 | 4 |
| umowa_o_dzielo | 2 | 2 | 4 |
| umowa_o_prace | 4 | 0 | 4 |
| umowa_o_zachowanie_poufnosci | 1 | 3 | 4 |
| umowa_pozyczki | 2 | 2 | 4 |
| umowa_przechowania | 1 | 3 | 4 |
| umowa_ramowa | 1 | 3 | 4 |
| umowa_sprzedazy | 4 | 0 | 4 |
| umowa_ubezpieczenia | 1 | 3 | 4 |
| umowa_zlecenia | 1 | 3 | 4 |
| wezwanie_do_zaplaty | 2 | 2 | 4 |

## Contamination check

- id collisions vs `sft_final_merged.jsonl` (2111 rows): **0**
- exact-question-text duplicates: **0**
- max SequenceMatcher similarity to any train question: **0.494** (threshold 0.5; warnings: 0)

Top-5 most-similar external questions (lower is better, just a sanity check):

| id | doc_type | max_sim | closest training question |
|---|---|---|---|
| ext_057 | wezwanie_do_zaplaty | 0.494 | Mam umowę z zapisem na sąd polubowny i kontrahent nie płaci mi od kilku miesięcy |
| ext_042 | umowa_o_zachowanie_poufnosci | 0.453 | Chcę podpisać umowę poufności z firmą, której pokażę moje dane techniczne i bizn |
| ext_005 | pelnomocnictwo | 0.439 | Kupuję działkę, ale nie będę w stanie osobiście pojawić się u notariusza. Czy mo |
| ext_058 | wezwanie_do_zaplaty | 0.422 | Zapłaciłem zaliczkę za usługę, ale umowa się nie zrealizowała i teraz druga stro |
| ext_012 | umowa_darowizny | 0.421 | Chcę poręczyć kredyt dla znajomego, ale boję się, że jeśli część spłaci, to i ta |

## Method

External questions are first-person Polish scenarios authored to match the corpus register. Each row uses the canonical system prompt and a user message of the new question. `tools` is copied from a reference row of the same doc_type (the eval script's doc_type enum patch in `eval_holdouts_v2_v3.py` rewrites the enum regardless).

The merged 100-row file `eval100_questions.jsonl` is the input for the eval script (once the script is pointed at it via a new `--input` flag — see eval script TODO).
