"""Offline tests for the eval100 judge pipeline (no network, no models)."""

import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from legal_drafter import judge as J


GOOD_JSON = json.dumps({
    "summary_a": "A covers sale.",
    "summary_b": "B covers sale with warranty.",
    "defects_a": "none found",
    "defects_b": "none found",
    "missing_a": "warranty",
    "missing_b": "none found",
    "per_rubric": "correctness A=ok B=ok",
    "verdict": "B",
    "confidence": "high",
    "reason": "B adds warranty.",
}, ensure_ascii=False)


def make_stream(text, split=11):
    for i in range(0, len(text), split):
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=text[i:i + split]))])


class FakeCompletions:
    def __init__(self, script):
        # script: list of ("ok", text) | ("raise", exc) consumed in order
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        kind, payload = self.script.pop(0) if self.script else ("ok", GOOD_JSON)
        if kind == "raise":
            raise payload
        return make_stream(payload)


class FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))


def make_spec(**kw):
    base = dict(base_url="http://router:1", api_key="k",
                model_a="judge-A", model_b="judge-B",
                timeout=5.0, watchdog=30.0, max_tokens=64, temperature=0.0)
    base.update(kw)
    return J.JudgeClientSpec(**base)


def make_record(**kw):
    rec = {
        "question_id": "ext_001",
        "doc_type": "kaucja",
        "source": "external",
        "pair": "base_vs_final",
        "slot_models": {"A": "base", "B": "final"},
        "scenario": "Client rents a flat.",
        "contract_a": "CONTRACT A TEXT art. 6.",
        "contract_b": "CONTRACT B TEXT art. 6.",
        "reference": "",
        "reference_kind": "none",
    }
    rec.update(kw)
    return rec


class TestParseVerdict(unittest.TestCase):
    def test_strict_ok(self):
        v, status = J.parse_verdict(GOOD_JSON)
        self.assertEqual(status, "ok")
        self.assertEqual(v["verdict"], "B")

    def test_salvaged_from_prose_and_fences(self):
        raw = "Here is my verdict:\n```json\n" + GOOD_JSON + "\n```\nDone."
        v, status = J.parse_verdict(raw)
        self.assertEqual(status, "salvaged")
        self.assertEqual(v["verdict"], "B")
        self.assertEqual(v["confidence"], "high")

    def test_failed_empty(self):
        v, status = J.parse_verdict("   ")
        self.assertEqual(status, "failed")
        self.assertIsNone(v)

    def test_failed_no_verdict_key(self):
        v, status = J.parse_verdict(json.dumps({"reason": "nope"}))
        self.assertEqual(status, "failed")

    def test_failed_bad_verdict_value(self):
        v, status = J.parse_verdict(json.dumps({"verdict": "both"}))
        self.assertEqual(status, "failed")

    def test_case_insensitive_verdict(self):
        raw = GOOD_JSON.replace('"B"', '"b"')
        v, status = J.parse_verdict(raw)
        self.assertEqual(v["verdict"], "B")


class TestMapVerdict(unittest.TestCase):
    def test_fwd_and_rev(self):
        self.assertEqual(J.map_verdict("A", "base", "final"), "base")
        self.assertEqual(J.map_verdict("B", "base", "final"), "final")
        self.assertEqual(J.map_verdict("A", "final", "base"), "final")
        self.assertEqual(J.map_verdict("tie", "base", "final"), "tie")


class TestCitations(unittest.TestCase):
    def test_extracts_numbers_in_order_deduped(self):
        keys = J.cited_articles("per art. 394 KC and art. 385(3)(23). Again art. 394!")
        self.assertEqual(keys, ["394", "385"])

    def test_superscript_suffix(self):
        keys = J.cited_articles("niedozwolone art. 385¹ KC")
        self.assertEqual(keys, ["385_1"])


class TestCleanContract(unittest.TestCase):
    PAYLOAD = json.dumps({"summary": "s", "contract": "K" * 600})

    def test_fenced_last_wins(self):
        raw = ("Thinking...\n```json\n" + json.dumps({"summary": "draft",
                 "contract": "D" * 600}) + "\n```\nMore thinking...\n```json\n"
               + self.PAYLOAD + "\n```")
        c, s, ok = J.extract_clean_contract(raw)
        self.assertTrue(ok)
        self.assertEqual(c, "K" * 600)
        self.assertEqual(s, "s")

    def test_inline_json_no_fences(self):
        raw = "Preamble here.\n" + self.PAYLOAD + "\nTrailing words."
        c, _s, ok = J.extract_clean_contract(raw)
        self.assertTrue(ok)
        self.assertEqual(c, "K" * 600)

    def test_unparseable_falls_back_raw(self):
        c, _s, ok = J.extract_clean_contract("just prose " * 100)
        self.assertFalse(ok)
        self.assertTrue(c.startswith("just prose"))

    def test_short_contract_rejected(self):
        raw = json.dumps({"summary": "s", "contract": "tiny"})
        c, _s, ok = J.extract_clean_contract(raw)
        self.assertFalse(ok)

    def test_trailing_garbage_salvaged(self):
        raw = "CoT here.\n" + self.PAYLOAD + "\n}"
        c, _s, ok = J.extract_clean_contract(raw)
        self.assertTrue(ok)
        self.assertEqual(c, "K" * 600)


class TestScenarioAndEmbeddedRef(unittest.TestCase):
    def test_scenario_is_user_only(self):
        q = {"messages": [{"role": "system", "content": "SYS SCAFFOLD"},
                          {"role": "user", "content": "CLIENT ASK"}]}
        self.assertEqual(J.extract_scenario(q), "CLIENT ASK")

    def test_embedded_ref_unpacks_chunk_json(self):
        chunks = [{"chunk_id": "c1", "display_address": "KC art. 1",
                   "text": "t" * 500},
                  {"chunk_id": "c2", "text": "tiny"}]
        q = {"messages": [{"role": "tool", "content": json.dumps(chunks)},
                          {"role": "tool", "content": "ok"}]}
        ref = J.extract_embedded_reference(q)
        self.assertIn("[KC art. 1]", ref)
        self.assertIn("t" * 100, ref)
        self.assertNotIn('"chunk_id"', ref)


class TestContractUsable(unittest.TestCase):
    def test_good(self):
        payload = json.dumps({"summary": "s", "contract": "K" * 600})
        self.assertTrue(J.contract_usable({"contract": payload,
                                           "empty": False}))

    def test_good_with_cot_preamble(self):
        payload = "Reasoning...\n```json\n" + json.dumps(
            {"summary": "s", "contract": "K" * 600}) + "\n```"
        self.assertTrue(J.contract_usable({"contract": payload,
                                           "empty": False}))

    def test_short_is_unusable(self):
        self.assertFalse(J.contract_usable({"contract": "text",
                                            "empty": False}))

    def test_error(self):
        self.assertFalse(J.contract_usable({"contract": "text",
                                            "error": "Server disconnected"}))

    def test_empty_flag(self):
        self.assertFalse(J.contract_usable({"contract": "text", "empty": True}))

    def test_blank(self):
        self.assertFalse(J.contract_usable({"contract": "   "}))


class TestEmbeddedReference(unittest.TestCase):
    def test_picks_tool_chunks_skips_tiny(self):
        row = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "x" * 500},
            {"role": "tool", "content": "ok"},
        ]}
        ref = J.extract_embedded_reference(row)
        self.assertIn("x" * 100, ref)
        self.assertNotIn("\nok\n", ref)


class TestPrepareInputs(unittest.TestCase):
    def _write(self, tmp, name, rows):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def test_skips_missing_final_pairs_keeps_base_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = {"id": "ext_001", "doc_type": "kaucja",
                 "messages": [{"role": "system", "content": "s"},
                              {"role": "user", "content": "u"}]}
            qp = self._write(tmp, "q.jsonl", [q])
            good = {"id": "ext_001", "contract": json.dumps(
                {"summary": "s", "contract": "K" * 600}), "empty": False}
            bad = {"id": "ext_001", "contract": "", "empty": False,
                   "error": "Server disconnected"}
            bp = self._write(tmp, "b.jsonl", [good])
            lp = self._write(tmp, "l.jsonl", [good])
            fp = self._write(tmp, "f.jsonl", [bad])
            out = os.path.join(tmp, "in.jsonl")
            stats = J.prepare_inputs(qp, {"base": bp, "lora250": lp,
                                          "final": fp}, out, qdrant_path=None)
            self.assertEqual(stats["inputs"], 1)
            self.assertEqual(stats["skipped_pairs"], 2)
            with open(out, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f]
            self.assertEqual(recs[0]["pair"], "base_vs_lora250")
            self.assertEqual(recs[0]["reference_kind"], "none")

    def test_embedded_reference_used_for_holdout_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = {"id": 13, "doc_type": "umowa",
                 "messages": [{"role": "system", "content": "s"},
                              {"role": "user", "content": "u"},
                              {"role": "tool", "content": "y" * 500}]}
            qp = self._write(tmp, "q.jsonl", [q])
            good = {"id": 13, "contract": json.dumps(
                {"summary": "s", "contract": "K" * 600}), "empty": False}
            paths = {m: self._write(tmp, f"{m}.jsonl", [good])
                     for m in ("b", "l", "f")}
            out = os.path.join(tmp, "in.jsonl")
            stats = J.prepare_inputs(
                qp, {"base": paths["b"], "lora250": paths["l"],
                     "final": paths["f"]}, out, qdrant_path=None)
            self.assertEqual(stats["inputs"], 3)
            with open(out, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f]
            self.assertTrue(all(r["reference_kind"] == "embedded" for r in recs))
            self.assertIn("y" * 100, recs[0]["reference"])


class TestCitationLookup(unittest.TestCase):
    FAKE_CHUNKS = [
        {"chunk_id": "sejm_eli_x_art_394", "title": "Kodeks cywilny",
         "display_address": "KC art. 394", "text": "zadatek text"},
        {"chunk_id": "sejm_eli_x_art_709", "title": "Kodeks cywilny",
         "display_address": "KC art. 709", "text": "leasing title text"},
    ]

    def setUp(self):
        J._CHUNK_CACHE["test-store"] = list(self.FAKE_CHUNKS)

    def tearDown(self):
        J._CHUNK_CACHE.pop("test-store", None)

    def test_exact_found(self):
        ref, cov = J.lookup_article_chunks(["394"], "test-store")
        self.assertEqual(cov, {"394": "found"})
        self.assertIn("zadatek text", ref)

    def test_flattened_superscript_falls_back_to_base(self):
        ref, cov = J.lookup_article_chunks(["7098"], "test-store")
        self.assertEqual(cov, {"7098": "found_base"})
        self.assertIn("leasing title text", ref)

    def test_missing_stays_missing(self):
        ref, cov = J.lookup_article_chunks(["9999"], "test-store")
        self.assertEqual(cov, {"9999": "missing"})
        self.assertEqual(ref, "")

    def test_budget_cut_is_honest(self):
        with mock.patch.object(J, "_REF_TOTAL", 10):
            ref, cov = J.lookup_article_chunks(["394"], "test-store")
            self.assertEqual(cov, {"394": "budget_cut"})
            self.assertEqual(ref, "")


class TestPromptVersions(unittest.TestCase):
    def test_v3_adds_reference_note(self):
        v2, _ = J.get_prompt("pl-contract-pairwise-v2")
        v3, _ = J.get_prompt("pl-contract-pairwise-v3")
        self.assertIn("REFERENCE NOTE", v3)
        self.assertNotIn("REFERENCE NOTE", v2)
        self.assertTrue(v3.startswith(v2))

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            J.get_prompt("v99")

    def test_row_pins_prompt_version(self):
        spec = make_spec()
        client = FakeClient([("ok", GOOD_JSON)])
        row = J.judge_one(client, "judge_a", spec, make_record(),
                          "fwd", 1, "run1", max_retries=0,
                          system_version="pl-contract-pairwise-v2")
        self.assertEqual(row["system_prompt_version"],
                         "pl-contract-pairwise-v2")


class TestJudgeOne(unittest.TestCase):
    def test_ok_first_try_rev_maps_slots(self):
        spec = make_spec()
        client = FakeClient([("ok", GOOD_JSON)])
        row = J.judge_one(client, "judge_a", spec, make_record(),
                          "rev", 1, "run1", max_retries=2)
        self.assertEqual(row["status"], "ok")
        # rev order: final sits in slot A; verdict B -> slot B -> base
        self.assertEqual(row["slot_a"], "final")
        self.assertEqual(row["verdict_mapped"], "base")
        self.assertEqual(row["attempts"], 1)
        # temperature 0 + max_tokens forwarded, stream enabled
        sent = client.chat.completions.calls[0]
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["temperature"], 0.0)

    def test_retry_then_success(self):
        spec = make_spec()
        client = FakeClient([("raise", ConnectionError("down")),
                             ("ok", GOOD_JSON)])
        with mock.patch("time.sleep") as slp:
            row = J.judge_one(client, "judge_a", spec, make_record(),
                              "fwd", 2, "run1", max_retries=2)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["attempts"], 2)
        slp.assert_called_once_with(2)  # min(2**1, 30)

    def test_exhausted_attempts_error_row_preserves_raw(self):
        spec = make_spec()
        client = FakeClient([("raise", ConnectionError("down")),
                             ("ok", "not json at all"),
                             ("raise", TimeoutError("t"))])
        with mock.patch("time.sleep"):
            row = J.judge_one(client, "judge_b", spec, make_record(),
                              "fwd", 1, "run1", max_retries=2)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["attempts"], 3)  # 1 initial + 2 retries
        self.assertIsNone(row["verdict_mapped"])
        self.assertIn("failed after 3 attempts", row["error"])
        # last raw output preserved (the unparseable text from attempt 2)
        self.assertEqual(row["raw_response"], "not json at all")


class TestResumeAndAppend(unittest.TestCase):
    def test_load_done_keys_and_retry_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jsonl")
            rows = [
                {"question_id": "ext_001", "pair": "base_vs_final",
                 "order": "fwd", "judge": "judge_a", "trial": 1,
                 "status": "ok"},
                {"question_id": "ext_001", "pair": "base_vs_final",
                 "order": "rev", "judge": "judge_a", "trial": 1,
                 "status": "error"},
            ]
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            done = J.load_done_keys(p)
            self.assertEqual(len(done), 2)
            done2 = J.load_done_keys(p, retry_errors=True)
            self.assertEqual(len(done2), 1)  # error key offered again

    def test_append_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"old": true}\n')
            st = J.RunState(annotations_path=p)
            st.append({"new": True, "status": "ok"})
            with open(p, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), {"old": True})


class TestConfigGuard(unittest.TestCase):
    @staticmethod
    def _load_cli():
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "judge_eval100.py")
        spec = importlib.util.spec_from_file_location("judge_eval100", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_env_defaults_and_cli_override(self):
        import os
        mod = self._load_cli()
        env = {
            "JUDGE_INPUTS": "/tmp/env_inputs.jsonl",
            "JUDGE_OUT_DIR": "/tmp/env_runs",
            "JUDGE_TRIALS": "5",
            "JUDGE_JUDGES": "judge_b",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            parser = mod.build_parser()
            args = parser.parse_args(["run"])
            self.assertEqual(args.inputs, "/tmp/env_inputs.jsonl")
            self.assertEqual(args.out_dir, "/tmp/env_runs")
            self.assertEqual(args.trials, 5)
            self.assertEqual(args.judges, "judge_b")
            # CLI flags win over env.
            args2 = parser.parse_args(["run", "--trials", "2",
                                       "--judges", "judge_a"])
            self.assertEqual(args2.trials, 2)
            self.assertEqual(args2.judges, "judge_a")

    def test_invalid_env_judges_falls_back_to_both(self):
        import os
        mod = self._load_cli()
        with mock.patch.dict(os.environ, {"JUDGE_JUDGES": "nonsense"}):
            args = mod.build_parser().parse_args(["run"])
            self.assertEqual(args.judges, "both")

    def test_missing_required_reports_option(self):
        import os
        mod = self._load_cli()
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("JUDGE_")}
        with mock.patch.dict(os.environ, clean, clear=True):
            with self.assertRaises(SystemExit):
                mod.main(["run"])

    def test_mismatched_settings_rejected(self):
        mod = self._load_cli()
        saved = {"judge_model_a": "A", "judge_model_b": "B", "trials": 2,
                 "system_prompt_version": "v1", "temperature": 0.0,
                 "max_tokens": 2048, "seed": None}
        same = dict(saved)
        ok, _ = mod._config_matches(saved, same)
        self.assertTrue(ok)
        changed = dict(saved, trials=5)
        ok, why = mod._config_matches(saved, changed)
        self.assertFalse(ok)
        self.assertIn("trials", why)


class TestTallyRule(unittest.TestCase):
    @staticmethod
    def _load_tally():
        path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "tally_judge.py")
        spec = importlib.util.spec_from_file_location("tally_judge", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_unanimous_win_high_confidence(self):
        mod = self._load_tally()
        votes = []
        for j in ("judge_a", "judge_b"):
            for o in ("fwd", "rev"):
                for t in (1, 2):
                    # fwd: slot A=base wins; rev: slot A=final, B wins -> base
                    w = "A" if o == "fwd" else "B"
                    mapped = "base" if w == ("A" if o == "fwd" else "B") else "?"
                    votes.append({"judge": j, "order": o, "trial": t,
                                  "verdict_mapped": "base"})
        rec = mod.tally_set(votes)
        self.assertEqual(rec["verdict"], "base")
        self.assertEqual(rec["confidence"], "high")
        self.assertTrue(rec["swap_consistent"])
        self.assertEqual(rec["flags"], [])

    def test_position_flip_becomes_tie(self):
        mod = self._load_tally()
        votes = []
        for j in ("judge_a", "judge_b"):
            for o in ("fwd", "rev"):
                for t in (1, 2):
                    # each judge reliably picks whoever sits in slot A
                    mapped = "base" if o == "fwd" else "final"
                    votes.append({"judge": j, "order": o, "trial": t,
                                  "verdict_mapped": mapped})
        rec = mod.tally_set(votes)
        self.assertEqual(rec["verdict"], "tie")
        self.assertIn("position_flip", rec["flags"])
        self.assertFalse(rec["swap_consistent"])

    def test_unstable_order_flags_noise(self):
        mod = self._load_tally()
        votes = []
        for j in ("judge_a", "judge_b"):
            for o in ("fwd", "rev"):
                vals = ("base", "final") if (j, o) == ("judge_a", "fwd") else ("base", "base")
                for t, v in zip((1, 2), vals):
                    votes.append({"judge": j, "order": o, "trial": t,
                                  "verdict_mapped": v})
        rec = mod.tally_set(votes)
        self.assertEqual(rec["verdict"], "tie")
        self.assertIn("unstable_order", rec["flags"])

    def test_cross_judge_disagreement_tie(self):
        mod = self._load_tally()
        votes = []
        for j in ("judge_a", "judge_b"):
            w = "base" if j == "judge_a" else "final"
            for o in ("fwd", "rev"):
                for t in (1, 2):
                    votes.append({"judge": j, "order": o, "trial": t,
                                  "verdict_mapped": w})
        rec = mod.tally_set(votes)
        self.assertEqual(rec["verdict"], "tie")
        self.assertIn("cross_judge_disagree", rec["flags"])


class TestCircuitBreaker(unittest.TestCase):
    def _br(self, **kw):
        d = dict(cooldowns=(0.05, 0.1), wait_burst=0.005,
                 max_tripped_seconds=10.0)
        d.update(kw)
        return J.CircuitBreaker(**d)

    def _go_fail(self, br, judge="judge_a", n=1):
        for _ in range(n):
            mode, rnd = br.before_task(judge)
            self.assertEqual(mode, "go")
            br.after_task(False, rnd)

    def test_consecutive_trip_then_probe_reset(self):
        br = self._br()
        self._go_fail(br, n=5)
        st = br.stats()
        self.assertEqual(st["trips"], 1)
        # pause expires quickly; both probes succeed -> clean slate
        m1, r1 = br.before_task("judge_a")
        m2, r2 = br.before_task("judge_b")
        self.assertEqual((m1, m2), ("probe", "probe"))
        self.assertEqual(r1, r2)
        br.after_task(True, r1)
        br.after_task(True, r2)
        st = br.stats()
        self.assertEqual(st["level"], 0)
        mode, _ = br.before_task("judge_a")
        self.assertEqual(mode, "go")

    def test_window_trip_without_consecutive(self):
        br = self._br()
        for ok in (True, False, False, False, False,
                   True, False, False, False, False):
            mode, rnd = br.before_task("judge_a")
            self.assertEqual(mode, "go")
            br.after_task(ok, rnd)
        self.assertEqual(br.stats()["trips"], 1)  # 8/10 failed, consec<=4

    def test_probe_failure_escalates(self):
        br = self._br()
        self._go_fail(br, n=5)
        m1, r1 = br.before_task("judge_a")
        m2, r2 = br.before_task("judge_b")
        br.after_task(False, r1)
        br.after_task(False, r2)
        st = br.stats()
        self.assertEqual(st["trips"], 2)
        self.assertEqual(st["level"], 2)

    def test_abort_after_budget(self):
        br = self._br(max_tripped_seconds=0.06)
        self._go_fail(br, n=5)  # trip 1, total 0.05
        m1, r1 = br.before_task("judge_a")
        m2, r2 = br.before_task("judge_b")
        br.after_task(False, r1)
        br.after_task(False, r2)  # trip 2, total 0.10 -> abort
        self.assertTrue(br.stats()["aborted"])
        mode, _ = br.before_task("judge_a")
        self.assertEqual(mode, "abort")

    def test_per_judge_breakers_are_independent(self):
        # Sick judge_a trips its own breaker while judge_b flows untouched.
        ba, bb = J.CircuitBreaker(), J.CircuitBreaker()
        for _ in range(5):
            mode, rnd = ba.before_task("judge_a")
            self.assertEqual(mode, "go")
            ba.after_task(False, rnd)
        self.assertEqual(ba.stats()["trips"], 1)
        self.assertEqual(bb.stats()["trips"], 0)
        mode, _ = bb.before_task("judge_b")
        self.assertEqual(mode, "go")

    def test_stale_probe_cannot_resolve_new_round(self):
        br = self._br()
        self._go_fail(br, n=5)  # round 1
        _, r1 = br.before_task("judge_a")  # claim, never complete yet
        _, r1b = br.before_task("judge_b")
        br.after_task(False, r1b)
        br.after_task(False, r1)  # round 1 fails -> round 2 armed
        self.assertEqual(br.stats()["trips"], 2)
        # stale completion from round 1 must not resolve round 2
        br.after_task(True, r1)
        self.assertEqual(br.probe_done, 0)
        # round 2 resolves normally
        _, r2a = br.before_task("judge_a")
        _, r2b = br.before_task("judge_b")
        br.after_task(True, r2a)
        br.after_task(True, r2b)
        self.assertEqual(br.stats()["level"], 0)


if __name__ == "__main__":
    unittest.main()
