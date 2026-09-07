#!/usr/bin/env python3
"""Blind pairwise judging for the eval100 benchmark (local machine, API judges).

Stages:
  prepare  join questions + per-model contracts (+ references) -> judge_inputs.jsonl
  run      judge every input with both judges, both orders, K trials -> annotations.jsonl

Resume-safe: ``run`` appends one row per verdict and skips keys already
recorded. Pass the same --run-id to continue an interrupted run (settings
are verified to match before a single new call is made).

Examples:
  PYTHONPATH=src python3 scripts/judge_eval100.py prepare \\
      --questions outputs/eval100/eval100_questions.jsonl \\
      --base outputs/eval100/eval100_base.jsonl \\
      --lora250 outputs/eval100/eval100_lora250.jsonl \\
      --final outputs/eval100/eval100_final.jsonl \\
      --out outputs/eval100/judge_inputs.jsonl

  PYTHONPATH=src python3 scripts/judge_eval100.py run \\
      --inputs outputs/eval100/judge_inputs.jsonl \\
      --out-dir outputs/judge_runs --trials 2 --workers 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from legal_drafter.judge import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    CircuitBreaker,
    JudgeClientSpec,
    RunState,
    done_key,
    get_prompt,
    judge_one,
    load_done_keys,
    prepare_inputs,
    summarize_annotations,
)


# ---------------------------------------------------------------- prepare ---

def cmd_prepare(args) -> int:
    stats = prepare_inputs(
        args.questions,
        {"base": args.base, "lora250": args.lora250, "final": args.final},
        args.out,
        qdrant_path=args.qdrant_path,
    )
    print(f"questions: {stats['questions']}")
    print(f"judge inputs written: {stats['inputs']} -> {args.out}")
    print(f"contracts sent raw (no JSON payload found): "
          f"{stats.get('raw_fallback_contracts', 0)}")
    print(f"skipped pairs (missing contract): {stats['skipped_pairs']}")
    for pair, n in stats["skipped_by_pair"].items():
        print(f"  {pair}: {n}")
    return 0


# -------------------------------------------------------------------- run ---

def _git_hash() -> str:
    try:
        import subprocess

        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def _config_matches(saved: dict, current: dict) -> tuple[bool, str]:
    for key in ("judge_model_a", "judge_model_b", "trials",
                "system_prompt_version", "temperature", "max_tokens", "seed"):
        if saved.get(key) != current.get(key):
            return False, (f"config mismatch on {key}: "
                           f"run has {saved.get(key)!r}, requested {current.get(key)!r}")
    return True, ""


def cmd_run(args) -> int:
    spec = JudgeClientSpec.from_env()
    judges = ["judge_a", "judge_b"] if args.judges == "both" else [args.judges]
    try:
        _, prompt_version = get_prompt(args.prompt_version)
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    run_id = args.run_id or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    annotations_path = os.path.join(run_dir, "annotations.jsonl")
    config_path = os.path.join(run_dir, "config.json")

    _breaker_off = (_env_str("JUDGE_BREAKER", "1").strip().lower()
                      in ("0", "false", "no"))
    # Global breaker (total outage) + one per judge (single-backend outage:
    # the healthy judge's successes would otherwise mask the sick one's
    # failures in a shared window).
    breakers: dict[str, CircuitBreaker] = {}
    if not _breaker_off:
        breakers = {"*": CircuitBreaker(),
                    "judge_a": CircuitBreaker(),
                    "judge_b": CircuitBreaker()}
        print(f"breaker armed (global + per-judge): "
              f"{breakers['*'].describe()}", flush=True)

    def _gate(judge: str):
        """(mode, probe_round, scope) with scope '*' or the judge name."""
        mode, rnd = breakers["*"].before_task("global")
        if mode != "go":
            return mode, rnd, "*"
        return (*breakers[judge].before_task(judge), judge)

    def _record(judge: str, ok: bool, gated: tuple) -> None:
        mode, rnd, scope = gated
        if scope == "*":
            breakers["*"].after_task(ok, rnd)
        else:
            breakers[judge].after_task(ok, rnd)
            # A finished probe also counts as global signal.
            breakers["*"].after_task(ok, -1)

    def _trip_note() -> None:
        for scope, br in breakers.items():
            st = br.stats()
            if st["trips"] > shared["trips_seen"].get(scope, 0):
                shared["trips_seen"][scope] = st["trips"]
                with print_lock:
                    print(f"[breaker:{scope}] TRIP #{st['trips']} "
                          f"(paused_total={st['tripped_min']}m)", flush=True)

    run_config = {
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "command": " ".join(sys.argv),
        "inputs": os.path.abspath(args.inputs),
        "judge_model_a": spec.model_a,
        "judge_model_b": spec.model_b,
        "judges": judges,
        "trials": args.trials,
        "workers": args.workers,
        "max_retries": args.retries,
        "retry_errors": args.retry_errors,
        "system_prompt_version": prompt_version,
        "breaker": ({s: br.describe() for s, br in breakers.items()}
                    if breakers else None),
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "seed": spec.seed,
        "router": {k: v for k, v in spec.masked().items() if k != "api_key"},
    }
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            saved = json.load(f)
        ok, why = _config_matches(saved, run_config)
        if not ok:
            print(f"ERROR: refusing to append to run {run_id}: {why}", flush=True)
            print("Use a fresh --run-id for different settings.", flush=True)
            return 2
        print(f"resuming run {run_id} (settings verified match)", flush=True)
    else:
        tmp = config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)
        os.replace(tmp, config_path)

    with open(args.inputs, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    done = load_done_keys(annotations_path, retry_errors=args.retry_errors)
    tasks = []
    # Judge is the innermost loop so both judges stay in flight together
    # whenever workers >= 2 (no judge ever waits behind the other's queue).
    for rec in records:
        for order in ("fwd", "rev"):
            for trial in range(1, args.trials + 1):
                for judge in judges:
                    key = done_key(rec["question_id"], rec["pair"], order,
                                   judge, trial)
                    if key not in done:
                        tasks.append((rec, judge, order, trial))
    total_planned = len(records) * len(judges) * 2 * args.trials
    print(f"run {run_id}: {len(tasks)} pending calls "
          f"({total_planned - len(tasks)}/{total_planned} already recorded)",
          flush=True)
    if not tasks:
        print("nothing to do.", flush=True)
        return 0

    clients = spec.make_clients()
    state = RunState(annotations_path=annotations_path)
    print_lock = threading.Lock()
    counter = {"n": total_planned - len(tasks)}

    shared = {"aborted": False, "trips_seen": {}}

    def one(task):
        rec, judge, order, trial = task
        gated = ("go", -1, "-")
        if breakers:
            mode, rnd, scope = _gate(judge)
            gated = (mode, rnd, scope)
            if mode == "abort":
                shared["aborted"] = True
                return None
            if mode == "probe":
                with print_lock:
                    print(f"[breaker:{scope}] probe ({judge}) dispatched",
                          flush=True)
        row = judge_one(clients[judge], judge, spec, rec, order, trial,
                        run_id, max_retries=args.retries,
                        system_version=prompt_version)
        if breakers:
            _record(judge, row.get("status") == "ok", gated)
            _trip_note()
        state.append(row)
        with print_lock:
            counter["n"] += 1
            verdict = row.get("verdict_mapped") or f"ERROR:{row.get('error', '')[:60]}"
            print(f"[{counter['n']}/{total_planned}] q={rec['question_id']} "
                  f"{rec['pair']} {order} {judge} t{trial} -> {verdict} "
                  f"({row.get('confidence')})", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for fut in as_completed(futs):
            fut.result()  # never raises (judge_one catches everything), re-raise bugs only

    summary = summarize_annotations(annotations_path)
    summary["run_id"] = run_id
    if breakers:
        summary["breaker"] = {s: br.stats() for s, br in breakers.items()}
    if shared["aborted"]:
        summary["aborted"] = True
        print(f"ABORTED run {run_id}: breaker budget exhausted; "
              f"recorded rows kept, re-run with --retry-errors to replay "
              f"failures.", flush=True)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"DONE run {run_id}: rows={summary['rows']} "
          f"by_status={summary['by_status']}", flush=True)
    return 0


def _env_str(name: str, default=None):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build judge_inputs.jsonl")
    p.add_argument("--questions", default=_env_str("JUDGE_QUESTIONS"))
    p.add_argument("--base", default=_env_str("JUDGE_BASE_RESULTS"))
    p.add_argument("--lora250", default=_env_str("JUDGE_LORA250_RESULTS"))
    p.add_argument("--final", default=_env_str("JUDGE_FINAL_RESULTS"))
    p.add_argument("--out", default=_env_str("JUDGE_PREPARE_OUT"))
    p.add_argument("--qdrant-path",
                   default=_env_str("JUDGE_QDRANT_PATH", ".rag/qdrant/qdrant"),
                   help="local read-only payload store for citation lookup "
                        "(no models loaded; empty/unreadable -> reference 'none')")
    p.set_defaults(fn=cmd_prepare,
                   _required=["questions", "base", "lora250", "final", "out"])

    judges_default = _env_str("JUDGE_JUDGES", "both")
    if judges_default not in ("both", "judge_a", "judge_b"):
        judges_default = "both"
    r = sub.add_parser("run", help="judge inputs -> annotations.jsonl")
    r.add_argument("--inputs", default=_env_str("JUDGE_INPUTS"))
    r.add_argument("--out-dir", default=_env_str("JUDGE_OUT_DIR",
                                                "outputs/judge_runs"))
    r.add_argument("--run-id", default=_env_str("JUDGE_RUN_ID"),
                   help="resume an existing run dir (settings verified)")
    r.add_argument("--trials", type=int, default=_env_int("JUDGE_TRIALS", 2))
    r.add_argument("--workers", type=int, default=_env_int("JUDGE_WORKERS", 1))
    r.add_argument("--retries", type=int, default=_env_int("JUDGE_RETRIES", 2),
                   help="retries per call after the initial attempt")
    r.add_argument("--retry-errors", action="store_true",
                   default=_env_flag("JUDGE_RETRY_ERRORS"),
                   help="re-attempt keys previously recorded as errors")
    r.add_argument("--judges", choices=["both", "judge_a", "judge_b"],
                   default=judges_default)
    r.add_argument("--prompt-version",
                   default=_env_str("JUDGE_PROMPT_VERSION",
                                    DEFAULT_PROMPT_VERSION),
                   help="pinned system prompt (recorded per row; resume "
                        "requires the same value)")
    r.set_defaults(fn=cmd_run, _required=["inputs"])
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    missing = [f"--{n.replace('_', '-')}" for n in
               getattr(args, "_required", []) if not getattr(args, n)]
    if missing:
        ap.error(f"missing required options (or set the JUDGE_* env): "
                 f"{', '.join(missing)}")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
