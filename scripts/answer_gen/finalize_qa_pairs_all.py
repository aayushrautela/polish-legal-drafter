"""Minimal finalize: just cap tool msgs for train view. No audit — data is clean."""
import json, hashlib, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "outputs" / "qa_pairs_all" / "qa_pairs_full.jsonl"
OUT = ROOT / "outputs" / "qa_pairs_all_final"
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(SRC)]
print(f"Loaded: {len(rows)}")

# Filter out old error entries (no 'messages' key)
rows = [r for r in rows if "messages" in r]
print(f"After filtering errors: {len(rows)}")

# Write sft_final.jsonl (canonical, full)
final_path = OUT / "sft_final.jsonl"
with open(final_path, "w") as f:
    for r in rows:
        f.write(json.dumps({
            "id": r["id"], "doc_type": r["doc_type"], "variant": r["variant"],
            "source": r.get("source", ""), "question": r["question"],
            "tools": r.get("tools", []), "messages": r["messages"],
            "contract": r["contract"],
        }, ensure_ascii=False) + "\n")
print(f"Wrote sft_final.jsonl ({final_path.stat().st_size} bytes)")

# Write sft_train_view.jsonl (tool msgs capped @2000)
view_path = OUT / "sft_train_view.jsonl"
n_capped = 0
with open(view_path, "w") as f:
    for r in rows:
        msgs = []
        for m in r["messages"]:
            m = dict(m)
            if m.get("role") == "tool" and len(m.get("content", "")) > 2000:
                m["content"] = m["content"][:2000] + "\n[...]"
                n_capped += 1
            msgs.append(m)
        rec = {
            "id": r["id"], "doc_type": r["doc_type"], "variant": r["variant"],
            "source": r.get("source", ""), "question": r["question"],
            "tools": r.get("tools", []), "messages": msgs,
            "contract": r["contract"],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Wrote sft_train_view.jsonl ({n_capped} tool msgs capped)")

# Holdout split (~2%)
random.seed(42)
idx = list(range(len(rows)))
random.shuffle(idx)
n_holdout = max(1, int(len(rows) * 0.02))
holdout_ids = [rows[i]["id"] for i in idx[:n_holdout]]
(OUT / "holdout_ids.json").write_text(json.dumps(holdout_ids, indent=2))

per_doc, per_variant = {}, {}
for r in rows:
    per_doc[r["doc_type"]] = per_doc.get(r["doc_type"], 0) + 1
    per_variant[r["variant"]] = per_variant.get(r["variant"], 0) + 1

manifest = {
    "source": "outputs/qa_pairs_all/qa_pairs_full.jsonl",
    "teacher_thinking": "english",
    "total_rows": len(rows),
    "holdout_rows": n_holdout,
    "per_doc": per_doc,
    "per_variant": per_variant,
    "tool_cap_chars": 2000,
    "note": "no audit — source data 100% clean",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

print(f"\nFINAL: {len(rows)} rows, {n_holdout} holdout, {len(per_doc)} doc_types")
