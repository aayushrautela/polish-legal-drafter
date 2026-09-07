"""Serve the student (base Qwen3.5-2B + LoRA adapters) with vLLM for eval.

Provenance: replicates the Vast-box serving that produced the eval100
drafts (diary 2026-09-02: base + ckpt250 on port 18000). The eval harness
(``eval_holdouts_v2_v3.py``) calls this endpoint like any OpenAI API and
selects configurations via the ``model`` field, so the served names MUST
match: base under its model path/ID, adapters under their short names
(ckpt250, final, ...).

Why these flags (each learned the hard way):
  --enable-lora --max-lora-rank 16 --lora-modules ... : serve base + all
      adapters in one engine; eval switches columns without restarts.
      Rank 16 matches the trained adapters.
  --enable-auto-tool-choice --tool-call-parser qwen3_coder : Qwen3.5 does
      not emit hermes-format tool calls; the default parser fails silently
      and the agentic loop dies.
  --max-model-len 32768 : research transcripts run past 20k tokens.
  --gpu-memory-utilization 0.85 : 1.0 OOM-killed the engine mid-eval under
      parallel load; 0.85 leaves headroom.

Configuration (all env-driven):
  VLLM_BASE_MODEL   base weights path or HF ID
                    (default: /workspace/models/Qwen3.5-2B)
  VLLM_SERVED_NAME  name the base is served under; must equal the eval
                    harness's base model string (default: = VLLM_BASE_MODEL)
  VLLM_LORA_MODULES comma-separated name=path pairs
                    (default: ckpt250=/workspace/checkpoints-2b/checkpoint-250,final=/workspace/checkpoints-2b/final)
  VLLM_PORT         (default: 18000)
  VLLM_MAX_MODEL_LEN (default: 32768)
  VLLM_MAX_LORA_RANK (default: 16)
  VLLM_GPU_MEM      (default: 0.85)

Usage (on the GPU box):
  python3 scripts/eval/serve_vllm.py
  python3 scripts/eval/serve_vllm.py --dry-run   # print command, don't launch
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def build_command(env=os.environ) -> list[str]:
    base = env.get("VLLM_BASE_MODEL", "/workspace/models/Qwen3.5-2B")
    served = env.get("VLLM_SERVED_NAME", base)
    port = env.get("VLLM_PORT", "18000")
    max_len = env.get("VLLM_MAX_MODEL_LEN", "32768")
    max_rank = env.get("VLLM_MAX_LORA_RANK", "16")
    gpu_mem = env.get("VLLM_GPU_MEM", "0.85")
    raw_modules = env.get(
        "VLLM_LORA_MODULES",
        "ckpt250=/workspace/checkpoints-2b/checkpoint-250,"
        "final=/workspace/checkpoints-2b/final",
    )
    modules = [m.strip() for m in raw_modules.split(",") if m.strip()]
    cmd = [
        "vllm", "serve", base,
        "--port", str(port),
        "--max-model-len", str(max_len),
        "--served-model-name", served,
        "--enable-lora",
        "--max-lora-rank", str(max_rank),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    for m in modules:
        cmd += ["--lora-modules", m]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the vllm command without launching")
    args = ap.parse_args(argv)

    # Fail fast on missing adapter dirs (before the slow vLLM startup).
    missing = []
    for m in os.environ.get(
            "VLLM_LORA_MODULES",
            "ckpt250=/workspace/checkpoints-2b/checkpoint-250,"
            "final=/workspace/checkpoints-2b/final").split(","):
        m = m.strip()
        if m and "=" in m and not os.path.isdir(m.split("=", 1)[1]):
            missing.append(m)
    if missing and not args.dry_run:
        print(f"[serve] missing adapter dirs: {missing}", file=sys.stderr)
        return 2

    cmd = build_command()
    print("[serve] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    if shutil.which("vllm") is None:
        print("[serve] 'vllm' binary not found on PATH", file=sys.stderr)
        return 2
    os.execvp("vllm", cmd)


if __name__ == "__main__":
    raise SystemExit(main())
