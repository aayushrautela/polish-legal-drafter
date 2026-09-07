#!/usr/bin/env python3
"""Local LoRA training script for Vast — supports single-GPU and multi-GPU DDP.

Downloads unsloth/Qwen3.5-4B from HF, trains bf16 LoRA on sft_final_merged.jsonl,
saves checkpoints at specific steps.

Single-GPU:
    python scripts/vast/train_lora_vast.py --model /workspace/models/Qwen3.5-4B ...

Multi-GPU (DDP):
    torchrun --nproc_per_node=6 scripts/vast/train_lora_vast.py --model /workspace/models/Qwen3.5-4B ...
"""
import argparse
import glob
import json
import os
import pathlib
import random
import time

def setup_ddp():
    """Initialize distributed training. Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    else:
        import torch
        torch.cuda.set_device(0)
        return 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/sft_final_merged.jsonl")
    ap.add_argument("--output", default="/artifacts/checkpoints")
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", default="2,5,10,20,50,100,150,200,250,350,500")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()

    # --- DDP setup ---
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0

    save_steps = set(int(s) for s in args.save_steps.split(","))

    # --- MUST import unsloth first (patches transformers/peft/trl) ---
    import unsloth
    import datasets
    import torch
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    t0 = time.time()
    if is_main:
        print(f"[setup] loading {args.model} bf16 LoRA, max_seq_length={args.max_seq}", flush=True)
        print(f"[setup] DDP: {world_size} GPUs, batch_size={args.batch_size}, grad_accum={args.grad_accum}", flush=True)

    # --- Load model on correct GPU ---
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        device_map=device_map,
    )
    tokenizer.truncation_side = "left"  # CRITICAL: default 'right' cuts contracts

    # --- load data ---
    if is_main:
        print(f"[data] reading {args.data}", flush=True)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_holdout = max(24, len(rows) // 50)
    train_rows = [rows[i] for i in idx[n_holdout:]]
    holdout_ids = [rows[i]["id"] for i in idx[:n_holdout]]

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        (out_dir / "holdout_ids.json").write_text(json.dumps(holdout_ids), encoding="utf-8")
        print(f"[data] train={len(train_rows)} holdout={n_holdout} ({len(rows)} total)", flush=True)

    # --- normalize tool-call args (Qwen3.5 chat template needs dicts, not JSON strings) ---
    def _normalize_toolcall_args(messages):
        import copy
        msgs = copy.deepcopy(messages)
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                a = tc["function"]["arguments"]
                tc["function"]["arguments"] = json.loads(a) if isinstance(a, str) else a
        return msgs

    def fmt(batch):
        texts = [
            tokenizer.apply_chat_template(
                _normalize_toolcall_args(r["messages"]),
                tools=r.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
            )
            for r in batch["rows"]
        ]
        return {"text": texts}

    ds = datasets.Dataset.from_dict({"rows": train_rows}).map(
        fmt, batched=True, remove_columns=["rows"], num_proc=2
    )

    # --- LoRA ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[setup] trainable={n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", flush=True)

    # --- training ---
    steps_per_epoch = len(train_rows) // (args.batch_size * args.grad_accum * world_size)
    total_steps = steps_per_epoch * args.epochs
    if is_main:
        print(f"[train] epochs={args.epochs} batch={args.batch_size} accum={args.grad_accum} "
              f"steps_per_epoch={steps_per_epoch} total={total_steps}", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=max(10, total_steps // 20),
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="no",  # we save manually via callback
            bf16=True,
            seed=args.seed,
            report_to="none",
            # DDP settings
            ddp_find_unused_parameters=False,
        ),
    )

    # --- train only on assistant responses (not user/tool observations) ---
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    # --- manual checkpoint callback at specific steps ---
    from transformers import TrainerCallback
    class SaveAtSteps(TrainerCallback):
        def __init__(self, save_steps, output_dir, model, tokenizer, trainer):
            self.save_steps = set(save_steps)
            self.output_dir = pathlib.Path(output_dir)
            self.model = model
            self.tokenizer = tokenizer
            self.trainer = trainer

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.save_steps and is_main:
                ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
                print(f"[ckpt] saving at step {state.global_step} -> {ckpt_dir}", flush=True)
                self.trainer.save_model(ckpt_dir)
                self.trainer.state.save_to_json(ckpt_dir / "trainer_state.json")
                self.tokenizer.save_pretrained(ckpt_dir)
            return control

    trainer.add_callback(SaveAtSteps(save_steps, out_dir, model, tokenizer, trainer))

    # --- resume from latest checkpoint if any ---
    resume_ckpt = None
    existing = sorted(glob.glob(f"{out_dir}/checkpoint-*"))
    if existing:
        resume_ckpt = existing[-1]
        if is_main:
            print(f"[resume] found checkpoint {resume_ckpt}, resuming from step {resume_ckpt.split('-')[-1]}", flush=True)

    if is_main:
        print(f"[train] starting...", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- save final (only from rank 0) ---
    if is_main:
        final_dir = out_dir / "final"
        print(f"[done] saving final -> {final_dir}", flush=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        elapsed = time.time() - t0
        print(f"[done] total wall={elapsed:.1f}s ({elapsed/60:.1f}min) "
              f"checkpoints at steps {sorted(save_steps)}", flush=True)


if __name__ == "__main__":
    main()
