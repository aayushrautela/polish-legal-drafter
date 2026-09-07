# Vast.ai H100 LoRA Training Recipe

Last updated: 2026-08-30 | Tested on: Vast.ai instance 49189071 (H100 80GB)

## Instance Requirements

- **GPU**: H100 80GB (700W TDP)
- **OS**: Ubuntu with Miniforge3 (conda) at `/opt/miniforge3/`
- **Python**: 3.13.13 (system default in miniforge)
- **Disk**: 64GB+
- **Image**: "Unsloth Studio" (pre-installed but has broken deps)

## Quick Start (2 min)

```bash
# 1. SSH in
ssh -i vast_ai_key -o StrictHostKeyChecking=no -p 50691 root@<IP>

# 2. Fix torchvision (Unsloth requires this)
/opt/miniforge3/bin/pip install --force-reinstall --no-deps --no-cache-dir 'torchvision==0.27.1'

# 3. Install latest unsloth from git (pip version too old for Qwen3.5)
/opt/miniforge3/bin/pip install git+https://github.com/unslothai/unsloth.git \
  git+https://github.com/unslothai/unsloth-zoo.git

# 4. Upgrade transformers (needs >=5.2.0 for Qwen3.5)
/opt/miniforge3/bin/pip install 'transformers>=5.3.0'

# 5. Install training deps
/opt/miniforge3/bin/pip install datasets accelerate peft trl bitsandbytes xformers

# 6. Verify
/opt/miniforge3/bin/python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
from unsloth import FastLanguageModel; print('unsloth OK')
from trl import SFTTrainer; print('trl OK')
import transformers; print('transformers', transformers.__version__)
"
```

## Data Upload

```bash
# From local machine
tar czf - scripts/vast/train_h100.py outputs/qa_pairs_all_final/sft_train_view.jsonl | \
  ssh -i vast_ai_key -p <PORT> root@<IP> "cd /workspace && tar xzf -"
```

## Launch Training

```bash
# On Vast instance (tmux survives disconnects)
tmux new-session -d -s train 'cd /workspace && \
  HF_HOME=/workspace/.hf_home \
  UNSLOTH_SKIP_TORCHVISION_CHECK=1 \
  /opt/miniforge3/bin/python scripts/vast/train_h100.py \
    --data outputs/qa_pairs_all_final/sft_train_view.jsonl \
    --model unsloth/Qwen3.5-2B \
    --out outputs/checkpoints_2b \
    --epochs 2 \
    --max-seq 32768 \
    2>&1 | tee outputs/train.log; echo TRAIN_DONE'
```

## Monitor

```bash
# Status
tmux has-session -t train && echo RUNNING || echo DEAD
tmux capture-pane -t train -p -S -10 | tail -10
tail -1 /workspace/outputs/run_metrics.jsonl

# GPU
nvidia-smi | grep MiB

# Checkpoints saved at steps: 1-10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250, 258
```

## Key Config

| Setting | Value | Notes |
|---------|-------|-------|
| MAX_SEQ | 32768 | Covers ~92% of data (avg 20k tok) |
| Batch | 1 × grad_accum 16 | Effective batch 16 |
| LR | 2e-4 | Cosine schedule |
| LoRA | r=16, alpha=16 | q/k/v/o/gate/up/down |
| Epochs | 2 | Checkpoints let us pick best |
| Truncation | left | Keeps contract at end |

## Pitfalls (learned 2026-08-30)

1. **torchvision mismatch**: Unsloth 2026.8.x bundles torch 2.12.1 but installs torchvision 0.28.0 (needs torch 2.13). Fix: force-reinstall torchvision==0.27.1
2. **transformers too old**: Unsloth pip version needs transformers 4.57.2, but Qwen3.5 needs >=5.2.0. Install unsloth from git + transformers>=5.3.0
3. **HF cache**: Set `HF_HOME=/workspace/.hf_home` — persists model across sessions
4. **UNSLOTH_SKIP_TORCHVISION_CHECK=1**: Bypass the torchvision probe (works fine even with warning)

## Checkpoint Retention

All checkpoints are saved to `outputs/checkpoints_2b/ckpt-{step}/` — never delete. Pick best by loss curve in `outputs/run_metrics.jsonl`.

## Artifact Retention (MANDatory)

After training:
1. Download final adapter: `outputs/qwen35-2b-lora-final/`
2. Download all checkpoints: `outputs/checkpoints_2b/`
3. Download metrics: `outputs/run_metrics.jsonl`
4. Download training data: `outputs/qa_pairs_all_final/`
5. Log to diary with paths
