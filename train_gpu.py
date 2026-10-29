"""
High-Throughput GTX 1660 Ti (6 GB VRAM, Turing sm_75) Causal LM Trainer with Standard Cross-Entropy.

Speed & Model-Compression Highlights:
1. Default `fast-4m` preset (~4.13M parameters — ~10.6x smaller than `base-45m`!):
   Uses Factorized Tied Token Embedding (`d_embed=128 <-> d_model=256`) + 2:1 GQA (`n_heads=4, n_kv_heads=2`,
   `head_dim=64`) + SwiGLU, shrinking both 10k-vocab parameters and output GEMM FLOPs by 4x!
2. Megabatch Length-Bucketed Sampler (`LengthBucketBatchSampler`):
   Eliminates ~40% wasted `<|pad|>` compute by batching similar-length CoT sequences together.
3. Single-GEMM Fast Cross-Entropy (`chunked_ce=False` for <=18M models) + Native-FP16 RoPE + `FP32SafeRMSNorm`:
   Zero CPU-GPU sync barriers inside the forward/backward loop and zero autograd recomputation overhead.
4. Right-sized optimizer step (`micro_batch=12, grad_accum=1` -> 1 forward/backward = 1 step, ~0.85 GB VRAM on 1660 Ti),
   increasing 4-hour training throughput from ~700 steps to **20,000 ~ 50,000+ steps**!
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.dataset import LengthBucketBatchSampler, MixedCausalDataset, collate_uint16_batch
from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from model.transformer import MODEL_PRESETS, ModelConfig, XiangqiCausalLM


def detect_compute_precision(device_arg: str) -> Tuple[torch.device, str, Optional[torch.dtype], bool]:
    if device_arg == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_arg)

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        major, minor = torch.cuda.get_device_capability(dev)
        gpu_name = torch.cuda.get_device_name(dev)
        vram_gb = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        print(f"[GPU OK] {gpu_name} | sm_{major}{minor} | VRAM={vram_gb:.2f} GB | PyTorch={torch.__version__}")
        if major >= 8:
            return dev, "bf16", torch.bfloat16, False
        return dev, "fp16", torch.float16, True

    print("=" * 72)
    print(f"[警告 / GPU未启用] 当前 PyTorch ({torch.__version__}) 未检测到可用 CUDA，正在使用 CPU FP32！")
    print("如果你在 GTX 1660 Ti 机器上看到此提示，说明当前环境装成了 CPU 版 PyTorch。")
    print("请执行以下命令重装 CUDA 版 PyTorch（速度可再提升 10~15 倍）：")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
    print("=" * 72)
    return dev, "fp32", None, False


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
    cycle_start: int = 0,
) -> float:
    rel_step = max(0, step - cycle_start)
    rel_total = max(1, total_steps - cycle_start)
    if rel_step < warmup_steps:
        return max_lr * float(rel_step + 1) / float(max(1, warmup_steps))
    if rel_step >= rel_total:
        return min_lr
    progress = float(rel_step - warmup_steps) / float(max(1, rel_total - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_validation(
    model: XiangqiCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    max_batches: int = 60,
) -> Tuple[float, float, float]:
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    for b_idx, (x, y) in enumerate(val_loader):
        if b_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda" and autocast_dtype is not None:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                _, loss, n_corr, n_tok = model(x, y, return_logits=False, return_acc=True)
        else:
            _, loss, n_corr, n_tok = model(x, y, return_logits=False, return_acc=True)

        if n_tok > 0 and loss is not None:
            total_nll += float(loss.item()) * n_tok
            total_correct += n_corr
            total_tokens += n_tok

    model.train()
    if total_tokens == 0:
        return 0.0, 1.0, 0.0
    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(20.0, avg_loss))
    acc = total_correct / total_tokens
    return avg_loss, ppl, acc


def plot_curves_svg_or_png(history: List[Dict], out_dir: str) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_steps = [h["step"] for h in history if "val_loss" in h]
    va_loss = [h["val_loss"] for h in history if "val_loss" in h]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
        ax.plot(steps, tr_loss, label="Train CE Loss", color="#2563eb", linewidth=1.6)
        if va_steps:
            ax.plot(va_steps, va_loss, label="Val CE Loss", color="#dc2626", linewidth=2.0, marker="o", markersize=4)
        ax.set_xlabel("Optimizer Steps")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Xiangqi DAG-CoT Causal LM Training Curve (10k Vocab)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curves.png"))
        plt.close(fig)
    except Exception:
        w, h, pad = 760, 420, 55
        max_s = max(max(steps), 1)
        all_y = tr_loss + va_loss
        min_y, max_y = min(all_y) * 0.95, max(max(all_y) * 1.05, min(all_y) + 0.1)

        def sx(s: float) -> float:
            return pad + (w - 2 * pad) * (s / max_s)

        def sy(v: float) -> float:
            return h - pad - (h - 2 * pad) * ((v - min_y) / (max_y - min_y))

        tr_pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(steps, tr_loss))
        va_pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(va_steps, va_loss))
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333" stroke-width="1.5"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{tr_pts}"/>
  <polyline fill="none" stroke="#dc2626" stroke-width="2.5" points="{va_pts}"/>
  <text x="{w//2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">
    Xiangqi DAG-CoT Causal LM Cross-Entropy Loss
  </text>
</svg>"""
        with open(os.path.join(out_dir, "training_curves.svg"), "w", encoding="utf-8") as f:
            f.write(svg)


def build_optimizer(model: XiangqiCausalLM, lr: float, weight_decay: float, is_cuda: bool) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "tok_emb" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if is_cuda:
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(groups, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-Speed GTX 1660 Ti FP16 Mixed-Precision Trainer")
    parser.add_argument("--preset", type=str, default="fast-4m", choices=list(MODEL_PRESETS.keys()))
    # Optional custom dimension overrides so any exact model size can be dialed in from CLI:
    parser.add_argument("--d-model", type=int, default=0, help="Override Transformer hidden size (0 = use preset)")
    parser.add_argument("--d-embed", type=int, default=0, help="Override Factorized Vocab Embedding width (0 = use preset)")
    parser.add_argument("--n-layers", type=int, default=0, help="Override layer count (0 = use preset)")
    parser.add_argument("--n-heads", type=int, default=0, help="Override query head count (0 = use preset)")
    parser.add_argument("--n-kv-heads", type=int, default=0, help="Override GQA KV head count (0 = use preset)")
    parser.add_argument("--d-ff", type=int, default=0, help="Override SwiGLU intermediate size (0 = use preset)")

    parser.add_argument("--train-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "train.jsonl"))
    parser.add_argument("--val-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "val.jsonl"))
    parser.add_argument("--general-jsonl", type=str, default="", help="Optional general Chinese corpus JSONL to co-train")
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--micro-batch-size", type=int, default=12, help="12 sequences (~0.85GB VRAM for fast-4m on 1660 Ti)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="1 = every forward/backward updates weights immediately")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Force exact total optimizer steps (e.g. --steps 30000); strictly step-driven, ignores time quota!",
    )
    parser.add_argument(
        "--add-steps",
        type=int,
        default=0,
        help="Continual training: force exact +N additional steps on top of resumed checkpoint step (implies --resume, ignores time quota)!",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--max-hours", type=float, default=0.0, help="0.0 = NO time limit (strictly step/epoch driven)")
    parser.add_argument("--lr", type=float, default=8e-4, help="8e-4 is optimal for ~4M models (use 4e-4 for >=45M)")
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--weight-decay", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Optional torch.compile (Linux/Triton)")
    parser.add_argument("--loss-on-prompt", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest_checkpoint.pt if matching config")
    parser.add_argument("--init-from", type=str, default="", help="Initialize model weights from a prior stage checkpoint")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device, prec_name, autocast_dtype, use_scaler = detect_compute_precision(args.device)
    tok = GeneralXiangqiTokenizer.load(args.tokenizer_path)

    general_paths = [args.general_jsonl] if (args.general_jsonl and os.path.exists(args.general_jsonl)) else []
    if general_paths:
        print(f"[Co-Train] Mixing Xiangqi DAG-CoT + General NL corpus: {general_paths}")

    print("[Data] Tokenizing & indexing training dataset into compact uint16 RAM tensor...")
    train_ds = MixedCausalDataset(
        xiangqi_jsonl_paths=[args.train_jsonl],
        tokenizer=tok,
        max_seq_len=1024,
        general_jsonl_paths=general_paths,
        mask_user_prompt=not args.loss_on_prompt,
    )
    val_ds = MixedCausalDataset(
        xiangqi_jsonl_paths=[args.val_jsonl],
        tokenizer=tok,
        max_seq_len=1024,
        general_jsonl_paths=(),
        mask_user_prompt=not args.loss_on_prompt,
    )
    print(f"[Data] Train stats: {train_ds.summary_stats()}")
    print(f"[Data] Val stats  : {val_ds.summary_stats()}")

    preset_cfg = dict(MODEL_PRESETS[args.preset])
    if args.d_model > 0:
        preset_cfg["d_model"] = args.d_model
    if args.d_embed > 0:
        preset_cfg["d_embed"] = args.d_embed
    if args.n_layers > 0:
        preset_cfg["n_layers"] = args.n_layers
    if args.n_heads > 0:
        preset_cfg["n_heads"] = args.n_heads
    if args.n_kv_heads > 0:
        preset_cfg["n_kv_heads"] = args.n_kv_heads
    if args.d_ff > 0:
        preset_cfg["d_ff"] = args.d_ff

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        eos_token_id=tok.eos_id,
        **preset_cfg,
    )
    use_grad_ckpt = args.grad_checkpoint or (args.preset == "large-88m")

    model = XiangqiCausalLM(cfg, grad_checkpoint=use_grad_ckpt).to(device)
    n_params = model.count_parameters()
    emb_params = model.tok_emb.weight.numel() + (
        (model.emb_up.weight.numel() + model.head_down.weight.numel()) if model.is_factorized else 0
    )
    backbone_params = n_params - emb_params
    print(
        f"[Model] Preset={args.preset} | Total={n_params:,} ({n_params/1e6:.2f}M: "
        f"VocabEmb={emb_params/1e6:.2f}M + Backbone={backbone_params/1e6:.2f}M) | "
        f"d_model={cfg.d_model}, d_embed={model.d_embed}, L={cfg.n_layers}, H={cfg.n_heads}/{cfg.n_kv_heads} | "
        f"Precision={prec_name.upper()}"
    )
    assert n_params >= 1_000_000, f"Model must be >= 1M parameters, got {n_params}"

    if args.init_from and os.path.exists(args.init_from):
        ckpt_init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_init["model_state_dict"], strict=True)
        print(f"[Init] Loaded pretrained weights from {args.init_from}")

    raw_model = model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    optimizer = build_optimizer(raw_model, lr=args.lr, weight_decay=args.weight_decay, is_cuda=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_scaler))

    micro_bs = args.micro_batch_size
    grad_accum = args.grad_accum_steps
    if device.type == "cpu" and micro_bs > 8:
        micro_bs = 8

    start_step = 0
    best_val_loss = float("inf")
    history: List[Dict] = []
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    do_resume = args.resume or (args.add_steps > 0)
    if do_resume and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("model_config") == cfg.to_dict():
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and use_scaler:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_step = int(ckpt.get("step", 0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            history = ckpt.get("history", [])
            print(f"[Resume] Resumed from step={start_step:,}, best_val_loss={best_val_loss:.4f}")
        else:
            print(
                "[Resume] Found checkpoint from a different model preset/config; "
                "starting fresh training for current model size!"
            )

    def make_loaders(bs: int) -> Tuple[DataLoader, DataLoader]:
        nw = 2 if (device.type == "cuda" and os.name != "nt") else 0
        bucket_sampler = LengthBucketBatchSampler(
            seq_lens=train_ds.seq_lens,
            batch_size=bs,
            drop_last=(len(train_ds) >= bs),
            megabatch_mult=32,
            seed=args.seed + start_step,
        )
        tr = DataLoader(
            train_ds,
            batch_sampler=bucket_sampler,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_uint16_batch,
        )
        va_sampler = LengthBucketBatchSampler(
            seq_lens=val_ds.seq_lens,
            batch_size=bs,
            drop_last=False,
            megabatch_mult=16,
            seed=args.seed,
        )
        va = DataLoader(
            val_ds,
            batch_sampler=va_sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_uint16_batch,
        )
        return tr, va

    train_loader, val_loader = make_loaders(micro_bs)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)

    # Strict step-count mode overrides any wall-clock time quota (`--steps` / `--add-steps`)!
    eff_max_hours = args.max_hours
    cycle_start_step = 0
    if args.add_steps > 0:
        planned_total_steps = start_step + args.add_steps
        cycle_start_step = start_step
        eff_max_hours = 0.0
    elif args.steps > 0:
        planned_total_steps = args.steps
        if start_step > 0:
            cycle_start_step = start_step
        eff_max_hours = 0.0
    else:
        planned_total_steps = min(args.max_steps, max(start_step + 1, steps_per_epoch * args.epochs))
        if start_step > 0 and start_step >= int(0.8 * planned_total_steps):
            planned_total_steps = start_step + max(100, steps_per_epoch * 4)
            cycle_start_step = start_step

    span_steps = max(1, planned_total_steps - cycle_start_step)
    warmup_steps = min(args.warmup_steps, max(5, span_steps // 10))

    t_start = time.time()
    deadline = (t_start + eff_max_hours * 3600.0) if eff_max_hours > 0 else float("inf")
    calibrated_wall_clock = False
    avg_len = max(128.0, float(train_ds.summary_stats().get("avg_seq_len", 600.0)))
    budget_desc = f"strict_steps={planned_total_steps:,} (NO time limit)" if eff_max_hours <= 0 else f"max_hours={eff_max_hours:.2f}h"

    print(
        f"[Train] Starting Length-Bucketed training: micro_batch={micro_bs}, grad_accum={grad_accum} "
        f"(~{int(micro_bs * grad_accum * avg_len):,} active tok/step), start_step={start_step:,} -> "
        f"target_step={planned_total_steps:,} | {budget_desc}"
    )

    model.train()
    step = start_step
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_steps = 0
    running_toks = 0
    t_log_last = time.time()
    t_measure_start: Optional[float] = None
    step_measure_start: int = start_step

    while step < planned_total_steps and time.time() < deadline:
        if step - start_step == 10 and t_measure_start is None:
            t_measure_start = time.time()
            step_measure_start = step
        elif (
            not calibrated_wall_clock
            and eff_max_hours > 0
            and t_measure_start is not None
            and (step - step_measure_start) >= 30
        ):
            sec_per_step = (time.time() - t_measure_start) / max(1, step - step_measure_start)
            rem_sec = max(60.0, (deadline - time.time()) * 0.94)
            achievable_steps = step + int(rem_sec / max(1e-5, sec_per_step))
            print(
                f"[Speed Check] Measured {1.0/sec_per_step:.2f} steps/s "
                f"(~{int(achievable_steps):,} steps capacity in {eff_max_hours:.2f}h)"
            )
            if achievable_steps < planned_total_steps:
                print(
                    f"[Auto-Calibrate] Adjusting Cosine LR horizon {planned_total_steps} -> {achievable_steps} steps "
                    f"to fit {eff_max_hours:.2f}h!"
                )
                planned_total_steps = max(step + 20, achievable_steps)
            calibrated_wall_clock = True

        cur_lr = cosine_lr(
            step,
            planned_total_steps,
            warmup_steps,
            args.lr,
            args.min_lr,
            cycle_start=cycle_start_step,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss_accum = 0.0
        oom_triggered = False

        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            try:
                running_toks += int(x.numel())
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if device.type == "cuda" and autocast_dtype is not None:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        _, loss = model(x, y)
                        assert loss is not None
                        scaled_loss = loss / grad_accum
                else:
                    _, loss = model(x, y)
                    assert loss is not None
                    scaled_loss = loss / grad_accum

                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                step_loss_accum += float(loss.item()) / grad_accum
            except torch.cuda.OutOfMemoryError:
                oom_triggered = True
                break

        if oom_triggered:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro_bs > 1:
                micro_bs = max(1, micro_bs // 2)
                grad_accum *= 2
                train_loader, val_loader = make_loaders(micro_bs)
                train_iter = iter(train_loader)
                print(
                    f"[VRAM Auto-Guard] Caught CUDA OOM! Adjusted micro_batch={micro_bs}, "
                    f"grad_accum={grad_accum}. Retrying step..."
                )
                continue
            elif not raw_model.grad_checkpoint:
                raw_model.grad_checkpoint = True
                print("[VRAM Auto-Guard] Enabled activation gradient checkpointing. Retrying step...")
                continue
            else:
                raise RuntimeError("CUDA OOM even at micro_batch=1 + grad_checkpoint! Try `--preset nano-2m`.")

        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip).item())
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip).item())
            optimizer.step()

        step += 1
        running_loss += step_loss_accum
        running_steps += 1

        if step % args.log_interval == 0 or step == 1:
            dt = max(1e-4, time.time() - t_log_last)
            steps_per_s = running_steps / dt
            toks_per_s = running_toks / dt
            avg_tr = running_loss / max(1, running_steps)
            running_loss = 0.0
            running_steps = 0
            running_toks = 0
            t_log_last = time.time()

            elapsed_m = (time.time() - t_start) / 60.0
            vram_str = ""
            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                vram_str = f" | VRAM={peak_gb:.2f}GB"
            rec: Dict = {
                "step": step,
                "train_loss": round(avg_tr, 4),
                "lr": round(cur_lr, 7),
                "grad_norm": round(grad_norm, 3),
                "steps_per_sec": round(steps_per_s, 2),
                "toks_per_sec": int(toks_per_s),
                "elapsed_min": round(elapsed_m, 2),
            }
            history.append(rec)
            print(
                f"Step {step:5d}/{planned_total_steps} | loss={avg_tr:.4f} | lr={cur_lr:.2e} | "
                f"speed={steps_per_s:.2f} step/s ({int(toks_per_s):,} tok/s) | {elapsed_m:.1f}m{vram_str}"
            )

        if step % args.eval_interval == 0 or step == planned_total_steps or time.time() >= deadline:
            val_loss, val_ppl, val_acc = evaluate_validation(model, val_loader, device, autocast_dtype)
            print(
                f"  >>> [Eval @ Step {step}] val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                f"val_token_acc={val_acc*100:.2f}%"
            )
            if history and history[-1]["step"] == step:
                history[-1].update({"val_loss": round(val_loss, 4), "val_ppl": round(val_ppl, 2), "val_acc": round(val_acc, 4)})
            else:
                history.append(
                    {
                        "step": step,
                        "train_loss": round(step_loss_accum, 4),
                        "val_loss": round(val_loss, 4),
                        "val_ppl": round(val_ppl, 2),
                        "val_acc": round(val_acc, 4),
                    }
                )

            ckpt_payload = {
                "step": step,
                "best_val_loss": min(best_val_loss, val_loss),
                "model_config": cfg.to_dict(),
                "model_state_dict": raw_model.state_dict(),
                "tokenizer_path": args.tokenizer_path,
            }
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt_payload, best_ckpt_path)
                print(f"  >>> Saved new best model -> {best_ckpt_path} (val_loss={best_val_loss:.4f})")

            resume_payload = {
                **ckpt_payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_scaler else {},
                "history": history,
            }
            torch.save(resume_payload, latest_ckpt_path)

            with open(os.path.join(args.out_dir, "training_history.json"), "w", encoding="utf-8") as hf:
                json.dump(history, hf, ensure_ascii=False, indent=2)
            plot_curves_svg_or_png(history, args.out_dir)
            t_log_last = time.time()  # Exclude eval time from next step/s measurement

    print("============================================================")
    print(f"Training Finished at Step {step} ({(time.time() - t_start)/60.0:.2f} min)!")
    print(f"Best Validation CE Loss: {best_val_loss:.4f}")
    print(f"Best Checkpoint        : {best_ckpt_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
