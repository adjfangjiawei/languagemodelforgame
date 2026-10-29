"""
One-Click ~4-Hour End-to-End Orchestrator for GTX 1660 Ti (6 GB VRAM) on Windows / Linux.

Schedule breakdown (~4.0 hours total wall-clock):
  [Stage 1] Parallel CPU Pikafish NNUE 4~5-ply DAG-CoT Dataset Generation (~35 min cap, or skipped if already built)
            -> Generates ~20,000–45,000 high-density transposition-merged CoT dialogues across CPU cores.
  [Stage 2] GTX 1660 Ti FP16 Mixed-Precision Training (`base-45m`, 43.8M params, 10,240 vocab, `max_seq_len=1024`)
            -> Uses remaining ~3.4 hours with dynamic Cosine LR calibration & Windows VRAM auto-guard.
  [Stage 3] Held-Out Validation Evaluation (`eval_metrics.json`) + Live Middlegame CoT Demo (~2 min).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd: list[str]) -> None:
    print("\n>>> Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="One-Click ~4h GTX 1660 Ti Training Pipeline")
    parser.add_argument("--total-hours", type=float, default=4.0, help="Total wall-clock budget in hours (ignored if --steps or --add-steps > 0)")
    parser.add_argument("--data-minutes", type=float, default=35.0, help="Max minutes for initial data gen (0 = unlimited)")
    parser.add_argument("--num-positions", type=int, default=22000, help="Target unique positions (~42,000 mirrored samples)")
    parser.add_argument("--add-positions", type=int, default=0, help="Explicitly generate +N additional new positions")
    parser.add_argument("--steps", type=int, default=0, help="Force exact total training steps (disables time budget)")
    parser.add_argument("--add-steps", type=int, default=0, help="Force exact +N additional training steps (disables time budget)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--preset", type=str, default="fast-4m")
    parser.add_argument("--micro-batch-size", type=int, default=12)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-eval", type=int, default=80)
    args = parser.parse_args()

    t0 = time.time()
    py = sys.executable

    tok_path = os.path.join(PROJECT_ROOT, "tokenizer_assets", "tokenizer_10k.json")
    if not os.path.exists(tok_path):
        run_cmd([py, os.path.join("scripts", "train_tokenizer.py"), "--no-hf-download"])

    train_jsonl = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
    val_jsonl = os.path.join(PROJECT_ROOT, "data", "val.jsonl")
    existing_n = count_lines(train_jsonl)
    strict_step_mode = (args.steps > 0) or (args.add_steps > 0)

    if args.add_positions > 0:
        run_cmd(
            [
                py,
                os.path.join("scripts", "build_dataset_parallel.py"),
                "--add-positions",
                str(args.add_positions),
                "--val-count",
                "500",
                "--time-limit-minutes",
                "0",
                "--workers",
                str(args.workers),
                "--resume",
            ]
        )
    elif existing_n < int(args.num_positions * 1.4):
        data_min = 0.0 if strict_step_mode else args.data_minutes
        run_cmd(
            [
                py,
                os.path.join("scripts", "build_dataset_parallel.py"),
                "--num-positions",
                str(args.num_positions),
                "--val-count",
                "500",
                "--time-limit-minutes",
                str(data_min),
                "--workers",
                str(args.workers),
                "--resume",
            ]
        )
    else:
        print(f"[Stage 1] Dataset already has {existing_n:,} training dialogues; proceeding directly to GPU training!")

    train_cmd = [
        py,
        "train_gpu.py",
        "--preset",
        args.preset,
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--resume",
    ]
    if args.add_steps > 0:
        train_cmd.extend(["--add-steps", str(args.add_steps), "--max-hours", "0"])
    elif args.steps > 0:
        train_cmd.extend(["--steps", str(args.steps), "--max-hours", "0"])
    else:
        elapsed_h = (time.time() - t0) / 3600.0
        train_budget_h = max(0.15, args.total_hours - elapsed_h - 0.06)
        train_cmd.extend(["--max-hours", f"{train_budget_h:.3f}"])

    run_cmd(train_cmd)

    if count_lines(val_jsonl) > 0:
        run_cmd(
            [
                py,
                "inference_chat.py",
                "--eval-val",
                "--num-eval",
                str(min(args.num_eval, count_lines(val_jsonl))),
            ]
        )
    run_cmd([py, "inference_chat.py"])

    print(f"\nAll stages completed in {(time.time() - t0) / 3600.0:.2f} hours!")


if __name__ == "__main__":
    main()
