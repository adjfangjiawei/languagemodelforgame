"""
One-Click Continual Data Expansion + Forced-Step SFT Resumption Orchestrator (Zero Wall-Clock Limit).

Workflow:
  1. Optionally generates `--add-positions N` brand-new unique canonical boards via Pikafish NNUE +
     5-strategy Variation-DAG CoT (`seen_canonical_u64.bin` guarantees 0% overlap with all prior batches!).
  2. Incrementally tokenizes ONLY the newly appended JSONL lines into the disk `np.memmap` cache.
  3. Resumes `checkpoints/latest_checkpoint.pt` (weights + AdamW moments + GradScaler) and trains
     for an EXACT forced step count (`--add-steps K` or `--steps TOTAL`) with NO time quota!

Examples:
  # Generate +5,000 new positions (~10,000 mirrored CoT samples) and train +12,000 additional steps:
  python continue_sft.py --add-positions 5000 --add-steps 12000

  # Skip data generation and force-train existing dataset up to exact step 50,000:
  python continue_sft.py --add-positions 0 --steps 50000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd: list[str]) -> None:
    print("\n>>> Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental CoT Data Expansion + Exact-Step Continual SFT")
    parser.add_argument("--add-positions", type=int, default=5000, help="Number of NEW unique positions to generate (0 = skip)")
    parser.add_argument("--add-steps", type=int, default=10000, help="Exact additional optimizer steps to train on top of current step")
    parser.add_argument("--steps", type=int, default=0, help="Alternatively specify exact absolute total target step (overrides --add-steps)")
    parser.add_argument("--preset", type=str, default="fast-4m")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--micro-batch-size", type=int, default=12)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--shard-size", type=int, default=0, help="Optional shard split size for 1M~10M+ datasets")
    parser.add_argument("--train-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "train*.jsonl"))
    parser.add_argument("--general-jsonl", type=str, default="", help="Optional general Chinese JSONL to mix during continual SFT")
    args = parser.parse_args()

    py = sys.executable

    if args.add_positions > 0:
        data_cmd = [
            py,
            os.path.join("scripts", "build_dataset_parallel.py"),
            "--add-positions",
            str(args.add_positions),
            "--workers",
            str(args.workers),
            "--time-limit-minutes",
            "0",
            "--resume",
        ]
        if args.shard_size > 0:
            data_cmd.extend(["--shard-size", str(args.shard_size)])
        run_cmd(data_cmd)

    train_cmd = [
        py,
        "train_gpu.py",
        "--preset",
        args.preset,
        "--train-jsonl",
        args.train_jsonl,
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--lr",
        str(args.lr),
        "--max-hours",
        "0",
        "--resume",
    ]
    if args.steps > 0:
        train_cmd.extend(["--steps", str(args.steps)])
    else:
        train_cmd.extend(["--add-steps", str(max(1, args.add_steps))])

    if args.general_jsonl:
        train_cmd.extend(["--general-jsonl", args.general_jsonl])

    run_cmd(train_cmd)


if __name__ == "__main__":
    main()
