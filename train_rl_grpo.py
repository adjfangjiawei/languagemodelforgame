"""
Post-Training Reinforcement Learning CoT Exploration Engine (DeepSeek-R1 / DAPO-Style GRPO).

Designed to run AFTER Supervised Fine-Tuning (whether after 20K or 10,000,000+ SFT positions) to let
the model actively explore novel reasoning paths (`<|think|> ... <|/think|>`) via Reinforcement Learning
with Verifiable Engine & Board-Rule Rewards (RLVR) — NO separate Critic/Value network required!

Core RL Algorithm (GRPO + DAPO enhancements):
1. Batched Parallel Group Rollout (`generate_group`, G=4..8 trajectories per board at T=0.85).
2. 4-Pillar Verifiable Xiangqi Reward Oracle `R(s, o)`:
   - `R_format`      : Valid `<|think|>...<|/think|>\n最佳走法：X` + readable Chinese connector preservation
   - `R_legal`       : Exact Xiangqi legality verification (`+0.30` legal, `-0.60` illegal)
   - `R_strength`    : Continuous Pikafish NNUE centipawn-regret reward (`+1.50` for Master optimal `Δcp <= 5`,
                       smooth decay for inaccuracies, negative penalty down to `-1.00` for blunders!)
   - `R_cot_process` : Process-verified CoT self-consistency (chosen move actually analyzed in `<|think|>`) +
                       grounded legal candidate exploration + DAG/Trie merging bonus!
3. DAPO Asymmetric Clip-Higher (`eps_low=0.20, eps_high=0.28`) + Dynamic Zero-Variance Group Resampling +
   Schulman Unbiased Reference-Model KL Penalty (`pi_ref` frozen in FP16 to prevent gibberish / collapse).
4. Strictly Step-Driven (`--steps N` / `--add-steps K`, zero wall-clock cutoff) + O(1)-RAM streaming over
   10M+-line sharded FEN/JSONL corpora + online self-play exploration states!

Usage examples:
  # Run 3,000 GRPO RL exploration steps starting from your SFT checkpoint:
  python train_rl_grpo.py --init-from checkpoints/best_model.pt --steps 3000 --group-size 6

  # Continue RL for +2,000 more forced steps from latest_rl_checkpoint.pt:
  python train_rl_grpo.py --resume --add-steps 2000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.dataset import expand_jsonl_paths
from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from model.transformer import MODEL_PRESETS, ModelConfig, XiangqiCausalLM
from scripts.build_dataset_parallel import OPENING_BOOK
from train_gpu import build_optimizer, cosine_lr, detect_compute_precision
from xiangqi.board import START_FEN, XiangqiBoard
from xiangqi.dag_cot import format_user_prompt
from xiangqi.engine_wrapper import MoveEval, PikafishEngine

CHINESE_MOVE_RE = re.compile(
    r"((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])"
)
FINAL_ANS_RE = re.compile(
    r"最佳走法\s*[：:]\s*((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])"
)


@dataclass
class RolloutEval:
    response_ids: List[int]
    decoded_text: str
    pred_move_zh: str
    reward_total: float
    r_format: float
    r_legal: float
    r_strength: float
    r_process: float
    is_legal: bool
    is_master_optimal: bool
    cp_regret: float
    self_consistent: bool


class VerifiableXiangqiRewardEngine:
    """
    Computes verifiable outcome + CoT process rewards using `XiangqiBoard` rules and `PikafishEngine` NNUE.
    Caches Pikafish evaluations by `state_key` so repeated positions & identical candidate moves across a
    G-rollout group only query the engine once!
    """

    def __init__(self, engine: PikafishEngine, eval_depth: int = 8):
        self.engine = engine
        self.eval_depth = eval_depth
        self.root_cache: "OrderedDict[str, Tuple[int, Dict[str, int]]]" = OrderedDict()
        self.after_move_cache: "OrderedDict[str, int]" = OrderedDict()

    def _get_root_eval(self, board: XiangqiBoard) -> Tuple[int, Dict[str, int]]:
        skey = board.state_key()
        if skey in self.root_cache:
            self.root_cache.move_to_end(skey)
            return self.root_cache[skey]

        evs: List[MoveEval] = self.engine.evaluate_multipv(board.to_fen(), depth=self.eval_depth, multipv=8)
        if not evs:
            return 0, {}
        best_cp = evs[0].score_cp
        uci_scores = {e.uci: e.score_cp for e in evs}
        self.root_cache[skey] = (best_cp, uci_scores)
        if len(self.root_cache) > 8192:
            self.root_cache.popitem(last=False)
        return best_cp, uci_scores

    def _score_specific_move(self, board: XiangqiBoard, uci: str, uci_scores: Dict[str, int]) -> int:
        if uci in uci_scores:
            return uci_scores[uci]
        nxt = board.copy()
        nxt.push_uci(uci)
        # Check immediate checkmate/stalemate delivery!
        if not nxt.generate_legal_moves():
            return 9999
        akey = nxt.state_key()
        if akey in self.after_move_cache:
            self.after_move_cache.move_to_end(akey)
            return self.after_move_cache[akey]

        opp_evs = self.engine.evaluate_multipv(nxt.to_fen(), depth=max(5, self.eval_depth - 1), multipv=1)
        mover_cp = -opp_evs[0].score_cp if opp_evs else 0
        self.after_move_cache[akey] = mover_cp
        if len(self.after_move_cache) > 16384:
            self.after_move_cache.popitem(last=False)
        return mover_cp

    def evaluate_group(
        self,
        board: XiangqiBoard,
        rollout_id_lists: List[List[int]],
        tokenizer: GeneralXiangqiTokenizer,
    ) -> List[RolloutEval]:
        legal_zh_to_uci: Dict[str, str] = {}
        for uci in board.generate_legal_moves():
            try:
                legal_zh_to_uci[board.uci_to_zh(uci)] = uci
            except Exception:
                pass

        best_cp, uci_scores = self._get_root_eval(board)
        results: List[RolloutEval] = []

        for r_ids in rollout_id_lists:
            # Recall prompt already ends with `<|think|>\n`, so prepend `<|think|>\n` for full structural view
            txt = "<|think|>\n" + tokenizer.decode(r_ids, skip_special_tokens=False).replace("<|eos|>", "")

            has_close_think = "<|/think|>" in txt
            cot_part, tail_part = txt.split("<|/think|>", 1) if has_close_think else (txt, "")
            cot_body = cot_part.replace("<|think|>", "").strip()

            m_ans = FINAL_ANS_RE.search(tail_part)
            pred_zh = m_ans.group(1).strip() if m_ans else ""

            # 1. Format & Readability Reward `R_format` in [-0.50, +0.20]
            if has_close_think and pred_zh:
                r_format = 0.15
                # Preserve natural Chinese connectors (penalize connector-stripping or degenerate token loops!)
                has_cn_conn = any(w in cot_body for w in ("若", "如果", "此后", "或", "应", "走"))
                unique_char_ratio = len(set(cot_body)) / max(1, min(len(cot_body), 120))
                if has_cn_conn and len(cot_body) >= 24 and unique_char_ratio >= 0.16:
                    r_format += 0.05
                elif unique_char_ratio < 0.09:
                    r_format -= 0.25
            else:
                r_format = -0.50

            # 2. Exact Rules Legality `R_legal` & Engine Regret `R_strength`
            is_legal = bool(pred_zh and pred_zh in legal_zh_to_uci)
            is_opt = False
            cp_regret = 999.0

            if not is_legal:
                r_legal = -0.60
                r_strength = -0.50
            else:
                r_legal = 0.30
                pred_uci = legal_zh_to_uci[pred_zh]
                move_cp = self._score_specific_move(board, pred_uci, uci_scores)
                cp_regret = max(0.0, float(best_cp - move_cp))

                if move_cp >= 9000 or cp_regret <= 5.0:
                    r_strength = 1.50
                    is_opt = True
                elif cp_regret <= 38.0:
                    r_strength = 0.85 + 0.65 * math.exp(-(cp_regret - 5.0) / 22.0)
                elif cp_regret <= 120.0:
                    r_strength = 0.65 * (1.0 - (cp_regret - 38.0) / 82.0)
                else:
                    r_strength = -min(1.00, (cp_regret - 120.0) / 280.0)

            # 3. Verifiable CoT Process Grounding `R_process` in [0.0, +0.45]
            r_process = 0.0
            self_consistent = False
            if is_legal and len(cot_body) >= 20:
                mentioned_moves = CHINESE_MOVE_RE.findall(cot_body)
                if pred_zh in mentioned_moves:
                    self_consistent = True
                    r_process += 0.15

                # Check grounded legal root candidates explored in `<|think|>`
                distinct_legal_roots: Set[str] = set()
                for mv in mentioned_moves[:25]:
                    if mv in legal_zh_to_uci:
                        distinct_legal_roots.add(mv)

                if len(distinct_legal_roots) >= 2:
                    r_process += 0.15
                elif len(distinct_legal_roots) == 1:
                    r_process += 0.07

                # Reward structured readable Variation-DAG / Trie merging phrases (only when outcome move is strong!)
                if cp_regret <= 45.0:
                    if ("同一盘面" in cot_body or "同形" in cot_body or ("如果" in cot_body and "或者" in cot_body)):
                        r_process += 0.10
                    if "此后" in cot_body and "或" in cot_body:
                        r_process += 0.05

            total_r = r_format + r_legal + r_strength + r_process
            results.append(
                RolloutEval(
                    response_ids=r_ids,
                    decoded_text=txt,
                    pred_move_zh=pred_zh,
                    reward_total=round(total_r, 5),
                    r_format=round(r_format, 4),
                    r_legal=round(r_legal, 4),
                    r_strength=round(r_strength, 4),
                    r_process=round(r_process, 4),
                    is_legal=is_legal,
                    is_master_optimal=is_opt,
                    cp_regret=round(cp_regret, 1),
                    self_consistent=self_consistent,
                )
            )
        return results


class RLPositionStream:
    """
    O(1)-RAM Infinite Xiangqi Position Environment for Large-Scale Post-Training RL:
      - `dataset`          : Streams FENs from 1..N JSONL shards (safe on 10,000,000+ line files!)
      - `selfplay_explore` : Generates fresh online opening/middlegame/endgame states via Pikafish rollouts
      - `hybrid`           : Alternates dataset FENs and live online exploration states!
    """

    def __init__(self, jsonl_pattern: str, engine: PikafishEngine, mode: str = "hybrid", seed: int = 2026):
        self.files = expand_jsonl_paths([jsonl_pattern]) if jsonl_pattern else []
        self.engine = engine
        self.mode = mode if self.files else "selfplay_explore"
        self.rng = random.Random(seed)
        self._file_idx = 0
        self._fp: Optional[object] = None
        self._online_buffer: List[str] = []

    def _next_from_files(self) -> Optional[str]:
        if not self.files:
            return None
        for _ in range(len(self.files) + 1):
            if self._fp is None:
                p = self.files[self._file_idx % len(self.files)]
                self._file_idx += 1
                self._fp = open(p, "r", encoding="utf-8")
            line = self._fp.readline()  # type: ignore
            if not line:
                self._fp.close()  # type: ignore
                self._fp = None
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fen = obj.get("fen", "")
                if fen:
                    b = XiangqiBoard(fen)
                    if b.is_nnue_valid() and len(b.generate_legal_moves()) >= 2:
                        return b.to_fen()
            except Exception:
                continue
        return None

    def _refill_online_selfplay(self) -> None:
        b = XiangqiBoard(START_FEN)
        if self.rng.random() < 0.8:
            _, moves = self.rng.choice(OPENING_BOOK)
            for uci in moves[: self.rng.randint(2, len(moves))]:
                if uci in b.generate_legal_moves():
                    b.push_uci(uci)
        plies = self.rng.randint(12, 48)
        for p in range(plies):
            legal = b.generate_legal_moves()
            if len(legal) < 2:
                break
            evs = self.engine.evaluate_multipv(b.to_fen(), depth=4, multipv=5)
            if not evs or abs(evs[0].score_cp) >= 9200:
                break
            pick = self.rng.choice(evs[: min(len(evs), 4)]).uci
            b.push_uci(pick)
            if p % 3 == 0 and b.is_nnue_valid() and len(b.generate_legal_moves()) >= 2:
                self._online_buffer.append(b.to_fen())
        self.rng.shuffle(self._online_buffer)

    def next_fen(self) -> str:
        use_online = (self.mode == "selfplay_explore") or (self.mode == "hybrid" and self.rng.random() < 0.35)
        if not use_online:
            fen = self._next_from_files()
            if fen:
                return fen
        if not self._online_buffer:
            self._refill_online_selfplay()
        if self._online_buffer:
            return self._online_buffer.pop()
        return START_FEN


def build_exempt_ids(tok: GeneralXiangqiTokenizer) -> Set[int]:
    exempt: Set[int] = set(range(16))
    for s in ["\n", "，", "。", "；", "：", "、", "（", "）", "【", "】", "或", "或者", "若", "则", "红", "黑"]:
        exempt.update(tok.encode(s))
    return exempt


def pack_rollout_tensors(
    prompt_ids_list: List[List[int]],
    response_ids_list: List[List[int]],
    advantages: List[float],
    pad_id: int,
    eos_id: int,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack variable-length `(prompt_ids, response_ids + [eos_id])` pairs into aligned `[N, T]` tensors:
      - `input_ids`   : `[N, T]`
      - `target_ids`  : `[N, T]`
      - `action_mask` : `[N, T]` float32 (`1.0` strictly on generated CoT + Move + EOS tokens, `0.0` on prompt/pad!)
      - `adv_tensor`  : `[N, 1]` float32 group-normalized advantage
    """
    N = len(prompt_ids_list)
    seqs: List[List[int]] = []
    p_lens: List[int] = []
    for p_ids, r_ids in zip(prompt_ids_list, response_ids_list):
        full = (p_ids + r_ids + [eos_id])[: max_seq_len + 1]
        seqs.append(full)
        p_lens.append(len(p_ids))

    max_t = max(len(s) - 1 for s in seqs)
    max_t = min(max_seq_len, ((max_t + 15) // 16) * 16)

    inp = torch.full((N, max_t), pad_id, dtype=torch.long, device=device)
    tgt = torch.full((N, max_t), pad_id, dtype=torch.long, device=device)
    msk = torch.zeros((N, max_t), dtype=torch.float32, device=device)

    for i, (s, plen) in enumerate(zip(seqs, p_lens)):
        L = min(len(s) - 1, max_t)
        inp[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)
        tgt[i, :L] = torch.tensor(s[1 : L + 1], dtype=torch.long, device=device)
        # `tgt[i, plen - 1]` is the FIRST generated token `r_ids[0]`!
        act_start = max(0, plen - 1)
        if L > act_start:
            msk[i, act_start:L] = 1.0

    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(1)
    return inp, tgt, msk, adv_t


def plot_rl_curves(history: List[Dict], out_dir: str) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    opt_rates = [h["master_optimal_rate"] * 100.0 for h in history]
    leg_rates = [h["legal_rate"] * 100.0 for h in history]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
        ax1.plot(steps, rewards, color="#16a34a", linewidth=1.8, label="Mean Verifiable Reward")
        ax1.set_xlabel("RL GRPO Steps")
        ax1.set_ylabel("Reward")
        ax1.set_title("GRPO Post-Training CoT Reward")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(steps, leg_rates, color="#2563eb", linewidth=1.6, label="Legal Move %")
        ax2.plot(steps, opt_rates, color="#d97706", linewidth=1.8, label="Master Optimal (Δcp<=5) %")
        ax2.set_xlabel("RL GRPO Steps")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Move Legality & Master Optimality")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rl_training_curves.png"))
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-R1 / DAPO Style GRPO RL CoT Post-Trainer")
    parser.add_argument(
        "--init-from",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
        help="Pretrained / SFT checkpoint to initialize policy & frozen reference model from",
    )
    parser.add_argument("--preset", type=str, default="fast-4m", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--fen-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "train*.jsonl"))
    parser.add_argument("--env-mode", type=str, default="hybrid", choices=["hybrid", "dataset", "selfplay_explore"])
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs"))
    parser.add_argument("--device", type=str, default="auto")

    # Strict Step-Count Control (No wall-clock cutoff!)
    parser.add_argument("--steps", type=int, default=2000, help="Exact total RL steps to run (strictly step-driven)")
    parser.add_argument("--add-steps", type=int, default=0, help="Exact +N additional RL steps on top of resumed step")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest_rl_checkpoint.pt")
    parser.add_argument("--sync-best-to-main", action="store_true", help="Also copy improved RL model to best_model.pt for instant Web UI play")

    # GRPO / DAPO Hyperparameters
    parser.add_argument("--group-size", type=int, default=6, help="G rollouts sampled per board position")
    parser.add_argument("--prompts-per-step", type=int, default=2, help="Number of distinct board positions per RL step")
    parser.add_argument("--max-cot-tokens", type=int, default=768, help="Max generated CoT + Move tokens per RL rollout")
    parser.add_argument("--temperature", type=float, default=0.85, help="Exploration sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=2, help="Clipped policy update epochs per rollout batch")
    parser.add_argument("--mini-batch-seqs", type=int, default=6, help="Sequences per backward sub-batch (fits 6GB VRAM easily)")
    parser.add_argument("--clip-eps-low", type=float, default=0.20)
    parser.add_argument("--clip-eps-high", type=float, default=0.28, help="DAPO Clip-Higher encourages novel reasoning exploration!")
    parser.add_argument("--kl-beta", type=float, default=0.02, help="Reference-model KL coefficient preserving natural Chinese CoT")
    parser.add_argument("--lr", type=float, default=2e-5, help="RL post-training peak learning rate")
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-depth", type=int, default=7, help="Pikafish NNUE search depth for reward oracle")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device, prec_name, autocast_dtype, use_scaler = detect_compute_precision(args.device)
    tok = GeneralXiangqiTokenizer.load(args.tokenizer_path)
    exempt_ids = build_exempt_ids(tok)
    force_close_ids = tok.encode("\n<|/think|>\n最佳走法：")

    # 1. Load Policy Model (from SFT `--init-from` or fresh preset if running unit test)
    if os.path.exists(args.init_from):
        sft_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(sft_ckpt["model_config"])
        policy = XiangqiCausalLM(cfg, grad_checkpoint=(cfg.d_model >= 512)).to(device)
        policy.load_state_dict(sft_ckpt["model_state_dict"], strict=True)
        print(f"[RL Init] Loaded SFT policy from {args.init_from} ({policy.count_parameters()/1e6:.2f}M params)")
    else:
        print(f"[RL Warning] SFT checkpoint not found at {args.init_from}; initializing from preset={args.preset}.")
        cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            pad_token_id=tok.pad_id,
            eos_token_id=tok.eos_id,
            **MODEL_PRESETS[args.preset],
        )
        policy = XiangqiCausalLM(cfg).to(device)

    # 2. Create Frozen Reference Policy `pi_ref` (anchored to initial SFT model!)
    ref_policy = copy.deepcopy(policy).to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    if device.type == "cuda" and autocast_dtype is not None:
        ref_policy = ref_policy.to(dtype=autocast_dtype)  # ~8 MB VRAM for fast-4m!

    optimizer = build_optimizer(policy, lr=args.lr, weight_decay=0.01, is_cuda=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_scaler))

    latest_rl_path = os.path.join(args.checkpoint_dir, "latest_rl_checkpoint.pt")
    best_rl_path = os.path.join(args.checkpoint_dir, "best_rl_model.pt")
    main_best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    start_step = 0
    best_moving_reward = -1e9
    history: List[Dict] = []

    do_resume = args.resume or (args.add_steps > 0)
    if do_resume and os.path.exists(latest_rl_path):
        rckpt = torch.load(latest_rl_path, map_location=device, weights_only=False)
        if rckpt.get("model_config") == cfg.to_dict():
            policy.load_state_dict(rckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
            if "scaler_state_dict" in rckpt and use_scaler:
                scaler.load_state_dict(rckpt["scaler_state_dict"])
            start_step = int(rckpt.get("step", 0))
            best_moving_reward = float(rckpt.get("best_moving_reward", -1e9))
            history = rckpt.get("history", [])
            print(f"[RL Resume] Resumed GRPO at step={start_step:,}, best_moving_reward={best_moving_reward:.4f}")

    if args.add_steps > 0:
        target_steps = start_step + args.add_steps
        cycle_start = start_step
    else:
        target_steps = args.steps
        cycle_start = start_step if start_step > 0 else 0

    warmup = min(args.warmup_steps, max(2, (target_steps - cycle_start) // 10))

    print(
        f"[GRPO Start] Steps: {start_step:,} -> {target_steps:,} (STRICT STEP MODE, NO TIME LIMIT) | "
        f"G={args.group_size} x P={args.prompts_per_step} = {args.group_size * args.prompts_per_step} rollouts/step | "
        f"T={args.temperature} | KL_beta={args.kl_beta}"
    )

    with PikafishEngine(hash_mb=48, threads=1) as eng:
        reward_oracle = VerifiableXiangqiRewardEngine(eng, eval_depth=args.eval_depth)
        pos_stream = RLPositionStream(args.fen_jsonl, eng, mode=args.env_mode, seed=args.seed + start_step)

        step = start_step
        moving_reward: Optional[float] = None
        acc_metrics = {
            "reward": 0.0,
            "max_r": 0.0,
            "legal": 0.0,
            "opt": 0.0,
            "regret": 0.0,
            "consist": 0.0,
            "kl": 0.0,
            "cot_toks": 0.0,
            "count": 0,
        }
        t0 = time.time()

        while step < target_steps:
            t_step = time.time()
            cur_lr = cosine_lr(step, target_steps, warmup, args.lr, args.min_lr, cycle_start=cycle_start)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            batch_prompt_ids: List[List[int]] = []
            batch_resp_ids: List[List[int]] = []
            batch_advantages: List[float] = []
            step_evals: List[RolloutEval] = []

            # Phase 1 & 2: Sample P boards and run batched G-rollout exploration per board!
            policy.eval()
            collected_groups = 0
            resample_tries = 0

            while collected_groups < args.prompts_per_step:
                fen = pos_stream.next_fen()
                board = XiangqiBoard(fen)
                u_prompt = format_user_prompt(board.to_fen())
                p_ids = tok.encode(f"<|user|>\n{u_prompt}\n<|assistant|>\n<|think|>\n")

                if device.type == "cuda" and autocast_dtype is not None:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        r_id_lists = policy.generate_group(
                            prompt_ids=p_ids,
                            group_size=args.group_size,
                            max_new_tokens=args.max_cot_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=1.10,
                            exempt_rep_ids=exempt_ids,
                            eos_token_id=tok.eos_id,
                            end_think_id=tok.end_think_id,
                            force_close_ids=force_close_ids,
                            device=device,
                        )
                else:
                    r_id_lists = policy.generate_group(
                        prompt_ids=p_ids,
                        group_size=args.group_size,
                        max_new_tokens=args.max_cot_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=1.10,
                        exempt_rep_ids=exempt_ids,
                        eos_token_id=tok.eos_id,
                        end_think_id=tok.end_think_id,
                        force_close_ids=force_close_ids,
                        device=device,
                    )

                evals = reward_oracle.evaluate_group(board, r_id_lists, tok)
                r_vals = torch.tensor([ev.reward_total for ev in evals], dtype=torch.float32)
                r_std = float(r_vals.std(unbiased=False).item())

                # DAPO Dynamic Zero-Variance Filter: if all G rollouts got identical reward, try another board (up to 2x)
                if r_std < 1e-4 and resample_tries < 2:
                    resample_tries += 1
                    continue

                r_mean = float(r_vals.mean().item())
                advs = ((r_vals - r_mean) / (r_std + 1e-5)).tolist() if r_std >= 1e-5 else [0.0] * len(evals)

                for ev, adv in zip(evals, advs):
                    batch_prompt_ids.append(p_ids)
                    batch_resp_ids.append(ev.response_ids)
                    batch_advantages.append(float(adv))
                    step_evals.append(ev)

                collected_groups += 1

            # Phase 3: Compute Old & Reference Per-Token Log-Probs under `no_grad`
            inp_all, tgt_all, msk_all, adv_all = pack_rollout_tensors(
                batch_prompt_ids,
                batch_resp_ids,
                batch_advantages,
                pad_id=tok.pad_id,
                eos_id=tok.eos_id,
                max_seq_len=cfg.max_seq_len,
                device=device,
            )
            N_seq = inp_all.shape[0]
            mb = max(1, min(args.mini_batch_seqs, N_seq))

            old_logp_chunks: List[torch.Tensor] = []
            ref_logp_chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for s_i in range(0, N_seq, mb):
                    ic = inp_all[s_i : s_i + mb]
                    tc = tgt_all[s_i : s_i + mb]
                    if device.type == "cuda" and autocast_dtype is not None:
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            olp, _ = policy.compute_token_logprobs(ic, tc)
                            rlp, _ = ref_policy.compute_token_logprobs(ic, tc)
                    else:
                        olp, _ = policy.compute_token_logprobs(ic, tc)
                        rlp, _ = ref_policy.compute_token_logprobs(ic, tc)
                    old_logp_chunks.append(olp)
                    ref_logp_chunks.append(rlp)
            old_logp_all = torch.cat(old_logp_chunks, dim=0)
            ref_logp_all = torch.cat(ref_logp_chunks, dim=0)

            # Phase 4: GRPO Clipped Surrogate + Unbiased KL Policy Updates!
            policy.train()
            total_active_tokens = msk_all.sum().clamp_min(1.0)
            kl_step_val = 0.0

            for _ in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True)
                kl_epoch_sum = 0.0

                for s_i in range(0, N_seq, mb):
                    ic = inp_all[s_i : s_i + mb]
                    tc = tgt_all[s_i : s_i + mb]
                    mc = msk_all[s_i : s_i + mb]
                    ac = adv_all[s_i : s_i + mb]
                    olp = old_logp_all[s_i : s_i + mb]
                    rlp = ref_logp_all[s_i : s_i + mb]

                    if device.type == "cuda" and autocast_dtype is not None:
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            new_lp, _ = policy.compute_token_logprobs(ic, tc)
                    else:
                        new_lp, _ = policy.compute_token_logprobs(ic, tc)

                    # Importance ratio r_{i,t}(theta) = exp(new_lp - old_lp)
                    log_ratio = (new_lp - olp).clamp(-10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps_low, 1.0 + args.clip_eps_high)

                    surr1 = ratio * ac
                    surr2 = clipped_ratio * ac
                    pg_obj = torch.minimum(surr1, surr2)

                    # Schulman's guaranteed non-negative unbiased KL estimator: exp(ref - new) - (ref - new) - 1
                    diff_ref = (rlp - new_lp).clamp(-10.0, 10.0)
                    kl_tok = torch.exp(diff_ref) - diff_ref - 1.0

                    per_tok_loss = -(pg_obj - args.kl_beta * kl_tok) * mc
                    mb_loss = per_tok_loss.sum() / total_active_tokens

                    if use_scaler:
                        scaler.scale(mb_loss).backward()
                    else:
                        mb_loss.backward()

                    kl_epoch_sum += float((kl_tok * mc).sum().detach().item())

                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    optimizer.step()

                kl_step_val = kl_epoch_sum / float(total_active_tokens.item())

            step += 1

            # Aggregate step telemetry
            mean_r = sum(e.reward_total for e in step_evals) / len(step_evals)
            max_r = max(e.reward_total for e in step_evals)
            leg_rate = sum(1 for e in step_evals if e.is_legal) / len(step_evals)
            opt_rate = sum(1 for e in step_evals if e.is_master_optimal) / len(step_evals)
            legal_regrets = [e.cp_regret for e in step_evals if e.is_legal]
            avg_reg = sum(legal_regrets) / len(legal_regrets) if legal_regrets else 999.0
            cons_rate = sum(1 for e in step_evals if e.self_consistent) / len(step_evals)
            avg_cot_len = sum(len(e.response_ids) for e in step_evals) / len(step_evals)

            moving_reward = mean_r if moving_reward is None else (0.9 * moving_reward + 0.1 * mean_r)

            acc_metrics["reward"] += mean_r
            acc_metrics["max_r"] += max_r
            acc_metrics["legal"] += leg_rate
            acc_metrics["opt"] += opt_rate
            acc_metrics["regret"] += avg_reg
            acc_metrics["consist"] += cons_rate
            acc_metrics["kl"] += kl_step_val
            acc_metrics["cot_toks"] += avg_cot_len
            acc_metrics["count"] += 1

            if step % args.log_interval == 0 or step == 1:
                c = max(1, acc_metrics["count"])
                rec = {
                    "step": step,
                    "mean_reward": round(acc_metrics["reward"] / c, 4),
                    "max_reward": round(acc_metrics["max_r"] / c, 4),
                    "legal_rate": round(acc_metrics["legal"] / c, 4),
                    "master_optimal_rate": round(acc_metrics["opt"] / c, 4),
                    "avg_cp_regret": round(acc_metrics["regret"] / c, 1),
                    "cot_consistent_rate": round(acc_metrics["consist"] / c, 4),
                    "kl_div": round(acc_metrics["kl"] / c, 5),
                    "avg_cot_tokens": round(acc_metrics["cot_toks"] / c, 1),
                    "lr": round(cur_lr, 8),
                    "sec_per_step": round(time.time() - t_step, 2),
                }
                history.append(rec)
                print(
                    f"[GRPO Step {step:5d}/{target_steps}] R={rec['mean_reward']:+.3f} (max={rec['max_reward']:+.2f}) | "
                    f"Legal={rec['legal_rate']*100:5.1f}% | MasterOpt={rec['master_optimal_rate']*100:5.1f}% | "
                    f"Δcp={rec['avg_cp_regret']:5.1f} | CoTConsist={rec['cot_consistent_rate']*100:4.0f}% | "
                    f"KL={rec['kl_div']:.4f} | len={rec['avg_cot_tokens']:.0f} | {rec['sec_per_step']:.2f}s/st"
                )
                for k in acc_metrics:
                    acc_metrics[k] = 0

            if step % args.save_interval == 0 or step == target_steps:
                ckpt_base = {
                    "stage": "rl_grpo",
                    "step": step,
                    "best_moving_reward": max(best_moving_reward, moving_reward),
                    "best_val_loss": 0.0,
                    "model_config": cfg.to_dict(),
                    "model_state_dict": policy.state_dict(),
                    "tokenizer_path": args.tokenizer_path,
                }
                if moving_reward >= best_moving_reward:
                    best_moving_reward = moving_reward
                    torch.save(ckpt_base, best_rl_path)
                    print(f"  >>> Saved new best RL CoT model -> {best_rl_path} (EMA_R={best_moving_reward:+.4f})")
                    if args.sync_best_to_main:
                        torch.save(ckpt_base, main_best_path)
                        print(f"  >>> Synced RL model -> {main_best_path} (ready for Web UI hot-reload!)")

                torch.save(
                    {
                        **ckpt_base,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_scaler else {},
                        "history": history,
                    },
                    latest_rl_path,
                )
                with open(os.path.join(args.out_dir, "rl_history.json"), "w", encoding="utf-8") as hf:
                    json.dump(history, hf, ensure_ascii=False, indent=2)
                plot_rl_curves(history, args.out_dir)

    print("============================================================")
    print(f"GRPO Post-Training RL Finished at Step {step:,} ({(time.time() - t0)/60.0:.2f} min)!")
    print(f"Best EMA Verifiable Reward : {best_moving_reward:+.4f}")
    print(f"Best RL Model Checkpoint   : {best_rl_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
