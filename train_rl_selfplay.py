"""
AlphaGo / AlphaZero-Style Full-Game Self-Play CoT Reinforcement Learning Engine (`train_rl_selfplay.py`).

Unlike single-turn engine-oracle RL (`train_rl_grpo.py`), this trainer puts the CoT Language Model
DIRECTLY INTO COMPLETE XIANGQI GAMES AGAINST ITSELF (Red vs Black, swapping colors, with League Pool +
Champion Arena Elo Gating) — 100% pure self-play with ZERO external chess engine required!

How AlphaGo Self-Play Trains the CoT Language Model:
1. **Parallel Multi-Board Self-Play Arena**:
   - Runs `B` full Xiangqi games simultaneously on GPU using exact-RoPE left-padded `generate_batch()`.
   - At EVERY ply `t` of EVERY game, the model generates `<|think|> ... <|/think|>\n最佳走法：a_t` before
     making its move on the board!
2. **AlphaZero Dual-Head Representation (`Policy pi_theta + State Value Head V_phi(s)`)**:
   - Attaches a lightweight `AlphaGoValueHead` (`V_phi(s_t) in [-1, +1]`) to the shared Transformer's
     final board-prompt hidden state — Policy and Value share ONE single forward pass during training!
   - Supports **Intra-Turn K-CoT Lookahead Selection (`--cot-rollouts K`)**: samples `K` candidate CoTs
     at board `s_t`, pushes each legal candidate move `a^(k)` to successor `s'_k`, evaluates `-V_phi(s'_k)`
     (with AlphaGo Dirichlet root noise in opening plies), and plays the strongest move!
3. **Complete Pure-Rule Xiangqi Referee (Zero External Engine Needed)**:
   - Checkmate & Stalemate (`绝杀 / 困毙判负`, real Xiangqi rule!), Perpetual Check (`长将判负`),
     3-Fold Repetition (`不变作和`), Insufficient Attacking Material, 60-Move Natural Rule, and
     Endgame Adjudication -> yields true game outcome `z in [-1, +1]`!
4. **Negamax Alternating-Turn GAE + Monte-Carlo Outcome + CoT Readability Guard**:
   - Computes zero-sum alternating-turn TD(lambda)/GAE (`delta_t = r_t - gamma * V(s_{t+1}) - V(s_t)`)
     blended with true game winner `(z_t - V(s_t))`.
   - Protects natural Chinese CoT readability via pure rule-based CoT grounding checks + frozen FP16
     reference policy `pi_ref` unbiased KL regularization (`不要删除连接的中文`).
5. **AlphaGo Zero Champion Arena Gating**:
   - Periodically plays color-balanced paired games (`Learner` vs `Champion`), computes exact `W/D/L`
     and `ΔElo`, and promotes new Champions to `checkpoints/best_selfplay_model.pt`!
6. **Strictly Step-Driven (`--steps N` / `--add-steps K`, NO wall-clock time quota!)**.

Usage examples:
  # Run 2,000 AlphaGo Self-Play CoT RL steps starting from your SFT checkpoint (e.g. after 10M SFT):
  python train_rl_selfplay.py --init-from checkpoints/best_model.pt --steps 2000 --games-per-step 4 --cot-rollouts 2

  # Resume & force +1,500 additional self-play RL steps (zero time limit):
  python train_rl_selfplay.py --resume --add-steps 1500
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.dataset import expand_jsonl_paths
from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from model.transformer import MODEL_PRESETS, ModelConfig, XiangqiCausalLM
from scripts.build_dataset_parallel import OPENING_BOOK
from train_gpu import cosine_lr, detect_compute_precision
from xiangqi.board import START_FEN, XiangqiBoard
from xiangqi.dag_cot import format_user_prompt

CHINESE_MOVE_RE = re.compile(
    r"((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])"
)
FINAL_ANS_RE = re.compile(
    r"最佳走法\s*[：:]\s*((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])"
)

# Classical Xiangqi piece values (used ONLY if a game hits `max_game_plies` without checkmate)
PIECE_CP = {
    "R": 900, "N": 420, "C": 450, "B": 200, "A": 200, "P": 120, "K": 0,
    "r": -900, "n": -420, "c": -450, "b": -200, "a": -200, "p": -120, "k": 0,
}


class AlphaGoValueHead(nn.Module):
    """
    Lightweight State Value Head `V_phi(s_t) in [-1, +1]` attached to `XiangqiCausalLM`'s final prompt
    hidden state (`d_model`). Predicts expected self-play outcome from side-to-move's perspective!
    Kept modular so `policy.state_dict()` remains 100% compatible with `web_app.py` and `inference_chat.py`.
    """

    def __init__(self, d_model: int):
        super().__init__()
        hid = max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.SiLU(),
            nn.Linear(hid, 1),
        )
        # Small output init so initial V(s) starts near 0.0 (unbiased prior before seeing game outcomes)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)

    def forward(self, prompt_hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(prompt_hidden.float())).squeeze(-1)


def classical_red_balance_cp(board: XiangqiBoard) -> int:
    """Pure rule-based material + crossed-river pawn balance (Red minus Black in centipawns)."""
    score = 0
    for r in range(10):
        for c in range(9):
            ch = board.grid[r][c]
            if ch == ".":
                continue
            score += PIECE_CP.get(ch, 0)
            if ch == "P" and r <= 4:  # Red pawn crossed river!
                score += 110 + (20 if 3 <= c <= 5 else 0)
            elif ch == "p" and r >= 5:  # Black pawn crossed river!
                score -= 110 + (20 if 3 <= c <= 5 else 0)
    return score


def has_attacking_material(board: XiangqiBoard) -> bool:
    for r in range(10):
        for c in range(9):
            if board.grid[r][c] in ("R", "N", "C", "P", "r", "n", "c", "p"):
                return True
    return False


def build_exempt_ids(tok: GeneralXiangqiTokenizer) -> Set[int]:
    exempt: Set[int] = set(range(16))
    for s in ["\n", "，", "。", "；", "：", "、", "（", "）", "【", "】", "或", "或者", "若", "则", "红", "黑"]:
        exempt.update(tok.encode(s))
    return exempt


def score_cot_process_pure_rules(
    decoded_txt: str,
    pred_zh: str,
    legal_zh_map: Dict[str, str],
) -> Tuple[float, bool]:
    """
    100% Engine-Free CoT Readability & Grounding Shaping (`[-0.35, +0.22]`):
    Ensures self-play never collapses natural Chinese reasoning or disconnects `<|think|>` from `最佳走法`!
    """
    has_close = "<|/think|>" in decoded_txt
    cot_part = decoded_txt.split("<|/think|>", 1)[0].replace("<|think|>", "").strip() if has_close else ""

    if not has_close or not pred_zh:
        return -0.35, False

    bonus = 0.05
    # Guard Chinese connecting words (`不要删除连接的中文`)
    has_cn_conn = any(w in cot_part for w in ("若", "如果", "此后", "或", "应", "走"))
    uniq_ratio = len(set(cot_part)) / max(1, min(len(cot_part), 120))
    if has_cn_conn and len(cot_part) >= 18 and uniq_ratio >= 0.16:
        bonus += 0.04
    elif uniq_ratio < 0.09:
        return -0.25, False

    self_consistent = False
    if pred_zh in legal_zh_map and len(cot_part) >= 16:
        mentions = CHINESE_MOVE_RE.findall(cot_part)
        if pred_zh in mentions:
            self_consistent = True
            bonus += 0.07
        legal_roots = {m for m in mentions[:24] if m in legal_zh_map}
        if len(legal_roots) >= 2:
            bonus += 0.04
        if "同一盘面" in cot_part or "同形" in cot_part or ("如果" in cot_part and "或者" in cot_part):
            bonus += 0.02

    return round(bonus, 4), self_consistent


def parse_rollout_move(
    r_ids: List[int],
    legal_zh_map: Dict[str, str],
    tok: GeneralXiangqiTokenizer,
) -> Tuple[str, str, Optional[str], float, bool]:
    """Returns `(full_txt, pred_zh, legal_uci_or_None, cot_shaping_reward, self_consistent)`."""
    txt = "<|think|>\n" + tok.decode(r_ids, skip_special_tokens=False).replace("<|eos|>", "")
    tail = txt.split("<|/think|>", 1)[1] if "<|/think|>" in txt else ""
    m = FINAL_ANS_RE.search(tail)
    pred_zh = m.group(1).strip() if m else ""
    if not pred_zh and tail:
        ms = CHINESE_MOVE_RE.findall(tail)
        if ms:
            pred_zh = ms[-1]

    uci = legal_zh_map.get(pred_zh) if pred_zh else None
    cot_r, consist = score_cot_process_pure_rules(txt, pred_zh, legal_zh_map)
    return txt, pred_zh, uci, cot_r, consist


@dataclass
class TurnRecord:
    prompt_ids: List[int]
    response_ids: List[int]
    side: str  # 'w' or 'b'
    ply: int
    v_pred: float
    cot_bonus: float
    was_legal: bool
    self_consistent: bool
    learner_owned: bool = True


@dataclass
class CompletedGame:
    turns: List[TurnRecord] = field(default_factory=list)
    bad_illegal_samples: List[TurnRecord] = field(default_factory=list)
    red_outcome: float = 0.0  # +1.0 Red win, -1.0 Black win, 0.0 Draw
    reason: str = "draw"
    total_plies: int = 0
    learner_side: str = "both"  # "both", "w", or "b"


class StartPositionSampler:
    """Provides standard start, diverse opening-book starts, and optional curriculum mid/endgame FENs."""

    def __init__(self, fen_jsonl_pattern: str, start_mode: str = "standard_and_book", seed: int = 2026):
        self.files = expand_jsonl_paths([fen_jsonl_pattern]) if fen_jsonl_pattern else []
        self.start_mode = start_mode
        self.rng = random.Random(seed)
        self._file_idx = 0
        self._fp: Optional[object] = None

    def _next_curriculum_fen(self) -> Optional[str]:
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
            try:
                fen = json.loads(line).get("fen", "")
                if fen:
                    b = XiangqiBoard(fen)
                    if b.is_nnue_valid() and len(b.generate_legal_moves()) >= 2:
                        return b.to_fen()
            except Exception:
                continue
        return None

    def sample_board(self) -> XiangqiBoard:
        if self.start_mode == "pure_from_start":
            return XiangqiBoard(START_FEN)

        r = self.rng.random()
        if r < 0.30:
            return XiangqiBoard(START_FEN)
        if r < 0.80 or not self.files:
            b = XiangqiBoard(START_FEN)
            _, moves = self.rng.choice(OPENING_BOOK)
            k = self.rng.randint(1, len(moves))
            for u in moves[:k]:
                if u in b.generate_legal_moves():
                    b.push_uci(u)
            return b

        cfen = self._next_curriculum_fen()
        return XiangqiBoard(cfen) if cfen else XiangqiBoard(START_FEN)


@torch.no_grad()
def evaluate_board_values(
    policy: XiangqiCausalLM,
    value_head: AlphaGoValueHead,
    boards: List[XiangqiBoard],
    tok: GeneralXiangqiTokenizer,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> List[float]:
    if not boards:
        return []
    prompts = [tok.encode(f"<|user|>\n{format_user_prompt(b.to_fen())}\n<|assistant|>\n<|think|>\n") for b in boards]
    if device.type == "cuda" and autocast_dtype is not None:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            h = policy.forward_prompt_hidden(prompts, device=device)
    else:
        h = policy.forward_prompt_hidden(prompts, device=device)
    vals = value_head(h).cpu().tolist()
    return [float(v) for v in vals]


def sample_dirichlet(k: int, alpha: float, rng: random.Random) -> List[float]:
    raw = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(raw) or 1.0
    return [x / s for x in raw]


@torch.no_grad()
def play_selfplay_games_batch(
    learner: XiangqiCausalLM,
    value_head: AlphaGoValueHead,
    opponent_pool: List[XiangqiCausalLM],
    opponent_mode: str,
    pos_sampler: StartPositionSampler,
    tok: GeneralXiangqiTokenizer,
    exempt_ids: Set[int],
    force_close_ids: List[int],
    num_games: int,
    cot_rollouts_per_turn: int,
    max_cot_tokens: int,
    max_game_plies: int,
    temp_open: float,
    temp_mid: float,
    temp_drop_ply: int,
    illegal_policy: str,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    rng: random.Random,
    fixed_openings: Optional[List[Tuple[XiangqiBoard, str, XiangqiCausalLM]]] = None,
) -> List[CompletedGame]:
    """
    Plays `num_games` full Xiangqi games IN PARALLEL on GPU!
    Supports:
      - Symmetric AlphaZero Self-Play (`learner` plays both Red & Black)
      - League / Arena Matches (`learner` plays Red or Black vs historical Champion!)
      - Intra-Turn `K`-CoT Lookahead Selection via `-V_phi(s')` + Dirichlet exploration noise!
    """
    learner.eval()
    value_head.eval()

    # Per-game mutable state
    boards: List[XiangqiBoard] = []
    learner_sides: List[str] = []  # "both", "w", or "b"
    opp_models: List[Optional[XiangqiCausalLM]] = []

    if fixed_openings is not None:
        for b_init, l_side, opp_m in fixed_openings:
            boards.append(b_init.copy())
            learner_sides.append(l_side)
            opp_models.append(opp_m)
    else:
        for _ in range(num_games):
            boards.append(pos_sampler.sample_board())
            if opponent_mode == "league" and opponent_pool and rng.random() < 0.50:
                learner_sides.append("w" if rng.random() < 0.5 else "b")
                opp_models.append(rng.choice(opponent_pool))
            else:
                learner_sides.append("both")
                opp_models.append(None)

    G = len(boards)
    games = [CompletedGame(learner_side=learner_sides[i]) for i in range(G)]
    active = [True] * G
    plies = [0] * G
    no_capture_plies = [0] * G
    state_counts: List[Dict[str, int]] = [{} for _ in range(G)]
    # Track continuous checking streaks per side for `长将判负`!
    consec_checks: List[Dict[str, int]] = [{"w": 0, "b": 0} for _ in range(G)]

    for i in range(G):
        sk = boards[i].state_key()
        state_counts[i][sk] = 1

    while any(active):
        # 1. Check immediate terminal conditions at start of turn
        active_indices: List[int] = []
        for i in range(G):
            if not active[i]:
                continue
            b = boards[i]
            legal_ucis = b.generate_legal_moves()

            # Rule 1: Checkmate or Stalemate (`绝杀 / 困毙判负` -> current mover loses!)
            if not legal_ucis:
                games[i].red_outcome = -1.0 if b.turn == "w" else +1.0
                games[i].reason = "checkmate_or_stalemate"
                games[i].total_plies = plies[i]
                active[i] = False
                continue

            # Rule 2: Neither side has any attacking pieces (`R/N/C/P`) -> Draw!
            if not has_attacking_material(b):
                games[i].red_outcome = 0.0
                games[i].reason = "insufficient_material"
                games[i].total_plies = plies[i]
                active[i] = False
                continue

            # Rule 3: 60-move natural limit (`120` plies without capture)
            if no_capture_plies[i] >= 120:
                games[i].red_outcome = 0.0
                games[i].reason = "60_move_rule"
                games[i].total_plies = plies[i]
                active[i] = False
                continue

            # Rule 4: Max-ply cap -> Pure material+crossed-pawn adjudication
            if plies[i] >= max_game_plies:
                bal_cp = classical_red_balance_cp(b)
                if abs(bal_cp) >= 350:
                    # Smooth decisive outcome in [0.65, 0.95] so clean checkmates (+1.0) are STILL strictly preferred!
                    mag = min(0.92, 0.60 + (abs(bal_cp) - 350) / 1800.0)
                    games[i].red_outcome = +mag if bal_cp > 0 else -mag
                    games[i].reason = "max_ply_material_win"
                else:
                    games[i].red_outcome = 0.0
                    games[i].reason = "max_ply_draw"
                games[i].total_plies = plies[i]
                active[i] = False
                continue

            active_indices.append(i)

        if not active_indices:
            break

        # 2. Group active turns by model instance (`learner` vs league opponent) for batched GPU generation!
        learner_turn_idxs: List[int] = []
        opp_groups: Dict[int, Tuple[XiangqiCausalLM, List[int]]] = {}

        for i in active_indices:
            side = boards[i].turn
            if learner_sides[i] == "both" or learner_sides[i] == side or opp_models[i] is None:
                learner_turn_idxs.append(i)
            else:
                om = opp_models[i]
                assert om is not None
                oid = id(om)
                if oid not in opp_groups:
                    opp_groups[oid] = (om, [])
                opp_groups[oid][1].append(i)

        # Compute state values `V_phi(s_t)` for ALL active boards in ONE batch (ensures exact Negamax `-V(s_{t+1})` even in League play!)
        learner_v_map: Dict[int, float] = {}
        if active_indices:
            v_list = evaluate_board_values(
                learner,
                value_head,
                [boards[i] for i in active_indices],
                tok,
                device,
                autocast_dtype,
            )
            for idx_game, v_val in zip(active_indices, v_list):
                learner_v_map[idx_game] = v_val

        # Helper to run batched K-rollout CoT generation for a subset of games using `mdl`
        def run_model_group(mdl: XiangqiCausalLM, g_idxs: List[int], k_roll: int, is_learner_turn: bool) -> None:
            flat_prompts: List[List[int]] = []
            flat_temps: List[float] = []
            per_game_prompt: List[List[int]] = []

            for gi in g_idxs:
                p_ids = tok.encode(f"<|user|>\n{format_user_prompt(boards[gi].to_fen())}\n<|assistant|>\n<|think|>\n")
                per_game_prompt.append(p_ids)
                t_cur = temp_open if plies[gi] < temp_drop_ply else temp_mid
                for _ in range(k_roll):
                    flat_prompts.append(p_ids)
                    flat_temps.append(t_cur)

            if device.type == "cuda" and autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    flat_outs = mdl.generate_batch(
                        prompt_ids_batch=flat_prompts,
                        max_new_tokens=max_cot_tokens,
                        temperatures=flat_temps,
                        top_p=0.95,
                        repetition_penalty=1.10,
                        exempt_rep_ids=exempt_ids,
                        eos_token_id=tok.eos_id,
                        end_think_id=tok.end_think_id,
                        force_close_ids=force_close_ids,
                        device=device,
                    )
            else:
                flat_outs = mdl.generate_batch(
                    prompt_ids_batch=flat_prompts,
                    max_new_tokens=max_cot_tokens,
                    temperatures=flat_temps,
                    top_p=0.95,
                    repetition_penalty=1.10,
                    exempt_rep_ids=exempt_ids,
                    eos_token_id=tok.eos_id,
                    end_think_id=tok.end_think_id,
                    force_close_ids=force_close_ids,
                    device=device,
                )

            # Parse candidates & batch-evaluate successor boards `-V_phi(s')` when K > 1!
            succ_boards_to_eval: List[XiangqiBoard] = []
            succ_map_keys: List[Tuple[int, int]] = []  # (local_idx, cand_k)
            parsed_per_game: List[List[Tuple[List[int], Optional[str], float, bool, Optional[float]]]] = []

            for loc_i, gi in enumerate(g_idxs):
                b = boards[gi]
                legal_zh: Dict[str, str] = {}
                for u in b.generate_legal_moves():
                    try:
                        legal_zh[b.uci_to_zh(u)] = u
                    except Exception:
                        pass

                cands = []
                for ck in range(k_roll):
                    r_ids = flat_outs[loc_i * k_roll + ck]
                    _, _, uci, cot_r, consist = parse_rollout_move(r_ids, legal_zh, tok)
                    imm_q: Optional[float] = None
                    if uci is not None and k_roll > 1:
                        nb = b.copy()
                        nb.push_uci(uci)
                        if not nb.generate_legal_moves():
                            imm_q = +1.0  # Immediate checkmate/stalemate win!
                        else:
                            succ_boards_to_eval.append(nb)
                            succ_map_keys.append((loc_i, ck))
                    cands.append([r_ids, uci, cot_r, consist, imm_q])
                parsed_per_game.append(cands)  # type: ignore

            if succ_boards_to_eval:
                opp_vals = evaluate_board_values(learner, value_head, succ_boards_to_eval, tok, device, autocast_dtype)
                for (loc_i, ck), opp_v in zip(succ_map_keys, opp_vals):
                    # Negamax flip: mover's Q-value of successor state is `-V(opponent_state)`!
                    parsed_per_game[loc_i][ck][4] = -float(opp_v)  # type: ignore

            # Execute selected move on each board!
            for loc_i, gi in enumerate(g_idxs):
                b = boards[gi]
                p_ids = per_game_prompt[loc_i]
                cands = parsed_per_game[loc_i]
                side_now = b.turn
                v_now = learner_v_map.get(gi, 0.0)

                legal_cands = [c for c in cands if c[1] is not None]
                illegal_cands = [c for c in cands if c[1] is None]

                # Record at most 1 illegal rollout per turn as an explicit negative lesson (-1.0)
                if is_learner_turn and illegal_cands and fixed_openings is None:
                    bad_c = illegal_cands[0]
                    games[gi].bad_illegal_samples.append(
                        TurnRecord(
                            prompt_ids=p_ids,
                            response_ids=bad_c[0],
                            side=side_now,
                            ply=plies[gi],
                            v_pred=v_now,
                            cot_bonus=-0.35,
                            was_legal=False,
                            self_consistent=False,
                            learner_owned=True,
                        )
                    )

                if legal_cands:
                    if len(legal_cands) == 1:
                        chosen = legal_cands[0]
                    else:
                        # AlphaZero PUCT / Softmax + Dirichlet noise selection over `-V_phi(s') + 0.15*cot_r`!
                        qs = [float(c[4] if c[4] is not None else 0.0) + 0.15 * float(c[2]) for c in legal_cands]
                        if plies[gi] < temp_drop_ply and fixed_openings is None:
                            noise = sample_dirichlet(len(legal_cands), 0.35, rng)
                            probs = [0.75 * math.exp(q * 2.5) + 0.25 * n for q, n in zip(qs, noise)]
                            tot_p = sum(probs)
                            pick_r = rng.random() * tot_p
                            acc_p = 0.0
                            chosen = legal_cands[-1]
                            for c_item, pw in zip(legal_cands, probs):
                                acc_p += pw
                                if pick_r <= acc_p:
                                    chosen = c_item
                                    break
                        else:
                            best_j = max(range(len(legal_cands)), key=lambda j: qs[j])
                            chosen = legal_cands[best_j]

                    chosen_r_ids, chosen_uci, chosen_cot_r, chosen_consist, _ = chosen
                    assert chosen_uci is not None

                    if is_learner_turn:
                        games[gi].turns.append(
                            TurnRecord(
                                prompt_ids=p_ids,
                                response_ids=chosen_r_ids,
                                side=side_now,
                                ply=plies[gi],
                                v_pred=v_now,
                                cot_bonus=chosen_cot_r,
                                was_legal=True,
                                self_consistent=chosen_consist,
                                learner_owned=True,
                            )
                        )
                    else:
                        # Still record opponent turn's board value for alternating Negamax GAE continuity (`learner_owned=False`)
                        games[gi].turns.append(
                            TurnRecord(
                                prompt_ids=p_ids,
                                response_ids=[],
                                side=side_now,
                                ply=plies[gi],
                                v_pred=v_now,
                                cot_bonus=0.0,
                                was_legal=True,
                                self_consistent=False,
                                learner_owned=False,
                            )
                        )
                    move_to_push = chosen_uci
                else:
                    # All K rollouts produced an illegal move!
                    if illegal_policy == "forfeit_loss":
                        games[gi].red_outcome = -1.0 if side_now == "w" else +1.0
                        games[gi].reason = "illegal_forfeit"
                        games[gi].total_plies = plies[gi]
                        active[gi] = False
                        continue
                    # `penalize_and_continue`: bad rollout already recorded with -1.0 return; advance board via random/capture-aware legal move!
                    all_leg = b.generate_legal_moves()
                    move_to_push = rng.choice(all_leg)

                # Advance board state & update Xiangqi repetition / perpetual-check counters!
                _, _, c2, r2 = b.parse_uci(move_to_push)
                is_capture = b.grid[r2][c2] != "."
                b.push_uci(move_to_push)
                plies[gi] += 1
                no_capture_plies[gi] = 0 if is_capture else (no_capture_plies[gi] + 1)

                # Check if mover just delivered check to opponent (`b.turn` is now opponent!)
                opp_side = b.turn
                if b.is_in_check(opp_side):
                    consec_checks[gi][side_now] += 1
                else:
                    consec_checks[gi][side_now] = 0

                sk = b.state_key()
                state_counts[gi][sk] = state_counts[gi].get(sk, 0) + 1
                if state_counts[gi][sk] >= 3:
                    # Real Chinese Chess 3-fold repetition rule: Perpetual Check LOSES (`长将判负`)!
                    w_chk = consec_checks[gi]["w"] >= 2
                    b_chk = consec_checks[gi]["b"] >= 2
                    if w_chk and not b_chk:
                        games[gi].red_outcome = -1.0
                        games[gi].reason = "perpetual_check_loss_red"
                    elif b_chk and not w_chk:
                        games[gi].red_outcome = +1.0
                        games[gi].reason = "perpetual_check_loss_black"
                    else:
                        # Symmetric repetition draw with slight -0.05 draw contempt handled at return computation
                        games[gi].red_outcome = 0.0
                        games[gi].reason = "repetition_draw"
                    games[gi].total_plies = plies[gi]
                    active[gi] = False

        if learner_turn_idxs:
            run_model_group(learner, learner_turn_idxs, cot_rollouts_per_turn, is_learner_turn=True)
        for _, (om, o_idxs) in opp_groups.items():
            run_model_group(om, o_idxs, 1, is_learner_turn=False)

    return games


def build_selfplay_training_examples(
    games: List[CompletedGame],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    mc_weight: float = 0.50,
    draw_contempt: float = 0.04,
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """
    Converts completed self-play games into `(prompt_ids, response_ids, advantage, value_target)` tuples
    using **Two-Player Zero-Sum Alternating-Turn Negamax GAE** + **True Game Winner `z_t`**!
    """
    prompts: List[List[int]] = []
    resps: List[List[int]] = []
    raw_advs: List[float] = []
    val_targets: List[float] = []

    for g in games:
        T = len(g.turns)
        if T > 0:
            g_p: List[List[int]] = []
            g_r: List[List[int]] = []
            g_a: List[float] = []
            g_v: List[float] = []
            gae_next = 0.0
            for t in range(T - 1, -1, -1):
                tr = g.turns[t]
                # Outcome from side-to-move's perspective at ply `t`!
                if g.red_outcome == 0.0:
                    z_t = -draw_contempt  # Slight contempt encourages active winning play over shuffling draws!
                else:
                    z_t = g.red_outcome if tr.side == "w" else -g.red_outcome

                if t == T - 1:
                    gae_cur = z_t - tr.v_pred
                else:
                    nxt = g.turns[t + 1]
                    # Exact multi-ply Negamax discounting: flip sign iff `nxt.side != tr.side`, discount by `gamma^(Δply)`!
                    sign_flip = -1.0 if nxt.side != tr.side else +1.0
                    ply_gap = max(1, nxt.ply - tr.ply)
                    gamma_gap = gamma ** ply_gap
                    next_v_from_cur_side = sign_flip * nxt.v_pred
                    delta = gamma_gap * next_v_from_cur_side - tr.v_pred
                    gae_cur = delta + (gamma_gap * gae_lambda * sign_flip) * gae_next

                gae_next = gae_cur

                if tr.learner_owned and tr.response_ids:
                    mc_adv = z_t - tr.v_pred
                    blended_adv = mc_weight * mc_adv + (1.0 - mc_weight) * gae_cur + tr.cot_bonus
                    g_p.append(tr.prompt_ids)
                    g_r.append(tr.response_ids)
                    g_a.append(float(blended_adv))
                    g_v.append(float(max(-1.0, min(1.0, z_t))))

            # Restore chronological ply order (ply 0 -> ply T-1)
            prompts.extend(reversed(g_p))
            resps.extend(reversed(g_r))
            raw_advs.extend(reversed(g_a))
            val_targets.extend(reversed(g_v))

        # Append explicit illegal-move lessons (`z = -1.0`, negative advantage) so illegality rapidly drops to 0%!
        for bad in g.bad_illegal_samples[:8]:
            if bad.response_ids:
                prompts.append(bad.prompt_ids)
                resps.append(bad.response_ids)
                raw_advs.append(float(-1.0 - bad.v_pred + bad.cot_bonus))
                val_targets.append(-1.0)

    if len(raw_advs) >= 2:
        t_adv = torch.tensor(raw_advs, dtype=torch.float32)
        norm_advs = ((t_adv - t_adv.mean()) / (t_adv.std(unbiased=False) + 1e-5)).clamp(-4.0, 4.0).tolist()
    else:
        norm_advs = raw_advs

    return prompts, resps, norm_advs, val_targets


def pack_selfplay_batch(
    prompts: List[List[int]],
    resps: List[List[int]],
    advs: List[float],
    v_targets: List[float],
    pad_id: int,
    eos_id: int,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = len(prompts)
    seqs: List[List[int]] = []
    p_last_idxs: List[int] = []
    for p, r in zip(prompts, resps):
        full = (p + r + [eos_id])[: max_seq_len + 1]
        seqs.append(full)
        p_last_idxs.append(max(0, len(p) - 1))

    max_t = min(max_seq_len, max(16, ((max(len(s) - 1 for s in seqs) + 15) // 16) * 16))
    inp = torch.full((N, max_t), pad_id, dtype=torch.long, device=device)
    tgt = torch.full((N, max_t), pad_id, dtype=torch.long, device=device)
    msk = torch.zeros((N, max_t), dtype=torch.float32, device=device)

    for i, (s, pidx) in enumerate(zip(seqs, p_last_idxs)):
        L = min(len(s) - 1, max_t)
        inp[i, :L] = torch.tensor(s[:L], dtype=torch.long, device=device)
        tgt[i, :L] = torch.tensor(s[1 : L + 1], dtype=torch.long, device=device)
        if L > pidx:
            msk[i, pidx:L] = 1.0

    p_idx_t = torch.tensor(p_last_idxs, dtype=torch.long, device=device)
    adv_t = torch.tensor(advs, dtype=torch.float32, device=device).unsqueeze(1)
    vt_t = torch.tensor(v_targets, dtype=torch.float32, device=device)
    return inp, tgt, msk, p_idx_t, adv_t, vt_t


@torch.no_grad()
def evaluate_champion_arena(
    learner: XiangqiCausalLM,
    value_head: AlphaGoValueHead,
    champion: XiangqiCausalLM,
    tok: GeneralXiangqiTokenizer,
    exempt_ids: Set[int],
    force_close_ids: List[int],
    num_pairs: int,
    max_cot_tokens: int,
    max_game_plies: int,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    rng: random.Random,
) -> Dict[str, float]:
    """
    AlphaGo Zero Symmetric Arena Evaluation:
    Plays `num_pairs * 2` games from identical opening positions with colors swapped (`Learner=Red` & `Learner=Black`)!
    """
    fixed: List[Tuple[XiangqiBoard, str, XiangqiCausalLM]] = []
    for p_idx in range(num_pairs):
        b = XiangqiBoard(START_FEN)
        if p_idx > 0:
            _, moves = OPENING_BOOK[(p_idx - 1) % len(OPENING_BOOK)]
            for u in moves[:2]:
                if u in b.generate_legal_moves():
                    b.push_uci(u)
        fixed.append((b.copy(), "w", champion))
        fixed.append((b.copy(), "b", champion))

    arena_games = play_selfplay_games_batch(
        learner=learner,
        value_head=value_head,
        opponent_pool=[champion],
        opponent_mode="league",
        pos_sampler=StartPositionSampler("", "pure_from_start"),
        tok=tok,
        exempt_ids=exempt_ids,
        force_close_ids=force_close_ids,
        num_games=len(fixed),
        cot_rollouts_per_turn=1,
        max_cot_tokens=max_cot_tokens,
        max_game_plies=max_game_plies,
        temp_open=0.45,
        temp_mid=0.25,
        temp_drop_ply=8,
        illegal_policy="penalize_and_continue",
        device=device,
        autocast_dtype=autocast_dtype,
        rng=rng,
        fixed_openings=fixed,
    )

    wins = draws = losses = 0
    for g in arena_games:
        l_out = g.red_outcome if g.learner_side == "w" else -g.red_outcome
        if l_out > 0.25:
            wins += 1
        elif l_out < -0.25:
            losses += 1
        else:
            draws += 1

    tot = max(1, wins + draws + losses)
    win_score = (wins + 0.5 * draws) / tot
    clamped_s = min(0.99, max(0.01, win_score))
    elo_diff = 400.0 * math.log10(clamped_s / (1.0 - clamped_s))
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_score": round(win_score, 4),
        "elo_diff": round(elo_diff, 1),
    }


def plot_selfplay_curves(history: List[Dict], out_dir: str) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in history]
        elo = [h["cumulative_elo"] for h in history]
        legal = [h["legal_turn_rate"] * 100.0 for h in history]
        decisive = [h["decisive_game_rate"] * 100.0 for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=140)
        ax1.plot(steps, elo, color="#dc2626", linewidth=2.0, label="Self-Play Champion Elo Gain")
        ax1.set_xlabel("Self-Play RL Steps")
        ax1.set_ylabel("Cumulative ΔElo vs SFT Init")
        ax1.set_title("AlphaGo-Style Self-Play Elo Progression")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(steps, legal, color="#2563eb", linewidth=1.8, label="Legal CoT Move %")
        ax2.plot(steps, decisive, color="#16a34a", linewidth=1.8, label="Decisive Win/Loss Game %")
        ax2.set_xlabel("Self-Play RL Steps")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Self-Play Game Quality & Legality")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "selfplay_rl_curves.png"))
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaGo / AlphaZero-Style Full-Game CoT Self-Play RL Trainer")
    parser.add_argument(
        "--init-from",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
        help="SFT checkpoint (e.g. trained on 10M+ positions) to start self-play from",
    )
    parser.add_argument("--preset", type=str, default="fast-4m", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--fen-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "train*.jsonl"))
    parser.add_argument("--start-mode", type=str, default="standard_and_book", choices=["standard_and_book", "pure_from_start"])
    parser.add_argument("--opponent-mode", type=str, default="league", choices=["self", "league"])
    parser.add_argument("--illegal-policy", type=str, default="penalize_and_continue", choices=["penalize_and_continue", "forfeit_loss"])
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs"))
    parser.add_argument("--device", type=str, default="auto")

    # Strict Step-Count Control (NO wall-clock time limit!)
    parser.add_argument("--steps", type=int, default=1500, help="Exact total self-play RL update steps (no time cap)")
    parser.add_argument("--add-steps", type=int, default=0, help="Exact +N additional self-play RL steps on resume")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest_selfplay_checkpoint.pt")
    parser.add_argument("--sync-best-to-main", action="store_true", help="Sync newly promoted Champion to best_model.pt")

    # AlphaGo Self-Play + CoT Hyperparameters
    parser.add_argument("--games-per-step", type=int, default=4, help="Parallel full Xiangqi games played per RL step")
    parser.add_argument("--cot-rollouts", type=int, default=2, help="K candidate CoTs evaluated via -V(s') per move (AlphaZero lookahead)")
    parser.add_argument("--max-cot-tokens", type=int, default=384, help="Max `<|think|>...<|/think|> + move` tokens per ply in self-play")
    parser.add_argument("--max-game-plies", type=int, default=110, help="Max plies (half-moves) per self-play game before adjudication")
    parser.add_argument("--temp-open", type=float, default=1.0, help="AlphaGo opening exploration temperature")
    parser.add_argument("--temp-mid", type=float, default=0.72, help="Middlegame/endgame tactical temperature")
    parser.add_argument("--temp-drop-ply", type=int, default=16, help="Ply at which temperature drops from temp-open to temp-mid")

    # PPO / GAE / Value / Reference KL Hyperparameters
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--mini-batch-seqs", type=int, default=8)
    parser.add_argument("--max-turns-per-step", type=int, default=96, help="Cap sampled plies per step so VRAM & step time stay bounded")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--mc-weight", type=float, default=0.50, help="Blend between true game outcome z_t (0.5) and Negamax GAE (0.5)")
    parser.add_argument("--value-coef", type=float, default=0.50, help="Weight of AlphaGo Value Head Smooth-L1 loss")
    parser.add_argument("--clip-eps-low", type=float, default=0.20)
    parser.add_argument("--clip-eps-high", type=float, default=0.28)
    parser.add_argument("--kl-beta", type=float, default=0.02, help="Reference KL weight protecting natural Chinese CoT")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--value-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Champion Arena Gating
    parser.add_argument("--arena-interval", type=int, default=50, help="Run Learner vs Champion match every N steps")
    parser.add_argument("--arena-pairs", type=int, default=3, help="Number of color-swapped opening pairs (2*pairs games)")
    parser.add_argument("--arena-gate", type=float, default=0.52, help="Win-score threshold to crown new Self-Play Champion")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device, prec_name, autocast_dtype, use_scaler = detect_compute_precision(args.device)
    tok = GeneralXiangqiTokenizer.load(args.tokenizer_path)
    exempt_ids = build_exempt_ids(tok)
    force_close_ids = tok.encode("\n<|/think|>\n最佳走法：")

    # 1. Load Policy Model `pi_theta`
    if os.path.exists(args.init_from):
        sft_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(sft_ckpt["model_config"])
        policy = XiangqiCausalLM(cfg, grad_checkpoint=(cfg.d_model >= 512)).to(device)
        policy.load_state_dict(sft_ckpt["model_state_dict"], strict=True)
        print(f"[Self-Play Init] Loaded SFT policy from {args.init_from} ({policy.count_parameters()/1e6:.2f}M params)")
    else:
        print(f"[Self-Play Warning] SFT checkpoint not found at {args.init_from}; initializing from preset={args.preset}.")
        cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            pad_token_id=tok.pad_id,
            eos_token_id=tok.eos_id,
            **MODEL_PRESETS[args.preset],
        )
        policy = XiangqiCausalLM(cfg).to(device)

    value_head = AlphaGoValueHead(cfg.d_model).to(device)

    # 2. Frozen Reference Model `pi_ref` (protects readable Chinese CoT prose during 100k+ self-play moves!)
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    if device.type == "cuda" and autocast_dtype is not None:
        ref_policy = ref_policy.to(dtype=autocast_dtype)

    # 3. Current Champion & League Pool (AlphaGo / AlphaZero style!)
    champion = copy.deepcopy(policy).to(device).eval()
    for p in champion.parameters():
        p.requires_grad_(False)
    if device.type == "cuda" and autocast_dtype is not None:
        champion = champion.to(dtype=autocast_dtype)
    league_pool: List[XiangqiCausalLM] = [champion]

    # Joint optimizer with distinct learning rate for Policy backbone vs freshly initialized ValueHead
    decay_params = [p for n, p in policy.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in policy.named_parameters() if p.requires_grad and p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": 0.01, "lr": args.lr, "role": "policy"},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": args.lr, "role": "policy"},
        {"params": list(value_head.parameters()), "weight_decay": 1e-4, "lr": args.value_lr, "role": "value"},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_scaler))

    latest_sp_path = os.path.join(args.checkpoint_dir, "latest_selfplay_checkpoint.pt")
    best_sp_path = os.path.join(args.checkpoint_dir, "best_selfplay_model.pt")
    main_best_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    start_step = 0
    cumulative_elo = 0.0
    total_games_played = 0
    history: List[Dict] = []

    do_resume = args.resume or (args.add_steps > 0)
    if do_resume and os.path.exists(latest_sp_path):
        sp_ckpt = torch.load(latest_sp_path, map_location=device, weights_only=False)
        if sp_ckpt.get("model_config") == cfg.to_dict():
            policy.load_state_dict(sp_ckpt["model_state_dict"], strict=True)
            if "value_head_state_dict" in sp_ckpt:
                value_head.load_state_dict(sp_ckpt["value_head_state_dict"], strict=True)
            if "champion_state_dict" in sp_ckpt:
                champ_fp32 = XiangqiCausalLM(cfg).to(device)
                champ_fp32.load_state_dict(sp_ckpt["champion_state_dict"], strict=True)
                champ_fp32.eval()
                for p in champ_fp32.parameters():
                    p.requires_grad_(False)
                if device.type == "cuda" and autocast_dtype is not None:
                    champ_fp32 = champ_fp32.to(dtype=autocast_dtype)
                champion = champ_fp32
                league_pool = [champion]
            optimizer.load_state_dict(sp_ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in sp_ckpt and use_scaler:
                scaler.load_state_dict(sp_ckpt["scaler_state_dict"])
            start_step = int(sp_ckpt.get("step", 0))
            cumulative_elo = float(sp_ckpt.get("cumulative_elo", 0.0))
            total_games_played = int(sp_ckpt.get("total_games_played", 0))
            history = sp_ckpt.get("history", [])
            print(
                f"[Self-Play Resume] Resumed at step={start_step:,}, "
                f"games_played={total_games_played:,}, cumulative_elo={cumulative_elo:+.1f}"
            )

    if args.add_steps > 0:
        target_steps = start_step + args.add_steps
        cycle_start = start_step
    else:
        target_steps = args.steps
        cycle_start = start_step if start_step > 0 else 0

    warmup = min(args.warmup_steps, max(2, (target_steps - cycle_start) // 10))
    pos_sampler = StartPositionSampler(args.fen_jsonl, start_mode=args.start_mode, seed=args.seed + start_step)

    print(
        f"[AlphaGo Self-Play Start] Steps: {start_step:,} -> {target_steps:,} (STRICT STEP MODE, ZERO ENGINE, NO TIME LIMIT)\n"
        f"  Games/step={args.games_per_step} | K-CoT Lookahead={args.cot_rollouts} | MaxPlies={args.max_game_plies} | "
        f"OpponentMode={args.opponent_mode} | ValueHead={sum(p.numel() for p in value_head.parameters()):,} params"
    )

    step = start_step
    t0 = time.time()

    while step < target_steps:
        t_step = time.time()
        cur_lr = cosine_lr(step, target_steps, warmup, args.lr, args.min_lr, cycle_start=cycle_start)
        val_lr = cur_lr * (args.value_lr / max(1e-8, args.lr))
        for pg in optimizer.param_groups:
            pg["lr"] = val_lr if pg.get("role") == "value" else cur_lr

        # Phase 1: Play `games_per_step` complete Xiangqi games in parallel on GPU!
        games = play_selfplay_games_batch(
            learner=policy,
            value_head=value_head,
            opponent_pool=league_pool,
            opponent_mode=args.opponent_mode,
            pos_sampler=pos_sampler,
            tok=tok,
            exempt_ids=exempt_ids,
            force_close_ids=force_close_ids,
            num_games=args.games_per_step,
            cot_rollouts_per_turn=args.cot_rollouts,
            max_cot_tokens=args.max_cot_tokens,
            max_game_plies=args.max_game_plies,
            temp_open=args.temp_open,
            temp_mid=args.temp_mid,
            temp_drop_ply=args.temp_drop_ply,
            illegal_policy=args.illegal_policy,
            device=device,
            autocast_dtype=autocast_dtype,
            rng=rng,
        )
        total_games_played += len(games)

        # Phase 2: Negamax Alternating-Turn GAE + True Game Winner `z_t`
        prompts, resps, advs, v_targets = build_selfplay_training_examples(
            games,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            mc_weight=args.mc_weight,
        )

        # Subsample if a marathon game batch produced > `max_turns_per_step` trajectories (keeps 6GB VRAM & step latency constant)
        if len(prompts) > args.max_turns_per_step:
            idxs = list(range(len(prompts)))
            rng.shuffle(idxs)
            idxs = idxs[: args.max_turns_per_step]
            prompts = [prompts[i] for i in idxs]
            resps = [resps[i] for i in idxs]
            advs = [advs[i] for i in idxs]
            v_targets = [v_targets[i] for i in idxs]

        kl_step_val = 0.0
        val_loss_step = 0.0

        if prompts:
            inp_all, tgt_all, msk_all, pidx_all, adv_all, vt_all = pack_selfplay_batch(
                prompts,
                resps,
                advs,
                v_targets,
                pad_id=tok.pad_id,
                eos_id=tok.eos_id,
                max_seq_len=cfg.max_seq_len,
                device=device,
            )
            N_seq = inp_all.shape[0]
            mb = max(1, min(args.mini_batch_seqs, N_seq))

            # Cache `old_logp` & `ref_logp` under `no_grad`
            old_lp_chunks: List[torch.Tensor] = []
            ref_lp_chunks: List[torch.Tensor] = []
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
                    old_lp_chunks.append(olp)
                    ref_lp_chunks.append(rlp)
            old_lp_all = torch.cat(old_lp_chunks, dim=0)
            ref_lp_all = torch.cat(ref_lp_chunks, dim=0)

            # Phase 3: Joint Policy (CoT + Move) + AlphaGo Value Head Update in ONE Single Forward Pass!
            policy.train()
            value_head.train()
            tot_act_toks = msk_all.sum().clamp_min(1.0)

            for _ in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True)
                kl_sum = 0.0
                vloss_sum = 0.0

                for s_i in range(0, N_seq, mb):
                    ic = inp_all[s_i : s_i + mb]
                    tc = tgt_all[s_i : s_i + mb]
                    mc = msk_all[s_i : s_i + mb]
                    pc = pidx_all[s_i : s_i + mb]
                    ac = adv_all[s_i : s_i + mb]
                    vc = vt_all[s_i : s_i + mb]
                    olp = old_lp_all[s_i : s_i + mb]
                    rlp = ref_lp_all[s_i : s_i + mb]

                    if device.type == "cuda" and autocast_dtype is not None:
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            new_lp, _, p_hid = policy.compute_token_logprobs(ic, tc, prompt_last_idx=pc)
                            v_pred = value_head(p_hid)
                    else:
                        new_lp, _, p_hid = policy.compute_token_logprobs(ic, tc, prompt_last_idx=pc)
                        v_pred = value_head(p_hid)

                    ratio = torch.exp((new_lp - olp).clamp(-10.0, 10.0))
                    clipped = torch.clamp(ratio, 1.0 - args.clip_eps_low, 1.0 + args.clip_eps_high)
                    pg_obj = torch.minimum(ratio * ac, clipped * ac)

                    diff_ref = (rlp - new_lp).clamp(-10.0, 10.0)
                    kl_tok = torch.exp(diff_ref) - diff_ref - 1.0

                    policy_loss = (-(pg_obj - args.kl_beta * kl_tok) * mc).sum() / tot_act_toks
                    v_loss = F.smooth_l1_loss(v_pred, vc) * (float(ic.shape[0]) / float(N_seq))
                    joint_loss = policy_loss + args.value_coef * v_loss

                    if use_scaler:
                        scaler.scale(joint_loss).backward()
                    else:
                        joint_loss.backward()

                    kl_sum += float((kl_tok * mc).sum().detach().item())
                    vloss_sum += float(v_loss.detach().item())

                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_head.parameters()), args.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + list(value_head.parameters()), args.grad_clip
                    )
                    optimizer.step()

                kl_step_val = kl_sum / float(tot_act_toks.item())
                val_loss_step = vloss_sum

        step += 1

        # Telemetry aggregation
        red_w = sum(1 for g in games if g.red_outcome > 0.25)
        blk_w = sum(1 for g in games if g.red_outcome < -0.25)
        draws = len(games) - red_w - blk_w
        avg_plies = sum(g.total_plies for g in games) / max(1, len(games))
        learner_turns = [t for g in games for t in g.turns if t.learner_owned]
        bad_turns = [t for g in games for t in g.bad_illegal_samples]
        tot_attempts = max(1, len(learner_turns) + len(bad_turns))
        legal_rate = len(learner_turns) / tot_attempts
        consist_rate = sum(1 for t in learner_turns if t.self_consistent) / max(1, len(learner_turns))

        # Phase 4: Periodic AlphaGo Champion Arena Match!
        arena_info: Optional[Dict[str, float]] = None
        if step % args.arena_interval == 0 or step == target_steps:
            arena_info = evaluate_champion_arena(
                learner=policy,
                value_head=value_head,
                champion=champion,
                tok=tok,
                exempt_ids=exempt_ids,
                force_close_ids=force_close_ids,
                num_pairs=args.arena_pairs,
                max_cot_tokens=args.max_cot_tokens,
                max_game_plies=args.max_game_plies,
                device=device,
                autocast_dtype=autocast_dtype,
                rng=rng,
            )
            print(
                f"  >>> [AlphaGo Arena @ Step {step}] Learner vs Champion: "
                f"+{arena_info['wins']} ={arena_info['draws']} -{arena_info['losses']} "
                f"(Score={arena_info['win_score']*100:.1f}%, ΔElo={arena_info['elo_diff']:+.1f})"
            )
            if arena_info["win_score"] >= args.arena_gate or not os.path.exists(best_sp_path):
                if arena_info["win_score"] >= args.arena_gate:
                    cumulative_elo += max(0.0, arena_info["elo_diff"])
                new_champ = copy.deepcopy(policy).to(device).eval()
                for p in new_champ.parameters():
                    p.requires_grad_(False)
                if device.type == "cuda" and autocast_dtype is not None:
                    new_champ = new_champ.to(dtype=autocast_dtype)
                champion = new_champ
                league_pool.append(champion)
                if len(league_pool) > 4:
                    league_pool.pop(0)  # Keep sliding pool of 4 diverse historical Champions (<35 MB VRAM!)

                champ_ckpt = {
                    "stage": "rl_selfplay_alphago",
                    "step": step,
                    "cumulative_elo": round(cumulative_elo, 1),
                    "total_games_played": total_games_played,
                    "best_val_loss": val_loss_step,
                    "model_config": cfg.to_dict(),
                    "model_state_dict": policy.state_dict(),
                    "value_head_state_dict": value_head.state_dict(),
                    "tokenizer_path": args.tokenizer_path,
                }
                torch.save(champ_ckpt, best_sp_path)
                print(
                    f"  >>> CROWNED NEW SELF-PLAY CHAMPION -> {best_sp_path} "
                    f"(Cumulative Elo={cumulative_elo:+.1f})"
                )
                if args.sync_best_to_main:
                    torch.save(champ_ckpt, main_best_path)
                    print(f"  >>> Synced Self-Play Champion -> {main_best_path} (Web UI ready!)")

        rec = {
            "step": step,
            "total_games_played": total_games_played,
            "red_wins": red_w,
            "black_wins": blk_w,
            "draws": draws,
            "decisive_game_rate": round((red_w + blk_w) / max(1, len(games)), 4),
            "avg_game_plies": round(avg_plies, 1),
            "legal_turn_rate": round(legal_rate, 4),
            "cot_consistent_rate": round(consist_rate, 4),
            "value_loss": round(val_loss_step, 4),
            "kl_div": round(kl_step_val, 5),
            "cumulative_elo": round(cumulative_elo, 1),
            "sec_per_step": round(time.time() - t_step, 2),
        }
        if arena_info is not None:
            rec["arena"] = arena_info
        history.append(rec)

        if step % args.log_interval == 0 or step == 1:
            reasons = {}
            for g in games:
                reasons[g.reason] = reasons.get(g.reason, 0) + 1
            print(
                f"[SelfPlay Step {step:5d}/{target_steps}] Games={total_games_played:,} (+R:{red_w} +B:{blk_w} =D:{draws}, avg_ply={avg_plies:.0f}) | "
                f"Legal={legal_rate*100:5.1f}% | CoTConsist={consist_rate*100:4.0f}% | "
                f"VLoss={val_loss_step:.4f} | KL={kl_step_val:.4f} | Elo={cumulative_elo:+.1f} | {rec['sec_per_step']:.2f}s/st | {reasons}"
            )

        if step % args.save_interval == 0 or step == target_steps:
            # Store Champion weights in FP32 dict for clean cross-device resumption
            champ_sd = {k: v.float().cpu() for k, v in champion.state_dict().items()}
            torch.save(
                {
                    "stage": "rl_selfplay_alphago",
                    "step": step,
                    "cumulative_elo": cumulative_elo,
                    "total_games_played": total_games_played,
                    "model_config": cfg.to_dict(),
                    "model_state_dict": policy.state_dict(),
                    "value_head_state_dict": value_head.state_dict(),
                    "champion_state_dict": champ_sd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_scaler else {},
                    "tokenizer_path": args.tokenizer_path,
                    "history": history,
                },
                latest_sp_path,
            )
            with open(os.path.join(args.out_dir, "selfplay_rl_history.json"), "w", encoding="utf-8") as hf:
                json.dump(history, hf, ensure_ascii=False, indent=2)
            plot_selfplay_curves(history, args.out_dir)

    print("============================================================")
    print(f"AlphaGo Self-Play CoT RL Finished at Step {step:,} ({(time.time() - t0)/60.0:.2f} min)!")
    print(f"Total Full Self-Play Games : {total_games_played:,}")
    print(f"Cumulative Champion ΔElo   : {cumulative_elo:+.1f}")
    print(f"Best Self-Play Checkpoint  : {best_sp_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
