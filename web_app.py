"""
Interactive Chinese Xiangqi + Live Streaming AI CoT Web Server (Zero External Web Dependencies).

Features:
  - Click-to-move SVG Xiangqi chessboard with full legal move highlighting, Check/Checkmate detection,
    Undo, Flip Board, Preset Famous Positions, and Custom FEN loader.
  - Real-time streaming `<|think|> ... <|/think|>` Chain-of-Thought panel supporting:
      1) `lm_smart`     : Trained Neural CoT LM (`checkpoints/best_model.pt`) + Best-of-N / Pikafish legality guard
      2) `lm_raw`       : Pure Trained Neural CoT LM (zero engine intervention)
      3) `pikafish_dag` : Live Pikafish NNUE Master Engine + 5-Strategy Transposition Variation-DAG CoT
  - Hot-reload button (`/api/reload_model`) so you can test your checkpoint live while `train_gpu.py` is still training!

Run locally on Windows / Linux:
  python web_app.py --port 7860
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlparse

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from model.transformer import ModelConfig, XiangqiCausalLM
from xiangqi.board import START_FEN, XiangqiBoard
from xiangqi.dag_cot import build_budgeted_dag_dialogue, format_user_prompt
from xiangqi.engine_wrapper import PikafishEngine, resolve_default_engine_path
from xiangqi.tree_search import traverse_feasible_tree

WEB_UI_PATH = os.path.join(PROJECT_ROOT, "web_ui", "index.html")


def serialize_board(board: XiangqiBoard) -> Dict[str, Any]:
    legal_list: List[Dict[str, Any]] = []
    for uci in board.generate_legal_moves():
        c1, r1, c2, r2 = board.parse_uci(uci)
        is_cap = board.grid[r2][c2] != "."
        try:
            zh = board.uci_to_chinese(uci)
        except Exception:
            zh = uci
        legal_list.append(
            {
                "uci": uci,
                "zh": zh,
                "c1": c1,
                "r1": r1,
                "c2": c2,
                "r2": r2,
                "is_capture": is_cap,
            }
        )
    return {
        "fen": board.to_fen(),
        "turn": board.turn,
        "grid": board.grid,
        "in_check": board.is_in_check(board.turn),
        "legal_moves": legal_list,
    }


def extract_predicted_move_zh(text: str) -> str:
    after = text.split("<|/think|>")[-1] if "<|/think|>" in text else text
    m = re.search(
        r"最佳走法\s*[：:]\s*((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])",
        after,
    )
    if m:
        return m.group(1).strip()
    ms = re.findall(
        r"((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])",
        after,
    )
    if ms:
        return ms[-1]
    ms_all = re.findall(
        r"((?:[前中后][车马炮兵卒][一二三四五六七八九1-9]?|[帅将仕士相象车马炮兵卒][一二三四五六七八九1-9])[进退平][一二三四五六七八九1-9])",
        text,
    )
    return ms_all[-1] if ms_all else ""


class AIBackendHub:
    """Thread-safe manager for both Neural CoT LM checkpoints and Pikafish NNUE DAG-CoT."""

    def __init__(self, explicit_ckpt: str = "", device_str: str = "auto"):
        self.lock = threading.Lock()
        self.explicit_ckpt = explicit_ckpt
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.lm_loaded: bool = False
        self.lm_kind: str = ""  # "gpu_10k" or "proto_1k5"
        self.model: Any = None
        self.tokenizer: Any = None
        self.exempt_ids: Set[int] = set()
        self.model_summary: str = "未加载"
        self.ckpt_path_loaded: str = ""

        self.pikafish: Optional[PikafishEngine] = None
        self.pikafish_lru: "OrderedDict[str, Any]" = OrderedDict()
        self._init_pikafish()
        self.reload_checkpoint()

    def _init_pikafish(self) -> None:
        try:
            eng_p = resolve_default_engine_path()
            if os.path.exists(eng_p):
                self.pikafish = PikafishEngine(engine_path=eng_p, hash_mb=32, threads=1)
        except Exception as e:
            print(f"[Web] Pikafish init warning: {e}")
            self.pikafish = None

    def reload_checkpoint(self) -> str:
        with self.lock:
            candidates = []
            if self.explicit_ckpt:
                candidates.append(("gpu_10k", self.explicit_ckpt))
            candidates.extend(
                [
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "best_selfplay_model.pt")),
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "best_rl_model.pt")),
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")),
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "latest_selfplay_checkpoint.pt")),
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "latest_rl_checkpoint.pt")),
                    ("gpu_10k", os.path.join(PROJECT_ROOT, "checkpoints", "latest_checkpoint.pt")),
                    ("proto_1k5", "/home/user/xiangqi_cot_lm/checkpoints/best_model.pt"),
                ]
            )

            for kind, path in candidates:
                if not path or not os.path.exists(path):
                    continue
                try:
                    if kind == "gpu_10k":
                        ckpt = torch.load(path, map_location=self.device, weights_only=False)
                        tok_p = ckpt.get("tokenizer_path", DEFAULT_TOKENIZER_PATH)
                        if not os.path.exists(tok_p):
                            tok_p = DEFAULT_TOKENIZER_PATH
                        tok = GeneralXiangqiTokenizer.load(tok_p)
                        cfg = ModelConfig.from_dict(ckpt["model_config"])
                        mdl = XiangqiCausalLM(cfg).to(self.device)
                        mdl.load_state_dict(ckpt["model_state_dict"], strict=True)
                        mdl.eval()

                        exempt = set(range(16))
                        for s in ["\n", "，", "。", "；", "：", "、", "（", "）", "【", "】", "或", "或者", "若", "则", "红", "黑"]:
                            exempt.update(tok.encode(s))

                        n_p = mdl.count_parameters() / 1e6
                        step = ckpt.get("step", "?")
                        stage = ckpt.get("stage", "sft")
                        self.model = mdl
                        self.tokenizer = tok
                        self.exempt_ids = exempt
                        self.lm_kind = "gpu_10k"
                        self.lm_loaded = True
                        self.ckpt_path_loaded = path
                        if stage == "rl_selfplay_alphago":
                            elo = float(ckpt.get("cumulative_elo", 0.0))
                            gms = int(ckpt.get("total_games_played", 0))
                            self.model_summary = (
                                f"[AlphaGo自对弈] {n_p:.2f}M (10k词表, step={step}, games={gms}, ΔElo={elo:+.0f}, {self.device.type.upper()})"
                            )
                        elif stage == "rl_grpo":
                            r_ema = float(ckpt.get("best_moving_reward", 0.0))
                            self.model_summary = f"[RL-GRPO] {n_p:.2f}M (10k词表, step={step}, R={r_ema:+.2f}, {self.device.type.upper()})"
                        else:
                            vloss = float(ckpt.get("best_val_loss", 0.0))
                            self.model_summary = f"[SFT] {n_p:.2f}M (10k词表, step={step}, val_loss={vloss:.3f}, {self.device.type.upper()})"
                        return f"已成功加载权重: {self.model_summary}"

                    elif kind == "proto_1k5":
                        proto_root = "/home/user/xiangqi_cot_lm"
                        tok_mod_spec = importlib.util.spec_from_file_location(
                            "proto_tok", os.path.join(proto_root, "model", "tokenizer.py")
                        )
                        tr_mod_spec = importlib.util.spec_from_file_location(
                            "proto_tr", os.path.join(proto_root, "model", "transformer.py")
                        )
                        assert tok_mod_spec and tok_mod_spec.loader and tr_mod_spec and tr_mod_spec.loader
                        tok_mod = importlib.util.module_from_spec(tok_mod_spec)
                        tr_mod = importlib.util.module_from_spec(tr_mod_spec)
                        sys.modules["proto_tok"] = tok_mod
                        sys.modules["proto_tr"] = tr_mod
                        tok_mod_spec.loader.exec_module(tok_mod)
                        tr_mod_spec.loader.exec_module(tr_mod)

                        tok = tok_mod.XiangqiTokenizer.load(os.path.join(proto_root, "checkpoints", "tokenizer.json"))
                        ckpt = torch.load(path, map_location=self.device, weights_only=False)
                        cfg = tr_mod.ModelConfig.from_dict(ckpt["config"])
                        mdl = tr_mod.XiangqiCoTForCausalLM(cfg).to(self.device)
                        mdl.load_state_dict(ckpt["model_state_dict"], strict=True)
                        mdl.eval()

                        exempt = set(range(9))
                        for s in ["\n", "，", "。", "；", "：", "、", "（", "）", "或", "若", "则"]:
                            exempt.update(tok.encode(s))

                        n_p = mdl.count_parameters() / 1e6
                        step = ckpt.get("step", 1290)
                        vloss = ckpt.get("val_loss", 1.841)
                        self.model = mdl
                        self.tokenizer = tok
                        self.exempt_ids = exempt
                        self.lm_kind = "proto_1k5"
                        self.lm_loaded = True
                        self.ckpt_path_loaded = path
                        self.model_summary = f"{n_p:.2f}M (已训原型权重, step={step}, val_loss={vloss:.3f}, {self.device.type.upper()})"
                        return f"已加载预置模型: {self.model_summary}"
                except Exception as e:
                    print(f"[Web] Failed loading {path}: {e}")

            self.lm_loaded = False
            self.model_summary = "未检测到已训权重"
            return "未找到可用模型权重，已就绪使用 Pikafish DAG-CoT 模式"

    @torch.no_grad()
    def _stream_proto_tokens(
        self,
        prompt_ids: List[int],
        temperature: float,
        force_close_ids: List[int],
    ) -> Iterator[int]:
        """Streaming KV-cache generator for the 1.5k-vocab prototype model."""
        mdl = self.model
        cfg = mdl.cfg
        dev = self.device
        stop_id = self.tokenizer.eos_id
        end_think_id = self.tokenizer.end_think_id

        ids = list(prompt_ids[: cfg.max_seq_len - 16])
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        T0 = inp.shape[1]
        x = mdl.tok_emb(inp)
        cos, sin = mdl.rope_cos[:T0], mdl.rope_sin[:T0]
        head_dim = cfg.d_model // cfg.n_heads
        empty_k = torch.empty((1, cfg.n_heads, 0, head_dim), dtype=x.dtype, device=dev)
        empty_v = torch.empty((1, cfg.n_heads, 0, head_dim), dtype=x.dtype, device=dev)
        caches = []
        for blk in mdl.blocks:
            x, c = blk(x, cos, sin, kv_cache=(empty_k, empty_v))
            caches.append(c)
        logits = mdl.lm_head(mdl.norm_f(x[:, -1:, :]))[:, 0, :].float()

        generated: List[int] = []
        cur_pos = T0
        reserve = len(force_close_ids) + 8

        def step_tok(tid: int) -> None:
            nonlocal caches, logits, cur_pos
            generated.append(tid)
            if cur_pos >= cfg.max_seq_len:
                return
            si = torch.tensor([[tid]], dtype=torch.long, device=dev)
            hx = mdl.tok_emb(si)
            cs, ss = mdl.rope_cos[cur_pos : cur_pos + 1], mdl.rope_sin[cur_pos : cur_pos + 1]
            new_c = []
            for idx, blk in enumerate(mdl.blocks):
                hx, c = blk(hx, cs, ss, kv_cache=caches[idx])
                new_c.append(c)
            caches = new_c
            logits = mdl.lm_head(mdl.norm_f(hx))[:, 0, :].float()
            cur_pos += 1

        while cur_pos < cfg.max_seq_len:
            if end_think_id not in generated and (cfg.max_seq_len - cur_pos) <= reserve:
                for cid in force_close_ids:
                    if cur_pos >= cfg.max_seq_len:
                        break
                    step_tok(cid)
                    yield cid
                continue

            sl = logits.clone()
            if generated:
                for tid in set(generated[-40:]) - self.exempt_ids:
                    sl[0, tid] = sl[0, tid] / 1.16 if sl[0, tid] > 0 else sl[0, tid] * 1.16

            if temperature <= 1e-5:
                nxt = int(torch.argmax(sl, dim=-1).item())
            else:
                probs = F.softmax(sl / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())

            if nxt == stop_id:
                break
            step_tok(nxt)
            yield nxt

    def stream_ai_turn(self, fen: str, mode: str, temperature: float) -> Iterator[Dict[str, Any]]:
        with self.lock:
            t0 = time.time()
            board = XiangqiBoard(fen)
            legal_map: Dict[str, str] = {}
            for uci in board.generate_legal_moves():
                try:
                    legal_map[board.uci_to_chinese(uci)] = uci
                except Exception:
                    pass
            if not legal_map:
                yield {"type": "error", "error": "当前局面已无合法着法（绝杀/困毙）"}
                return

            use_pikafish_mode = (mode == "pikafish_dag") or (not self.lm_loaded)

            if use_pikafish_mode:
                if self.pikafish is None:
                    yield {"type": "error", "error": "Pikafish 引擎未就绪"}
                    return
                res = traverse_feasible_tree(
                    fen=board.to_fen(),
                    engine=self.pikafish,
                    target_plies=4,
                    max_branch=5,
                    score_threshold_cp=38,
                    oscillation_threshold_cp=65,
                    root_depth=8,
                    inner_depth=6,
                    max_routes=350,
                    max_node_evals=68,
                    lru_cache=self.pikafish_lru,
                )
                if res is None:
                    best_uci = next(iter(legal_map.values()))
                    best_zh = board.uci_to_chinese(best_uci)
                    cot_body = f"当前仅存稳健应着 {best_zh}。"
                else:
                    best_uci = res.best_move_uci
                    best_zh = res.best_move_zh
                    _, cot_body, _, _ = build_budgeted_dag_dialogue(res, max_seq_tokens=980, max_routes=320)

                full_display = f"<|think|>\n{cot_body}\n<|/think|>\n最佳走法：{best_zh}"
                # Stream smoothly in readable chunks so user sees the live structure unfold
                chunk_sz = 28
                emitted = 0
                for start in range(0, len(full_display), chunk_sz):
                    piece = full_display[start : start + chunk_sz]
                    emitted += len(piece) // 2
                    dt = max(1e-3, time.time() - t0)
                    yield {
                        "type": "delta",
                        "text": piece,
                        "tokens": emitted,
                        "tok_per_s": int(emitted / dt),
                    }
                    time.sleep(0.018)

                c1, r1, c2, r2 = board.parse_uci(best_uci)
                is_cap = board.grid[r2][c2] != "."
                board.push_uci(best_uci)
                dt = max(1e-3, time.time() - t0)
                yield {
                    "type": "done",
                    "display_cot": full_display,
                    "move_zh": best_zh,
                    "move_uci": best_uci,
                    "source_badge": "👑 Pikafish 大师引擎 + 五重同形合流 DAG",
                    "tokens": emitted,
                    "elapsed_s": round(dt, 2),
                    "tok_per_s": int(emitted / dt),
                    "is_capture": is_cap,
                    "move_coords": {"c1": c1, "r1": r1, "c2": c2, "r2": r2},
                    "board": serialize_board(board),
                }
                return

            # Neural CoT LM Streaming Path (`gpu_10k` or `proto_1k5`)
            if self.lm_kind == "gpu_10k":
                u_prompt = format_user_prompt(board.to_fen())
                prompt_str = f"<|user|>\n{u_prompt}\n<|assistant|>\n<|think|>\n"
                prefix_init = "<|think|>\n"
                prompt_ids = self.tokenizer.encode(prompt_str)
                force_close = self.tokenizer.encode("\n<|/think|>\n最佳走法：")
                token_iter = self.model.stream_generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=self.model.cfg.max_seq_len - len(prompt_ids),
                    temperature=temperature,
                    top_p=0.92,
                    repetition_penalty=1.12,
                    rep_window=40,
                    no_repeat_ngram_size=10,
                    exempt_rep_ids=self.exempt_ids,
                    eos_token_id=self.tokenizer.eos_id,
                    end_think_id=self.tokenizer.end_think_id,
                    force_close_ids=force_close,
                    device=self.device,
                )
            else:
                side_zh = "红方" if board.turn == "w" else "黑方"
                u_prompt = f"盘面FEN：{board.to_fen()}\n当前轮到{side_zh}走棋，请推演可行路线并给出下一步最佳走法。"
                prompt_str = f"<|user|>\n{u_prompt}\n<|assistant|>\n"
                prefix_init = ""
                prompt_ids = self.tokenizer.encode(prompt_str)
                force_close = self.tokenizer.encode("\n<|/think|>\n最佳走法：")
                token_iter = self._stream_proto_tokens(prompt_ids, temperature, force_close)

            gen_ids: List[int] = []
            emitted_str = prefix_init
            if prefix_init:
                yield {"type": "delta", "text": prefix_init, "tokens": 1, "tok_per_s": 0}

            for tid in token_iter:
                gen_ids.append(tid)
                if len(gen_ids) % 3 == 0:
                    decoded = prefix_init + self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                    decoded = re.sub(r"(?:<\|think\|>\s*){2,}", "<|think|>\n", decoded)
                    decoded = decoded.replace("<|eos|>", "").rstrip("\ufffd")
                    if len(decoded) > len(emitted_str):
                        delta = decoded[len(emitted_str) :]
                        emitted_str = decoded
                        dt = max(1e-3, time.time() - t0)
                        yield {
                            "type": "delta",
                            "text": delta,
                            "tokens": len(gen_ids),
                            "tok_per_s": int(len(gen_ids) / dt),
                        }

            # Flush final trailing tokens
            full_decoded = prefix_init + self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            full_decoded = re.sub(r"(?:<\|think\|>\s*){2,}", "<|think|>\n", full_decoded)
            full_decoded = full_decoded.replace("<|eos|>", "")
            if len(full_decoded) > len(emitted_str):
                yield {
                    "type": "delta",
                    "text": full_decoded[len(emitted_str) :],
                    "tokens": len(gen_ids),
                    "tok_per_s": int(len(gen_ids) / max(1e-3, time.time() - t0)),
                }

            pred_zh = extract_predicted_move_zh(full_decoded)
            source_badge = "✅ 自研神经网络 CoT 原生着法"

            if pred_zh in legal_map:
                chosen_zh = pred_zh
                chosen_uci = legal_map[chosen_zh]
            else:
                chosen_zh = ""
                chosen_uci = ""
                # Try rapid Best-of-3 stochastic resampling from Neural LM
                for samp_t in (0.55, 0.75):
                    if self.lm_kind == "gpu_10k":
                        s_ids = self.model.generate(
                            prompt_ids=prompt_ids,
                            max_new_tokens=self.model.cfg.max_seq_len - len(prompt_ids),
                            temperature=samp_t,
                            top_p=0.92,
                            repetition_penalty=1.12,
                            exempt_rep_ids=self.exempt_ids,
                            eos_token_id=self.tokenizer.eos_id,
                            end_think_id=self.tokenizer.end_think_id,
                            force_close_ids=force_close,
                            device=self.device,
                        )
                    else:
                        s_ids = list(self._stream_proto_tokens(prompt_ids, samp_t, force_close))
                    s_txt = self.tokenizer.decode(s_ids, skip_special_tokens=False)
                    cand_zh = extract_predicted_move_zh(s_txt)
                    if cand_zh in legal_map:
                        chosen_zh = cand_zh
                        chosen_uci = legal_map[cand_zh]
                        source_badge = f"🎯 自研模型 Best-of-N 采样命中（贪婪原测: {pred_zh or '无'}）"
                        full_decoded += f"\n\n[注：贪婪解码着法「{pred_zh or '未闭合'}」当前盘面不合法，经模型采样修正为合法着法「{chosen_zh}」]"
                        break

                if not chosen_uci:
                    if mode == "lm_smart" and self.pikafish is not None and board.is_nnue_valid():
                        evs = self.pikafish.evaluate_multipv(board.to_fen(), depth=6, multipv=1)
                        if evs and evs[0].uci in legal_map.values():
                            chosen_uci = evs[0].uci
                            chosen_zh = board.uci_to_chinese(chosen_uci)
                            source_badge = f"🛡️ 合法性守护兜底（模型原测非法: {pred_zh or '无'}）"
                            full_decoded += f"\n\n[合法守护提示：当前模型检查点预测着法「{pred_zh or '无'}」不合法，已由合法守护修正为「{chosen_zh}」以继续对局]"
                    if not chosen_uci:
                        chosen_zh, chosen_uci = next(iter(legal_map.items()))
                        source_badge = f"⚠️ 规则合法兜底（模型原测非法: {pred_zh or '无'}）"

            c1, r1, c2, r2 = board.parse_uci(chosen_uci)
            is_cap = board.grid[r2][c2] != "."
            board.push_uci(chosen_uci)
            dt = max(1e-3, time.time() - t0)

            yield {
                "type": "done",
                "display_cot": full_decoded,
                "move_zh": chosen_zh,
                "move_uci": chosen_uci,
                "source_badge": source_badge,
                "tokens": len(gen_ids),
                "elapsed_s": round(dt, 2),
                "tok_per_s": int(len(gen_ids) / dt),
                "is_capture": is_cap,
                "move_coords": {"c1": c1, "r1": r1, "c2": c2, "r2": r2},
                "board": serialize_board(board),
            }


HUB: Optional[AIBackendHub] = None


class XiangqiWebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"[Web] {self.address_string()} - {fmt % args}\n")
        sys.stdout.flush()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        assert HUB is not None
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            with open(WEB_UI_PATH, "rb") as f:
                html_bytes = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(html_bytes)
        elif parsed.path == "/api/status":
            self._send_json(
                {
                    "lm_loaded": HUB.lm_loaded,
                    "lm_kind": HUB.lm_kind,
                    "model_summary": HUB.model_summary,
                    "pikafish_ready": HUB.pikafish is not None,
                }
            )
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        assert HUB is not None
        parsed = urlparse(self.path)

        if parsed.path == "/api/reload_model":
            msg = HUB.reload_checkpoint()
            self._send_json({"ok": True, "message": msg, "model_summary": HUB.model_summary})
            return

        if parsed.path == "/api/board":
            try:
                body = self._read_json_body()
                fen = body.get("fen", START_FEN)
                b = XiangqiBoard(fen)
                self._send_json(serialize_board(b))
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if parsed.path == "/api/apply_move":
            try:
                body = self._read_json_body()
                b = XiangqiBoard(body["fen"])
                uci = body["uci"]
                if uci not in b.generate_legal_moves():
                    self._send_json({"error": f"非法着法: {uci}"}, status=400)
                    return
                zh = b.uci_to_chinese(uci)
                c1, r1, c2, r2 = b.parse_uci(uci)
                is_cap = b.grid[r2][c2] != "."
                b.push_uci(uci)
                self._send_json(
                    {
                        "ok": True,
                        "move_uci": uci,
                        "move_zh": zh,
                        "is_capture": is_cap,
                        "move_coords": {"c1": c1, "r1": r1, "c2": c2, "r2": r2},
                        "board": serialize_board(b),
                    }
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if parsed.path == "/api/ai_move_stream":
            try:
                body = self._read_json_body()
                fen = body.get("fen", START_FEN)
                mode = body.get("mode", "lm_smart")
                temperature = float(body.get("temperature", 0.0))

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self._send_cors_headers()
                self.end_headers()

                for event in HUB.stream_ai_turn(fen, mode, temperature):
                    chunk = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as e:
                try:
                    err = (json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")
                    self.wfile.write(err)
                    self.wfile.flush()
                except Exception:
                    pass
            return

        self._send_json({"error": "Not found"}, status=404)


def main() -> None:
    global HUB
    parser = argparse.ArgumentParser(description="Xiangqi AI CoT Interactive Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to specific .pt checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser on local PC")
    args = parser.parse_args()

    HUB = AIBackendHub(explicit_ckpt=args.checkpoint, device_str=args.device)
    server = ThreadingHTTPServer((args.host, args.port), XiangqiWebHandler)

    local_url = f"http://127.0.0.1:{args.port}"
    print("=" * 68, flush=True)
    print("  中国象棋 · AI 思维链 (CoT) 可视化对弈网页已启动！", flush=True)
    print(f"  本地浏览器访问地址 : {local_url}", flush=True)
    print(f"  当前神经网络状态   : {HUB.model_summary}", flush=True)
    print(f"  Pikafish 引擎状态  : {'已就绪' if HUB.pikafish else '未找到'}", flush=True)
    print("=" * 68, flush=True)

    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        server.server_close()


if __name__ == "__main__":
    main()
