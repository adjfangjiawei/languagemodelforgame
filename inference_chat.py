"""
Interactive Dialogue & Quantitative Pikafish Evaluation CLI for the 10k-Vocab Xiangqi DAG-CoT LM.

Usage examples:
  1) Evaluate held-out validation boards after training:
     python inference_chat.py --eval-val --num-eval 100
  2) Analyze a specific Xiangqi FEN with full DAG-CoT reasoning:
     python inference_chat.py --fen "r1bakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 5"
  3) Launch interactive play / FEN / general-chat CLI:
     python inference_chat.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from model.transformer import ModelConfig, XiangqiCausalLM
from xiangqi.board import START_FEN, XiangqiBoard
from xiangqi.dag_cot import format_user_prompt


def build_exempt_token_ids(tok: GeneralXiangqiTokenizer) -> Set[int]:
    """Exempt special chat/CoT tags, newlines, Chinese punctuation, and connectors from repetition penalty."""
    exempt: Set[int] = set(range(16))
    for s in ["\n", "，", "。", "；", "：", "、", "（", "）", "【", "】", "或", "或者", "若", "则", "红", "黑"]:
        exempt.update(tok.encode(s))
    return exempt


def extract_cot_and_move(response_text: str) -> Tuple[str, str]:
    cot = ""
    m_think = re.search(r"<\|think\|>\s*(.*?)\s*<\|/think\|>", response_text, flags=re.DOTALL)
    if m_think:
        cot = m_think.group(1).strip()

    after_think = response_text.split("<|/think|>")[-1] if "<|/think|>" in response_text else response_text
    m_move = re.search(r"最佳走法\s*[：:]\s*([^\s<。，；\n]{3,5})", after_think)
    if m_move:
        return cot, m_move.group(1).strip()

    m_any = re.findall(r"([车马炮相仕帅兵象士将卒前后中][一二三四五六七八九1-9进退平]{2,4})", after_think)
    if m_any:
        return cot, m_any[-1]
    return cot, ""


def legal_zh_map(board: XiangqiBoard) -> Dict[str, str]:
    """Map every legal Chinese move string -> UCI move string on `board`."""
    mapping: Dict[str, str] = {}
    for uci in board.generate_legal_moves():
        try:
            zh = board.uci_to_chinese(uci)
            mapping[zh] = uci
        except Exception:
            continue
    return mapping


def load_trained_model(
    ckpt_path: str, tokenizer_path: str, device_str: str = "auto"
) -> Tuple[XiangqiCausalLM, GeneralXiangqiTokenizer, torch.device]:
    dev = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else (device_str if device_str != "auto" else "cpu"))
    tok = GeneralXiangqiTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = XiangqiCausalLM(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, tok, dev


def generate_xiangqi_reply(
    model: XiangqiCausalLM,
    tok: GeneralXiangqiTokenizer,
    fen: str,
    device: torch.device,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Tuple[str, str, str]:
    prompt_str = f"<|user|>\n{format_user_prompt(fen)}\n<|assistant|>\n<|think|>\n"
    prompt_ids = tok.encode(prompt_str)
    force_close_ids = tok.encode("\n<|/think|>\n最佳走法：")
    exempt_ids = build_exempt_token_ids(tok)

    gen_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=model.cfg.max_seq_len - len(prompt_ids),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.12,
        rep_window=40,
        no_repeat_ngram_size=10,
        exempt_rep_ids=exempt_ids,
        eos_token_id=tok.eos_id,
        end_think_id=tok.end_think_id,
        force_close_ids=force_close_ids,
        device=device,
    )
    full_assistant = "<|think|>\n" + tok.decode(gen_ids, skip_special_tokens=False)
    full_assistant = full_assistant.replace("<|eos|>", "").strip()
    cot, pred_zh = extract_cot_and_move(full_assistant)
    return full_assistant, cot, pred_zh


def run_val_benchmark(
    model: XiangqiCausalLM,
    tok: GeneralXiangqiTokenizer,
    val_jsonl: str,
    device: torch.device,
    num_eval: int,
    out_path: str,
) -> None:
    rows: List[Dict] = []
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= num_eval:
                break

    n = len(rows)
    if n == 0:
        print("No validation records found!")
        return

    cot_format_ok = 0
    dag_or_merge_ok = 0
    transposition_phrase_ok = 0
    greedy_legal = 0
    greedy_feasible = 0
    greedy_top1 = 0
    bo4_legal = 0
    bo4_feasible = 0
    bo4_top1 = 0

    for i, rec in enumerate(rows, 1):
        fen = rec["fen"]
        board = XiangqiBoard(fen)
        zh2uci = legal_zh_map(board)
        feasible_set = set(rec.get("feasible_root_moves_zh", [rec["best_move_zh"]]))
        gold_zh = rec["best_move_zh"]

        raw_txt, cot, pred_zh = generate_xiangqi_reply(model, tok, fen, device, temperature=0.0)
        if "<|think|>" in raw_txt and "<|/think|>" in raw_txt and "最佳走法：" in raw_txt:
            cot_format_ok += 1
        if ("或" in cot) or ("、" in cot):
            dag_or_merge_ok += 1
        if ("同形" in cot) or ("同一盘面" in cot) or ("或者" in cot):
            transposition_phrase_ok += 1

        g_leg = pred_zh in zh2uci
        if g_leg:
            greedy_legal += 1
            if pred_zh in feasible_set:
                greedy_feasible += 1
            if pred_zh == gold_zh:
                greedy_top1 += 1

        # Best-of-4 legality-verified candidate selection (greedy + up to 3 stochastic samples)
        chosen_zh = pred_zh if g_leg else ""
        if not chosen_zh:
            for temp in (0.45, 0.65, 0.80):
                _, _, s_zh = generate_xiangqi_reply(model, tok, fen, device, temperature=temp, top_p=0.92)
                if s_zh in zh2uci:
                    chosen_zh = s_zh
                    break

        if chosen_zh in zh2uci:
            bo4_legal += 1
            if chosen_zh in feasible_set:
                bo4_feasible += 1
            if chosen_zh == gold_zh:
                bo4_top1 += 1

        if i % 10 == 0 or i == n:
            print(
                f"[Eval {i}/{n}] format={cot_format_ok/i:.1%} | greedy_legal={greedy_legal/i:.1%} "
                f"| bo4_legal={bo4_legal/i:.1%} | bo4_feasible={bo4_feasible/i:.1%} | bo4_top1={bo4_top1/i:.1%}"
            )

    metrics = {
        "num_evaluated": n,
        "cot_tag_format_rate": round(cot_format_ok / n, 4),
        "cot_dag_or_merge_rate": round(dag_or_merge_ok / n, 4),
        "cot_transposition_phrase_rate": round(transposition_phrase_ok / n, 4),
        "greedy_legal_move_rate": round(greedy_legal / n, 4),
        "greedy_pikafish_feasible_rate": round(greedy_feasible / n, 4),
        "greedy_pikafish_top1_rate": round(greedy_top1 / n, 4),
        "bo4_legal_move_rate": round(bo4_legal / n, 4),
        "bo4_pikafish_feasible_rate": round(bo4_feasible / n, 4),
        "bo4_pikafish_top1_rate": round(bo4_top1 / n, 4),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)
    print("\n================ Validation Benchmark Summary ================")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def interactive_loop(model: XiangqiCausalLM, tok: GeneralXiangqiTokenizer, device: torch.device) -> None:
    board = XiangqiBoard(START_FEN)
    print("============================================================")
    print(" 中国象棋 DAG-CoT 对话模型交互终端 (10,240 通用词表)")
    print(" 输入指令：")
    print("   - 直接输入红方走法（如 `炮二平五` 或 `h2e2`）与模型对弈")
    print("   - `fen <FEN串>` 载入任意残局/中局盘面并让模型推演")
    print("   - `go` 让模型为当前盘面生成同形合流 CoT 并走子")
    print("   - `reset` 重置初始盘面 | `quit` 退出")
    print("============================================================")

    while True:
        print("\n当前FEN:", board.to_fen())
        try:
            cmd = input("请输入走法或指令 > ").strip()
        except EOFError:
            break
        if not cmd or cmd.lower() in ("quit", "exit", "q"):
            break
        if cmd.lower() == "reset":
            board = XiangqiBoard(START_FEN)
            continue
        if cmd.lower().startswith("fen "):
            board = XiangqiBoard(cmd[4:].strip())
            cmd = "go"

        if cmd.lower() != "go":
            zh2uci = legal_zh_map(board)
            if cmd in zh2uci:
                board.push_uci(zh2uci[cmd])
            elif cmd in board.generate_legal_moves():
                board.push_uci(cmd)
            else:
                # Treat as free-form natural language query (for Stage-2 general chat models!)
                prompt_ids = tok.encode(f"<|user|>\n{cmd}\n<|assistant|>\n")
                out_ids = model.generate(prompt_ids=prompt_ids, max_new_tokens=512, temperature=0.5, device=device)
                print("\n[模型回复]\n" + tok.decode(out_ids, skip_special_tokens=False).replace("<|eos|>", ""))
                continue

        raw_txt, cot, pred_zh = generate_xiangqi_reply(model, tok, board.to_fen(), device, temperature=0.0)
        zh2uci = legal_zh_map(board)
        if pred_zh not in zh2uci:
            for t in (0.5, 0.7):
                raw_txt, cot, pred_zh = generate_xiangqi_reply(model, tok, board.to_fen(), device, temperature=t)
                if pred_zh in zh2uci:
                    break
        print("\n---------------- 模型思维链 (DAG-CoT) ----------------")
        print(cot)
        print("------------------------------------------------------")
        print(f"模型最佳走法：{pred_zh}")
        if pred_zh in zh2uci:
            board.push_uci(zh2uci[pred_zh])


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiangqi DAG-CoT Inference & Evaluation")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"))
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--val-jsonl", type=str, default=os.path.join(PROJECT_ROOT, "data", "val.jsonl"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fen", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--eval-val", action="store_true")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--out-metrics", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "eval_metrics.json"))
    args = parser.parse_args()

    model, tok, dev = load_trained_model(args.checkpoint, args.tokenizer_path, args.device)

    if args.eval_val:
        run_val_benchmark(model, tok, args.val_jsonl, dev, args.num_eval, args.out_metrics)
    elif args.interactive:
        interactive_loop(model, tok, dev)
    else:
        test_fen = args.fen or "r1bakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 5"
        raw_txt, cot, pred_zh = generate_xiangqi_reply(model, tok, test_fen, dev, temperature=0.0)
        b = XiangqiBoard(test_fen)
        zh2uci = legal_zh_map(b)
        print(f"FEN: {test_fen}")
        print("\n<|think|>")
        print(cot)
        print("<|/think|>")
        print(f"最佳走法：{pred_zh}  (合法着法校验: {pred_zh in zh2uci})")


if __name__ == "__main__":
    main()
