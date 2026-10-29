"""
Build the 10,240-vocab Universal Byte-Level BPE Tokenizer (`tokenizer_assets/tokenizer_10k.json`).

Combines:
1. Full GB2312 standard Chinese character set (6,763 characters: 3,755 Level-1 + 3,008 Level-2) so general
   Chinese text never fragments common characters into raw UTF-8 bytes.
2. High-frequency General Chinese multi-character words + English/Math/Code vocabulary (seeded from Qwen2.5
   BPE rank order or optional user `--extra-corpus` text/jsonl files).
3. Xiangqi FEN + DAG-CoT reasoning phrases & standard Chinese move notations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer, get_gb2312_level1_and_2_chars
from xiangqi.board import START_FEN, XiangqiBoard


def generate_all_xiangqi_move_notations() -> List[str]:
    """Generate exhaustive Chinese Xiangqi move strings ('炮二平五', '马8进7', '前车进二', etc.) for vocabulary seeding."""
    moves: List[str] = []
    r_nums = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    b_nums = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    actions = ["进", "退", "平"]

    for p, nums in [
        ("车", r_nums), ("马", r_nums), ("炮", r_nums), ("兵", r_nums), ("相", r_nums), ("仕", r_nums), ("帅", r_nums),
        ("车", b_nums), ("马", b_nums), ("炮", b_nums), ("卒", b_nums), ("象", b_nums), ("士", b_nums), ("将", b_nums),
    ]:
        for n1 in nums:
            for act in actions:
                for n2 in nums[:4]:
                    moves.append(f"{p}{n1}{act}{n2}")
        if p in ("车", "马", "炮", "兵", "卒"):
            for tag in ("前", "后"):
                for act in actions:
                    for n2 in nums[:3]:
                        moves.append(f"{tag}{p}{act}{n2}")
    return moves


def extract_general_vocab_seeds_from_hf(limit_rank: int = 32000) -> List[str]:
    """
    If available (cached or online), decode top `limit_rank` tokens from Qwen/Qwen2.5-0.5B's BPE table
    to obtain genuine real-world high-frequency Chinese 2~4 char words, English words, and code tokens.
    """
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        q_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "tokenizer.json")
        q_tok = Tokenizer.from_file(q_path)
        seeds: List[str] = []
        total = min(limit_rank, q_tok.get_vocab_size())
        # Weight earlier (higher-frequency) BPE merges more heavily
        for tid in range(256, total):
            s = q_tok.decode([tid], skip_special_tokens=True)
            if not s or "\ufffd" in s:
                continue
            repeat = 4 if tid < 6000 else (2 if tid < 16000 else 1)
            for _ in range(repeat):
                seeds.append(s)
        return seeds
    except Exception as e:
        print(f"[Info] Offline fallback (could not load HF general seed: {e})")
        return []


def build_training_stream(extra_corpus_paths: List[str], xiangqi_jsonl: str) -> Iterator[str]:
    # 1. Guarantee all 6,763 GB2312 Chinese characters merge cleanly into single tokens
    gb_chars = get_gb2312_level1_and_2_chars()
    lvl1 = gb_chars[:3755]
    lvl2 = gb_chars[3755:]
    # Yield newline-separated items so Chinese chars/words DO NOT acquire an artificial leading space ('Ġ')!
    lvl1_block = "\n".join(lvl1)
    lvl2_block = "\n".join(lvl2)
    for _ in range(18):
        yield lvl1_block
    for _ in range(8):
        yield lvl2_block

    # 2. General Chinese & English/Code multi-character words
    gen_seeds = extract_general_vocab_seeds_from_hf(limit_rank=28000)
    if gen_seeds:
        chunk = 500
        for i in range(0, len(gen_seeds), chunk):
            yield "\n".join(gen_seeds[i : i + chunk])

    # 3. Optional user-supplied general Chinese corpus files (.txt or .jsonl)
    for p in extra_corpus_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if p.endswith(".jsonl"):
                        try:
                            obj = json.loads(line)
                            yield obj.get("text", "") or obj.get("cot", "") or line
                        except Exception:
                            yield line
                    else:
                        yield line

    # 4. Xiangqi move vocabulary & DAG-CoT reasoning phrases (~600-800 chess merges)
    xq_moves = " ".join(generate_all_xiangqi_move_notations())
    yield xq_moves

    dag_phrases = (
        "中国象棋局面FEN 当前行棋方：红方 黑方 请推演后续可行变化并给出下一步最佳走法 "
        "当前轮到红方行棋 当前轮到黑方行棋 双方大致均势 红方占据主动优势 红方胜势明显 "
        "经评估筛选，当前仅有唯一不震荡的稳健走法 其余着法均会导致局势明显受损 "
        "按同形归并与前缀树合并推演4至5步变化 【同形合流】 如果 或者 则双方归于同一盘面 "
        "此后红可走 此后黑可走 若黑改走 若红改走 最佳走法： 综合以上 可行路线分析"
    )
    for _ in range(30):
        yield dag_phrases

    if xiangqi_jsonl and os.path.exists(xiangqi_jsonl):
        with open(xiangqi_jsonl, "r", encoding="utf-8") as xf:
            for idx, line in enumerate(xf):
                if idx >= 400:
                    break
                obj = json.loads(line)
                yield f"{obj.get('user_prompt', '')}\n{obj.get('cot', '')}\n最佳走法：{obj.get('best_move_zh', '')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 10,240-vocab Universal Chinese + Xiangqi BPE Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=10240)
    parser.add_argument("--out", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--xiangqi-jsonl", type=str, default="/home/user/xiangqi_cot_lm/data/train.jsonl")
    parser.add_argument("--extra-corpus", nargs="*", default=[], help="Optional general Chinese .txt/.jsonl files")
    args = parser.parse_args()

    print(f"Training {args.vocab_size}-vocab Byte-Level BPE tokenizer (General Chinese + English/Code + Xiangqi)...")
    tok = GeneralXiangqiTokenizer.train_universal_bpe(
        corpus_iterator=build_training_stream(args.extra_corpus, args.xiangqi_jsonl),
        vocab_size=args.vocab_size,
    )
    tok.save(args.out)
    print(f"Saved tokenizer -> {args.out} (vocab_size={tok.vocab_size})")

    # Verify both General Natural Language AND Xiangqi DAG-CoT compression!
    test_general_zh = (
        "自然语言处理与深度学习模型的训练通常需要大规模高质量语料。通过注入类似中国象棋博弈树的"
        "多步严谨逻辑推理链（Chain-of-Thought），模型能够显著提升复杂规划与分支推演能力！"
    )
    test_code_en = "def train_step(model, batch, scaler):\n    with torch.amp.autocast('cuda', dtype=torch.float16):\n        loss = model(batch)\n    return loss"
    test_xq = (
        f"<|user|>\n中国象棋局面FEN：{START_FEN}\n<|assistant|>\n<|think|>\n"
        "【同形合流】如果红走炮二平五、黑应马8进7、红续马二进三，或者红走马二进三、黑应马8进7、红续炮二平五，"
        "则双方归于同一盘面，黑走车9平8，此后红可走车一平二或兵七进一。\n<|/think|>\n最佳走法：炮二平五<|eos|>"
    )
    for label, s in [("通用中文段落", test_general_zh), ("英文代码片段", test_code_en), ("象棋同形DAG-CoT", test_xq)]:
        ids = tok.encode(s)
        dec = tok.decode(ids, skip_special_tokens=False)
        assert dec == s, f"Decode mismatch on {label}!"
        assert tok.unk_id not in ids, f"UNK found on {label}!"
        print(f"  [{label}] chars={len(s)} -> tokens={len(ids)} ({len(ids)/len(s):.2f} tok/char) | Lossless=True | UNK=0")


if __name__ == "__main__":
    main()
