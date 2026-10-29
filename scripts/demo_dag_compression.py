"""
Quantitative Ablation & Benchmark of Readable Chinese CoT Merging Strategies on Real 5-Ply Pikafish Trees.

Compares 5 progressive levels of readable Chinese CoT compression (ZERO removal of Chinese connectors):
  - L0 (Flat):        逐条路线完整中文平铺枚举（无前缀共享、无同局面合并）
  - L1 (Trie):        前缀字典树归并 + 唯一应着顺承 (`dag_cot` 关闭同形/同构/同子合并)
  - L2 (+Transpos):   + 跨分支/子分支同局面合流 (`如果[路径1]，或者[路径2]，则双方归于同一盘面...`)
  - L3 (+SiblingIso): + 自底向上兄弟子树同构后缀归并 (`若应M1或M2，均...`)
  - L4 (Full 5-DAG):  + 同子多落点自然省主语归并 (`炮8平6、平7、平9或车1进1`)

Measures:
  - Character count & compression ratio
  - 10,240-vocab Byte-Level BPE token count & compression ratio
  - Feasible 5-ply route capacity inside `max_seq_len = 1024` tokens!
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from xiangqi.board import START_FEN
from xiangqi.dag_cot import (
    DAGStats,
    build_budgeted_dag_dialogue,
    build_dag_cot,
    format_full_dialogue,
    prioritize_routes_across_roots,
    render_dag_cot_from_search,
)
from xiangqi.engine_wrapper import PikafishEngine
from xiangqi.tree_search import SearchResult, traverse_feasible_tree

BENCHMARK_POSITIONS: List[Tuple[str, str]] = [
    ("开局初形（先手多路布局）", START_FEN),
    (
        "仙人指路对卒底炮（高频同形合流）",
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/6P2/P1P1P3P/1C5C1/9/RNBAKABNR b - - 0 1",
    ),
    (
        "中炮对屏风马进七兵（复杂中局）",
        "r1bakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 5",
    ),
    (
        "五七炮对屏风马双车对峙",
        "1rbakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/2C1C1N2/9/RNBAKAB1R b - - 0 6",
    ),
    (
        "顺炮直车对横车中局搏杀",
        "r1bakab2/8r/1cn1c4/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 6",
    ),
    (
        "车马炮多兵残局转换",
        "2bak4/4a4/4b4/p3p3p/2pn5/2P3B2/P3P3P/3AC1N2/4K4/2r3R2 w - - 0 22",
    ),
]


def format_flat_chinese_cot(res: SearchResult, max_routes: int = 500) -> str:
    """L0: Readable Chinese flat route enumeration (zero prefix or transposition sharing)."""
    side = "红方" if res.root_turn == "w" else "黑方"
    opp = "黑方" if res.root_turn == "w" else "红方"
    cands = "、".join(c.zh for c in res.root_candidates)
    ordered = prioritize_routes_across_roots(res.routes, max_routes=max_routes)
    lines = [f"当前轮到{side}行棋，初筛可行候选着法为：{cands}。逐条推演如下："]
    for idx, r in enumerate(ordered[:max_routes], 1):
        m = [sm.zh for sm in r]
        clauses = []
        for p_idx, mv in enumerate(m):
            actor = side if p_idx % 2 == 0 else opp
            if p_idx == 0:
                clauses.append(f"如果{actor}走{mv}")
            elif p_idx < len(m) - 1:
                clauses.append(f"{actor}应以{mv}")
            else:
                clauses.append(f"则此后{actor}可走{mv}")
        lines.append(f"路线{idx}：" + "，".join(clauses) + "。")
    lines.append(f"综合以上全部变例推演，最佳走法为{res.best_move_zh}。")
    return "\n".join(lines)


def count_routes_within_1024(res: SearchResult, tok: GeneralXiangqiTokenizer, mode: str) -> int:
    """Binary-search how many prioritized routes fit inside `max_seq_len=1024` tokens."""
    ordered = prioritize_routes_across_roots(res.routes, max_routes=500)
    total = len(ordered)
    if total == 0:
        return 0
    if mode == "L4":
        _, _, n_used, _ = build_budgeted_dag_dialogue(res, tok.count_tokens, max_seq_tokens=1020, max_routes=500)
        return n_used

    def make_dlg(k: int) -> int:
        if mode == "L0":
            cot = format_flat_chinese_cot(res, max_routes=k)
        elif mode == "L1":
            cot = render_dag_cot_from_search(
                res,
                selected_routes=ordered[:k],
                enable_transposition=False,
                enable_sibling_iso=False,
                enable_same_piece_leaf=False,
            )
        elif mode == "L2":
            cot = render_dag_cot_from_search(
                res,
                selected_routes=ordered[:k],
                enable_transposition=True,
                enable_sibling_iso=False,
                enable_same_piece_leaf=False,
            )
        elif mode == "L3":
            cot = render_dag_cot_from_search(
                res,
                selected_routes=ordered[:k],
                enable_transposition=True,
                enable_sibling_iso=True,
                enable_same_piece_leaf=False,
            )
        else:
            raise ValueError(mode)
        return tok.count_tokens(format_full_dialogue(res.fen, cot, res.best_move_zh))

    if make_dlg(total) <= 1020:
        return total
    lo, hi, best = 1, total, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if make_dlg(mid) <= 1020:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> None:
    tok = GeneralXiangqiTokenizer.load(DEFAULT_TOKENIZER_PATH)
    summary_rows: List[Dict] = []

    with PikafishEngine(hash_mb=32, threads=1) as eng:
        lru = OrderedDict()
        for title, fen in BENCHMARK_POSITIONS:
            res = traverse_feasible_tree(
                fen=fen,
                engine=eng,
                target_plies=5,
                max_branch=5,
                score_threshold_cp=38,
                oscillation_threshold_cp=65,
                root_depth=8,
                inner_depth=6,
                max_routes=500,
                max_node_evals=125,
                lru_cache=lru,
            )
            assert res is not None
            n_routes = len(res.routes)

            cot_l0 = format_flat_chinese_cot(res, max_routes=n_routes)
            cot_l1, _ = build_dag_cot(
                res, max_routes=n_routes, enable_transposition=False, enable_sibling_iso=False, enable_same_piece_leaf=False
            )
            cot_l2, _ = build_dag_cot(
                res, max_routes=n_routes, enable_transposition=True, enable_sibling_iso=False, enable_same_piece_leaf=False
            )
            cot_l3, _ = build_dag_cot(
                res, max_routes=n_routes, enable_transposition=True, enable_sibling_iso=True, enable_same_piece_leaf=False
            )
            cot_l4, st_l4 = build_dag_cot(
                res, max_routes=n_routes, enable_transposition=True, enable_sibling_iso=True, enable_same_piece_leaf=True
            )

            toks = {
                "L0_flat": tok.count_tokens(cot_l0),
                "L1_trie": tok.count_tokens(cot_l1),
                "L2_transposition": tok.count_tokens(cot_l2),
                "L3_sibling_iso": tok.count_tokens(cot_l3),
                "L4_full_dag": tok.count_tokens(cot_l4),
            }
            chars = {
                "L0_flat": len(cot_l0),
                "L1_trie": len(cot_l1),
                "L2_transposition": len(cot_l2),
                "L3_sibling_iso": len(cot_l3),
                "L4_full_dag": len(cot_l4),
            }
            caps = {
                "L0_flat": count_routes_within_1024(res, tok, "L0"),
                "L1_trie": count_routes_within_1024(res, tok, "L1"),
                "L2_transposition": count_routes_within_1024(res, tok, "L2"),
                "L3_sibling_iso": count_routes_within_1024(res, tok, "L3"),
                "L4_full_dag": count_routes_within_1024(res, tok, "L4"),
            }

            row = {
                "position": title,
                "routes": n_routes,
                "tt_hits_in_search": res.transposition_hits,
                "transposition_merges": st_l4.transposition_merges,
                "sibling_iso_merges": st_l4.isomorphic_sibling_merges,
                "same_piece_leaf_merges": st_l4.same_piece_leaf_merges,
                "chars": chars,
                "tokens_10k": toks,
                "token_compression_vs_flat": round(toks["L0_flat"] / max(1, toks["L4_full_dag"]), 2),
                "token_saving_vs_trie_pct": round(
                    100.0 * (toks["L1_trie"] - toks["L4_full_dag"]) / max(1, toks["L1_trie"]), 1
                ),
                "routes_in_1024_ctx": caps,
            }
            summary_rows.append(row)
            print(
                f"[{title}] routes={n_routes} | "
                f"10k-Tokens: L0(Flat)={toks['L0_flat']} -> L1(Trie)={toks['L1_trie']} -> "
                f"L2(+同形)={toks['L2_transposition']} -> L3(+同构)={toks['L3_sibling_iso']} -> "
                f"L4(全策略)={toks['L4_full_dag']} ({row['token_compression_vs_flat']}x vs Flat, "
                f"-{row['token_saving_vs_trie_pct']}% vs Trie) | "
                f"1024上下文容纳路线数: Flat={caps['L0_flat']} -> Trie={caps['L1_trie']} -> FullDAG={caps['L4_full_dag']}"
            )

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dag_compression_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved full benchmark table to {out_path}")


if __name__ == "__main__":
    main()
