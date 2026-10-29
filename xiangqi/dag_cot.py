"""
Transposition-Aware Variation DAG (有向无环博弈图) Natural-Language CoT Narrator.

Implements 5 lossless, natural-Chinese-preserving compression strategies over Pikafish 4~5-ply variation sets:
1. [S1: 前缀字典树归并 (Prefix Trie)]
   `A B C D E` + `A B C D F` -> `A B C D，此后可走 E 或 F` (zero prefix repetition).
2. [S2: 同局面殊途同归归并 (Exact Board-State Transposition / DAG Convergence)]
   When distinct move sequences `A B C` and `H I J` reach the EXACT SAME `after_state_key` (`grid + turn`),
   their prefixes are merged upfront via `"如果...，或者...，则双方归于同一盘面，..."` and their identical
   downstream continuation graph is narrated ONLY ONCE! Works both across root moves and recursively inside subtrees.
3. [S3: 兄弟同构后缀归并 (Bottom-Up Merkle Continuation Isomorphism)]
   When sibling moves `M1` and `M2` (even with different FENs, e.g. `士4进5` vs `士6进5`) yield the exact same
   downstream continuation text, they are merged into `"若黑应士4进5或士6进5，..."`.
4. [S4: 同子多落点自然归并 (Same-Piece Multi-Target Leaf Grouping)]
   `[炮8平6, 炮8平7, 炮8平9, 车1进1]` -> `"此后黑可走炮8平6、平7、平9或车1进1"`, matching authentic Chinese chess books.
5. [S5: 唯一应着线性顺承 (Forced-Reply Chain Flow)]
   Unbranched forcing sequences flow as natural clauses without redundant structural markers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .board import XiangqiBoard
from .tree_search import SearchResult, StepMove


@dataclass
class DAGStats:
    raw_routes: int = 0
    raw_move_tokens: int = 0
    transposition_merges: int = 0
    isomorphic_sibling_merges: int = 0
    same_piece_leaf_merges: int = 0


def _side_zh(turn: str, short: bool = True) -> str:
    if short:
        return "红" if turn == "w" else "黑"
    return "红方" if turn == "w" else "黑方"


def _describe_score(score_cp: int, root_turn: str) -> str:
    side = _side_zh(root_turn, short=False)
    opp = _side_zh("b" if root_turn == "w" else "w", short=False)
    if score_cp >= 9000:
        plies = 10000 - score_cp
        return f"{side}已呈绝杀之势（约{max(1, (plies + 1) // 2)}步杀）"
    if score_cp <= -9000:
        return f"{side}面临杀棋威胁，亟需顽强解杀"
    sign = f"+{score_cp}" if score_cp >= 0 else f"{score_cp}"
    if score_cp >= 280:
        return f"{side}胜势明显（评分{sign}）"
    if score_cp >= 90:
        return f"{side}占据主动优势（评分{sign}）"
    if score_cp >= 30:
        return f"{side}局面略优（评分{sign}）"
    if score_cp > -30:
        return f"双方大致均势（评分{sign}）"
    if score_cp > -90:
        return f"{opp}稍占先机（评分{sign}）"
    if score_cp > -280:
        return f"{side}局势偏紧（评分{sign}）"
    return f"{side}处于劣势，需谨守防线（评分{sign}）"


def _format_score_tag(score_cp: int) -> str:
    if score_cp >= 9000:
        return "胜势/杀棋"
    if score_cp <= -9000:
        return "败势"
    return f"+{score_cp}分" if score_cp >= 0 else f"{score_cp}分"


def _join_with_or(items: List[str], sep: str = "、", or_word: str = "或") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}{or_word}{items[1]}"
    return f"{sep.join(items[:-1])}{or_word}{items[-1]}"


def _split_piece_and_action(move_zh: str) -> Tuple[str, str]:
    """
    Split a standard Chinese move like '炮8平6' -> ('炮8', '平6'),
    '前车进二' -> ('前车', '进二'), '前兵三进一' -> ('前兵三', '进一').
    """
    for idx, ch in enumerate(move_zh):
        if ch in ("进", "退", "平") and idx >= 2:
            return move_zh[:idx], move_zh[idx:]
    return move_zh, ""


def format_compact_leaf_moves(moves: List[str], stats: Optional[DAGStats] = None) -> str:
    """
    [Strategy 4: 同子多落点自然归并]
    Group moves sharing the same piece identifier while preserving 100% readable Chinese:
      ['炮8平6', '炮8平7', '炮8平9', '车1进1'] -> '炮8平6、平7、平9或车1进1'
    """
    if len(moves) <= 1:
        return moves[0] if moves else ""

    # Group by piece prefix preserving first-appearance order
    groups: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
    for mv in moves:
        p_head, act_tail = _split_piece_and_action(mv)
        if not act_tail:
            groups.setdefault(mv, []).append((mv, ""))
        else:
            groups.setdefault(p_head, []).append((mv, act_tail))

    flat_phrases: List[str] = []
    for p_head, items in groups.items():
        if len(items) == 1 or not items[0][1]:
            flat_phrases.append(items[0][0])
        else:
            if stats is not None:
                stats.same_piece_leaf_merges += len(items) - 1
            # First item keeps full move (e.g. '炮8平6'), subsequent items of same piece use '平7', '平9'
            first_full = items[0][0]
            rest_tails = [act for _, act in items[1:]]
            # If this piece group is the ONLY group in `moves`, join directly with '、' + '或'!
            if len(groups) == 1:
                all_parts = [first_full] + rest_tails
                return _join_with_or(all_parts, sep="、", or_word="或")
            else:
                flat_phrases.append("、".join([first_full] + rest_tails))

    return _join_with_or(flat_phrases, sep="、", or_word="或")


def _narrate_step(step: StepMove, idx_in_chain: int) -> str:
    s = _side_zh(step.mover, short=True)
    verbs = ["走", "应", "续", "走", "接"]
    v = verbs[min(idx_in_chain, len(verbs) - 1)]
    return f"{s}{v}{step.zh}"


def _format_prefix_alternatives(prefixes: List[List[StepMove]]) -> str:
    """
    Format 2+ converging move sequences `[P_1, P_2, ...]` into natural Chinese `"如果...，或者..."`,
    factoring any shared leading sub-prefix inside the alternatives themselves so nothing repeats!
    """
    # Group by first step just in case 3 paths converge where 2 share the 1st move
    by_first: "OrderedDict[str, List[List[StepMove]]]" = OrderedDict()
    for p in prefixes:
        by_first.setdefault(p[0].zh, []).append(p)

    clauses: List[str] = []
    for first_zh, sub_list in by_first.items():
        first_step = sub_list[0][0]
        head_str = f"{_side_zh(first_step.mover, short=True)}走{first_zh}"
        if len(sub_list) == 1:
            rest = [_narrate_step(st, i + 1) for i, st in enumerate(sub_list[0][1:])]
            clauses.append("、".join([head_str] + rest) if rest else head_str)
        else:
            inner = []
            for p in sub_list:
                rest = [_narrate_step(st, i + 1) for i, st in enumerate(p[1:])]
                inner.append("、".join(rest))
            clauses.append(f"{head_str}后（{'，或'.join(inner)}）")

    if len(clauses) == 2:
        return f"如果{clauses[0]}，或者{clauses[1]}"
    return "如果" + "；或者".join(clauses[:-1]) + f"；抑或{clauses[-1]}"


def _extract_transposition_groups(
    routes: List[List[StepMove]],
) -> Tuple[List[Tuple[List[List[StepMove]], List[List[StepMove]]]], List[List[StepMove]]]:
    """
    [Strategy 2: 同局面殊途同归检测]
    Scan relative depths k in (2, 3, 4) for sets of >=2 distinct prefixes (with >=2 distinct first moves
    so shared-first-move prefixes are handled recursively inside their child subtree without repeating m_0!)
    that land on the EXACT SAME `after_state_key` AND have non-empty suffix moves.

    Returns:
      - `merged_groups`: list of `(converging_prefixes, unified_suffix_routes)`
      - `leftover_routes`: routes that did not belong to any extracted transposition group at this level.
    """
    if len(routes) < 2:
        return [], routes

    used_route_indices: Set[int] = set()
    merged_groups: List[Tuple[List[List[StepMove]], List[List[StepMove]]]] = []

    # Check shortest convergence depth first (k = 2, then 3, then 4) so earliest board equality merges maximal suffixes!
    for k in (2, 3, 4):
        state_to_route_ids: "OrderedDict[str, List[int]]" = OrderedDict()
        for r_idx, r in enumerate(routes):
            if r_idx in used_route_indices:
                continue
            if len(r) > k:  # Must have at least 1 suffix move to share!
                skey = r[k - 1].after_state_key
                state_to_route_ids.setdefault(skey, []).append(r_idx)

        for skey, r_ids in state_to_route_ids.items():
            if len(r_ids) < 2:
                continue
            # Collect distinct k-step move sequences reaching `skey`
            prefix_map: "OrderedDict[Tuple[str, ...], List[StepMove]]" = OrderedDict()
            first_moves: Set[str] = set()
            for r_idx in r_ids:
                pf = routes[r_idx][:k]
                pf_key = tuple(x.zh for x in pf)
                if pf_key not in prefix_map:
                    prefix_map[pf_key] = pf
                first_moves.add(pf[0].zh)

            # Only merge at THIS level if >= 2 distinct first moves converge to `skey`!
            # (If all share `pf[0]`, recursion into `pf[0]` will factor `pf[0]` first and merge at depth k-1!)
            if len(prefix_map) >= 2 and len(first_moves) >= 2:
                cand_set = set(r_ids)
                other_routes = [
                    routes[idx]
                    for idx in range(len(routes))
                    if idx not in used_route_indices and idx not in cand_set
                ]

                # Exact Prefix-Break vs Suffix-Gain check:
                # Never break a deep Prefix-Trie trunk (`s` shared steps with `other_routes`) just to merge a tiny 1-leaf suffix!
                break_cost_chars = 14  # connector overhead ("如果...或者...归于同形")
                total_lcp = 0
                for pf in prefix_map.values():
                    max_lcp = 0
                    for ort in other_routes:
                        lcp = 0
                        while lcp < len(pf) and lcp < len(ort) and pf[lcp].zh == ort[lcp].zh:
                            lcp += 1
                        if lcp > max_lcp:
                            max_lcp = lcp
                    total_lcp += max_lcp
                    break_cost_chars += 6 * max_lcp

                seen_suf: Set[Tuple[str, ...]] = set()
                unified_suffixes: List[List[StepMove]] = []
                for r_idx in r_ids:
                    suf = routes[r_idx][k:]
                    suf_key = tuple(x.zh for x in suf)
                    if suf and suf_key not in seen_suf:
                        seen_suf.add(suf_key)
                        unified_suffixes.append(suf)

                if unified_suffixes:
                    est_suf_chars = len(_render_continuation_routes(unified_suffixes, stats=None))
                    saved_chars = (len(prefix_map) - 1) * est_suf_chars
                    if saved_chars >= break_cost_chars or (total_lcp == 0 and k <= 3):
                        for r_idx in r_ids:
                            used_route_indices.add(r_idx)
                        merged_groups.append((list(prefix_map.values()), unified_suffixes))

    leftover = [r for idx, r in enumerate(routes) if idx not in used_route_indices]
    return merged_groups, leftover


def _render_continuation_routes(
    routes: List[List[StepMove]],
    stats: Optional[DAGStats] = None,
    enable_transposition: bool = True,
    enable_sibling_iso: bool = True,
    enable_same_piece_leaf: bool = True,
) -> str:
    """
    Recursively render a non-empty list of relative continuation routes `[s_1, s_2, ...]`
    combining:
      - Intra-subtree same-position transposition merging (S2)
      - Unary prefix chain compression (S1 & S5)
      - Leaf sibling + same-piece grouping (S1 & S4)
      - Bottom-up sibling continuation isomorphism merging (S3)
    """
    routes = [r for r in routes if r]
    if not routes:
        return ""

    # 1. Check if ALL routes are single-step leaves (`len(r) == 1`)!
    if all(len(r) == 1 for r in routes):
        seen_m: List[str] = []
        for r in routes:
            if r[0].zh not in seen_m:
                seen_m.append(r[0].zh)
        mover_s = _side_zh(routes[0][0].mover, short=True)
        if len(seen_m) == 1:
            return f"{mover_s}接{seen_m[0]}"
        compact_str = (
            format_compact_leaf_moves(seen_m, stats=stats)
            if enable_same_piece_leaf
            else _join_with_or(seen_m, sep="、", or_word="或")
        )
        return f"此后{mover_s}可走{compact_str}"

    # 2. Check if ALL routes share a common unary chain `[c_1, ..., c_m]`!
    common_chain: List[StepMove] = []
    while True:
        if any(len(r) == len(common_chain) for r in routes):
            break
        nxt_zh = routes[0][len(common_chain)].zh
        if all(r[len(common_chain)].zh == nxt_zh for r in routes):
            common_chain.append(routes[0][len(common_chain)])
        else:
            break

    if common_chain:
        chain_text = "，".join(_narrate_step(st, i + 1) for i, st in enumerate(common_chain))
        trimmed = [r[len(common_chain) :] for r in routes if len(r) > len(common_chain)]
        if not trimmed:
            return chain_text
        rest_text = _render_continuation_routes(
            trimmed,
            stats=stats,
            enable_transposition=enable_transposition,
            enable_sibling_iso=enable_sibling_iso,
            enable_same_piece_leaf=enable_same_piece_leaf,
        )
        if rest_text.startswith(("此后", "如果", "（1）", "①", "若")):
            sep = "：" if rest_text.startswith(("（1）", "①")) else "，"
            return f"{chain_text}{sep}{rest_text}"
        return f"{chain_text}，{rest_text}"

    # 3. Check for intra-subtree Same-Position Transposition Convergence [S2] at this branch point!
    if enable_transposition:
        trans_groups, leftover = _extract_transposition_groups(routes)
    else:
        trans_groups, leftover = [], routes
    parts: List[str] = []

    for prefixes, suffixes in trans_groups:
        if stats is not None:
            stats.transposition_merges += len(prefixes) - 1
        cond_str = _format_prefix_alternatives(prefixes)
        suf_str = _render_continuation_routes(
            suffixes,
            stats=stats,
            enable_transposition=enable_transposition,
            enable_sibling_iso=enable_sibling_iso,
            enable_same_piece_leaf=enable_same_piece_leaf,
        )
        parts.append(f"{cond_str}，则归于同形，{suf_str}")

    if not leftover:
        return "；".join(parts)

    # 4. Partition `leftover` by first move `r[0].zh`
    leaf_moves: List[str] = []
    branch_buckets: "OrderedDict[str, Tuple[StepMove, List[List[StepMove]]]]" = OrderedDict()

    for r in leftover:
        m0 = r[0]
        if len(r) == 1:
            if m0.zh not in leaf_moves:
                leaf_moves.append(m0.zh)
        else:
            if m0.zh not in branch_buckets:
                branch_buckets[m0.zh] = (m0, [])
            branch_buckets[m0.zh][1].append(r[1:])

    mover_s = _side_zh(leftover[0][0].mover, short=True)
    cur_ply = leftover[0][0].ply

    # 5. [Strategy 3: Bottom-Up Sibling Subtree Isomorphism Merging!]
    iso_items: List[Tuple[str, List[str]]] = []
    if enable_sibling_iso:
        iso_groups: "OrderedDict[str, List[str]]" = OrderedDict()
        for mv_zh, (_, sub_rs) in branch_buckets.items():
            cont_txt = _render_continuation_routes(
                sub_rs,
                stats=stats,
                enable_transposition=enable_transposition,
                enable_sibling_iso=enable_sibling_iso,
                enable_same_piece_leaf=enable_same_piece_leaf,
            )
            iso_groups.setdefault(cont_txt, []).append(mv_zh)
        iso_items = list(iso_groups.items())
    else:
        for mv_zh, (_, sub_rs) in branch_buckets.items():
            cont_txt = _render_continuation_routes(
                sub_rs,
                stats=stats,
                enable_transposition=enable_transposition,
                enable_sibling_iso=enable_sibling_iso,
                enable_same_piece_leaf=enable_same_piece_leaf,
            )
            iso_items.append((cont_txt, [mv_zh]))

    for b_idx, (cont_txt, sibling_mvs) in enumerate(iso_items):
        if stats is not None and len(sibling_mvs) > 1:
            stats.isomorphic_sibling_merges += len(sibling_mvs) - 1

        if len(sibling_mvs) > 1:
            mv_expr = (
                format_compact_leaf_moves(sibling_mvs, stats=None)
                if enable_same_piece_leaf
                else _join_with_or(sibling_mvs, sep="、", or_word="或")
            )
        else:
            mv_expr = sibling_mvs[0]
        both_word = "均" if len(sibling_mvs) > 1 else ""

        if cur_ply == 2:
            lead = f"（{b_idx + 1}）{mover_s}若应{mv_expr}" if b_idx == 0 else f"（{b_idx + 1}）{mover_s}若改应{mv_expr}"
            parts.append(f"{lead}，{both_word}{cont_txt}" if cont_txt else lead)
        elif cur_ply == 3:
            circled = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"][min(b_idx, 7)]
            lead = f"{circled}{mover_s}续{mv_expr}" if b_idx == 0 else f"{circled}{mover_s}另可{mv_expr}"
            parts.append(f"{lead}，{both_word}{cont_txt}" if cont_txt else lead)
        else:
            lead = f"若{mover_s}走{mv_expr}" if b_idx == 0 else f"若{mover_s}改走{mv_expr}"
            if cont_txt.startswith("此后"):
                c_body = f"则{both_word}" + cont_txt[2:]
            else:
                c_body = f"则{both_word}{cont_txt}" if cont_txt else ""
            parts.append(f"{lead}{c_body}")

    if leaf_moves:
        compact_leaves = (
            format_compact_leaf_moves(leaf_moves, stats=stats)
            if enable_same_piece_leaf
            else _join_with_or(leaf_moves, sep="、", or_word="或")
        )
        if not branch_buckets and not trans_groups:
            parts.append(f"此后{mover_s}可走{compact_leaves}")
        else:
            parts.append(f"此外{mover_s}亦可直接走{compact_leaves}")

    return "；".join(parts)


def render_dag_cot_from_search(
    res: SearchResult,
    selected_routes: Optional[Sequence[Sequence[StepMove]]] = None,
    max_routes: int = 500,
    stats_out: Optional[DAGStats] = None,
    enable_transposition: bool = True,
    enable_sibling_iso: bool = True,
    enable_same_piece_leaf: bool = True,
) -> str:
    """
    Synthesize natural-language Chinese CoT from `res` using full Variation-DAG merging:
      - Root & Intra-branch Same-Board Transposition Merging (`如果...，或者...，则双方归于同一盘面`)
      - Prefix Trie Sharing
      - Sibling Isomorphic Continuation Merging
      - Same-Piece Multi-Target Leaf Grouping
    """
    seen_seq: Set[Tuple[str, ...]] = set()
    routes: List[List[StepMove]] = []
    src = selected_routes if selected_routes is not None else res.routes
    for r in src:
        if not r:
            continue
        k = tuple(s.zh for s in r)
        if k not in seen_seq:
            seen_seq.add(k)
            routes.append(list(r))
            if len(routes) >= max_routes:
                break

    stats = stats_out if stats_out is not None else DAGStats()
    stats.raw_routes = len(routes)
    stats.raw_move_tokens = sum(len(r) for r in routes)

    root_side = _side_zh(res.root_turn, short=False)
    root_short = _side_zh(res.root_turn, short=True)
    eval_desc = _describe_score(res.root_best_score_cp, res.root_turn)
    cand_score_map: Dict[str, int] = {c.zh: c.score_cp for c in res.root_candidates}

    distinct_roots_in_routes: List[str] = []
    for r in routes:
        if r[0].zh not in distinct_roots_in_routes:
            distinct_roots_in_routes.append(r[0].zh)

    lines: List[str] = []

    # 1. Opening Assessment
    if res.is_unique_forced or len(distinct_roots_in_routes) <= 1:
        only_zh = distinct_roots_in_routes[0] if distinct_roots_in_routes else res.best_move_zh
        if res.is_unique_forced:
            lines.append(
                f"当前轮到{root_side}行棋，{eval_desc}。经评估筛选，当前仅有唯一不震荡的稳健走法{only_zh}，"
                f"其余着法均会导致局势明显受损，推演后续变化如下："
            )
        else:
            lines.append(f"当前轮到{root_side}行棋，{eval_desc}。围绕核心着法{only_zh}推演后续变化：")
    else:
        cand_list_str = "、".join(
            f"{m}（{_format_score_tag(cand_score_map.get(m, res.root_best_score_cp))}）"
            for m in distinct_roots_in_routes
        )
        lines.append(
            f"当前轮到{root_side}行棋，{eval_desc}。阈值内共有{len(distinct_roots_in_routes)}种可行初着：{cand_list_str}。"
            f"按同形归并与前缀树合并推演4至5步变化："
        )

    # 2. Extract Cross-Root Same-Position Transposition Groups FIRST (`A B C == H I J` -> `如果...或者...`)
    if enable_transposition:
        cross_trans_groups, leftover_routes = _extract_transposition_groups(routes)
    else:
        cross_trans_groups, leftover_routes = [], routes

    for t_idx, (prefixes, suffixes) in enumerate(cross_trans_groups, 1):
        stats.transposition_merges += len(prefixes) - 1
        cond_str = _format_prefix_alternatives(prefixes)
        suf_body = _render_continuation_routes(
            suffixes,
            stats=stats,
            enable_transposition=enable_transposition,
            enable_sibling_iso=enable_sibling_iso,
            enable_same_piece_leaf=enable_same_piece_leaf,
        )
        tag = f"【同形合流{t_idx if len(cross_trans_groups) > 1 else ''}】"
        lines.append(f"{tag}{cond_str}，则双方归于同一盘面，{suf_body}。")

    # 3. Render remaining non-transposing routes grouped by Root Candidate `r[0].zh`
    root_buckets: "OrderedDict[str, List[List[StepMove]]]" = OrderedDict()
    for r in leftover_routes:
        root_buckets.setdefault(r[0].zh, []).append(r[1:])

    # Also apply Root-level Sibling Isomorphism if two root moves have identical remaining continuations!
    cn_idx = ["一", "二", "三", "四", "五", "六"]
    has_prior_trans = len(cross_trans_groups) > 0

    for idx, (r_zh, sub_rs) in enumerate(root_buckets.items()):
        sc_tag = _format_score_tag(cand_score_map.get(r_zh, res.root_best_score_cp))
        non_empty_subs = [sr for sr in sub_rs if sr]
        body = (
            _render_continuation_routes(
                non_empty_subs,
                stats=stats,
                enable_transposition=enable_transposition,
                enable_sibling_iso=enable_sibling_iso,
                enable_same_piece_leaf=enable_same_piece_leaf,
            )
            if non_empty_subs
            else ""
        )

        if len(distinct_roots_in_routes) <= 1 and not has_prior_trans:
            if body:
                lines.append(f"若{root_short}走{r_zh}，{body}。")
            else:
                lines.append(f"{root_short}走{r_zh}即可锁定胜局。")
        else:
            tag_num = cn_idx[min(idx, len(cn_idx) - 1)]
            extra_note = "其余支线" if has_prior_trans else f"{sc_tag}"
            verb = "先走" if (idx == 0 and not has_prior_trans) else ("走" if has_prior_trans else "改走")
            if body:
                lines.append(f"【路线{tag_num}（{extra_note}）】若{root_short}{verb}{r_zh}：{body}。")
            else:
                lines.append(f"【路线{tag_num}（{extra_note}）】{root_short}{verb}{r_zh}，直接成势。")

    # 4. Closing Conclusion
    n_total = len(routes)
    if len(distinct_roots_in_routes) <= 1:
        lines.append(
            f"综上推演（共涵盖{n_total}条可行路线），{res.best_move_zh}可确保局势稳固不震荡，故下一步应走{res.best_move_zh}。"
        )
    else:
        others = "、".join(m for m in distinct_roots_in_routes if m != res.best_move_zh)[:30]
        best_sc = _format_score_tag(res.root_best_score_cp)
        lines.append(
            f"综合以上{n_total}条可行路线分析，走{res.best_move_zh}后续应对最为稳健（维持{best_sc}），"
            f"整体优于{others}，故最佳走法为{res.best_move_zh}。"
        )

    return "\n".join(lines)


def build_dag_cot(
    res: SearchResult,
    max_routes: int = 500,
    enable_transposition: bool = True,
    enable_sibling_iso: bool = True,
    enable_same_piece_leaf: bool = True,
) -> Tuple[str, DAGStats]:
    """Convenience wrapper returning `(cot_text, DAGStats)` with configurable ablation flags."""
    ordered = prioritize_routes_across_roots(res.routes, max_routes=max_routes)
    st = DAGStats()
    cot = render_dag_cot_from_search(
        res=res,
        selected_routes=ordered,
        max_routes=max_routes,
        stats_out=st,
        enable_transposition=enable_transposition,
        enable_sibling_iso=enable_sibling_iso,
        enable_same_piece_leaf=enable_same_piece_leaf,
    )
    return cot, st


def prioritize_routes_across_roots(routes: Sequence[Sequence[StepMove]], max_routes: int = 500) -> List[List[StepMove]]:
    buckets: "OrderedDict[str, List[List[StepMove]]]" = OrderedDict()
    for r in routes[:max_routes]:
        if r:
            buckets.setdefault(r[0].zh, []).append(list(r))

    if len(buckets) <= 1:
        return [list(r) for r in routes[:max_routes]]

    keys = list(buckets.keys())
    ptrs = {k: 0 for k in keys}
    ordered: List[List[StepMove]] = []

    for k in keys:
        if ptrs[k] < len(buckets[k]):
            ordered.append(buckets[k][ptrs[k]])
            ptrs[k] += 1

    while len(ordered) < min(len(routes), max_routes):
        progressed = False
        for idx, k in enumerate(keys):
            quota = 2 if idx == 0 else 1
            for _ in range(quota):
                if ptrs[k] < len(buckets[k]):
                    ordered.append(buckets[k][ptrs[k]])
                    ptrs[k] += 1
                    progressed = True
        if not progressed:
            break
    return ordered


def format_user_prompt(fen: str) -> str:
    board = XiangqiBoard(fen)
    side = "红方" if board.turn == "w" else "黑方"
    return f"中国象棋局面FEN：{board.to_fen()}\n当前行棋方：{side}。请推演后续可行变化并给出下一步最佳走法。"


def format_full_dialogue(fen: str, cot_text: str, best_move_zh: str) -> str:
    user_msg = format_user_prompt(fen)
    return f"<|user|>\n{user_msg}\n<|assistant|>\n<|think|>\n{cot_text}\n<|/think|>\n最佳走法：{best_move_zh}<|eos|>"


def build_budgeted_dag_dialogue(
    res: SearchResult,
    token_counter: Optional[Callable[[str], int]] = None,
    max_seq_tokens: int = 1020,
    max_routes: int = 500,
) -> Tuple[str, str, int, DAGStats]:
    """
    Binary-search maximum number of feasible routes `K <= max_routes` whose DAG-compressed
    dialogue (`<|user|>...<|eos|>`) fits cleanly within `max_seq_tokens` (inside 1024 context).
    """
    ordered = prioritize_routes_across_roots(res.routes, max_routes=max_routes)
    stats = DAGStats()
    if not ordered:
        cot = render_dag_cot_from_search(res, selected_routes=[], max_routes=max_routes, stats_out=stats)
        return format_full_dialogue(res.fen, cot, res.best_move_zh), cot, 0, stats

    full_cot = render_dag_cot_from_search(res, selected_routes=ordered, max_routes=max_routes, stats_out=stats)
    full_txt = format_full_dialogue(res.fen, full_cot, res.best_move_zh)
    if token_counter is None or token_counter(full_txt) <= max_seq_tokens:
        return full_txt, full_cot, len(ordered), stats

    lo, hi = 1, len(ordered)
    best_k = 1
    best_stats = DAGStats()
    best_cot = render_dag_cot_from_search(res, selected_routes=ordered[:1], max_routes=max_routes, stats_out=best_stats)
    best_txt = format_full_dialogue(res.fen, best_cot, res.best_move_zh)

    while lo <= hi:
        mid = (lo + hi) // 2
        cur_stats = DAGStats()
        cand_cot = render_dag_cot_from_search(res, selected_routes=ordered[:mid], max_routes=max_routes, stats_out=cur_stats)
        cand_txt = format_full_dialogue(res.fen, cand_cot, res.best_move_zh)
        if token_counter(cand_txt) <= max_seq_tokens:
            best_k = mid
            best_cot = cand_cot
            best_txt = cand_txt
            best_stats = cur_stats
            lo = mid + 1
        else:
            hi = mid - 1

    return best_txt, best_cot, best_k, best_stats
