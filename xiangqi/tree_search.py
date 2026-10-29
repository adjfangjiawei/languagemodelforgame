"""
Transposition-Cached 4~5-Ply Score-Threshold & Non-Oscillation Variation Search.

Key upgrades for DAG-CoT:
1. Every `StepMove` records `after_state_key` (`"{grid_fen} {turn}"`), enabling exact board-equivalence
   detection (`A B C == H I J`) across arbitrary move orders and captures.
2. Node evaluations are cached by `board.state_key()` (transposition table in Python + Pikafish TT),
   guaranteeing that identical board positions reached via different move orders share 100% consistent
   downstream continuations while cutting Pikafish search calls significantly!
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .board import XiangqiBoard, mirror_uci
from .engine_wrapper import MoveEval, PikafishEngine


@dataclass
class StepMove:
    ply: int
    mover: str               # 'w' or 'b'
    uci: str                 # e.g. 'h2e2'
    zh: str                  # e.g. '炮二平五'
    root_score_cp: int       # Root side-to-move perspective
    after_state_key: str     # Exact board state after this move ("{grid_fen} {turn}")


@dataclass
class CandidateRoot:
    uci: str
    zh: str
    score_cp: int


@dataclass
class SearchResult:
    fen: str
    root_turn: str
    root_best_score_cp: int
    best_move_uci: str
    best_move_zh: str
    target_plies: int
    score_threshold_cp: int
    oscillation_threshold_cp: int
    is_unique_forced: bool
    root_candidates: List[CandidateRoot]
    routes: List[List[StepMove]] = field(default_factory=list)
    engine_calls: int = 0
    transposition_hits: int = 0


def _filter_candidates(
    evals: List[MoveEval],
    mover: str,
    root_turn: str,
    parent_root_score: int,
    score_threshold_cp: int,
    oscillation_threshold_cp: int,
    is_root: bool = False,
) -> List[Tuple[MoveEval, int]]:
    if not evals:
        return []

    best_cp = evals[0].score_cp
    kept: List[Tuple[MoveEval, int]] = []

    for idx, ev in enumerate(evals):
        gap = best_cp - ev.score_cp
        if best_cp >= 9000:
            if ev.score_cp < 9000 or gap > 4:
                continue
        elif best_cp <= -9000:
            if gap > 2:
                continue
        else:
            if gap > score_threshold_cp:
                continue

        root_persp = ev.score_cp if mover == root_turn else -ev.score_cp

        if not is_root and idx > 0 and abs(best_cp) < 8500:
            if abs(root_persp - parent_root_score) > oscillation_threshold_cp:
                continue

        kept.append((ev, root_persp))

    if not kept:
        ev0 = evals[0]
        kept.append((ev0, ev0.score_cp if mover == root_turn else -ev0.score_cp))

    return kept


def traverse_feasible_tree(
    fen: str,
    engine: PikafishEngine,
    target_plies: int = 5,
    max_branch: int = 5,
    score_threshold_cp: int = 38,
    oscillation_threshold_cp: int = 65,
    root_depth: int = 9,
    inner_depth: int = 6,
    max_routes: int = 500,
    max_node_evals: int = 110,
    lru_cache: Optional["OrderedDict[str, List[MoveEval]]"] = None,
) -> Optional[SearchResult]:
    """
    Traverse `target_plies` (4 or 5 plies) of feasible, non-oscillating variations
    with state-key transposition caching so transposed positions get identical subtrees.
    """
    root_board = XiangqiBoard(fen)
    legal_root = root_board.generate_legal_moves()
    if not legal_root:
        return None

    root_turn = root_board.turn
    local_cache: Dict[str, List[MoveEval]] = {}
    engine_calls = 0
    transposition_hits = 0

    def query_node(board: XiangqiBoard, depth: int) -> List[MoveEval]:
        nonlocal engine_calls, transposition_hits
        skey = f"{board.state_key()}|d{depth}|m{max_branch}"
        if skey in local_cache:
            transposition_hits += 1
            return local_cache[skey]
        if lru_cache is not None and skey in lru_cache:
            transposition_hits += 1
            lru_cache.move_to_end(skey)
            val = lru_cache[skey]
            local_cache[skey] = val
            return val

        res = engine.evaluate_multipv(board.to_fen(), depth=depth, multipv=max_branch)
        local_cache[skey] = res
        if lru_cache is not None:
            lru_cache[skey] = res
            if len(lru_cache) > 16384:
                lru_cache.popitem(last=False)
        engine_calls += 1
        return res

    # Ply 1: Root evaluation
    root_evals = query_node(root_board, depth=root_depth)
    if not root_evals:
        return None

    root_best_cp = root_evals[0].score_cp
    root_kept = _filter_candidates(
        root_evals,
        mover=root_turn,
        root_turn=root_turn,
        parent_root_score=root_best_cp,
        score_threshold_cp=score_threshold_cp,
        oscillation_threshold_cp=oscillation_threshold_cp,
        is_root=True,
    )

    root_candidates: List[CandidateRoot] = []
    frontier: List[Tuple[float, int, XiangqiBoard, List[StepMove]]] = []
    completed_routes: List[Tuple[int, float, List[StepMove]]] = []

    for r_idx, (ev, rp_score) in enumerate(root_kept):
        zh = root_board.uci_to_zh(ev.uci)
        root_candidates.append(CandidateRoot(uci=ev.uci, zh=zh, score_cp=rp_score))
        nb = root_board.copy()
        nb.push_uci(ev.uci)
        step1 = StepMove(
            ply=1,
            mover=root_turn,
            uci=ev.uci,
            zh=zh,
            root_score_cp=rp_score,
            after_state_key=nb.state_key(),
        )
        penalty = float(root_best_cp - rp_score)
        if target_plies <= 1 or not nb.generate_legal_moves():
            completed_routes.append((r_idx, penalty, [step1]))
        else:
            frontier.append((penalty, r_idx, nb, [step1]))

    for ply in range(2, target_plies + 1):
        if not frontier:
            break

        by_root: Dict[int, List[Tuple[float, int, XiangqiBoard, List[StepMove]]]] = {}
        for item in frontier:
            by_root.setdefault(item[1], []).append(item)
        for lst in by_root.values():
            lst.sort(key=lambda x: x[0])

        ordered_frontier: List[Tuple[float, int, XiangqiBoard, List[StepMove]]] = []
        max_len = max(len(v) for v in by_root.values())
        for i in range(max_len):
            for r_idx in sorted(by_root.keys()):
                if i < len(by_root[r_idx]):
                    ordered_frontier.append(by_root[r_idx][i])

        next_frontier: List[Tuple[float, int, XiangqiBoard, List[StepMove]]] = []
        is_last_ply = ply == target_plies

        for penalty, r_idx, cur_board, path in ordered_frontier:
            if len(completed_routes) >= max_routes and is_last_ply:
                break

            skey_check = f"{cur_board.state_key()}|d{inner_depth}|m{max_branch}"
            already_cached = (skey_check in local_cache) or (lru_cache is not None and skey_check in lru_cache)

            if engine_calls >= max_node_evals and not already_cached:
                if len(path) >= 4:
                    completed_routes.append((r_idx, penalty, path))
                continue

            evals = query_node(cur_board, depth=inner_depth)
            if not evals:
                completed_routes.append((r_idx, penalty, path))
                continue

            parent_score = path[-1].root_score_cp
            kept = _filter_candidates(
                evals,
                mover=cur_board.turn,
                root_turn=root_turn,
                parent_root_score=parent_score,
                score_threshold_cp=score_threshold_cp,
                oscillation_threshold_cp=oscillation_threshold_cp,
                is_root=False,
            )

            best_here = evals[0].score_cp
            for ev, rp_score in kept:
                zh = cur_board.uci_to_zh(ev.uci)
                nb = cur_board.copy()
                nb.push_uci(ev.uci)
                step = StepMove(
                    ply=ply,
                    mover=cur_board.turn,
                    uci=ev.uci,
                    zh=zh,
                    root_score_cp=rp_score,
                    after_state_key=nb.state_key(),
                )
                new_path = path + [step]
                local_drop = max(0, best_here - ev.score_cp)
                osc = abs(rp_score - parent_score) * 0.20
                new_penalty = penalty + local_drop + osc

                if is_last_ply:
                    completed_routes.append((r_idx, new_penalty, new_path))
                else:
                    if not nb.generate_legal_moves():
                        completed_routes.append((r_idx, new_penalty, new_path))
                    else:
                        next_frontier.append((new_penalty, r_idx, nb, new_path))

        frontier = next_frontier

    completed_routes.sort(key=lambda item: (item[0], item[1]))
    seen_seqs = set()
    unique_routes: List[List[StepMove]] = []
    for _, _, route in completed_routes:
        seq_key = tuple(s.zh for s in route)
        if seq_key not in seen_seqs:
            seen_seqs.add(seq_key)
            unique_routes.append(route)
            if len(unique_routes) >= max_routes:
                break

    best_cand = root_candidates[0]
    return SearchResult(
        fen=root_board.to_fen(),
        root_turn=root_turn,
        root_best_score_cp=best_cand.score_cp,
        best_move_uci=best_cand.uci,
        best_move_zh=best_cand.zh,
        target_plies=target_plies,
        score_threshold_cp=score_threshold_cp,
        oscillation_threshold_cp=oscillation_threshold_cp,
        is_unique_forced=(len(root_candidates) == 1 and len(legal_root) > 1),
        root_candidates=root_candidates,
        routes=unique_routes,
        engine_calls=engine_calls,
        transposition_hits=transposition_hits,
    )


def mirror_search_result(res: SearchResult) -> SearchResult:
    """Exact horizontal board reflection preserving all `after_state_key` transposition equivalences."""
    m_root_board = XiangqiBoard(res.fen).mirrored()
    m_best_uci = mirror_uci(res.best_move_uci)
    m_best_zh = m_root_board.uci_to_zh(m_best_uci)

    m_cands = [
        CandidateRoot(
            uci=mirror_uci(c.uci),
            zh=m_root_board.uci_to_zh(mirror_uci(c.uci)),
            score_cp=c.score_cp,
        )
        for c in res.root_candidates
    ]

    m_routes: List[List[StepMove]] = []
    seen = set()
    for route in res.routes:
        b = m_root_board.copy()
        m_steps: List[StepMove] = []
        for s in route:
            mu = mirror_uci(s.uci)
            mzh = b.uci_to_zh(mu)
            b.push_uci(mu)
            m_steps.append(
                StepMove(
                    ply=s.ply,
                    mover=s.mover,
                    uci=mu,
                    zh=mzh,
                    root_score_cp=s.root_score_cp,
                    after_state_key=b.state_key(),
                )
            )
        key = tuple(x.zh for x in m_steps)
        if key not in seen:
            seen.add(key)
            m_routes.append(m_steps)

    return SearchResult(
        fen=m_root_board.to_fen(),
        root_turn=res.root_turn,
        root_best_score_cp=res.root_best_score_cp,
        best_move_uci=m_best_uci,
        best_move_zh=m_best_zh,
        target_plies=res.target_plies,
        score_threshold_cp=res.score_threshold_cp,
        oscillation_threshold_cp=res.oscillation_threshold_cp,
        is_unique_forced=res.is_unique_forced,
        root_candidates=m_cands,
        routes=m_routes,
        engine_calls=0,
        transposition_hits=res.transposition_hits,
    )
