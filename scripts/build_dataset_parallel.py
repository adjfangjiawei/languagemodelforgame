"""
Multi-Core Parallel Pikafish Master-Level DAG-CoT Dataset Builder (10M+ Scale & Incremental Ready).

Key Capabilities:
1. Incremental Continual Data Expansion (`--add-positions N`):
   Appends `N` brand-new unique canonical board positions (~`2N` mirrored DAG-CoT dialogues) directly onto
   an existing dataset without redoing any previously seen board state!
2. 10M+-Scale Binary 64-Bit Canonical Hash Index (`seen_canonical_u64.bin`):
   Stores 8-byte BLAKE2b canonical board hashes on disk so resuming even on 10,000,000+ positions takes <0.2s.
3. Optional Multi-Shard JSONL Output (`--shard-size`, e.g. 100000 lines/shard for multi-million datasets).
4. Strict canonical deduplication (`min(grid, mirror_grid) + turn`) BEFORE Train/Val split (zero mirror leakage).
5. Full 5-Strategy Variation-DAG CoT (`dag_cot.py`: Same-Position Merging `如果...或者...`, Prefix Trie,
   Sibling Subtree Isomorphism, Same-Piece Leaf Grouping) budgeted cleanly to `max_seq_len=1024`.
"""

from __future__ import annotations

import argparse
import array
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.tokenizer import DEFAULT_TOKENIZER_PATH, GeneralXiangqiTokenizer
from xiangqi.board import START_FEN, XiangqiBoard
from xiangqi.dag_cot import build_budgeted_dag_dialogue, format_user_prompt
from xiangqi.engine_wrapper import MoveEval, PikafishEngine
from xiangqi.tree_search import SearchResult, mirror_search_result, traverse_feasible_tree

OPENING_BOOK: List[Tuple[str, List[str]]] = [
    ("中炮对屏风马进七兵", ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "c3c4", "c6c5"]),
    ("中炮急进中兵对屏风马", ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "e3e4", "c9e7", "e4e5"]),
    ("中炮过河车对屏风马平炮兑车", ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "c3c4", "c9e7", "h0h6", "b7d7"]),
    ("中炮七路马对屏风马双炮过河", ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "g3g4", "b7b3"]),
    ("五七炮对屏风马", ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "b2c2", "a9b9", "b0c2"]),
    ("五八炮对屏风马", ["h2e2", "h9g7", "h0g2", "b9c7", "i0h0", "i9h9", "b2b4", "a9b9", "b0a2"]),
    ("五六炮对屏风马", ["h2e2", "h9g7", "h0g2", "b9c7", "i0h0", "i9h9", "b2d2", "c9e7", "b0c2"]),
    ("五六炮对反宫马", ["h2e2", "b9c7", "h0g2", "b7d7", "b2d2", "h9g7", "i0h0", "i9h9", "b0c2"]),
    ("中炮进三兵对反宫马", ["h2e2", "b9c7", "h0g2", "b7d7", "g3g4", "c9e7", "i0h0", "h9g7"]),
    ("中炮对顺炮直车", ["h2e2", "h7e7", "h0g2", "h9g7", "i0h0", "a9b9", "b0c2", "b9b4"]),
    ("中炮对顺炮横车", ["h2e2", "h7e7", "h0g2", "i9i8", "i0h0", "i8d8", "c3c4", "h9g7"]),
    ("中炮对列炮", ["h2e2", "b7e7", "h0g2", "h9g7", "i0h0", "b9c7", "b0a2", "i9h9"]),
    ("中炮对半途列炮", ["h2e2", "h9g7", "h0g2", "c6c5", "i0h0", "b7e7", "b0c2", "b9c7"]),
    ("中炮对单提马", ["h2e2", "b9c7", "h0g2", "h9i7", "i0h0", "i9h9", "c3c4", "d9e8"]),
    ("飞相局对左中炮", ["c0e2", "b7e7", "h0g2", "b9c7", "i0h0", "a9b9", "c3c4", "h9g7"]),
    ("飞相局对过宫炮", ["c0e2", "h7d7", "g3g4", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7"]),
    ("飞相局对士角炮", ["c0e2", "h7f7", "c3c4", "h9g7", "h0g2", "i9h9", "b0c2", "b9c7"]),
    ("仙人指路对卒底炮", ["c3c4", "h7e7", "h0g2", "h9g7", "i0h0", "b9c7", "b2e2", "a9b9"]),
    ("仙人指路对飞象", ["c3c4", "c9e7", "h0g2", "h9g7", "b0c2", "g6g5", "a0b0", "b9c7"]),
    ("仙人指路对金钩炮", ["c3c4", "b7g7", "h0g2", "b9c7", "b0c2", "h9g7", "e3e4", "i9h9"]),
    ("过宫炮对起马", ["h2d2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "b0c2", "g6g5"]),
    ("过宫炮对左中炮", ["h2d2", "b7e7", "h0g2", "b9c7", "i0h0", "h9g7", "c0e2", "i9h9"]),
    ("起马局对挺卒", ["h0g2", "g6g5", "g3g4", "h9g7", "c0e2", "b9c7", "i0h0", "i9h9"]),
    ("士角炮对右中炮", ["h2f2", "b7e7", "h0g2", "b9c7", "i0h0", "a9b9", "c0e2", "h9g7"]),
]

_W_ENGINE: Optional[PikafishEngine] = None
_W_LRU: Optional["OrderedDict[str, List[MoveEval]]"] = None
_W_TOK: Optional[GeneralXiangqiTokenizer] = None


def _init_worker(tok_path: str, hash_mb: int) -> None:
    global _W_ENGINE, _W_LRU, _W_TOK
    _W_ENGINE = PikafishEngine(hash_mb=hash_mb, threads=1)
    _W_LRU = OrderedDict()
    _W_TOK = GeneralXiangqiTokenizer.load(tok_path)


def canonical_key(board: XiangqiBoard) -> str:
    return f"{min(board.grid_fen(), board.mirrored().grid_fen())} {board.turn}"


def canonical_u64(board: XiangqiBoard) -> int:
    digest = hashlib.blake2b(canonical_key(board).encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rollout_game_worker(seed_and_games: Tuple[int, int]) -> List[Tuple[str, str]]:
    assert _W_ENGINE is not None
    seed, n_games = seed_and_games
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []

    for g in range(n_games):
        b = XiangqiBoard(START_FEN)
        if g % 4 != 0:
            name, moves = rng.choice(OPENING_BOOK)
            k = rng.randint(1, len(moves))
            for uci in moves[:k]:
                if uci in b.generate_legal_moves():
                    b.push_uci(uci)
                    if rng.random() < 0.35 and b.is_nnue_valid():
                        out.append((b.to_fen(), f"opening:{name}"))
        else:
            name = "freestyle"
            for _ in range(rng.randint(1, 3)):
                evs = _W_ENGINE.evaluate_multipv(b.to_fen(), depth=4, multipv=8)
                if evs:
                    b.push_uci(rng.choice(evs).uci)

        max_plies = rng.randint(56, 96)
        for ply in range(max_plies):
            legal = b.generate_legal_moves()
            if len(legal) < 2:
                break
            evs = _W_ENGINE.evaluate_multipv(b.to_fen(), depth=4, multipv=5)
            if not evs:
                break
            if abs(evs[0].score_cp) >= 9500:
                if b.is_nnue_valid():
                    out.append((b.to_fen(), "mate_tactic"))
                break

            r_val = rng.random()
            if r_val < 0.56:
                viable = [e for e in evs[:2] if evs[0].score_cp - e.score_cp <= 40] or [evs[0]]
                mv = rng.choice(viable).uci
            elif r_val < 0.84:
                viable = [e for e in evs[:4] if evs[0].score_cp - e.score_cp <= 88] or [evs[0]]
                mv = rng.choice(viable).uci
            else:
                mv = evs[min(len(evs) - 1, rng.randint(1, 4))].uci

            b.push_uci(mv)
            if b.is_nnue_valid() and (ply % 2 == 0 or r_val >= 0.84):
                phase = "middlegame" if ply < 38 else "endgame"
                if r_val >= 0.84:
                    phase += "_tactical"
                out.append((b.to_fen(), phase))

    return out


def _serialize_result(category: str, res: SearchResult, tok: GeneralXiangqiTokenizer, max_seq_len: int) -> Dict:
    full_dlg, cot_text, n_used, dag_stats = build_budgeted_dag_dialogue(
        res=res,
        token_counter=tok.count_tokens,
        max_seq_tokens=max_seq_len - 4,
        max_routes=500,
    )
    return {
        "fen": res.fen,
        "category": category,
        "turn": res.root_turn,
        "target_plies": res.target_plies,
        "root_best_score_cp": res.root_best_score_cp,
        "is_unique_forced": res.is_unique_forced,
        "feasible_root_moves_zh": [c.zh for c in res.root_candidates],
        "feasible_root_moves_uci": [c.uci for c in res.root_candidates],
        "total_feasible_routes": len(res.routes),
        "included_dag_routes": n_used,
        "transposition_merges": dag_stats.transposition_merges,
        "isomorphic_sibling_merges": dag_stats.isomorphic_sibling_merges,
        "same_piece_leaf_merges": dag_stats.same_piece_leaf_merges,
        "user_prompt": format_user_prompt(res.fen),
        "cot": cot_text,
        "best_move_zh": res.best_move_zh,
        "best_move_uci": res.best_move_uci,
        "token_count": tok.count_tokens(full_dlg),
    }


def _search_and_format_worker(task: Tuple[str, str, int, bool, int, int]) -> Optional[List[Dict]]:
    assert _W_ENGINE is not None and _W_TOK is not None
    fen, category, target_plies, do_mirror, max_seq_len, max_node_evals = task
    res = traverse_feasible_tree(
        fen=fen,
        engine=_W_ENGINE,
        target_plies=target_plies,
        max_branch=5,
        score_threshold_cp=38,
        oscillation_threshold_cp=65,
        root_depth=8,
        inner_depth=6,
        max_routes=500,
        max_node_evals=max_node_evals,
        lru_cache=_W_LRU,
    )
    if res is None or not res.routes:
        return None

    records = [_serialize_result(category, res, _W_TOK, max_seq_len)]
    if do_mirror:
        m_res = mirror_search_result(res)
        if m_res.fen.split()[0] != res.fen.split()[0]:
            records.append(_serialize_result(f"{category}:mirror", m_res, _W_TOK, max_seq_len))
    return records


def _load_or_init_dedup_index(
    out_dir: str, train_files: List[str], val_path: str
) -> Tuple[Set[int], int, int]:
    """Load 64-bit canonical deduplication index (`seen_canonical_u64.bin` + `.stats.json`) or bootstrap from JSONLs."""
    bin_path = os.path.join(out_dir, "seen_canonical_u64.bin")
    meta_path = os.path.join(out_dir, "dataset_progress.json")
    seen_u64: Set[int] = set()
    existing_train = 0
    existing_val = 0

    if os.path.exists(bin_path) and os.path.exists(meta_path):
        arr = array.array("Q")
        with open(bin_path, "rb") as bf:
            n_items = os.path.getsize(bin_path) // 8
            arr.fromfile(bf, n_items)
        seen_u64 = set(arr)
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            existing_train = int(meta.get("train_dialogues", 0))
            existing_val = int(meta.get("val_dialogues", 0))
        return seen_u64, existing_train, existing_val

    # Bootstrap once from any existing train/val JSONL files
    new_hashes: List[int] = []
    for p in train_files:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    h = canonical_u64(XiangqiBoard(obj["fen"]))
                    if h not in seen_u64:
                        seen_u64.add(h)
                        new_hashes.append(h)
                    existing_train += 1

    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                h = canonical_u64(XiangqiBoard(obj["fen"]))
                if h not in seen_u64:
                    seen_u64.add(h)
                    new_hashes.append(h)
                existing_val += 1

    if new_hashes:
        arr = array.array("Q", new_hashes)
        with open(bin_path, "wb") as bf:
            arr.tofile(bf)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(
                {"unique_canonical_boards": len(seen_u64), "train_dialogues": existing_train, "val_dialogues": existing_val},
                mf,
                indent=2,
            )
    return seen_u64, existing_train, existing_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Scalable & Incremental Pikafish DAG-CoT Dataset Builder")
    parser.add_argument("--num-positions", type=int, default=20000, help="Target total unique canonical board positions")
    parser.add_argument(
        "--add-positions",
        type=int,
        default=0,
        help="Explicitly generate N ADDITIONAL new unique canonical positions and append to existing data (implies --resume)!",
    )
    parser.add_argument("--val-count", type=int, default=500, help="Target held-out validation positions")
    parser.add_argument("--time-limit-minutes", type=float, default=0.0, help="0 = NO time limit (strictly count-driven)")
    parser.add_argument("--shard-size", type=int, default=0, help="If >0 (e.g. 100000), split into train_shard_XXXX.jsonl")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-node-evals", type=int, default=105)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--resume", action="store_true", help="Keep existing data and append only unseen canonical FENs")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from xiangqi.engine_wrapper import resolve_default_engine_path
    eng_check = resolve_default_engine_path()
    if not os.path.exists(eng_check):
        raise FileNotFoundError(f"Pikafish engine binary not found at: {eng_check}")

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    bin_path = os.path.join(args.out_dir, "seen_canonical_u64.bin")
    meta_path = os.path.join(args.out_dir, "dataset_progress.json")

    do_resume = args.resume or (args.add_positions > 0)

    t0 = time.time()
    deadline = (t0 + args.time_limit_minutes * 60.0) if args.time_limit_minutes > 0 else float("inf")

    seen_u64: Set[int] = set()
    existing_train = 0
    existing_val = 0

    existing_shards = sorted(glob.glob(os.path.join(args.out_dir, "train_shard_*.jsonl")))
    all_train_files = ([train_path] if os.path.exists(train_path) else []) + existing_shards

    if do_resume:
        seen_u64, existing_train, existing_val = _load_or_init_dedup_index(args.out_dir, all_train_files, val_path)
        print(
            f"[Incremental Resume] Loaded {len(seen_u64):,} seen canonical board hashes "
            f"({existing_train:,} train dialogues, {existing_val:,} val dialogues)."
        )
    else:
        for p in [bin_path, meta_path]:
            if os.path.exists(p):
                os.remove(p)

    if args.add_positions > 0:
        needed_unique = args.add_positions
    else:
        needed_unique = max(0, args.num_positions - len(seen_u64))

    if needed_unique == 0:
        print(
            f"Target unique position count ({args.num_positions:,}) already reached! "
            f"Tip: pass `--add-positions 5000` to append 5,000 more positions."
        )
        return

    # Mix seed with len(seen_u64) so incremental runs always explore fresh game trajectories!
    eff_seed = args.seed + len(seen_u64) * 131 + int(time.time() * 10) % 9973
    rng = random.Random(eff_seed)

    print(f"[1/2] Collecting {needed_unique:,} NEW unique Xiangqi positions across {args.workers} parallel Pikafish workers...")
    collected: List[Tuple[str, int, str]] = []  # (fen, u64_hash, category)

    start_b = XiangqiBoard(START_FEN)
    start_h = canonical_u64(start_b)
    if start_h not in seen_u64:
        seen_u64.add(start_h)
        collected.append((start_b.to_fen(), start_h, "opening:start"))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.tokenizer_path, args.hash_mb)) as pool:
        batch_idx = 0
        while len(collected) < needed_unique and time.time() < deadline - 30:
            batch_idx += 1
            rollout_tasks = [(eff_seed + batch_idx * 1009 + w, 6) for w in range(args.workers * 2)]
            for cand_list in pool.imap_unordered(_rollout_game_worker, rollout_tasks):
                for fen, cat in cand_list:
                    if len(collected) >= needed_unique:
                        break
                    h = canonical_u64(XiangqiBoard(fen))
                    if h not in seen_u64:
                        seen_u64.add(h)
                        collected.append((fen, h, cat))
                if len(collected) >= needed_unique:
                    break
            print(f"      Rollout progress: {len(collected):,}/{needed_unique:,} new unique positions ({time.time() - t0:.1f}s)")

        rng.shuffle(collected)
        val_needed = max(0, min(args.val_count - existing_val, max(1, len(collected) // 10) if existing_val == 0 else 0))
        if existing_val < args.val_count:
            val_needed = min(args.val_count - existing_val, max(0, len(collected) // 8))
        val_slice = collected[:val_needed]
        train_slice = collected[val_needed:]

        print(
            f"[2/2] Running 4~5-ply DAG-CoT search on {len(val_slice):,} Val + {len(train_slice):,} Train roots "
            f"(with mirror augmentation on Train)..."
        )

        val_mode = "a" if do_resume else "w"
        train_mode = "a" if do_resume else "w"

        val_tasks = [
            (fen, cat, (5 if i % 5 != 0 else 4), False, args.max_seq_len, args.max_node_evals)
            for i, (fen, _, cat) in enumerate(val_slice)
        ]
        train_tasks = [
            (fen, cat, (5 if i % 5 != 0 else 4), True, args.max_seq_len, args.max_node_evals)
            for i, (fen, _, cat) in enumerate(train_slice)
        ]

        committed_hashes: List[int] = []
        written_val = existing_val
        if val_tasks:
            with open(val_path, val_mode, encoding="utf-8") as vf:
                for (fen, h, _), recs in zip(val_slice, pool.imap(_search_and_format_worker, val_tasks, chunksize=4)):
                    if recs:
                        for r in recs:
                            vf.write(json.dumps(r, ensure_ascii=False) + "\n")
                            written_val += 1
                        committed_hashes.append(h)
                        vf.flush()

        written_train = existing_train
        completed_roots = 0
        t_search = time.time()
        trans_merges_total = 0

        # Open primary train file or active shard
        shard_idx = len(existing_shards)
        lines_in_cur_shard = 0
        if args.shard_size > 0:
            active_train_path = os.path.join(args.out_dir, f"train_shard_{shard_idx:04d}.jsonl")
            tf = open(active_train_path, "a" if do_resume else "w", encoding="utf-8")
        else:
            active_train_path = train_path
            tf = open(active_train_path, train_mode, encoding="utf-8")

        try:
            # Pair hash with task via enumerate over `imap` so exact committed hashes are saved even on early stop
            for idx_done, recs in enumerate(pool.imap(_search_and_format_worker, train_tasks, chunksize=4)):
                completed_roots += 1
                _, root_hash, _ = train_slice[idx_done]
                if recs:
                    committed_hashes.append(root_hash)
                    for r in recs:
                        tf.write(json.dumps(r, ensure_ascii=False) + "\n")
                        written_train += 1
                        lines_in_cur_shard += 1
                        trans_merges_total += r.get("transposition_merges", 0)

                        if args.shard_size > 0 and lines_in_cur_shard >= args.shard_size:
                            tf.flush()
                            tf.close()
                            shard_idx += 1
                            lines_in_cur_shard = 0
                            active_train_path = os.path.join(args.out_dir, f"train_shard_{shard_idx:04d}.jsonl")
                            tf = open(active_train_path, "w", encoding="utf-8")

                    if completed_roots % 25 == 0:
                        tf.flush()

                if completed_roots % 100 == 0 or completed_roots == len(train_tasks):
                    rate = completed_roots / max(0.1, time.time() - t_search)
                    print(
                        f"      Search progress: {completed_roots:,}/{len(train_tasks):,} roots "
                        f"({rate:.1f} roots/s) -> total train={written_train:,} | "
                        f"transposition merges={trans_merges_total:,}"
                    )

                if time.time() >= deadline:
                    print(f"      Reached optional time limit ({args.time_limit_minutes:.1f} min); flushing cleanly.")
                    break
        finally:
            tf.flush()
            tf.close()

    # Persist newly committed canonical 64-bit hashes in append mode (`ab`)!
    if committed_hashes:
        arr_out = array.array("Q", committed_hashes)
        with open(bin_path, "ab") as bf:
            arr_out.tofile(bf)
    total_unique = (os.path.getsize(bin_path) // 8) if os.path.exists(bin_path) else len(committed_hashes)
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "unique_canonical_boards": total_unique,
                "train_dialogues": written_train,
                "val_dialogues": written_val,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            mf,
            indent=2,
        )

    print("============================================================")
    print(f"DAG-CoT Dataset Build Complete in {time.time() - t0:.1f}s!")
    print(f"  Total unique canonical boards : {total_unique:,}")
    print(f"  Total Train dialogues         : {written_train:,} -> {active_train_path}")
    print(f"  Total Val dialogues           : {written_val:,} -> {val_path}")
    print(f"  Same-pos merges (this batch)  : {trans_merges_total:,}")
    print("============================================================")


if __name__ == "__main__":
    main()
