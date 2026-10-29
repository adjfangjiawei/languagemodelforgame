"""
Scalable Incremental Disk-Cached (`np.memmap`) `uint16` Dataset for 10K~10M+ CoT & General Dialogues.

Key Upgrades for Continual Data Expansion & Multi-Million Scale:
1. Incremental Byte-Offset Token Cache (`<jsonl>.tokcache.{bin,idx.npy,meta.json}`):
   - First load tokenizes `train.jsonl` into a flat `uint16` binary file on disk.
   - When you append new CoT data (`build_dataset_parallel.py --add-positions N`), subsequent training runs
     `f.seek(prev_byte_size)` and **ONLY tokenize the newly appended lines**, appending directly to `.tokcache.bin`!
2. Zero-Copy `np.memmap` Paging:
   Even at **10,000,000+ dialogues** (~12 GB of tokens), system RAM stays well under 1 GB because token arrays
   are memory-mapped read-only from disk rather than duplicated in Python heap memory.
3. Glob & Multi-Shard Support (`data/train*.jsonl`) + `LengthBucketBatchSampler` (~40% less `<pad>` waste).
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import GeneralXiangqiTokenizer


def expand_jsonl_paths(paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for p in paths:
        if not p:
            continue
        for sub in p.split(","):
            sub = sub.strip()
            if not sub:
                continue
            if any(ch in sub for ch in ("*", "?", "[")):
                matches = sorted(glob.glob(sub))
                expanded.extend(m for m in matches if os.path.isfile(m) and m.endswith(".jsonl"))
            elif os.path.isdir(sub):
                matches = sorted(glob.glob(os.path.join(sub, "train*.jsonl")))
                expanded.extend(m for m in matches if os.path.isfile(m))
            elif os.path.isfile(sub):
                expanded.append(sub)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for m in expanded:
        ab = os.path.abspath(m)
        if ab not in seen:
            seen.add(ab)
            uniq.append(m)
    return uniq


class _MemmapShard:
    """Represents one incrementally cached JSONL file backed by a read-only `uint16` `np.memmap`."""

    def __init__(
        self,
        jsonl_path: str,
        is_xiangqi: bool,
        tokenizer: GeneralXiangqiTokenizer,
        max_seq_len: int,
        mask_user_prompt: bool,
        use_disk_cache: bool = True,
    ):
        self.jsonl_path = jsonl_path
        self.is_xiangqi = is_xiangqi
        self.max_seq_len = max_seq_len
        self.mask_user_prompt = mask_user_prompt

        tag = f"L{max_seq_len}_M{int(mask_user_prompt)}_V{tokenizer.vocab_size}"
        self.bin_path = f"{jsonl_path}.{tag}.tokcache.bin"
        self.idx_path = f"{jsonl_path}.{tag}.tokcache.idx.npy"
        self.meta_path = f"{jsonl_path}.{tag}.tokcache.meta.json"

        if use_disk_cache:
            self._sync_incremental_cache(tokenizer)
            meta = self._read_meta()
            self.stats = meta["stats"]
            idx_mat = np.load(self.idx_path, mmap_mode="r")  # shape (N, 3): [start, end, mask_len]
            self.starts = idx_mat[:, 0]
            self.ends = idx_mat[:, 1]
            self.mask_lens = idx_mat[:, 2]
            total_u16 = os.path.getsize(self.bin_path) // 2
            self.tokens_mmap = np.memmap(self.bin_path, dtype=np.uint16, mode="r", shape=(total_u16,)) if total_u16 > 0 else np.zeros((0,), dtype=np.uint16)
        else:
            arrs, idx_rows, st = self._tokenize_range(tokenizer, start_byte=0, prev_token_offset=0)
            self.tokens_mmap = np.concatenate(arrs) if arrs else np.zeros((0,), dtype=np.uint16)
            idx_mat = np.asarray(idx_rows, dtype=np.int64) if idx_rows else np.zeros((0, 3), dtype=np.int64)
            self.starts = idx_mat[:, 0]
            self.ends = idx_mat[:, 1]
            self.mask_lens = idx_mat[:, 2]
            self.stats = st

    def _read_meta(self) -> Dict:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _tokenize_range(
        self,
        tokenizer: GeneralXiangqiTokenizer,
        start_byte: int,
        prev_token_offset: int,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]], Dict]:
        end_think_seq = tokenizer.encode("\n<|/think|>\n")
        max_raw = self.max_seq_len + 1

        chunks: List[np.ndarray] = []
        idx_rows: List[Tuple[int, int, int]] = []
        cur_off = prev_token_offset

        p_sum = 0
        c_sum = 0
        a_sum = 0
        n_count = 0

        with open(self.jsonl_path, "rb") as fb:
            fb.seek(start_byte)
            for raw_line in fb:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.is_xiangqi and "cot" in rec and "best_move_zh" in rec:
                    prefix_str = f"<|user|>\n{rec['user_prompt']}\n<|assistant|>\n"
                    cot_str = f"<|think|>\n{rec['cot']}\n<|/think|>\n"
                    ans_str = f"最佳走法：{rec['best_move_zh']}<|eos|>"

                    p_ids = tokenizer.encode(prefix_str)
                    c_ids = tokenizer.encode(cot_str)
                    a_ids = tokenizer.encode(ans_str)

                    if len(p_ids) + len(c_ids) + len(a_ids) > max_raw:
                        avail_c = max(8, max_raw - len(p_ids) - len(a_ids))
                        c_ids = c_ids[: max(2, avail_c - len(end_think_seq))] + end_think_seq

                    full = np.asarray(p_ids + c_ids + a_ids, dtype=np.uint16)
                    if len(full) < 4:
                        continue
                    mlen = max(0, len(p_ids) - 1) if self.mask_user_prompt else 0
                    p_sum += len(p_ids)
                    c_sum += len(c_ids)
                    a_sum += len(a_ids)
                else:
                    p_ids, r_ids = MixedCausalDataset._encode_general_record(rec, tokenizer, max_raw)
                    full = np.asarray((p_ids + r_ids)[:max_raw], dtype=np.uint16)
                    if len(full) < 4:
                        continue
                    mlen = max(0, len(p_ids) - 1) if (self.mask_user_prompt and p_ids) else 0
                    p_sum += len(p_ids)
                    a_sum += len(r_ids)

                chunks.append(full)
                nxt_off = cur_off + len(full)
                idx_rows.append((cur_off, nxt_off, mlen))
                cur_off = nxt_off
                n_count += 1

        st = {
            "count": n_count,
            "prompt_tok_sum": p_sum,
            "cot_tok_sum": c_sum,
            "ans_tok_sum": a_sum,
        }
        return chunks, idx_rows, st

    def _sync_incremental_cache(self, tokenizer: GeneralXiangqiTokenizer) -> None:
        cur_file_size = os.path.getsize(self.jsonl_path)
        cache_valid = (
            os.path.exists(self.bin_path)
            and os.path.exists(self.idx_path)
            and os.path.exists(self.meta_path)
        )

        if cache_valid:
            try:
                meta = self._read_meta()
                cached_size = int(meta.get("parsed_byte_size", -1))
                if cached_size == cur_file_size:
                    return  # 100% up-to-date! Zero tokenization needed!
                if 0 < cached_size < cur_file_size:
                    # Incremental append! Only tokenize newly added tail `[cached_size : cur_file_size]`!
                    prev_u16 = os.path.getsize(self.bin_path) // 2
                    old_idx = np.load(self.idx_path)
                    new_chunks, new_rows, delta_st = self._tokenize_range(
                        tokenizer, start_byte=cached_size, prev_token_offset=prev_u16
                    )
                    if new_chunks:
                        with open(self.bin_path, "ab") as bf:
                            for arr in new_chunks:
                                bf.write(arr.tobytes())
                        new_idx = np.vstack([old_idx, np.asarray(new_rows, dtype=np.int64)])
                        np.save(self.idx_path, new_idx)
                    old_st = meta["stats"]
                    merged_st = {
                        k: int(old_st.get(k, 0)) + int(delta_st.get(k, 0))
                        for k in ("count", "prompt_tok_sum", "cot_tok_sum", "ans_tok_sum")
                    }
                    with open(self.meta_path, "w", encoding="utf-8") as mf:
                        json.dump({"parsed_byte_size": cur_file_size, "stats": merged_st}, mf)
                    print(
                        f"[TokenCache] Incrementally tokenized +{delta_st['count']:,} new samples from "
                        f"{os.path.basename(self.jsonl_path)} (total={merged_st['count']:,})!"
                    )
                    return
            except Exception:
                pass  # Rebuild cleanly if file was overwritten/truncated

        # Full initial cache build
        chunks, idx_rows, st = self._tokenize_range(tokenizer, start_byte=0, prev_token_offset=0)
        with open(self.bin_path, "wb") as bf:
            for arr in chunks:
                bf.write(arr.tobytes())
        idx_arr = np.asarray(idx_rows, dtype=np.int64) if idx_rows else np.zeros((0, 3), dtype=np.int64)
        np.save(self.idx_path, idx_arr)
        with open(self.meta_path, "w", encoding="utf-8") as mf:
            json.dump({"parsed_byte_size": cur_file_size, "stats": st}, mf)

    def __len__(self) -> int:
        return len(self.starts)


class MixedCausalDataset(Dataset):
    """
    Multi-shard disk-backed `np.memmap` dataset supporting both Xiangqi DAG-CoT JSONLs and
    General Chinese SFT/Pretraining JSONLs up to tens of millions of samples.
    """

    def __init__(
        self,
        xiangqi_jsonl_paths: Sequence[str],
        tokenizer: GeneralXiangqiTokenizer,
        max_seq_len: int = 1024,
        general_jsonl_paths: Sequence[str] = (),
        mask_user_prompt: bool = True,
        use_disk_cache: bool = True,
    ):
        assert tokenizer.vocab_size <= 65535, "uint16 storage requires vocab_size <= 65535"
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_user_prompt = mask_user_prompt

        xq_files = expand_jsonl_paths(xiangqi_jsonl_paths)
        gen_files = expand_jsonl_paths(general_jsonl_paths)

        self.shards: List[_MemmapShard] = []
        for p in xq_files:
            self.shards.append(_MemmapShard(p, True, tokenizer, max_seq_len, mask_user_prompt, use_disk_cache))
        for p in gen_files:
            self.shards.append(_MemmapShard(p, False, tokenizer, max_seq_len, mask_user_prompt, use_disk_cache))

        shard_offsets = [0]
        seq_len_parts: List[np.ndarray] = []
        xq_count = 0
        gen_count = 0
        p_sum = 0
        c_sum = 0
        a_sum = 0

        for sh in self.shards:
            n_sh = len(sh)
            shard_offsets.append(shard_offsets[-1] + n_sh)
            if n_sh > 0:
                seq_len_parts.append((sh.ends - sh.starts - 1).astype(np.int32))
            if sh.is_xiangqi:
                xq_count += int(sh.stats["count"])
                p_sum += int(sh.stats["prompt_tok_sum"])
                c_sum += int(sh.stats["cot_tok_sum"])
                a_sum += int(sh.stats["ans_tok_sum"])
            else:
                gen_count += int(sh.stats["count"])

        self.shard_offsets = np.asarray(shard_offsets, dtype=np.int64)
        self.seq_lens = np.concatenate(seq_len_parts) if seq_len_parts else np.zeros((0,), dtype=np.int32)

        denom_xq = max(1, xq_count)
        tot_xq = max(1, p_sum + c_sum + a_sum)
        self._stats = {
            "total_samples": int(len(self.seq_lens)),
            "xiangqi_samples": xq_count,
            "general_nl_samples": gen_count,
            "num_shards": len(self.shards),
            "total_tokens": int(self.seq_lens.sum()) if len(self.seq_lens) else 0,
            "avg_prompt_tokens": round(p_sum / denom_xq, 1),
            "avg_cot_tokens": round(c_sum / denom_xq, 1),
            "avg_answer_tokens": round(a_sum / denom_xq, 1),
            "avg_seq_len": round(float(self.seq_lens.mean()), 1) if len(self.seq_lens) else 0.0,
            "max_seq_len_observed": int(self.seq_lens.max()) if len(self.seq_lens) else 0,
            "cot_token_ratio": round(c_sum / tot_xq, 4),
        }

    @staticmethod
    def _encode_general_record(
        rec: Dict, tokenizer: GeneralXiangqiTokenizer, max_raw: int
    ) -> Tuple[List[int], List[int]]:
        if "messages" in rec and isinstance(rec["messages"], list):
            msgs = rec["messages"]
            p_parts: List[str] = []
            ans_str = ""
            for i, m in enumerate(msgs):
                role = m.get("role", "user")
                content = m.get("content", "")
                if i == len(msgs) - 1 and role == "assistant":
                    ans_str = f"{content}<|eos|>"
                elif role == "system":
                    p_parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    p_parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    p_parts.append(f"<|assistant|>\n{content}<|eos|>")
            prefix_s = "\n".join(p_parts) + "\n<|assistant|>\n"
            return tokenizer.encode(prefix_s), tokenizer.encode(ans_str)
        elif "prompt" in rec and "response" in rec:
            p_s = f"<|user|>\n{rec['prompt']}\n<|assistant|>\n"
            r_s = f"{rec['response']}<|eos|>"
            return tokenizer.encode(p_s), tokenizer.encode(r_s)
        else:
            txt = rec.get("text", "")
            return [], tokenizer.encode(f"<|bos|>{txt}<|eos|>")[:max_raw]

    def summary_stats(self) -> Dict:
        return dict(self._stats)

    def __len__(self) -> int:
        return len(self.seq_lens)

    def get_length(self, idx: int) -> int:
        return int(self.seq_lens[idx])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        sh_idx = int(np.searchsorted(self.shard_offsets, idx, side="right") - 1)
        local_idx = idx - int(self.shard_offsets[sh_idx])
        sh = self.shards[sh_idx]
        s = int(sh.starts[local_idx])
        e = int(sh.ends[local_idx])
        m = int(sh.mask_lens[local_idx])
        return np.asarray(sh.tokens_mmap[s:e], dtype=np.uint16), m


def collate_uint16_batch(
    batch: List[Tuple[np.ndarray, int]],
    pad_token_id: int = 0,
    pad_to_multiple_of: int = 32,
    max_seq_len: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(arr) - 1 for arr, _ in batch)
    if pad_to_multiple_of > 1:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    max_len = min(max_len, max_seq_len)

    B = len(batch)
    inp_np = np.full((B, max_len), pad_token_id, dtype=np.int64)
    lbl_np = np.full((B, max_len), -100, dtype=np.int64)

    for i, (arr, mask_until) in enumerate(batch):
        L = min(len(arr) - 1, max_len)
        inp_np[i, :L] = arr[:L]
        lbl_np[i, :L] = arr[1 : L + 1]
        if mask_until > 0:
            m = min(mask_until, L)
            lbl_np[i, :m] = -100

    return torch.from_numpy(inp_np), torch.from_numpy(lbl_np)


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Megabatch Length-Bucketed Sampler eliminating ~40% wasted `<|pad|>` compute."""

    def __init__(
        self,
        seq_lens: np.ndarray,
        batch_size: int,
        drop_last: bool = True,
        megabatch_mult: int = 32,
        seed: int = 2026,
    ):
        self.seq_lens = seq_lens
        self.batch_size = max(1, int(batch_size))
        self.drop_last = drop_last
        self.megabatch_size = max(self.batch_size * 4, self.batch_size * megabatch_mult)
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.seq_lens)
        if self.drop_last:
            return max(1, n // self.batch_size) if n > 0 else 0
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.seq_lens)
        if n == 0:
            return
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        indices = list(range(n))
        rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, n, self.megabatch_size):
            mega = indices[start : start + self.megabatch_size]
            mega.sort(key=lambda idx: int(self.seq_lens[idx]))
            for b_start in range(0, len(mega), self.batch_size):
                b = mega[b_start : b_start + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last or n < self.batch_size:
                    batches.append(b)

        rng.shuffle(batches)
        for b in batches:
            yield b
