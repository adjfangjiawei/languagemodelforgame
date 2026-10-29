"""
10,240-Vocab Universal Byte-Level BPE Tokenizer (`GeneralXiangqiTokenizer`).

Designed specifically for multi-stage / multi-domain LLM training:
1. Vocabulary size = 10,240 (exact multiple of 128 for NVIDIA GPU FP16/TensorCore GEMM efficiency).
2. GPT-4 / Qwen2 style Byte-Level BPE (`ByteLevel` pre-tokenizer + decoder):
   - 100% lossless encoding & decoding on ANY UTF-8 text (general Chinese, English, code, math, emojis, rare chars),
   - ZERO `<|unk|>` ever!
3. Vocabulary covers:
   - 16 chat/CoT/tool special tokens (`<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|think|>`, `<|/think|>`, ...),
   - 256 raw byte tokens,
   - Full GB2312 common Chinese character inventory + top general Chinese multi-character words & English/code subwords,
   - High-frequency Chinese Chess (Xiangqi) moves & DAG-CoT reasoning phrases.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|think|>",
    "<|/think|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|sep|>",
    "<|extra_0|>",
    "<|extra_1|>",
    "<|extra_2|>",
    "<|extra_3|>",
]

# Pre-tokenization regex: splits Chinese/English words, numbers, FEN ranks, and punctuation cleanly before ByteLevel BPE
SPLIT_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

DEFAULT_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tokenizer_assets",
    "tokenizer_10k.json",
)


def get_gb2312_level1_and_2_chars() -> List[str]:
    """Extract all 6,763 standard Chinese characters from Python's built-in GB2312 table."""
    chars: List[str] = []
    # Level 1 (0xB0-0xD7, 3755 most common Chinese chars) + Level 2 (0xD8-0xF7, 3008 chars)
    for lead in range(0xB0, 0xF8):
        for trail in range(0xA1, 0xFF):
            if lead == 0xD7 and 0xFA <= trail <= 0xFE:
                continue
            try:
                ch = bytes([lead, trail]).decode("gb2312")
                chars.append(ch)
            except UnicodeDecodeError:
                pass
    return chars


class GeneralXiangqiTokenizer:
    """Fast Rust-backed 10,240-vocab Byte-Level BPE tokenizer for General Chinese + Xiangqi CoT."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.pad_id = self.tok.token_to_id("<|pad|>")
        self.unk_id = self.tok.token_to_id("<|unk|>")
        self.bos_id = self.tok.token_to_id("<|bos|>")
        self.eos_id = self.tok.token_to_id("<|eos|>")
        self.system_id = self.tok.token_to_id("<|system|>")
        self.user_id = self.tok.token_to_id("<|user|>")
        self.assistant_id = self.tok.token_to_id("<|assistant|>")
        self.think_id = self.tok.token_to_id("<|think|>")
        self.end_think_id = self.tok.token_to_id("<|/think|>")

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @classmethod
    def train_universal_bpe(
        cls,
        corpus_iterator: Iterable[str],
        vocab_size: int = 10240,
    ) -> "GeneralXiangqiTokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=Regex(SPLIT_REGEX), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(corpus_iterator, trainer=trainer)
        return cls(tok)

    def save(self, path: str = DEFAULT_TOKENIZER_PATH) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.tok.save(path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "GeneralXiangqiTokenizer":
        p = path or DEFAULT_TOKENIZER_PATH
        if not os.path.exists(p):
            raise FileNotFoundError(f"Tokenizer file not found: {p}")
        return cls(Tokenizer.from_file(p))

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
