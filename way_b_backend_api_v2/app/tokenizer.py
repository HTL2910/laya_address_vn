"""Dual-granularity Vietnamese tokenizer (Word/Subword + Character) for Laya Address VN v2.

Pure Python + NumPy implementation with zero PyTorch dependency so the FastAPI
runtime loads and runs with minimal overhead.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
NUM_TOKEN = "<NUM>"
NUM_BIG_TOKEN = "<NUM_BIG>"
PHONE_TOKEN = "<PHONE>"
YEAR_TOKEN = "<YEAR>"
ALNUM_HOUSE_TOKEN = "<HOUSE_ALNUM>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    CLS_TOKEN,
    NUM_TOKEN,
    NUM_BIG_TOKEN,
    PHONE_TOKEN,
    YEAR_TOKEN,
    ALNUM_HOUSE_TOKEN,
]

# Vietnamese diacritic stripping map (handles both đ/Đ and decomposed marks)
_VI_EXTRA_MAP = str.maketrans({"đ": "d", "Đ": "D"})

# Token splitting: preserves punctuation separators that matter for VN addresses (/, ., ,, -, @, :)
# Splits compound abbreviations like "P.Ben", "Q.1", "214/19", "TP.HCM", "x.loc", "p.12" into clean tokens.
_TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+\.\S+"
    r"|[^\W\d_]+|\d+[a-zA-Z]{1,2}\b|\d+|[.,/\-:@#()]",
    re.UNICODE,
)

_PHONE_RE = re.compile(r"^(?:0\d{8,10}|84\d{8,10})$")
_ALNUM_HOUSE_RE = re.compile(r"^\d{1,4}[a-z]{1,2}$")
_MEAS_RE = re.compile(r"^\d+(?:h|k|g|gr|kg|ml|l|cm|mm|m|gb|tb|p|ph|d|đ)$")
_SMALL_KEEP_NUMS = {str(i) for i in range(0, 32)}


def strip_accents(text: str) -> str:
    """Fold Vietnamese diacritics to plain ASCII equivalents."""
    text = (text or "").translate(_VI_EXTRA_MAP)
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """Unicode NFC normalization and whitespace cleanup."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_word_token(raw_tok: str) -> str:
    tok = raw_tok.lower()
    if tok.startswith(("http://", "https://", "www.")):
        return "<URL>"
    if "@" in tok and "." in tok:
        return "<EMAIL>"
    if tok.isdigit():
        if _PHONE_RE.match(tok):
            return PHONE_TOKEN
        if len(tok) == 4 and tok.startswith(("19", "20")):
            return YEAR_TOKEN
        if tok in _SMALL_KEEP_NUMS:
            return tok
        if len(tok) >= 5:
            return NUM_BIG_TOKEN
        return NUM_TOKEN
    if _ALNUM_HOUSE_RE.match(tok):
        # Preserve unit suffixes like 7h, 30k, 500g, 500gr, 2l, 15m, 10p as distinct semantic tokens
        if _MEAS_RE.match(tok):
            return "<MEAS_" + re.sub(r"^\d+", "", tok) + ">"
        return ALNUM_HOUSE_TOKEN
    return tok


def tokenize_words(text: str) -> List[str]:
    """Tokenize Vietnamese text into normalized word/punctuation tokens."""
    norm = normalize_text(text)
    if not norm:
        return []
    return [_normalize_word_token(m.group(0)) for m in _TOKEN_RE.finditer(norm)]


class DualGranularityTokenizer:
    """Shared word/subword + character tokenizer with deterministic vocab serialization."""

    def __init__(
        self,
        word_to_id: Dict[str, int],
        char_to_id: Dict[str, int],
        max_word_len: int = 64,
        max_char_len: int = 160,
        min_char_len: int = 8,
    ) -> None:
        self.word_to_id = word_to_id
        self.char_to_id = char_to_id
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.min_char_len = min_char_len

        self.pad_id = word_to_id[PAD_TOKEN]
        self.unk_id = word_to_id[UNK_TOKEN]
        self.cls_id = word_to_id[CLS_TOKEN]
        self.char_pad_id = char_to_id[PAD_TOKEN]
        self.char_unk_id = char_to_id[UNK_TOKEN]

    @property
    def word_vocab_size(self) -> int:
        return len(self.word_to_id)

    @property
    def char_vocab_size(self) -> int:
        return len(self.char_to_id)

    @classmethod
    def build_from_corpus(
        cls,
        texts: Iterable[str],
        max_word_vocab: int = 6500,
        min_word_freq: int = 2,
        max_char_vocab: int = 220,
        min_char_freq: int = 3,
        max_word_len: int = 64,
        max_char_len: int = 160,
        admin_texts: Optional[Iterable[str]] = None,
    ) -> "DualGranularityTokenizer":
        word_counter: Counter[str] = Counter()
        char_counter: Counter[str] = Counter()

        for text in texts:
            norm = normalize_text(text)
            for tok in tokenize_words(norm):
                word_counter[tok] += 1
                folded = strip_accents(tok)
                if folded != tok:
                    word_counter[folded] += 1
            for ch in norm.lower():
                char_counter[ch] += 1

        required_admin_tokens: Set[str] = set()
        if admin_texts is not None:
            for text in admin_texts:
                norm = normalize_text(text)
                for tok in tokenize_words(norm):
                    required_admin_tokens.add(tok)
                    folded = strip_accents(tok)
                    if folded != tok:
                        required_admin_tokens.add(folded)
                for ch in norm.lower():
                    char_counter[ch] += 1

        word_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        extra_specials = [
            "<URL>",
            "<EMAIL>",
            "<MEAS_h>",
            "<MEAS_k>",
            "<MEAS_g>",
            "<MEAS_gr>",
            "<MEAS_kg>",
            "<MEAS_ml>",
            "<MEAS_l>",
            "<MEAS_cm>",
            "<MEAS_mm>",
            "<MEAS_m>",
            "<MEAS_gb>",
            "<MEAS_tb>",
            "<MEAS_p>",
            "<MEAS_ph>",
            "<MEAS_d>",
            "<MEAS_đ>",
        ]
        for tok in extra_specials:
            if tok not in word_to_id:
                word_to_id[tok] = len(word_to_id)

        remaining_admin = sorted(required_admin_tokens - set(word_to_id))
        remaining_admin_set = set(remaining_admin)

        # Fill top frequent corpus words while guaranteeing slots for all Vietnamese admin syllables
        for tok, freq in word_counter.most_common():
            if freq < min_word_freq and tok not in remaining_admin_set:
                continue
            if tok in word_to_id:
                continue
            if tok in remaining_admin_set:
                word_to_id[tok] = len(word_to_id)
                remaining_admin_set.remove(tok)
            elif len(word_to_id) + len(remaining_admin_set) < max_word_vocab:
                word_to_id[tok] = len(word_to_id)
            if len(word_to_id) >= max_word_vocab:
                break

        for tok in sorted(remaining_admin_set):
            if len(word_to_id) >= max_word_vocab:
                break
            if tok not in word_to_id:
                word_to_id[tok] = len(word_to_id)

        char_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for ch, freq in char_counter.most_common():
            if freq < min_char_freq:
                break
            if ch not in char_to_id:
                char_to_id[ch] = len(char_to_id)
                if len(char_to_id) >= max_char_vocab:
                    break

        return cls(
            word_to_id=word_to_id,
            char_to_id=char_to_id,
            max_word_len=max_word_len,
            max_char_len=max_char_len,
        )

    def encode_single(self, text: str) -> Tuple[List[int], List[int]]:
        """Encode one text into (word_ids, char_ids) without padding."""
        norm = normalize_text(text)
        tokens = tokenize_words(norm)

        word_ids = [self.cls_id]
        for tok in tokens[: self.max_word_len - 1]:
            wid = self.word_to_id.get(tok)
            if wid is None:
                folded = strip_accents(tok)
                wid = self.word_to_id.get(folded, self.unk_id)
            word_ids.append(wid)

        lower = norm.lower()[: self.max_char_len]
        char_ids = [self.char_to_id.get(ch, self.char_unk_id) for ch in lower]
        if not char_ids:
            char_ids = [self.char_pad_id]
        return word_ids, char_ids

    def encode_batch(
        self,
        texts: Sequence[str],
        fixed_length: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a batch into (word_ids, attention_mask, char_ids) int64 NumPy arrays.

        Uses dynamic batch max length (up to max_word_len / max_char_len) by default
        for faster CPU inference while respecting min_char_len for Conv1d kernels.
        """
        batch_size = len(texts)
        if batch_size == 0:
            return (
                np.zeros((0, 1), dtype=np.int64),
                np.zeros((0, 1), dtype=np.int64),
                np.zeros((0, self.min_char_len), dtype=np.int64),
            )

        encoded = [self.encode_single(t) for t in texts]
        if fixed_length:
            w_len = self.max_word_len
            c_len = self.max_char_len
        else:
            w_len = min(self.max_word_len, max(len(w) for w, _ in encoded))
            c_len = min(
                self.max_char_len,
                max(self.min_char_len, max(len(c) for _, c in encoded)),
            )

        word_ids = np.full((batch_size, w_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, w_len), dtype=np.int64)
        char_ids = np.full((batch_size, c_len), self.char_pad_id, dtype=np.int64)

        for i, (w, c) in enumerate(encoded):
            wl = min(len(w), w_len)
            cl = min(len(c), c_len)
            word_ids[i, :wl] = w[:wl]
            attention_mask[i, :wl] = 1
            char_ids[i, :cl] = c[:cl]

        return word_ids, attention_mask, char_ids

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "max_word_len": self.max_word_len,
            "max_char_len": self.max_char_len,
            "min_char_len": self.min_char_len,
            "word_to_id": self.word_to_id,
            "char_to_id": self.char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DualGranularityTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            word_to_id=data["word_to_id"],
            char_to_id=data["char_to_id"],
            max_word_len=int(data.get("max_word_len", 64)),
            max_char_len=int(data.get("max_char_len", 160)),
            min_char_len=int(data.get("min_char_len", 8)),
        )
