"""Pure Python + NumPy tokenizer and feature encoder for Vietnamese address slot parsing."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

BIO_TAGS: Tuple[str, ...] = (
    "O",
    "B-HNO",
    "I-HNO",
    "B-STR",
    "I-STR",
    "B-WAR",
    "I-WAR",
    "B-DIS",
    "I-DIS",
    "B-PRO",
    "I-PRO",
    "B-PER",
    "I-PER",
    "B-PHN",
    "I-PHN",
    "B-NOTE",
    "I-NOTE",
)

TAG2ID: Dict[str, int] = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
ID2TAG: Dict[int, str] = {idx: tag for idx, tag in enumerate(BIO_TAGS)}
SLOT_TYPES: Tuple[str, ...] = ("HNO", "STR", "WAR", "DIS", "PRO", "PER", "PHN", "NOTE")


def build_bio_transition_mask() -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean (valid_start [17], valid_trans [17, 17]) BIO constraints."""
    n_tags = len(BIO_TAGS)
    valid_start = np.ones((n_tags,), dtype=bool)
    valid_trans = np.ones((n_tags, n_tags), dtype=bool)

    for idx, tag in enumerate(BIO_TAGS):
        if tag.startswith("I-"):
            valid_start[idx] = False

    for prev_idx, prev_tag in enumerate(BIO_TAGS):
        for next_idx, next_tag in enumerate(BIO_TAGS):
            if next_tag.startswith("I-"):
                slot = next_tag[2:]
                if prev_tag not in (f"B-{slot}", f"I-{slot}"):
                    valid_trans[prev_idx, next_idx] = False

    return valid_start, valid_trans

MAX_WORD_LEN = 64
MAX_CHAR_PER_WORD = 16
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1
CHAR_PAD_ID = 0
CHAR_UNK_ID = 1

# Preserves phone numbers (+84..., 0912.345.678), house/alley numbers (12/4/5, 45/2B),
# abbreviations with internal dots (x.loc, p.12, TP.HCM), apostrophes (B'Lao), and keeps
# commas/hyphens/parentheses/punctuation as separate tokens.
_TOKEN_RE = re.compile(r"\+\d+(?:[./]\d+)*|\w+(?:[./']\w+)*|[^\w\s]")
_DIGIT_RE = re.compile(r"\d")
_DIGIT_RUN_RE = re.compile(r"\d+")


def strip_vietnamese_accents(text: str) -> str:
    """Strip Vietnamese tone and vowel diacritics while mapping đ/Đ to d/D."""
    s = unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def tokenize_words(text: str) -> List[str]:
    """Fast whitespace + punctuation-aware tokenizer preserving address/phone tokens."""
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text)
    return _TOKEN_RE.findall(normalized)


def word_shape_variants(token: str) -> Tuple[str, str, str]:
    """Return (lowercased, fixed-digit-0-shape, collapsed-digit-#-shape) forms."""
    low = token.lower()
    zero_shape = _DIGIT_RE.sub("0", low)
    hash_shape = _DIGIT_RUN_RE.sub("#", low)
    return low, zero_shape, hash_shape


class SlotParserTokenizer:
    """Pure Python + NumPy word and character ID encoder for BIO slot tagging."""

    def __init__(
        self,
        word2id: Dict[str, int],
        char2id: Dict[str, int],
        max_word_len: int = MAX_WORD_LEN,
        max_char_per_word: int = MAX_CHAR_PER_WORD,
    ) -> None:
        self.word2id = dict(word2id)
        self.char2id = dict(char2id)
        self.max_word_len = int(max_word_len)
        self.max_char_per_word = int(max_char_per_word)
        self.pad_id = self.word2id.get(PAD_TOKEN, PAD_ID)
        self.unk_id = self.word2id.get(UNK_TOKEN, UNK_ID)
        self.char_pad_id = self.char2id.get(PAD_TOKEN, CHAR_PAD_ID)
        self.char_unk_id = self.char2id.get(UNK_TOKEN, CHAR_UNK_ID)

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    @property
    def char_vocab_size(self) -> int:
        return len(self.char2id)

    def _lookup_word_id(self, token: str) -> int:
        low = token.lower()
        wid = self.word2id.get(low)
        if wid is not None:
            return wid
        if any("0" <= ch <= "9" for ch in low):
            zero_shape = _DIGIT_RE.sub("0", low)
            wid = self.word2id.get(zero_shape)
            if wid is not None:
                return wid
            hash_shape = _DIGIT_RUN_RE.sub("#", low)
            wid = self.word2id.get(hash_shape)
            if wid is not None:
                return wid
        return self.unk_id

    def encode_tokens(
        self, tokens: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a single token sequence into fixed-width NumPy arrays."""
        word_ids = np.full((self.max_word_len,), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((self.max_word_len,), dtype=np.int64)
        char_ids = np.full(
            (self.max_word_len, self.max_char_per_word),
            self.char_pad_id,
            dtype=np.int64,
        )

        limit = min(len(tokens), self.max_word_len)
        lookup_word = self._lookup_word_id
        char2id = self.char2id
        char_unk = self.char_unk_id
        max_chars = self.max_char_per_word

        for idx in range(limit):
            tok = tokens[idx]
            word_ids[idx] = lookup_word(tok)
            attention_mask[idx] = 1
            for c_idx, ch in enumerate(tok[:max_chars]):
                char_ids[idx, c_idx] = char2id.get(ch, char_unk)

        return word_ids, attention_mask, char_ids

    def encode_batch(
        self, items: Sequence[Union[str, Sequence[str]]]
    ) -> Dict[str, np.ndarray]:
        """Encode a batch of strings or pre-tokenized sequences into [B, 64] / [B, 64, 16]."""
        batch_size = len(items)
        word_ids = np.full((batch_size, self.max_word_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, self.max_word_len), dtype=np.int64)
        char_ids = np.full(
            (batch_size, self.max_word_len, self.max_char_per_word),
            self.char_pad_id,
            dtype=np.int64,
        )

        for b_idx, item in enumerate(items):
            tokens = tokenize_words(item) if isinstance(item, str) else item
            w_arr, m_arr, c_arr = self.encode_tokens(tokens)
            word_ids[b_idx] = w_arr
            attention_mask[b_idx] = m_arr
            char_ids[b_idx] = c_arr

        return {
            "word_ids": word_ids,
            "attention_mask": attention_mask,
            "char_ids": char_ids,
        }

    def encode(self, text_or_tokens: Union[str, Sequence[str]]) -> Dict[str, np.ndarray]:
        """Encode one string or token sequence with leading batch dimension B=1."""
        return self.encode_batch([text_or_tokens])

    def save_vocab(self, path: Union[str, Path]) -> None:
        """Persist word and character vocabularies plus dimensions to JSON."""
        payload = {
            "max_word_len": self.max_word_len,
            "max_char_per_word": self.max_char_per_word,
            "word2id": self.word2id,
            "char2id": self.char2id,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_vocab_file(cls, path: Union[str, Path]) -> "SlotParserTokenizer":
        """Load tokenizer vocabulary and shape configuration from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            word2id=payload["word2id"],
            char2id=payload["char2id"],
            max_word_len=payload.get("max_word_len", MAX_WORD_LEN),
            max_char_per_word=payload.get("max_char_per_word", MAX_CHAR_PER_WORD),
        )

    @classmethod
    def build_from_corpus(
        cls,
        token_sequences: Iterable[Sequence[str]],
        max_vocab_size: int = 10000,
        min_word_freq: int = 2,
        max_word_len: int = MAX_WORD_LEN,
        max_char_per_word: int = MAX_CHAR_PER_WORD,
    ) -> "SlotParserTokenizer":
        """Build frequency-ranked word + digit-shape and character vocabularies."""
        word_counts: Dict[str, int] = {}
        char_counts: Dict[str, int] = {}

        for seq in token_sequences:
            for tok in seq:
                low, zero_shape, hash_shape = word_shape_variants(tok)
                word_counts[low] = word_counts.get(low, 0) + 1
                if zero_shape != low:
                    word_counts[zero_shape] = word_counts.get(zero_shape, 0) + 1
                if hash_shape != low and hash_shape != zero_shape:
                    word_counts[hash_shape] = word_counts.get(hash_shape, 0) + 1
                for ch in tok:
                    char_counts[ch] = char_counts.get(ch, 0) + 1

        word2id: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        ranked_words = sorted(
            (item for item in word_counts.items() if item[1] >= min_word_freq),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for word, _ in ranked_words[: max(0, max_vocab_size - len(word2id))]:
            if word not in word2id:
                word2id[word] = len(word2id)

        char2id: Dict[str, int] = {PAD_TOKEN: CHAR_PAD_ID, UNK_TOKEN: CHAR_UNK_ID}
        for ch, _ in sorted(char_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if ch not in char2id:
                char2id[ch] = len(char2id)

        return cls(
            word2id=word2id,
            char2id=char2id,
            max_word_len=max_word_len,
            max_char_per_word=max_char_per_word,
        )
