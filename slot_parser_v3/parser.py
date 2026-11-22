"""ONNX INT8 + pure NumPy Viterbi Vietnamese address slot parser."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .tokenizer import (
    BIO_TAGS,
    ID2TAG,
    SLOT_TYPES,
    SlotParserTokenizer,
    build_bio_transition_mask,
    tokenize_words,
)

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_INVALID_SCORE = -1e4
_VALID_START_MASK, _VALID_TRANS_MASK = build_bio_transition_mask()
_SPACE_BEFORE_DOT_RE = re.compile(r"\s+\.")
_SPACE_BEFORE_COMMA_RE = re.compile(r"\s+,")
_HYPHEN_SPACE_RE = re.compile(r"\s*-\s*")


@dataclass
class ParsedAddress:
    """Structured address and chat slot extraction result."""

    house_num: str = ""
    street: str = ""
    ward: str = ""
    district: str = ""
    province: str = ""
    person: str = ""
    phone: str = ""
    delivery_note: str = ""
    clean_address_for_goong: str = ""
    tokens: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize parsed fields into a plain JSON-safe dictionary."""
        return asdict(self)


def viterbi_decode(
    emissions: np.ndarray,
    transitions: np.ndarray,
    length: int,
    start_transitions: Optional[np.ndarray] = None,
    end_transitions: Optional[np.ndarray] = None,
) -> List[int]:
    """Run exact BIO-constrained NumPy Viterbi decoding for a single sequence."""
    if length <= 0:
        return []

    emit = np.asarray(emissions, dtype=np.float32)
    trans = np.asarray(transitions, dtype=np.float32)
    eff_len = min(int(length), emit.shape[0])
    num_tags = emit.shape[1]

    # Enforce strict BIO constraints regardless of caller-provided transition values.
    constrained_trans = np.where(_VALID_TRANS_MASK, trans, _INVALID_SCORE)
    if start_transitions is not None:
        start_vec = np.where(
            _VALID_START_MASK,
            np.asarray(start_transitions, dtype=np.float32),
            _INVALID_SCORE,
        )
    else:
        start_vec = np.where(_VALID_START_MASK, 0.0, _INVALID_SCORE).astype(np.float32)

    if end_transitions is not None:
        end_vec = np.asarray(end_transitions, dtype=np.float32)
    else:
        end_vec = np.zeros((num_tags,), dtype=np.float32)

    score = start_vec + emit[0]
    if eff_len == 1:
        return [int(np.argmax(score + end_vec))]

    backpointers = np.empty((eff_len - 1, num_tags), dtype=np.int64)
    for t in range(1, eff_len):
        step_scores = score[:, None] + constrained_trans + emit[t][None, :]
        best_prev = np.argmax(step_scores, axis=0)
        backpointers[t - 1] = best_prev
        score = step_scores[best_prev, np.arange(num_tags)]

    best_last = int(np.argmax(score + end_vec))
    path = [best_last]
    for t in range(eff_len - 2, -1, -1):
        best_last = int(backpointers[t, best_last])
        path.append(best_last)
    path.reverse()
    return path


def extract_bio_spans(
    tags: Sequence[str],
) -> List[Tuple[str, int, int]]:
    """Extract contiguous (slot_type, start_idx, end_idx_exclusive) BIO spans."""
    spans: List[Tuple[str, int, int]] = []
    cur_slot: Optional[str] = None
    cur_start = 0

    for idx, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            if cur_slot is not None:
                spans.append((cur_slot, cur_start, idx))
                cur_slot = None
            continue

        prefix, slot = tag.split("-", 1)
        if prefix == "B":
            if cur_slot is not None:
                spans.append((cur_slot, cur_start, idx))
            cur_slot = slot
            cur_start = idx
        elif prefix == "I":
            if cur_slot != slot:
                if cur_slot is not None:
                    spans.append((cur_slot, cur_start, idx))
                cur_slot = slot
                cur_start = idx

    if cur_slot is not None:
        spans.append((cur_slot, cur_start, len(tags)))
    return spans


def detokenize_span(tokens: Sequence[str], slot_type: str) -> str:
    """Reconstruct readable text from tokens belonging to one entity span."""
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = _SPACE_BEFORE_DOT_RE.sub(".", text)
    text = _SPACE_BEFORE_COMMA_RE.sub(",", text)
    if slot_type == "PHN":
        text = _HYPHEN_SPACE_RE.sub("-", text)
    return text.strip()


def assemble_clean_address(slots: Dict[str, str]) -> str:
    """Build clean Goong geocoding address excluding person, phone, and delivery notes."""
    street_line = " ".join(p for p in (slots.get("HNO", ""), slots.get("STR", "")) if p).strip()
    parts = [
        part
        for part in (
            street_line,
            slots.get("WAR", ""),
            slots.get("DIS", ""),
            slots.get("PRO", ""),
        )
        if part
    ]
    return ", ".join(parts)


def extract_slots_and_clean_address(
    tokens: Sequence[str], tags: Sequence[str]
) -> Dict[str, str]:
    """Aggregate BIO spans into slot strings plus clean_address_for_goong."""
    grouped: Dict[str, List[str]] = {slot: [] for slot in SLOT_TYPES}
    for slot_type, start, end in extract_bio_spans(tags):
        if slot_type in grouped:
            span_text = detokenize_span(tokens[start:end], slot_type)
            if span_text:
                grouped[slot_type].append(span_text)

    slots: Dict[str, str] = {}
    for slot_type, parts in grouped.items():
        if not parts:
            slots[slot_type] = ""
        elif len(parts) == 1:
            slots[slot_type] = parts[0]
        elif slot_type in ("HNO", "STR", "PER"):
            slots[slot_type] = " ".join(parts)
        else:
            slots[slot_type] = ", ".join(parts)

    slots["clean_address_for_goong"] = assemble_clean_address(slots)
    return slots


class AddressSlotParser:
    """Low-latency Vietnamese e-commerce address & chat slot parser (ONNX INT8 + NumPy)."""

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = DEFAULT_ARTIFACTS_DIR,
        intra_op_threads: int = 1,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        vocab_path = self.artifacts_dir / "vocab.json"
        crf_path = self.artifacts_dir / "crf_transitions.json"
        onnx_path = self.artifacts_dir / "slot_parser_v3_int8.onnx"

        self.tokenizer = SlotParserTokenizer.from_vocab_file(vocab_path)

        crf_data = json.loads(crf_path.read_text(encoding="utf-8"))
        self.transitions = np.asarray(crf_data["transitions"], dtype=np.float32)
        self.start_transitions = np.asarray(
            crf_data.get("start_transitions", np.zeros(len(BIO_TAGS))), dtype=np.float32
        )
        self.end_transitions = np.asarray(
            crf_data.get("end_transitions", np.zeros(len(BIO_TAGS))), dtype=np.float32
        )

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(intra_op_threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Warm up CPU arena once so first live .parse() call has steady latency.
        dummy = self.tokenizer.encode("")
        self.session.run(["emissions"], dummy)

    def _build_parsed_result(
        self, tokens: List[str], emissions: np.ndarray, latency_ms: float
    ) -> ParsedAddress:
        eff_len = min(len(tokens), self.tokenizer.max_word_len)
        tag_ids = viterbi_decode(
            emissions=emissions,
            transitions=self.transitions,
            length=eff_len,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )
        tags = [ID2TAG[tid] for tid in tag_ids]
        if len(tokens) > len(tags):
            tags.extend(["O"] * (len(tokens) - len(tags)))

        slots = extract_slots_and_clean_address(tokens, tags)
        return ParsedAddress(
            house_num=slots["HNO"],
            street=slots["STR"],
            ward=slots["WAR"],
            district=slots["DIS"],
            province=slots["PRO"],
            person=slots["PER"],
            phone=slots["PHN"],
            delivery_note=slots["NOTE"],
            clean_address_for_goong=slots["clean_address_for_goong"],
            tokens=tokens,
            tags=tags,
            latency_ms=latency_ms,
        )

    def parse(self, text: str) -> ParsedAddress:
        """Parse a single Vietnamese address/chat string into structured slots."""
        t0 = time.perf_counter()
        tokens = tokenize_words(text)
        if not tokens:
            return ParsedAddress(latency_ms=(time.perf_counter() - t0) * 1000.0)

        inputs = self.tokenizer.encode(tokens)
        emissions = self.session.run(["emissions"], inputs)[0][0]
        parsed = self._build_parsed_result(tokens, emissions, 0.0)
        parsed.latency_ms = (time.perf_counter() - t0) * 1000.0
        return parsed

    def parse_batch(self, texts: Sequence[str]) -> List[ParsedAddress]:
        """Parse a batch of texts using vectorized ONNX inference + Viterbi decoding."""
        if not texts:
            return []

        t0 = time.perf_counter()
        tokens_list = [tokenize_words(txt) for txt in texts]
        non_empty_indices = [i for i, toks in enumerate(tokens_list) if toks]
        results: List[Optional[ParsedAddress]] = [None] * len(texts)

        if non_empty_indices:
            batch_tokens = [tokens_list[i] for i in non_empty_indices]
            inputs = self.tokenizer.encode_batch(batch_tokens)
            batch_emissions = self.session.run(["emissions"], inputs)[0]
            total_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = total_ms / len(texts)
            for out_idx, orig_idx in enumerate(non_empty_indices):
                results[orig_idx] = self._build_parsed_result(
                    tokens_list[orig_idx], batch_emissions[out_idx], per_item_ms
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(texts)
        return [
            res if res is not None else ParsedAddress(latency_ms=elapsed_ms)
            for res in results
        ]
