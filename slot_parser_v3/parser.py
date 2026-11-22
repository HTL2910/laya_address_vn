"""ONNX INT8 + pure NumPy Viterbi Vietnamese address slot parser (v3.1)."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

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

_EDGE_SEP_TOKENS: FrozenSet[str] = frozenset({"-", "–", "—", ",", ";", ":", "|", ".", "/"})
_STANDALONE_PHONE_RE = re.compile(r"^(?:\+?84|0)[35789](?:[.\-]?\d){8,9}$")
_HOUSE_NUM_TOKEN_RE = re.compile(r"^\d{1,5}[a-zA-Z]?(?:/\d{1,4}[a-zA-Z]?)*$")
_CHAT_FOOD_NEG_RE = re.compile(
    r"\b(?:xin\s+gia|bao\s+gia|gia\s+bao\s+nhieu|banh\s+mi|tra\s+sua|ca\s+phe|"
    r"bun\s+bo|bun\s+dau|com\s+tam|chai\s+nuong|nuong\s+bbq|ao\s+thun|quan\s+jean|"
    r"pass\s+wifi|freeship|con\s+hang|het\s+hang|cam\s+on|xin\s+chao|chot\s+don|"
    r"pizza|burger|sushi|size\s+[smlx]{1,3})\b"
)
_ADDR_CUE_RE = re.compile(
    r"(?:^|[\s,])(?:"
    r"\d{1,5}[a-z]?(?:/\d{1,4}[a-z]?)+"
    r"|so\s+nha\b|so\s+\d+"
    r"|duong|pho|dai\s+lo|ngo|ngach|hem|kiet|quoc\s+lo|tinh\s+lo|ql\s*\d+|tl\s*\d+"
    r"|thon|ap|xom|ban|buon|lang|doi|cum|khu\s+pho|kp\b|tdp\b|to\s+\d+|khu\s+cong\s+nghiep|kcn\b|kdc\b|kdt\b"
    r"|chung\s+cu|cc\b|toa\b|block\b|tang\b|lau\b|phong\b|lo\s+[a-z0-9]+"
    r"|phuong\b|xa\b|thi\s+tran\b|dac\s+khu\b|quan\b|huyen\b|thi\s+xa\b|tinh\b"
    r"|p\.?\s*\d{1,2}\b|q\.?\s*\d{1,2}\b"
    r")",
    re.IGNORECASE,
)

_PROV_PREFIX_SINGLE: FrozenSet[str] = frozenset({"tinh", "tp", "t"})
_DIST_PREFIX_SINGLE: FrozenSet[str] = frozenset({"quan", "huyen", "q", "h", "tx", "tp"})
_WARD_PREFIX_SINGLE: FrozenSet[str] = frozenset({"phuong", "xa", "p", "f", "x", "tt", "dk"})
_TRAILING_POSTAL_RE = re.compile(r"^\d{5,6}$")


def _strip_accents_ascii(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").replace("đ", "d").replace("Đ", "D").lower()
    if not s.isascii():
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return s


def _find_converter_json(artifacts_dir: Path) -> Optional[Path]:
    pkg_dir = Path(__file__).resolve().parent
    candidates = (
        artifacts_dir / "vn_admin_v3_converter.json",
        pkg_dir.parent / "shared_data_v3" / "vn_admin_v3_converter.json",
        pkg_dir.parent / "post_goong" / "artifacts" / "vn_admin_v3_converter.json",
        pkg_dir.parent / "way_c_post_goong_v3" / "artifacts" / "vn_admin_v3_converter.json",
    )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


class _RightToLeftAdminScanner:
    """Pure-stdlib right-to-left VN admin scanner over legacy (63) & 2025 (34) gazetteer."""

    def __init__(self, converter_path: Path) -> None:
        data = json.loads(converter_path.read_text(encoding="utf-8"))
        self.po_to_idx: Dict[str, int] = {name: i for i, name in enumerate(data["po"])}
        self.pn_to_idx: Dict[str, int] = {name: i for i, name in enumerate(data["pn"])}
        self.do_to_idx: Dict[str, int] = {name: i for i, name in enumerate(data["do"])}
        self.wo_to_idx: Dict[str, int] = {name: i for i, name in enumerate(data["wo"])}
        self.wn_to_idx: Dict[str, int] = {name: i for i, name in enumerate(data["wn"])}
        self.pm: List[int] = data["pm"]
        self.dp: List[Set[int]] = [set(x) for x in data["dp"]]

        # Province lookup maps alias/stem -> canonical province stem and key sets.
        self.prov_stem: Dict[str, str] = {}
        self.prov_po_keys: Dict[str, Set[int]] = {}
        self.prov_pn_keys: Dict[str, Set[int]] = {}

        for pid_o, stem in enumerate(data["po"]):
            pid_n = self.pm[pid_o]
            for variant in {stem, stem.replace(" - ", " ")}:
                self.prov_stem[variant] = stem
                self.prov_po_keys.setdefault(variant, set()).add(pid_o)
                self.prov_pn_keys.setdefault(variant, set()).add(pid_n)

        for pid_n, stem in enumerate(data["pn"]):
            for variant in {stem, stem.replace(" - ", " ")}:
                self.prov_stem.setdefault(variant, stem)
                self.prov_pn_keys.setdefault(variant, set()).add(pid_n)

        pa_raw = dict(data.get("pa", {}))
        pa_raw.setdefault("tp ho chi minh", "ho chi minh")
        pa_raw.setdefault("tp ha noi", "ha noi")
        pa_raw.setdefault("br vt", "ba ria - vung tau")
        pa_raw.setdefault("dac lac", "dak lak")
        # City-districts that are districts first when District precedes them, keep in pa only if not colliding with do_to_idx
        self.prov_ambig_with_dist: Set[str] = set()
        for alias, target in pa_raw.items():
            norm_alias = alias.replace("-", " ").strip()
            norm_alias = re.sub(r"\s+", " ", norm_alias)
            if norm_alias in self.do_to_idx and norm_alias not in self.po_to_idx and norm_alias not in self.pn_to_idx:
                self.prov_ambig_with_dist.add(norm_alias)
            if target in self.po_to_idx:
                pid_o = self.po_to_idx[target]
                self.prov_stem.setdefault(norm_alias, target)
                self.prov_po_keys.setdefault(norm_alias, set()).add(pid_o)
                self.prov_pn_keys.setdefault(norm_alias, set()).add(self.pm[pid_o])
            elif target in self.pn_to_idx:
                pid_n = self.pn_to_idx[target]
                self.prov_stem.setdefault(norm_alias, target)
                self.prov_pn_keys.setdefault(norm_alias, set()).add(pid_n)

        # District lookup
        self.dist_to_dids: Dict[str, Set[int]] = {}
        for did_o, stem in enumerate(data["do"]):
            for variant in {stem, stem.replace(" - ", " ")}:
                self.dist_to_dids.setdefault(variant, set()).add(did_o)
        for alias, target in dict(data.get("da", {})).items():
            did_o = self.do_to_idx.get(target)
            if did_o is not None:
                norm_alias = re.sub(r"\s+", " ", alias.replace("-", " ").strip())
                self.dist_to_dids.setdefault(norm_alias, set()).add(did_o)

        # Ward lookup: normalized stem -> (allowed old dids, allowed po, allowed pn)
        self.ward_dids: Dict[str, Set[int]] = {}
        self.ward_pos: Dict[str, Set[int]] = {}
        self.ward_pns: Dict[str, Set[int]] = {}

        def _reg_ward(stem: str, did_o: Optional[int], pid_o: Optional[int], pid_n: Optional[int]) -> None:
            for variant in {stem, stem.replace(" - ", " ")}:
                if did_o is not None:
                    self.ward_dids.setdefault(variant, set()).add(did_o)
                if pid_o is not None:
                    self.ward_pos.setdefault(variant, set()).add(pid_o)
                if pid_n is not None:
                    self.ward_pns.setdefault(variant, set()).add(pid_n)

        for wo_idx, entries in enumerate(data["eo"]):
            stem = data["wo"][wo_idx]
            for did_o, pid_o, _wn_idx, pid_n, _is_div, _is_def in entries:
                _reg_ward(stem, did_o, pid_o, pid_n)
        for wn_idx, entries in enumerate(data["en"]):
            stem = data["wn"][wn_idx]
            for pid_n, _lat, _lon, _area, _code, _wt in entries:
                _reg_ward(stem, None, None, pid_n)

        self.max_ngram = 6

    @staticmethod
    def _tokens_to_atoms(tokens: Sequence[str]) -> Tuple[List[str], List[int]]:
        """Flatten tokens into lowercase ASCII word atoms mapped to original token indices."""
        atoms: List[str] = []
        orig_idx: List[int] = []
        n = len(tokens)
        i = 0
        while i < n:
            s = _strip_accents_ascii(tokens[i]).strip()
            if not s or s in _EDGE_SEP_TOKENS or s in ("(", ")", "\"", "'"):
                i += 1
                continue
            # Handle 'tp . hcm' or 'tp.' + 'hcm' cleanly at atom level
            if "." in s and re.fullmatch(r"[a-z0-9]+(?:\.[a-z0-9]+)+", s):
                parts = [p for p in s.split(".") if p]
                for p in parts:
                    if p == "tphcm":
                        for a in ("tp", "ho", "chi", "minh"):
                            atoms.append(a)
                            orig_idx.append(i)
                    elif p == "tphn":
                        for a in ("tp", "ha", "noi"):
                            atoms.append(a)
                            orig_idx.append(i)
                    else:
                        atoms.append(p)
                        orig_idx.append(i)
                i += 1
                continue
            if s == "tphcm":
                for a in ("tp", "ho", "chi", "minh"):
                    atoms.append(a)
                    orig_idx.append(i)
            elif s == "tphn":
                for a in ("tp", "ha", "noi"):
                    atoms.append(a)
                    orig_idx.append(i)
            elif s == "brvt":
                for a in ("ba", "ria", "vung", "tau"):
                    atoms.append(a)
                    orig_idx.append(i)
            else:
                m_w = re.fullmatch(r"[pf]0*(\d{1,2})", s)
                m_d = re.fullmatch(r"q0*(\d{1,2})", s)
                if m_w and 1 <= int(m_w.group(1)) <= 28:
                    atoms.append("phuong")
                    orig_idx.append(i)
                    atoms.append(str(int(m_w.group(1))))
                    orig_idx.append(i)
                elif m_d and 1 <= int(m_d.group(1)) <= 12:
                    atoms.append("quan")
                    orig_idx.append(i)
                    atoms.append(str(int(m_d.group(1))))
                    orig_idx.append(i)
                else:
                    clean = re.sub(r"[^a-z0-9/]", "", s)
                    if clean:
                        atoms.append(clean)
                        orig_idx.append(i)
            i += 1
        return atoms, orig_idx

    @staticmethod
    def _consume_prefix(atoms: Sequence[str], start_pos: int, level: str) -> int:
        if start_pos <= 0:
            return start_pos
        prev1 = atoms[start_pos - 1]
        if start_pos >= 2:
            prev2 = f"{atoms[start_pos - 2]} {prev1}"
            if level == "prov" and prev2 == "thanh pho":
                return start_pos - 2
            if level == "dist" and prev2 in ("thi xa", "thanh pho"):
                return start_pos - 2
            if level == "ward" and prev2 in ("thi tran", "dac khu"):
                return start_pos - 2
        if level == "prov" and prev1 in _PROV_PREFIX_SINGLE:
            return start_pos - 1
        if level == "dist" and prev1 in _DIST_PREFIX_SINGLE:
            return start_pos - 1
        if level == "ward" and prev1 in _WARD_PREFIX_SINGLE:
            return start_pos - 1
        return start_pos

    def scan_tail(self, tokens: Sequence[str]) -> Dict[str, Any]:
        """Match trailing [WAR] [DIS] [PRO] right-to-left using the legacy+2025 gazetteer."""
        atoms, atom_to_tok = self._tokens_to_atoms(tokens)
        end_pos = len(atoms)
        while end_pos > 1:
            last = atoms[end_pos - 1]
            if last in ("vn", "vietnam") or _TRAILING_POSTAL_RE.fullmatch(last):
                end_pos -= 1
            elif end_pos >= 2 and f"{atoms[end_pos - 2]} {last}" == "viet nam":
                end_pos -= 2
            else:
                break

        tok_end = len(tokens) if end_pos == len(atoms) else atom_to_tok[end_pos]
        pro_span: Optional[Tuple[int, int]] = None
        dis_span: Optional[Tuple[int, int]] = None
        war_span: Optional[Tuple[int, int]] = None

        matched_pos: Set[int] = set()
        matched_pns: Set[int] = set()
        matched_dids: Set[int] = set()

        # 1. Province at right end
        for k in range(min(self.max_ngram, end_pos), 0, -1):
            start_pos = end_pos - k
            cand = " ".join(atoms[start_pos:end_pos])
            if cand not in self.prov_stem:
                continue
            span_start = self._consume_prefix(atoms, start_pos, "prov")
            has_pfx = span_start < start_pos
            # Require explicit prefix if it overlaps with a district/city name unless no ward/dist conflict exists
            if cand in self.prov_ambig_with_dist and not has_pfx and span_start > 0:
                continue
            if len(cand.split()) == 1 and not has_pfx and cand not in ("hue",):
                continue
            matched_pos = set(self.prov_po_keys.get(cand, ()))
            matched_pns = set(self.prov_pn_keys.get(cand, ()))
            t0 = atom_to_tok[span_start]
            t1 = atom_to_tok[end_pos - 1] + 1
            pro_span = (t0, t1)
            end_pos = span_start
            tok_end = t0
            break

        # 2. District before province (or at right end)
        for k in range(min(self.max_ngram, end_pos), 0, -1):
            start_pos = end_pos - k
            cand = " ".join(atoms[start_pos:end_pos])
            dids = self.dist_to_dids.get(cand)
            if not dids:
                continue
            span_start = self._consume_prefix(atoms, start_pos, "dist")
            has_pfx = span_start < start_pos
            compat_dids = set(dids)
            if matched_pos or matched_pns:
                compat_dids = {
                    d
                    for d in dids
                    if (self.dp[d] & matched_pos) or any(self.pm[p] in matched_pns for p in self.dp[d])
                }
                if not compat_dids:
                    continue
            elif len(cand.split()) == 1 and not has_pfx:
                continue
            matched_dids = compat_dids
            t0 = atom_to_tok[span_start]
            t1 = atom_to_tok[end_pos - 1] + 1
            dis_span = (t0, t1)
            end_pos = span_start
            tok_end = t0
            break

        # 3. Ward before district/province
        for k in range(min(self.max_ngram, end_pos), 0, -1):
            start_pos = end_pos - k
            cand = " ".join(atoms[start_pos:end_pos])
            if cand not in self.ward_dids and cand not in self.ward_pns:
                continue
            span_start = self._consume_prefix(atoms, start_pos, "ward")
            has_pfx = span_start < start_pos
            w_dids = self.ward_dids.get(cand, set())
            w_pos = self.ward_pos.get(cand, set())
            w_pns = self.ward_pns.get(cand, set())
            compat = True
            if matched_dids:
                compat = bool(w_dids & matched_dids)
            elif matched_pos or matched_pns:
                compat = bool((w_pos & matched_pos) or (w_pns & matched_pns))
            if not compat:
                continue
            # Guard bare single-token ward names unless explicit prefix or parent confirmed
            if len(cand.split()) == 1 and not has_pfx and not (matched_dids or matched_pos or matched_pns):
                continue
            # Also avoid mistaking bare 1-word 'phu'/'nam'/'son' inside street names even with parent unless prefix
            if len(cand.split()) == 1 and not has_pfx and span_start >= 1 and atoms[span_start - 1] in ("duong", "pho", "ngo", "hem"):
                continue
            t0 = atom_to_tok[span_start]
            t1 = atom_to_tok[end_pos - 1] + 1
            war_span = (t0, t1)
            end_pos = span_start
            tok_end = t0
            break

        levels_found = int(pro_span is not None) + int(dis_span is not None) + int(war_span is not None)
        return {
            "levels_found": levels_found,
            "war_span": war_span,
            "dis_span": dis_span,
            "pro_span": pro_span,
            "left_end_tok": tok_end,
        }


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
    used_fallback: bool = False
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


def _trim_edge_seps(tokens: Sequence[str]) -> List[str]:
    """Strip leading and trailing separator tokens from a token slice."""
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in _EDGE_SEP_TOKENS:
        start += 1
    while end > start and tokens[end - 1] in _EDGE_SEP_TOKENS:
        end -= 1
    return list(tokens[start:end])


def _split_left_hno_str(left_tokens: Sequence[str]) -> Tuple[str, str]:
    """Split leading house number tokens from remaining street/hamlet tokens."""
    cleaned = _trim_edge_seps(left_tokens)
    if not cleaned:
        return "", ""
    first_ascii = _strip_accents_ascii(cleaned[0])
    if first_ascii == "so" and len(cleaned) >= 2:
        second_ascii = _strip_accents_ascii(cleaned[1])
        if second_ascii == "nha" and len(cleaned) >= 3 and _HOUSE_NUM_TOKEN_RE.fullmatch(cleaned[2]):
            return detokenize_span(cleaned[:3], "HNO"), detokenize_span(cleaned[3:], "STR")
        if _HOUSE_NUM_TOKEN_RE.fullmatch(cleaned[1]):
            return detokenize_span(cleaned[:2], "HNO"), detokenize_span(cleaned[2:], "STR")
    if _HOUSE_NUM_TOKEN_RE.fullmatch(cleaned[0]):
        return cleaned[0], detokenize_span(cleaned[1:], "STR")
    return "", detokenize_span(cleaned, "STR")


def _apply_empty_clean_fallback(
    tokens: Sequence[str],
    tags: Sequence[str],
    slots: Dict[str, str],
    scanner: Optional[_RightToLeftAdminScanner],
) -> bool:
    """Populate clean_address_for_goong via subtractive + R2L 2025 fallback when CRF emits empty address."""
    rem_tokens: List[str] = []
    for tok, tag in zip(tokens, tags):
        if tag.endswith(("-PHN", "-PER", "-NOTE")):
            continue
        if _STANDALONE_PHONE_RE.fullmatch(tok.strip()):
            continue
        rem_tokens.append(tok)

    rem_tokens = _trim_edge_seps(rem_tokens)
    while rem_tokens and rem_tokens[0] in ("(", ")"):
        rem_tokens.pop(0)
    while rem_tokens and rem_tokens[-1] in ("(", ")"):
        rem_tokens.pop()
    rem_tokens = _trim_edge_seps(rem_tokens)

    rem_text = detokenize_span(rem_tokens, "STR")
    if not rem_tokens or len(rem_text) <= 6:
        return False

    norm_rem = re.sub(r"\s+", " ", _strip_accents_ascii(rem_text)).strip()
    is_chat_food = bool(_CHAT_FOOD_NEG_RE.search(norm_rem))
    has_addr_cue = bool(_ADDR_CUE_RE.search(norm_rem))

    scan = (
        scanner.scan_tail(rem_tokens)
        if scanner is not None
        else {"levels_found": 0, "war_span": None, "dis_span": None, "pro_span": None, "left_end_tok": len(rem_tokens)}
    )
    levels_found = int(scan["levels_found"])

    if levels_found >= 2 or (levels_found == 1 and not is_chat_food and (has_addr_cue or len(norm_rem) > 15)):
        hno, street = _split_left_hno_str(rem_tokens[: int(scan["left_end_tok"])])
        if hno and not slots.get("HNO"):
            slots["HNO"] = hno
        if street and not slots.get("STR"):
            slots["STR"] = street
        if scan["war_span"] is not None and not slots.get("WAR"):
            w0, w1 = scan["war_span"]
            slots["WAR"] = detokenize_span(_trim_edge_seps(rem_tokens[w0:w1]), "WAR")
        if scan["dis_span"] is not None and not slots.get("DIS"):
            d0, d1 = scan["dis_span"]
            slots["DIS"] = detokenize_span(_trim_edge_seps(rem_tokens[d0:d1]), "DIS")
        if scan["pro_span"] is not None and not slots.get("PRO"):
            p0, p1 = scan["pro_span"]
            slots["PRO"] = detokenize_span(_trim_edge_seps(rem_tokens[p0:p1]), "PRO")
        rebuilt = assemble_clean_address(slots)
        if rebuilt:
            slots["clean_address_for_goong"] = rebuilt
            return True

    if levels_found == 0 and not is_chat_food and has_addr_cue and len(rem_tokens) >= 3:
        slots["clean_address_for_goong"] = rem_text
        if not slots.get("HNO") and not slots.get("STR"):
            hno, street = _split_left_hno_str(rem_tokens)
            slots["HNO"] = hno
            slots["STR"] = street
        return True

    return False


def _enforce_phone_tags(tokens: Sequence[str], tags: List[str]) -> List[str]:
    """Deterministically move standalone VN phone tokens out of address tags into B-PHN."""
    fixed = list(tags)
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if _STANDALONE_PHONE_RE.fullmatch(tok.strip()) and not fixed[i].endswith("-PHN"):
            fixed[i] = "B-PHN"
            if i + 1 < n and fixed[i + 1].startswith("I-"):
                fixed[i + 1] = "B-" + fixed[i + 1][2:]
    return fixed


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

        conv_path = _find_converter_json(self.artifacts_dir)
        self._r2l_scanner: Optional[_RightToLeftAdminScanner] = (
            _RightToLeftAdminScanner(conv_path) if conv_path is not None else None
        )

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
        tags = _enforce_phone_tags(tokens, tags)

        slots = extract_slots_and_clean_address(tokens, tags)
        used_fallback = False
        if slots["clean_address_for_goong"] == "":
            used_fallback = _apply_empty_clean_fallback(tokens, tags, slots, self._r2l_scanner)
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
            used_fallback=used_fallback,
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
