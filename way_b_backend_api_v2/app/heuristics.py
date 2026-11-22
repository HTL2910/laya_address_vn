"""Upgraded Tier-1 Vietnamese Gazetteer & Sufficiency Heuristics (< 0.3 ms) for Way B v2.

Loads `artifacts/vn_admin_gazetteer.json` (63 provinces, 656 districts, 7,044 wards) and
performs exact span-disjoint hierarchical matching across accented, unaccented, no-comma,
and abbreviated forms (x./p./tt./q./h./tx./tp./hn/hcm/sg/q1..12/q.tb).

Routes inputs into 4 outcomes:
- 'clear_reject'       -> sufficiency = 'not_address',          is_address = False
- 'clear_insufficient' -> sufficiency = 'address_insufficient', is_address = False
- 'clear_sufficient'   -> sufficiency = 'address_sufficient',   is_address = True
- 'ambiguous'          -> route to Tier-2 ONNX INT8 model (or heuristic fallback in heuristic_only)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

from app.tokenizer import normalize_text, strip_accents

RouteDecision = Literal["clear_reject", "clear_insufficient", "clear_sufficient", "ambiguous"]
SufficiencyLabel = Literal["not_address", "address_insufficient", "address_sufficient"]

DEFAULT_GAZETTEER_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "vn_admin_gazetteer.json"


@dataclass(frozen=True)
class HeuristicResult:
    route: RouteDecision
    is_address: bool
    sufficiency: SufficiencyLabel
    probability: float
    sufficiency_probs: Dict[str, float]
    reason: str


# ---------------------------------------------------------------------------
# Fast Compiled Lexical Patterns (operating on lowercased, accent-stripped text)
# ---------------------------------------------------------------------------

_URL_ONLY_RE = re.compile(
    r"^(?:https?://\S+|www\.\S+|[a-z0-9\-]+\.(?:vn|com|net|org|io|edu\.vn|gov\.vn)(?:/\S*)?)$",
    re.IGNORECASE,
)
_EMAIL_ONLY_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", re.IGNORECASE)
_PHONE_ONLY_RE = re.compile(
    r"^(?:(?:sdt|so dt|zalo|lh|lien he|tel|phone)\s*[:.]?\s*)?(?:\+?84|0)[\d\s.\-]{8,13}$",
    re.IGNORECASE,
)
_PURE_NUM_DATE_PRICE_RE = re.compile(
    r"^(?:[\d\s.,:/\-+xX*%]+(?:\s*(?:vnd|đ|d|dong|k|tr|trieu|kg|gr|g|ml|l|cm|mm|m|gb|tb|h|gio|phut|ngay|thang|nam))?)$",
    re.IGNORECASE,
)
_TRACKING_ID_RE = re.compile(r"^(?:[a-z]{2,6}\d{6,16}|#?[a-z]{1,4}\d{5,14})$", re.IGNORECASE)

_PREFIX_DOT_SPLIT_RE = re.compile(r"\b(tp|tx|tt|kp|tdp|sn|ql|tl|dk|p|q|h|x|t)\.\s*", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"[a-z0-9']+|[.,;:\-()]", re.IGNORECASE)

_WARD_PREFIX_FULL = {
    "phuong": "phuong",
    "xa": "xa",
    "thi tran": "thi tran",
    "dac khu": "dac khu",
    "p.": "phuong",
    "x.": "xa",
    "tt.": "thi tran",
    "dk": "dac khu",
}
_DIST_PREFIX_FULL = {
    "quan": "quan",
    "huyen": "huyen",
    "thi xa": "thi xa",
    "thanh pho": "thanh pho",
    "q.": "quan",
    "h.": "huyen",
    "tx.": "thi xa",
    "tp.": "thanh pho",
}
_PROV_PREFIXES = {"tinh", "thanh pho", "tp.", "tp", "t."}

# Food / product / e-commerce lexical cues (avoid colliding with "thành phố Hà Nội", "Cầu Giấy", "Lý Thường Kiệt", "Sán Chải", "gọi trước")
_FOOD_PRODUCT_RE = re.compile(
    r"\b(?:banh mi|banh dau xanh|banh trang|banh pia|banh com|banh cuon|banh it|banh tet|"
    r"banh chung|banh da cua|banh hoi|banh cay|banh gai|banh tam|banh bao|"
    r"pho bo|pho ga|pho tai|an pho|quan pho|to pho|bat pho|"
    r"bun bo|bun cha|bun dau|bun ca|com tam|hu tieu|mi quang|mi chu|"
    r"nuoc mam|keo dua|keo cu do|nem chua|nem nuong|cha muc|cha bo|cha ca|cha lua|"
    r"ca phe|tra sua|tra moc cau|tra tan cuong|che thai nguyen|muc mot nang|muc kho|muc ca mau|muc tuoi|muc ong|muc rim|"
    r"yen sao|toi den|toi co don|hat dieu|tieu den|tieu xanh|sau rieng|xoai cat|"
    r"buoi da xanh|buoi phuc trach|vai thieu|nhan long|gao st25|ruou sim|ruou bau da|"
    r"ca kho|thit trau|bo to|muoi tom|me xung|tom chua|o mai|com chay|de nui|cam sanh|cam vinh|"
    r"nuoc hoa|son moi|sua rua mat|kem chong nang|tay trang|dau goi|"
    r"ao thun|ao phong|ao khoac|ao so mi|so mi|quan jean|giay sneaker|giay chay|giay da|giay the thao|"
    r"tui xach|balo|tai nghe|ban phim|chuot|noi chien|may xay|iphone|samsung|laptop|"
    r"sac du phong|binh giu nhiet|hop com|op lung|the nho|cap sac|den hoc|den ban|tham tap|thuc an|sua tuoi|ngu coc|"
    r"mon an|hop khau vi|dac san|chai nuong|chai nuoc|chai dau|chai ruou|nuong bbq|bbq\b|size\s*(?:s|m|l|xl|xxl|\d{2})\b|"
    r"mau\s+(?:den|trang|xanh|do|vang|be|xam)|"
    r"\d+\s*(?:kg|gr|g|ml|l|gb|tb|chai|hop|goi|bich|cai|chiec|suat)\b)\b",
    re.IGNORECASE,
)

# Accented-only food/product words whose accent-stripped forms collide with address/chat terms
_ACCENTED_FOOD_RE = re.compile(r"\b(?:phở|mực|giày|gói|áo|quần)\b", re.IGNORECASE)

# Customer chat / order / fee / question cues (without delivery intent)
_CHAT_QUESTION_RE = re.compile(
    r"\b(?:xin gia|bao gia|inbox gia|gia bao nhieu|gia sao|bao nhieu tien|het bao nhieu|"
    r"phi ship|cuoc ship|gia ship|cuoc van chuyen|cuoc gui|ship cod|freeship|giam gia|ma giam gia|"
    r"ho tro giao|co giao hang|co giao|co ship|co chi nhanh|cua hang o|kho hang|"
    r"mo cua|may gio|con hang|het hang|chot don|huy don|kiem tra don|kiem tra giup|"
    r"don hang|doi tra|bao hanh|chinh hang|loai 1|dac biet|thom ngon|an ngon|"
    r"chuyen khoan|chuyen tien|so tai khoan|stk\b|ma otp|ma van don|gui link|xem anh|"
    r"o cho cu|nhu don|nha cu|nhan sau|nhan dia chi sau|cho cu|nhu tuan truoc|"
    r"bao nhieu|may ngay|the nao|la gi|sao vay|khong a|khong shop|khong ban|ok shop|ship nhanh giup|sai roi|"
    r"hen thu \d|hen chu nhat)\b",
    re.IGNORECASE,
)

# Narrative / weather / travel / sports / historical / bio / traffic red flags
_NARRATIVE_RE = re.compile(
    r"\b(?:hom nay|ngay mai|hom qua|toi nay|chieu nay|sang nay|cuoi tuan|mua nay|mua thu|mua he|"
    r"mua nuoc noi|tro lanh|se lanh|mua rao|mua to|nang dep|nang nong|suong mu|thoi tiet|"
    r"du lich|nghi mat|ve may bay|gia ve|tau cao toc|ve tau|xe khach|san bay|cong tac|lich trinh|tham quan|"
    r"doi tuyen|bong da|thi dau|ban phao hoa|le hoi|ket xe|un u|cam xe|kinh khung|"
    r"tuyen dung|tuyen nhan vien|lam viec tai|chi nhanh|van phong dai dien|khai truong|kiem ke|nghi trua|"
    r"danh nhan|anh hung|chien si|cach mang|tieu su|lich su|sach giao khoa|thuyet trinh|cuoc doi|tung song o|"
    r"benh vien|diem chuan|truong thpt|dai hoc|hoc cap 3|sinh vien|ky thi|"
    r"que o|que ngoai|que noi|ve que|an tet|noi tieng|thac nuoc|gia dat|gia nha|tang manh|giam nhe|"
    r"cup dien|ngap duong|chuc cac ban|hen gap|gap nhau|vua goi|nhan tin hoi|"
    r"ban tap hoa|mua xe|hoc lop|sinh nam|truong phong|khach san|het phong|trong nhieu|"
    r"phong tro|o ngoai bac|phun lua)\b",
    re.IGNORECASE,
)

# Delivery-intent indicators (suggests user IS providing a shipping destination)
_DELIVERY_INTENT_RE = re.compile(
    r"\b(?:giao cho|giao hang|giao qua|giao ve|giao toi|ship giup|ship toi|ship qua|ship ve|"
    r"gui ve|gui toi|nhan tai|nhan hang tai|dia chi|nguoi nhan|cong sau|nha tro|nha so)\b",
    re.IGNORECASE,
)

_PHONE_ANY_RE = re.compile(r"(?:\+?84|0)\d{8,10}\b")

# Granular fragment / unit prefixes (avoid matching "bạn ơi" as "bản", or "Buôn Hồ" town as hamlet)
_GRANULAR_PREFIX_RE = re.compile(
    r"\b(?:so nha|so\s+\d+|sn\s+\d+|ngo\s+\d+|ngach\s+\d+|hem\s+\d+|kiet\s+\d+|"
    r"thon\s+(?:\d+|[a-z]+)|ap\s+(?:\d+|[a-z]+)|xom\s+\d+|buon\s+(?!ho\b|ma\s+thuot\b)[a-z]+|"
    r"khom\s+\d+|khu pho\s*\d+|kp\.?\s*\d+|to\s+\d+|to dan pho\s*\d+|tdp\.?\s*\d+|"
    r"tang\s+\d+|lau\s+\d+|can ho\s+[a-z0-9\-]+|can\s+[a-z]\d+|phong\s+\d+|"
    r"lo\s+[a-z0-9\-]+|block\s+[a-z0-9]+|thap\s+[a-z0-9.]+|"
    r"\d{1,4}/\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?)?)\b",
    re.IGNORECASE,
)
_ACCENTED_GRANULAR_RE = re.compile(r"\bbản\s+[^\W\d_]+\b", re.IGNORECASE)

_STREET_BUILDING_RE = re.compile(
    r"\b(?:duong|pho|dai lo|quoc lo|ql\s*\d+[a-z]?|tinh lo|tl\s*\d+|xa lo|"
    r"chung cu|toa nha|khu do thi|kdt\b|khu cong nghiep|kcn\b|kcx\b)\b",
    re.IGNORECASE,
)

_HOUSE_STREET_RE = re.compile(
    r"(?:^|[\s,:])(?:so\s+|sn\s+)?\d{1,4}(?:[a-z]|/\d{1,3}[a-z]?){0,2}\s+(?:duong\s+|pho\s+|dai lo\s+)?[a-z]{2,}",
    re.IGNORECASE,
)

_VN_SURNAMES = {
    "nguyen", "tran", "le", "pham", "hoang", "huynh", "phan", "vu", "vo",
    "dang", "bui", "do", "ho", "ngo", "duong", "ly", "dinh", "doan",
    "truong", "mai", "vuong", "trinh", "dao", "luu", "ta", "ha", "chu",
}
_VN_PERSON_TITLES = {"anh", "chi", "ong", "ba", "co", "chu", "em", "ban", "chau", "thay", "bs", "ts", "mr", "ms"}
_VN_COMMON_MIDDLES = {
    "thi", "van", "minh", "quoc", "hoang", "ngoc", "duc", "huu", "gia",
    "thanh", "xuan", "hai", "anh", "kim", "bao", "nhat", "phuong", "thao", "cong",
}

_NON_STREET_WORDS = {
    "ngay", "dem", "gio", "phut", "thang", "nam", "tuan", "kg", "gr", "ml",
    "chai", "hop", "goi", "bich", "cai", "chiec", "ly", "suat", "trieu", "ngan", "dong", "vnd",
    "ao", "quan", "vay", "dam", "so", "tui", "vi", "non", "mu", "dep", "sach", "vo", "but",
}


# ---------------------------------------------------------------------------
# Gazetteer Index (63 provinces, 656 districts, 7,044 wards)
# ---------------------------------------------------------------------------

class GazetteerIndex:
    """Span-disjoint hierarchical matcher backed by MIT Vietnam Admin Gazetteer."""

    def __init__(self, gazetteer_path: str | Path = DEFAULT_GAZETTEER_PATH) -> None:
        path = Path(gazetteer_path)
        if not path.exists():
            raise FileNotFoundError(f"Gazetteer not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))

        self.provinces: List[str] = raw["p"]
        self.districts: List[str] = raw["d"]
        self.wards: List[str] = raw["w"]
        self.ward_edges: List[Set[Tuple[int, int]]] = [
            {(int(did), int(pid)) for did, pid in pairs}
            for pairs in raw["e"]
        ]
        self.district_provinces: List[Set[int]] = [
            {int(pid) for pid in pids}
            for pids in raw["dp"]
        ]

        self.prov_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(self.provinces)}
        self.dist_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(self.districts)}
        self.ward_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(self.wards)}

        self.dist_names_set: Set[str] = set(self.districts)
        self.prov_names_set: Set[str] = set(self.provinces)

        # Separate true province/city aliases from provincial-city aliases (which are districts in `d`)
        self.true_prov_aliases: Dict[str, int] = {}
        self.city_prov_aliases: Dict[str, int] = {}
        for alias, target in raw.get("pa", {}).items():
            if target not in self.prov_to_id:
                continue
            pid = self.prov_to_id[target]
            if alias in self.dist_to_id:
                self.city_prov_aliases[alias] = pid
            else:
                self.true_prov_aliases[alias] = pid

        self.dist_aliases: Dict[str, int] = {}
        for alias, target in raw.get("da", {}).items():
            if target in self.dist_to_id:
                self.dist_aliases[alias] = self.dist_to_id[target]

        self.max_ngram = 6

    @staticmethod
    def tokenize_for_match(folded: str) -> List[str]:
        expanded = _PREFIX_DOT_SPLIT_RE.sub(lambda m: m.group(1).lower() + ". ", folded)
        raw_toks = _TOKEN_SPLIT_RE.findall(expanded)
        tokens: List[str] = []
        i = 0
        while i < len(raw_toks):
            tok = raw_toks[i]
            if (
                i + 1 < len(raw_toks)
                and raw_toks[i + 1] == "."
                and tok in {"p", "x", "tt", "q", "h", "tx", "tp", "t", "kp", "tdp"}
            ):
                tokens.append(tok + ".")
                i += 2
            else:
                tokens.append(tok)
                i += 1
        return tokens

    def match_entities(self, tokens: Sequence[str]) -> Dict[str, List[Tuple[int, int, int, bool]]]:
        """Extract candidate ward, district, and province spans: (start, end, id, has_prefix)."""
        n = len(tokens)
        ward_spans: List[Tuple[int, int, int, bool]] = []
        dist_spans: List[Tuple[int, int, int, bool]] = []
        prov_spans: List[Tuple[int, int, int, bool]] = []

        sep_tokens = {",", ";", ":", "(", ")"}

        def _clean_phrase(slice_toks: Sequence[str]) -> Optional[str]:
            if any(t in sep_tokens for t in slice_toks):
                return None
            # Collapse "ba ria - vung tau" or "phan rang - thap cham" cleanly
            return " ".join(slice_toks)

        for i in range(n):
            tok = tokens[i]
            if tok in sep_tokens or tok == "-":
                continue

            # Check prefix lengths 1 or 2 tokens before span
            prefix_candidates: List[Tuple[int, str, str]] = [(i, "", "")]
            if tok in _WARD_PREFIX_FULL:
                prefix_candidates.append((i + 1, "ward", _WARD_PREFIX_FULL[tok]))
            if tok in _DIST_PREFIX_FULL:
                prefix_candidates.append((i + 1, "dist", _DIST_PREFIX_FULL[tok]))
            if tok in _PROV_PREFIXES:
                prefix_candidates.append((i + 1, "prov", tok))
            if i + 1 < n:
                two = f"{tok} {tokens[i + 1]}"
                if two in _WARD_PREFIX_FULL:
                    prefix_candidates.append((i + 2, "ward", _WARD_PREFIX_FULL[two]))
                if two in _DIST_PREFIX_FULL:
                    prefix_candidates.append((i + 2, "dist", _DIST_PREFIX_FULL[two]))
                if two in _PROV_PREFIXES:
                    prefix_candidates.append((i + 2, "prov", two))

            # Compact glued p12 / p14 / q1..q12 without dot
            m_p = re.fullmatch(r"p(\d{1,2})", tok)
            if m_p:
                num = int(m_p.group(1))
                if 1 <= num <= 28:
                    w_name = f"phuong {num}"
                    if w_name in self.ward_to_id:
                        ward_spans.append((i, i + 1, self.ward_to_id[w_name], True))

            if tok in self.dist_aliases:
                dist_spans.append((i, i + 1, self.dist_aliases[tok], True))

            for body_start, kind, full_pref in prefix_candidates:
                if body_start >= n:
                    continue
                has_pref = kind != ""
                for length in range(1, self.max_ngram + 1):
                    end = body_start + length
                    if end > n:
                        break
                    phrase = _clean_phrase(tokens[body_start:end])
                    if phrase is None:
                        break

                    # Also normalize hyphenated province/district lookup
                    phrase_no_dash = re.sub(r"\s*-\s*", " ", phrase)

                    # 1. Ward candidate
                    if kind in ("", "ward"):
                        w_cand = None
                        if has_pref and phrase.isdigit():
                            num = int(phrase)
                            if 1 <= num <= 28:
                                cand_name = f"{full_pref} {num}"
                                if cand_name in self.ward_to_id:
                                    w_cand = self.ward_to_id[cand_name]
                        elif phrase in self.ward_to_id:
                            if has_pref or length >= 2:
                                w_cand = self.ward_to_id[phrase]
                        if w_cand is not None:
                            ward_spans.append((i, end, w_cand, has_pref))

                    # 2. District candidate
                    if kind in ("", "dist"):
                        d_cand = None
                        if has_pref and phrase.isdigit():
                            cand_name = f"{full_pref} {phrase}"
                            if cand_name in self.dist_to_id:
                                d_cand = self.dist_to_id[cand_name]
                        elif has_pref and f"{tok[0]} {phrase}" in self.dist_aliases:
                            d_cand = self.dist_aliases[f"{tok[0]} {phrase}"]
                        elif phrase in self.dist_to_id:
                            # Guard bare single-word ambiguous tokens like "chu" / "lak" / "vinh"
                            if has_pref or (length >= 2) or (phrase in {"pleiku", "kbang", "m'drak"}):
                                d_cand = self.dist_to_id[phrase]
                            elif length == 1 and phrase == "vinh":
                                d_cand = self.dist_to_id[phrase]
                        elif phrase in self.dist_aliases:
                            d_cand = self.dist_aliases[phrase]

                        if d_cand is not None:
                            dist_spans.append((i, end, d_cand, has_pref))

                    # 3. Province candidate (skip provincial-city alias if exact same span already matched a district)
                    if kind in ("", "prov"):
                        p_cand = None
                        if phrase in self.prov_to_id:
                            p_cand = self.prov_to_id[phrase]
                        elif phrase_no_dash in self.prov_to_id:
                            p_cand = self.prov_to_id[phrase_no_dash]
                        elif phrase in self.true_prov_aliases:
                            # Require prefix or multi-char safety for 2-letter aliases dn/hp/ct/bd
                            if phrase in {"dn", "hp", "ct", "bd"}:
                                if has_pref or n >= 3:
                                    p_cand = self.true_prov_aliases[phrase]
                            else:
                                p_cand = self.true_prov_aliases[phrase]
                        elif phrase_no_dash in self.true_prov_aliases:
                            p_cand = self.true_prov_aliases[phrase_no_dash]
                        elif phrase in self.city_prov_aliases and has_pref and phrase not in self.dist_to_id:
                            p_cand = self.city_prov_aliases[phrase]
                        if p_cand is not None:
                            prov_spans.append((i, end, p_cand, has_pref))

        # Suppress unprefixed child/sibling matches strictly contained inside a prefixed parent span
        pref_spans = [
            (s, e, "w") for s, e, _, pref in ward_spans if pref
        ] + [
            (s, e, "d") for s, e, _, pref in dist_spans if pref
        ] + [
            (s, e, "p") for s, e, _, pref in prov_spans if pref
        ]

        def _is_suppressed(s: int, e: int, pref: bool, own_kind: str) -> bool:
            if pref:
                return False
            for ps, pe, pk in pref_spans:
                if pk != own_kind and ps <= s and e <= pe:
                    return True
            return False

        ward_spans = [sp for sp in ward_spans if not _is_suppressed(sp[0], sp[1], sp[3], "w")]
        dist_spans = [sp for sp in dist_spans if not _is_suppressed(sp[0], sp[1], sp[3], "d")]
        prov_spans = [sp for sp in prov_spans if not _is_suppressed(sp[0], sp[1], sp[3], "p")]

        return {"wards": ward_spans, "districts": dist_spans, "provinces": prov_spans}

    def has_valid_ward_district(
        self, spans: Dict[str, List[Tuple[int, int, int, bool]]]
    ) -> Optional[Tuple[Tuple[int, int, int, bool], Tuple[int, int, int, bool]]]:
        """Find best span-disjoint (ward, parent_district) pair linked in gazetteer."""
        best = None
        best_score = -1
        for ws, we, wid, w_pref in spans["wards"]:
            edges = self.ward_edges[wid]
            for ds, de, did, d_pref in spans["districts"]:
                if we <= ds or de <= ws:  # Strictly span-disjoint
                    if any(ed_d == did for ed_d, _ in edges):
                        score = (we - ws) + (de - ds) + int(w_pref) + int(d_pref)
                        if score > best_score:
                            best_score = score
                            best = ((ws, we, wid, w_pref), (ds, de, did, d_pref))
        return best

    def has_valid_ward_province(
        self, spans: Dict[str, List[Tuple[int, int, int, bool]]]
    ) -> Optional[Tuple[Tuple[int, int, int, bool], Tuple[int, int, int, bool]]]:
        """Find best span-disjoint (ward, parent_province) pair linked in gazetteer."""
        best = None
        best_score = -1
        for ws, we, wid, w_pref in spans["wards"]:
            edges = self.ward_edges[wid]
            w_name = self.wards[wid]
            # If ward lacks explicit prefix and shares name with a district/province, skip here
            if not w_pref and (w_name in self.dist_names_set or w_name in self.prov_names_set):
                continue
            for ps, pe, pid, p_pref in spans["provinces"]:
                if we <= ps or pe <= ws:
                    if any(ed_p == pid for _, ed_p in edges):
                        score = (we - ws) + (pe - ps) + int(w_pref) + int(p_pref)
                        if score > best_score:
                            best_score = score
                            best = ((ws, we, wid, w_pref), (ps, pe, pid, p_pref))
        return best

    def has_valid_district_province(
        self, spans: Dict[str, List[Tuple[int, int, int, bool]]]
    ) -> Optional[Tuple[Tuple[int, int, int, bool], Tuple[int, int, int, bool]]]:
        """Find span-disjoint (district, parent_province) pair linked in dp."""
        best = None
        best_score = -1
        for ds, de, did, d_pref in spans["districts"]:
            pids = self.district_provinces[did]
            for ps, pe, pid, p_pref in spans["provinces"]:
                if de <= ps or pe <= ds:
                    if pid in pids:
                        score = (de - ds) + (pe - ps) + int(d_pref) + int(p_pref)
                        if score > best_score:
                            best_score = score
                            best = ((ds, de, did, d_pref), (ps, pe, pid, p_pref))
        return best


@lru_cache(maxsize=4)
def get_gazetteer_index(gazetteer_path: str = str(DEFAULT_GAZETTEER_PATH)) -> GazetteerIndex:
    return GazetteerIndex(gazetteer_path)


def _make_result(
    route: RouteDecision,
    sufficiency: SufficiencyLabel,
    probs_tuple: Tuple[float, float, float],
    reason: str,
) -> HeuristicResult:
    p0, p1, p2 = probs_tuple
    total = max(1e-9, p0 + p1 + p2)
    p0, p1, p2 = round(p0 / total, 6), round(p1 / total, 6), round(p2 / total, 6)
    return HeuristicResult(
        route=route,
        is_address=(sufficiency == "address_sufficient"),
        sufficiency=sufficiency,
        probability=p2,
        sufficiency_probs={
            "not_address": p0,
            "address_insufficient": p1,
            "address_sufficient": p2,
        },
        reason=reason,
    )


def _looks_like_person_name(
    tokens: Sequence[str],
    has_digit: bool,
    has_comma: bool,
    spans: Dict[str, List[Tuple[int, int, int, bool]]],
) -> bool:
    if has_digit or has_comma:
        return False
    clean = [t for t in tokens if t not in {",", ";", ":", "(", ")", "-"}]
    if not (2 <= len(clean) <= 5):
        return False
    if len(clean) == 2:
        # 2-word names like "Nguyễn Trãi", "Lê Lợi" are colliding street/person names;
        # never misclassify a 2-word province/district like "Hà Nội" or "Hồ Chí Minh"
        if spans["provinces"] or spans["districts"]:
            return False
        return clean[0] in _VN_SURNAMES
    if clean[0] in _VN_SURNAMES and (clean[1] in _VN_COMMON_MIDDLES or len(clean) >= 3):
        return True
    if len(clean) >= 3 and clean[0] in _VN_PERSON_TITLES and clean[1] in _VN_SURNAMES:
        return True
    return False


def _has_plausible_house_street(folded: str) -> bool:
    m = _HOUSE_STREET_RE.search(folded)
    if not m:
        return False
    # Guard against "3 ngay 2 dem", "500g", "2 chai", etc.
    after_num = m.group(0).strip().split()[-1].lower()
    if after_num in _NON_STREET_WORDS:
        return False
    return True


def evaluate_heuristic(
    text: str,
    gazetteer: Optional[GazetteerIndex] = None,
) -> HeuristicResult:
    """Evaluate fast Tier-1 Vietnamese Gazetteer & Sufficiency Rules (< 0.3 ms)."""
    norm = normalize_text(text)
    if not norm:
        return _make_result("clear_reject", "not_address", (0.998, 0.001, 0.001), "empty_input")

    folded = strip_accents(norm).lower()

    # 1. Hard non-address technical primitives (URL, email, phone, tracking ID, pure numbers/prices/dates)
    if _URL_ONLY_RE.match(norm):
        return _make_result("clear_reject", "not_address", (0.997, 0.002, 0.001), "url_only")
    if _EMAIL_ONLY_RE.match(norm):
        return _make_result("clear_reject", "not_address", (0.997, 0.002, 0.001), "email_only")
    if _PHONE_ONLY_RE.match(folded):
        return _make_result("clear_reject", "not_address", (0.996, 0.002, 0.002), "phone_only")
    if _TRACKING_ID_RE.match(folded):
        return _make_result("clear_reject", "not_address", (0.995, 0.003, 0.002), "tracking_code")
    if _PURE_NUM_DATE_PRICE_RE.match(folded):
        return _make_result("clear_reject", "not_address", (0.996, 0.002, 0.002), "numeric_or_date_only")

    # Ultra-short inputs (< 4 chars): allow known city/ward abbrs (HN, HCM, SG, p.1) while rejecting "K", "Hi", "Ok", "73"
    if len(norm) < 4 and folded not in {"hn", "hcm", "sg"} and not re.fullmatch(r"p\.?\d{1,2}", folded):
        return _make_result("clear_reject", "not_address", (0.995, 0.003, 0.002), "too_short_non_admin")

    gaz = gazetteer if gazetteer is not None else get_gazetteer_index()
    tokens = gaz.tokenize_for_match(folded)
    non_sep_tokens = [t for t in tokens if t not in {",", ";", ":", "(", ")", "-"}]
    word_count = len(non_sep_tokens)

    has_digit = any(ch.isdigit() for ch in folded)
    has_comma = "," in norm
    has_question = "?" in norm
    has_parens_note = "(" in norm and ")" in norm
    has_phone = bool(_PHONE_ANY_RE.search(folded))

    has_food_product = bool(_FOOD_PRODUCT_RE.search(folded)) or bool(_ACCENTED_FOOD_RE.search(norm.lower()))
    has_chat_question = bool(_CHAT_QUESTION_RE.search(folded)) or has_question
    has_narrative = bool(_NARRATIVE_RE.search(folded))
    has_anti_cue = has_food_product or has_chat_question or has_narrative

    # Question/fee inquiries ("Phí ship về Bình Phước bao nhiêu?") are NOT delivery addresses
    has_delivery_intent = bool(_DELIVERY_INTENT_RE.search(folded)) and not has_chat_question

    has_granular = bool(_GRANULAR_PREFIX_RE.search(folded)) or bool(_ACCENTED_GRANULAR_RE.search(norm.lower()))
    has_street_bldg = bool(_STREET_BUILDING_RE.search(folded))
    has_house_street = _has_plausible_house_street(folded)

    spans = gaz.match_entities(tokens)
    wd_pair = gaz.has_valid_ward_district(spans)
    wp_pair = gaz.has_valid_ward_province(spans)
    dp_pair = gaz.has_valid_district_province(spans)

    has_ward_pref = any(w[3] for w in spans["wards"])
    has_dist_pref = any(d[3] for d in spans["districts"])
    has_prov_pref = any(p[3] for p in spans["provinces"])
    has_any_admin_prefix = has_ward_pref or has_dist_pref or has_prov_pref

    # 2. Pure food / product / chat / question / narrative without delivery + multi-tier address
    if has_anti_cue and not has_delivery_intent:
        if (wd_pair or (has_house_street and (spans["districts"] or spans["provinces"]))) and has_comma and not has_question and not has_food_product:
            return _make_result(
                "ambiguous",
                "not_address",
                (0.55, 0.15, 0.30),
                "anti_cue_with_address_structure",
            )
        return _make_result(
            "clear_reject",
            "not_address",
            (0.985, 0.010, 0.005),
            "food_product_chat_or_narrative",
        )

    # 3. Person / historical-figure guard (e.g. "Nguyễn Trãi", "Lê Lợi", "Hai Bà Trưng", "Nguyễn Thị Thanh Xuân", "Trần Văn Phú")
    if folded == "hai ba trung":
        return _make_result(
            "clear_reject",
            "not_address",
            (0.975, 0.015, 0.010),
            "historical_person_bare",
        )
    if _looks_like_person_name(tokens, has_digit=has_digit, has_comma=has_comma, spans=spans) and not has_any_admin_prefix and not has_granular and not has_street_bldg:
        if wd_pair is None and wp_pair is None:
            return _make_result(
                "clear_reject",
                "not_address",
                (0.982, 0.012, 0.006),
                "person_name_guard",
            )

    # 4. Delivery freetext with customer names / phones / notes -> route to Tier-2 ONNX INT8
    is_freetext_style = (
        has_delivery_intent
        or has_phone
        or has_parens_note
        or (" - " in norm and has_digit)
    )
    if is_freetext_style:
        fallback_suff: SufficiencyLabel = "not_address"
        probs_fb = (0.65, 0.15, 0.20)
        if wd_pair or wp_pair or (has_house_street and (spans["districts"] or spans["provinces"])) or (has_street_bldg and (spans["districts"] or spans["provinces"])):
            fallback_suff = "address_sufficient"
            probs_fb = (0.04, 0.08, 0.88)
        elif has_granular or spans["wards"] or spans["districts"] or spans["provinces"]:
            fallback_suff = "address_insufficient"
            probs_fb = (0.15, 0.68, 0.17)
        return _make_result("ambiguous", fallback_suff, probs_fb, "delivery_freetext_route_model")

    # 5. Single-word non-admin garbage ("Jean", "Xom", "Ok", etc.)
    if word_count == 1 and not has_any_admin_prefix and not spans["provinces"] and not spans["districts"] and not spans["wards"] and not has_granular:
        return _make_result("clear_reject", "not_address", (0.990, 0.006, 0.004), "single_word_non_admin")

    # 6. Clear Address Sufficient (no anti-cues, passes name guard)
    # 6a. Exact span-disjoint (ward + parent_district)
    if wd_pair is not None and word_count <= 24:
        # Ensure unmatched tokens don't look like an arbitrary long sentence
        (ws, we, _, _), (ds, de, _, _) = wd_pair
        matched_len = (we - ws) + (de - ds) + sum((pe - ps) for ps, pe, _, _ in spans["provinces"][:1])
        unmatched = max(0, len(tokens) - matched_len - norm.count(","))
        if unmatched <= 8 or has_granular or has_street_bldg or has_house_street:
            return _make_result(
                "clear_sufficient",
                "address_sufficient",
                (0.003, 0.008, 0.989),
                "gazetteer_ward_parent_district",
            )

    # 6b. Exact span-disjoint (ward + parent_province)
    if wp_pair is not None and word_count <= 24:
        (ws, we, _, w_pref), (ps, pe, _, _) = wp_pair
        # Require either explicit ward prefix OR clean 2-tier short/street form
        if w_pref or has_comma or has_granular or has_house_street or has_street_bldg or word_count <= 6:
            return _make_result(
                "clear_sufficient",
                "address_sufficient",
                (0.004, 0.011, 0.985),
                "gazetteer_ward_parent_province",
            )

    # 6c. House number + street + valid district or province (or building + district + province)
    if (has_house_street and (spans["districts"] or spans["provinces"])) or (has_street_bldg and spans["districts"] and spans["provinces"]):
        if word_count <= 22:
            return _make_result(
                "clear_sufficient",
                "address_sufficient",
                (0.006, 0.014, 0.980),
                "house_street_admin_sufficient",
            )

    # 6d. Explicit granular unit + >= 1 valid district/province (unless it's strictly a bare district+province like "Buôn Hồ, Đắk Lắk")
    if has_granular and ((spans["districts"] and spans["provinces"]) or (has_street_bldg and (spans["districts"] or spans["provinces"]))):
        if word_count <= 22 and not (dp_pair is not None and not spans["wards"] and not has_street_bldg and not has_house_street):
            return _make_result(
                "clear_sufficient",
                "address_sufficient",
                (0.007, 0.018, 0.975),
                "granular_building_admin_sufficient",
            )

    # 7. Clear Address Insufficient (no ward+parent, no house+street+admin)
    # Track exact token coverage so extra narrative words aren't misclassified as pure admin units
    covered: Set[int] = set()
    for s_idx, e_idx, _, _ in spans["wards"] + spans["districts"] + spans["provinces"]:
        for k in range(s_idx, e_idx):
            covered.add(k)
    uncovered_non_sep = sum(
        1 for idx, t in enumerate(tokens)
        if idx not in covered and t not in {",", ";", ":", "(", ")", "-", "."}
    )

    # 7a. Explicit prefixed ward ONLY (no district, no province)
    if has_ward_pref and not spans["districts"] and not spans["provinces"] and uncovered_non_sep <= 1:
        return _make_result(
            "clear_insufficient",
            "address_insufficient",
            (0.015, 0.970, 0.015),
            "ward_only_insufficient",
        )

    # 7b. District + Province without ward/street/house/granular
    if dp_pair is not None and wd_pair is None and wp_pair is None and not has_ward_pref and not has_granular and not has_house_street and not has_street_bldg and uncovered_non_sep == 0:
        return _make_result(
            "clear_insufficient",
            "address_insufficient",
            (0.018, 0.962, 0.020),
            "district_province_no_ward",
        )

    # 7c. District-only (no disjoint province/ward pair, e.g. "Quận 1", "Huyện Lộc Ninh", "Phú Lộc")
    if spans["districts"] and wd_pair is None and wp_pair is None and dp_pair is None and not has_ward_pref and not has_prov_pref and not has_granular and not has_house_street and not has_street_bldg:
        if uncovered_non_sep == 0 and (has_dist_pref or (word_count <= 4 and not spans["provinces"])):
            return _make_result(
                "clear_insufficient",
                "address_insufficient",
                (0.020, 0.965, 0.015),
                "district_only_insufficient",
            )

    # 7d. Province-only (covers whole input, no ward/district pair, e.g. "Hà Nội", "Tỉnh Bình Phước", "Bến Tre")
    if spans["provinces"] and wd_pair is None and wp_pair is None and dp_pair is None and not has_ward_pref and not has_dist_pref and not has_granular and not has_house_street and not has_street_bldg:
        if uncovered_non_sep == 0:
            return _make_result(
                "clear_insufficient",
                "address_insufficient",
                (0.020, 0.968, 0.012),
                "province_only_insufficient",
            )

    # 7e. Bare granular or street/building fragment without ward+parent or district/province
    if (has_granular or has_street_bldg) and not has_ward_pref and wd_pair is None and wp_pair is None and not spans["districts"] and not spans["provinces"] and word_count <= 8:
        return _make_result(
            "clear_insufficient",
            "address_insufficient",
            (0.025, 0.960, 0.015),
            "bare_fragment_insufficient",
        )

    # 8. Unclear / Short non-admin without any geo signal
    if not has_digit and not has_comma and word_count <= 5 and not spans["wards"] and not spans["districts"] and not spans["provinces"] and not has_granular and not has_street_bldg:
        return _make_result(
            "clear_reject",
            "not_address",
            (0.975, 0.018, 0.007),
            "no_address_cues",
        )

    # 9. Everything else routes to Tier-2 ONNX INT8 model
    if has_granular or has_street_bldg or spans["wards"] or spans["districts"] or spans["provinces"]:
        return _make_result(
            "ambiguous",
            "address_insufficient",
            (0.25, 0.55, 0.20),
            "ambiguous_route_to_model",
        )
    return _make_result(
        "ambiguous",
        "not_address",
        (0.75, 0.15, 0.10),
        "ambiguous_route_to_model",
    )
