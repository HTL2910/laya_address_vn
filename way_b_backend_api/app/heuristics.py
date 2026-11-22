"""Fast Tier-1 Vietnamese Heuristic & Dictionary Filter (< 0.2 ms) for Laya Address VN.

Classifies inputs into:
- 'clear_reject': Unambiguously non-address (URL, email, phone, bank note, bare short chatter)
- 'clear_address': Unambiguously structured multi-level Vietnamese address with zero narrative ambiguity
- 'ambiguous': Route to Tier-2 INT8 ONNX Transformer+CharCNN model
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.tokenizer import normalize_text, strip_accents

RouteDecision = Literal["clear_reject", "clear_address", "ambiguous"]


@dataclass(frozen=True)
class HeuristicResult:
    route: RouteDecision
    is_address: bool
    probability: float
    reason: str


# Fast compiled patterns (operating on lowercased + accent-stripped string)
_URL_ONLY_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_EMAIL_ONLY_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", re.IGNORECASE)
_PHONE_ONLY_RE = re.compile(r"^(?:zalo\s*[:\-]?\s*)?(?:\+?84|0)[\d\s.\-]{8,13}$", re.IGNORECASE)
_PURE_NUM_DATE_PRICE_RE = re.compile(
    r"^(?:[\d\s.,:/\-+xX*%]+(?:\s*(?:vnd|đ|d|k|tr|trieu|kg|g|ml|l|cm|mm|m|h|gio|phut|ngay|thang|nam))?)$",
    re.IGNORECASE,
)
_BANK_OR_CODE_RE = re.compile(
    r"\b(?:chuyen tien|chuyen khoan|ck tien|thanh toan hoa don|ma otp|otp cua ban|"
    r"ma van don|so tai khoan|stk\b|techcombank|vietcombank|mb bank|bidv|vpbank|acb\b|tpbank)\b",
    re.IGNORECASE,
)
_TRACKING_ID_RE = re.compile(r"^(?:[a-z]{2,6}\d{6,16}|#?[a-z]{1,4}\d{5,14})$", re.IGNORECASE)

# Explicit granular address indicators (accent-stripped)
_GRANULAR_PREFIX_RE = re.compile(
    r"(?:\bso\s+\d+[a-z]?(?:/\d+[a-z]?)*\b"
    r"|\bngo\s+\d+[a-z]?(?:/\d+[a-z]?)*\b"
    r"|\bngach\s+\d+[a-z]?(?:/\d+[a-z]?)*\b"
    r"|\bhem\s+\d+[a-z]?(?:/\d+[a-z]?)*\b"
    r"|\bkiet\s+\d+[a-z]?(?:/\d+[a-z]?)*\b"
    r"|\bthon\s+(?:\d+|[a-z]+)\b"
    r"|\bap\s+(?:\d+|[a-z]+)\b"
    r"|\bxom\s+\d+\b"
    r"|\bbuon\s+[a-z]+\b"
    r"|\bkhu pho\s*\d+\b"
    r"|\bkp\.?\s*\d+\b"
    r"|\bto\s*\d+\b"
    r"|\btdp\.?\s*\d+\b"
    r"|\btang\s*\d+\b"
    r"|\blau\s*\d+\b"
    r"|\bcan ho\s*[a-z0-9\-]+\b"
    r"|\bphong\s*\d{2,4}\b"
    r"|\blo\s+[a-z0-9\-]+\b"
    r"|\b\d{1,4}/\d{1,4}[a-z]?(?:/\d{1,3}[a-z]?)?\b)",
    re.IGNORECASE,
)

# Explicit administrative level markers (accent-stripped)
_WARD_RE = re.compile(
    r"(?:\bphuong\s+[a-z0-9]+|\bp\.\s*[a-z0-9]+|\bp\d{1,2}\b|\bxa\s+[a-z]+|\bx\.\s*[a-z]+|\bthi tran\s+[a-z]+|\btt\.\s*[a-z]+)",
    re.IGNORECASE,
)
_DISTRICT_RE = re.compile(
    r"(?:\bquan\s+(?:\d{1,2}|[a-z]+)|\bq\.\s*(?:\d{1,2}|[a-z]+)|\bq\d{1,2}\b|\bhuyen\s+[a-z]+|\bh\.\s*[a-z]+|\bthi xa\s+[a-z]+|\btx\.\s*[a-z]+)",
    re.IGNORECASE,
)
_CITY_PROVINCE_RE = re.compile(
    r"(?:\btp\.?\s*(?:hcm|ho chi minh|ha noi|da nang|thu duc|bien hoa|nha trang|buon ma thuot|can tho|hue|da lat|vinh|ha long|vung tau|quy nhon|thuan an|di an|thu dau mot)\b"
    r"|\b(?:tphcm|hcm|ha noi|hanoi|ho chi minh|da nang|hai phong|can tho|binh duong|dong nai|khanh hoa|dak lak|lam dong|nghe an|thanh hoa|quang ninh|bac ninh|long an|tien giang|an giang|kien giang|ca mau|tay ninh|binh dinh|quang nam|quang ngai|binh thuan|ba ria vung tau|vung tau|thua thien hue|thai nguyen|nam dinh|hai duong|hung yen|vinh phuc)\b"
    r"|\btinh\s+[a-z]+\b"
    r"|\bhn\b)",
    re.IGNORECASE,
)
_STREET_WORD_RE = re.compile(
    r"\b(?:duong|pho|dai lo|quoc lo|ql\d+[a-z]?|tinh lo|tl\d+|xa lo|toa nha|chung cu|kcn|kdt|khu do thi|tower|building|vinhomes|keangnam|bitexco|landmark|masteri|times city|royal city)\b",
    re.IGNORECASE,
)

# Narrative / non-address red-flag cues: if present, Tier-1 NEVER auto-accepts (let ONNX model decide)
_NARRATIVE_RED_FLAGS_RE = re.compile(
    r"\b(?:hom nay|ngay mai|toi nay|cuoi tuan|mua nay|tro lanh|mua chieu|nang dep|"
    r"du lich|ve may bay|xe khach|cong tac|doi tuyen|bong da|ban phao hoa|ket xe|"
    r"chi nhanh|tuyen dung|danh nhan|anh hung|lich su|benh vien|diem chuan|"
    r"gap nhau|hen moi nguoi|quan ca phe|tra sua|phi ship|bao nhieu tien|khuyen mai|"
    r"bao hanh|chinh hang|size\s*(?:s|m|l|xl|xxl)|goi\s*\d+g|hop\s*\d+g|chai\s*\d+ml|"
    r"chuyen tien|chuyen khoan|don hang|huy don|kiem tra giup|tim phong tro|que o|"
    r"tang manh|di an|di choi|di hoc|dai hoc|sach ve|la danh nhan|sap chay)\b",
    re.IGNORECASE,
)

# Known Vietnamese major district/location terms (accent-stripped) for soft reject guard
_KNOWN_GEO_TERMS = {
    "ha noi", "hanoi", "sai gon", "saigon", "ho chi minh", "hcm", "tphcm", "da nang",
    "hai phong", "can tho", "binh duong", "dong nai", "bien hoa", "nha trang", "khanh hoa",
    "hue", "da lat", "vung tau", "quy nhon", "buon ma thuot", "dak lak", "phu quoc",
    "nghe an", "thanh hoa", "quang ninh", "bac ninh", "long an", "ca mau", "an giang",
    "ba dinh", "hoan kiem", "dong da", "hai ba trung", "hoang mai", "thanh xuan",
    "cau giay", "tay ho", "long bien", "nam tu liem", "bac tu liem", "ha dong",
    "binh thanh", "go vap", "phu nhuan", "tan binh", "tan phu", "binh tan", "thu duc",
    "cu chi", "hoc mon", "binh chanh", "nha be", "hai chau", "thanh khe", "son tra",
    "ben nghe", "ben thanh", "da kao", "thao dien", "lang ha", "thai ha", "nguyen trai",
    "nguyen hue", "le loi", "tran hung dao", "dien bien phu", "pham hung", "my dinh",
    "keangnam", "vinhomes", "times city", "bitexco", "landmark", "ea kao", "tam hoa",
    "tan thong hoi", "nguyen van nguyen", "xuan thuy", "kim ma", "giai phong", "minh khai",
}


def _has_any_address_cue(folded: str) -> bool:
    if (
        _GRANULAR_PREFIX_RE.search(folded)
        or _WARD_RE.search(folded)
        or _DISTRICT_RE.search(folded)
        or _CITY_PROVINCE_RE.search(folded)
        or _STREET_WORD_RE.search(folded)
    ):
        return True
    for term in _KNOWN_GEO_TERMS:
        if term in folded:
            return True
    return False


def evaluate_heuristic(text: str) -> HeuristicResult:
    """Evaluate fast Tier-1 Vietnamese heuristic rules (< 0.2 ms)."""
    norm = normalize_text(text)
    if len(norm) < 4:
        return HeuristicResult(
            route="clear_reject",
            is_address=False,
            probability=0.002,
            reason="too_short",
        )

    folded = strip_accents(norm).lower()

    # 1. Hard non-address primitives (URL, email, phone, tracking ID, pure numbers/dates/prices)
    if _URL_ONLY_RE.match(norm):
        return HeuristicResult("clear_reject", False, 0.001, "url_only")
    if _EMAIL_ONLY_RE.match(norm):
        return HeuristicResult("clear_reject", False, 0.001, "email_only")
    if _PHONE_ONLY_RE.match(folded):
        return HeuristicResult("clear_reject", False, 0.002, "phone_only")
    if _TRACKING_ID_RE.match(folded):
        return HeuristicResult("clear_reject", False, 0.003, "tracking_code")
    if _PURE_NUM_DATE_PRICE_RE.match(folded):
        return HeuristicResult("clear_reject", False, 0.003, "numeric_or_date_only")

    has_granular = bool(_GRANULAR_PREFIX_RE.search(folded))
    has_ward = bool(_WARD_RE.search(folded))
    has_district = bool(_DISTRICT_RE.search(folded))
    has_city = bool(_CITY_PROVINCE_RE.search(folded))
    has_street_word = bool(_STREET_WORD_RE.search(folded))
    has_narrative_flag = bool(_NARRATIVE_RED_FLAGS_RE.search(folded))

    # Bank transfer / OTP notes with zero structured address markers -> clear_reject
    if _BANK_OR_CODE_RE.search(folded) and not (has_granular or has_ward or has_district or has_street_word):
        return HeuristicResult("clear_reject", False, 0.004, "bank_or_order_note")

    # Zero digits, zero commas, zero address cues, and short -> clear_reject (e.g., personal names, "Ok bạn nhé")
    has_digit = any(ch.isdigit() for ch in folded)
    has_comma = "," in folded
    word_count = len(folded.split())
    if not has_digit and not has_comma and word_count <= 7 and not _has_any_address_cue(folded):
        return HeuristicResult("clear_reject", False, 0.006, "no_address_cues")

    # 2. Conservative Tier-1 clear_address (zero narrative flags + strong multi-level VN address structure)
    admin_levels = int(has_ward) + int(has_district) + int(has_city)
    if not has_narrative_flag and len(norm) <= 180:
        # Case A: Granular unit (Số/Ngõ/Hẻm/Thôn/Ấp/KP/Tầng/214/19) + >= 2 admin levels
        if has_granular and admin_levels >= 2:
            return HeuristicResult("clear_address", True, 0.992, "granular_plus_multi_admin")
        # Case B: Explicit Ward + District + City (e.g. "P.Ben Nghe, Q.1, TP.HCM")
        if has_ward and has_district and has_city and word_count <= 18:
            return HeuristicResult("clear_address", True, 0.985, "three_explicit_admin_levels")
        # Case C: Granular unit + explicit street/building + at least 1 admin level + comma/compact
        if has_granular and has_street_word and admin_levels >= 1 and word_count <= 20:
            return HeuristicResult("clear_address", True, 0.980, "granular_street_admin")

    # 3. Everything else is ambiguous -> route to ONNX CPU model in hybrid mode;
    # compute heuristic fallback score for heuristic_only mode.
    score = 0.0
    if has_granular:
        score += 0.42
    if has_ward:
        score += 0.25
    if has_district:
        score += 0.20
    if has_city:
        score += 0.14
    if has_street_word:
        score += 0.16
    if has_comma:
        score += 0.08
    if has_digit and not has_granular:
        score += 0.10
    if has_narrative_flag:
        score -= 0.48
    if word_count <= 2 and not (has_granular or has_ward):
        score -= 0.25

    prob = max(0.02, min(0.95, 0.08 + score))
    return HeuristicResult(
        route="ambiguous",
        is_address=prob >= 0.5,
        probability=round(prob, 4),
        reason="ambiguous_route_to_model",
    )
