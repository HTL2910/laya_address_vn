"""Vietnamese text normalization, conservative Telex/VNI repair, and O(1) SymSpell fuzzy matcher."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Fast translation for Vietnamese D-stroke before NFD decomposition.
_D_STROKE_TRANS = str.maketrans({"đ": "d", "Đ": "D"})
_DASH_TRANS = str.maketrans({"–": "-", "—": "-", "−": "-"})
_NORM_PUNCT_RE = re.compile(r"[,.;:!?()\[\]{}\"'`~@#$%^&*+=|\\<>]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Tokenization pattern for Telex repair:
# Match URLs, emails, or alphanumeric+slash tokens as atomic units first so
# URLs, emails, house numbers (45/2B), and phone numbers (0912345678) are untouched.
_TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+\.\S+|[A-Za-zÀ-ỹ0-9_./-]+",
    re.UNICODE,
)
_PURE_ASCII_WORD_RE = re.compile(r"^[A-Za-z]{2,}$")
_URL_OR_EMAIL_RE = re.compile(r"^(?:https?://|www\.|\S+@\S+)", re.IGNORECASE)
_DOMAIN_LIKE_RE = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", re.IGNORECASE)

# Standard Vietnamese syllable validator (accent-stripped lowercase).
_VN_SYLLABLE_RE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
    r"(?:"
    r"(?:oai|oao|oay|oeo|uai|uao|uay|uoi|uou|ieu|yeu|uya|uyu|ai|ao|au|ay|eo|eu|ia|iu|oi|ui|uu)"
    r"|(?:uye|uoa|ie|uo|ye|oa|oe|ua|ue|uy|[aeiouy])(?:ng|nh|ch|[cmnpt])?"
    r")$"
)
_VN_EXTRA_SYLLABLES: frozenset[str] = frozenset({"dak", "lak", "krong"})

# Minority / native gazetteer spellings that naturally contain doubled letters or 's'/'r'
# and must not be altered when seen unmodified.
_GAZETTEER_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "bar",
        "boong",
        "chais",
        "choah",
        "choong",
        "cooih",
        "dee",
        "dooc",
        "ee",
        "hoong",
        "khoong",
        "kroong",
        "lee",
        "loong",
        "moong",
        "ngoong",
        "nhoong",
        "noong",
        "poo",
        "rooi",
        "toong",
    }
)

# Conservative English product/address/chat word protection set.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "after", "again", "air", "all", "also", "am", "an", "and", "any",
        "app", "apt", "are", "area", "as", "asap", "at", "atm", "auto", "ave",
        "baby", "back", "bag", "bags", "bank", "banner", "bar", "bars", "base", "bbq",
        "be", "bear", "bed", "beds", "beef", "been", "beer", "beers", "best", "big",
        "bike", "bill", "bird", "black", "block", "blue", "blvd", "book", "books",
        "boot", "boss", "both", "box", "boxes", "boy", "boys", "brand", "brief",
        "brother", "brown", "burger", "burgers", "bus", "business", "but", "buy",
        "buyer", "by", "bye",
        "cafe", "cake", "cakes", "call", "can", "cap", "caps", "car", "card", "cards",
        "cars", "case", "cash", "cat", "cats", "center", "central", "chair", "chat",
        "check", "cheese", "chef", "chief", "city", "class", "clean", "clear", "click",
        "cliff", "close", "club", "coat", "cod", "code", "coffee", "coffees", "color",
        "com", "combo", "come", "complex", "cool", "copy", "corner", "corp", "cost",
        "cover", "craft", "cross", "crown", "cup",
        "dark", "data", "date", "day", "days", "deal", "deep", "def", "delivery",
        "dept", "desk", "did", "do", "doctor", "does", "dog", "door", "doors", "down",
        "dress", "drink", "drinks", "driver", "drop", "duplex", "dust",
        "ear", "east", "edu", "email", "end", "enter", "error", "ever", "ext", "extra",
        "face", "facebook", "fair", "far", "fast", "father", "fax", "fb", "feel",
        "feet", "file", "filter", "find", "fine", "fire", "firm", "first", "fish",
        "fix", "flex", "floor", "floors", "flow", "flower", "food", "foods", "foot",
        "for", "ford", "form", "four", "fox", "free", "from", "front", "full", "fun",
        "game", "gas", "gate", "gear", "get", "gif", "gift", "gifts", "girl", "girls",
        "give", "glass", "glow", "go", "gojek", "gold", "golf", "good", "goods",
        "google", "gov", "grab", "gram", "gray", "green", "grey", "grill", "group",
        "grow", "gym",
        "had", "hair", "half", "hall", "hand", "hard", "has", "hat", "hats", "have",
        "he", "head", "header", "hear", "help", "her", "here", "hey", "hi", "high",
        "his", "home", "homes", "host", "hot", "hotel", "hotpot", "hour", "hours",
        "house", "houses", "how", "hrs", "http", "https",
        "id", "if", "in", "inbox", "inc", "index", "info", "inner", "insta", "into",
        "ios", "iron", "is", "it", "item", "items", "its",
        "jacket", "jar", "jean", "jeans", "job", "jpg", "jsc", "juice", "just",
        "keep", "key", "kg", "kgs", "kid", "kids", "king", "km", "kms", "know",
        "land", "lane", "large", "last", "late", "later", "layer", "lazada", "lead",
        "leaf", "led", "left", "less", "letter", "level", "life", "lift", "light",
        "like", "line", "link", "linux", "list", "liter", "little", "live", "llc",
        "local", "lock", "login", "look", "looks", "loss", "lost", "lot", "love",
        "low", "lower", "ltd", "lux",
        "madam", "mail", "main", "make", "mall", "man", "manager", "map", "maps",
        "mark", "market", "mart", "master", "max", "meal", "meat", "meet", "member",
        "men", "menu", "mess", "message", "meter", "milk", "mini", "mins", "miss",
        "mix", "ml", "mobile", "mode", "model", "moon", "more", "most", "motel",
        "mother", "motor", "move", "mr", "mrs", "ms", "msg", "much", "must", "my",
        "nail", "name", "near", "need", "net", "never", "new", "news", "next", "nice",
        "night", "no", "noon", "nor", "nos", "not", "note", "notes", "now", "number",
        "of", "off", "offer", "office", "oil", "ok", "okay", "old", "on", "one",
        "only", "open", "or", "orange", "order", "orders", "org", "other", "our",
        "out", "outer", "over", "own", "owner",
        "pack", "packs", "page", "pair", "pants", "paper", "par", "park", "part",
        "partner", "pass", "pay", "pc", "pcs", "pdf", "pen", "per", "pet", "phone",
        "photo", "pic", "pick", "pin", "pink", "pizza", "pizzas", "place", "plan",
        "play", "player", "plaza", "please", "pls", "plus", "plz", "pm", "png",
        "pool", "poor", "pork", "port", "pos", "post", "poster", "pour", "power",
        "press", "price", "print", "printer", "pro", "pub", "puff", "purple", "push",
        "qr", "queen", "quick",
        "radio", "rail", "rain", "read", "real", "red", "ref", "relax", "rent",
        "reply", "resort", "rest", "rice", "right", "ring", "rm", "road", "roof",
        "room", "rooms", "root", "rose", "router", "row", "run", "rust",
        "sale", "sales", "salon", "same", "save", "scanner", "screen", "sea", "seat",
        "sec", "secs", "see", "seed", "self", "sell", "seller", "send", "sent",
        "server", "service", "set", "sets", "sex", "share", "she", "shift", "ship",
        "ships", "shirt", "shirts", "shoe", "shoes", "shoot", "shop", "shops",
        "short", "should", "show", "shower", "shown", "side", "sign", "silver", "sir",
        "sister", "site", "six", "size", "sizes", "skirt", "sleep", "slow", "small",
        "smart", "smooth", "sms", "snow", "so", "soft", "sold", "some", "soon",
        "sort", "sound", "sour", "south", "spa", "space", "speed", "sport", "spot",
        "square", "staff", "stair", "star", "stars", "start", "state", "station",
        "stay", "steak", "step", "sticker", "stone", "stop", "store", "stores",
        "story", "street", "stress", "stuff", "style", "suite", "summer", "sun",
        "super", "sure", "sushi", "sweet", "system",
        "table", "tag", "take", "talk", "tar", "tax", "taxi", "tea", "team", "teas",
        "teen", "tel", "term", "test", "text", "than", "thank", "thanks", "that",
        "the", "their", "them", "then", "there", "these", "they", "thief", "thing",
        "this", "those", "three", "throw", "tiff", "tiki", "tiktok", "time", "times",
        "to", "today", "too", "tool", "tools", "top", "tops", "tour", "tours",
        "tower", "town", "toy", "toys", "trade", "train", "tree", "trees", "truck",
        "true", "trust", "try", "tv", "two", "type",
        "under", "unit", "unix", "up", "update", "upper", "url", "us", "use", "user",
        "van", "very", "vex", "view", "villa", "vip", "vow", "vs",
        "wall", "want", "war", "warm", "was", "watch", "water", "wax", "way", "we",
        "wear", "web", "week", "weeks", "well", "were", "west", "what", "when",
        "where", "which", "white", "who", "why", "wife", "will", "win", "window",
        "wine", "winter", "with", "without", "wolf", "woman", "women", "wood",
        "word", "work", "world", "would", "wow", "www",
        "yahoo", "yard", "year", "years", "yellow", "yes", "yet", "yoga", "you",
        "your", "youtube",
        "zalo", "zip", "zone", "zoo",
    }
)

# Mid-word VNI marker pattern: letter + (6/7/8/9 or tone 1-5 after vowel) + letter(s)
_VNI_WORD_RE = re.compile(
    r"^(?:[A-Za-z]+[6789])+[A-Za-z]+[1-5]?$|^[A-Za-z]*[aeiouyAEIOUY][1-5][A-Za-z]+$"
)
_TELEX_VOWEL_OR_DD_RE = re.compile(r"^dd(?=[aeiouy])|uwow|uow|uw|aa|aw|ee|oo|ow")
_MID_TELEX_TONE_RE = re.compile(r"^([a-z]*[aeiouy])([sfrxj])([a-z]+)$")

LEVEL_BITS: Dict[str, int] = {
    "wo": 1,
    "wn": 2,
    "do": 4,
    "po": 8,
    "pn": 16,
    "ward": 3,
    "district": 4,
    "province": 24,
}


def strip_accents(text: str) -> str:
    """Strip Vietnamese diacritics and map đ/Đ -> d/D."""
    if not text:
        return ""
    if text.isascii():
        return text
    mapped = text.translate(_D_STROKE_TRANS)
    decomposed = unicodedata.normalize("NFD", mapped)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese administrative text to lowercase accent-stripped spacing."""
    if not text:
        return ""
    s = strip_accents(text).lower().translate(_DASH_TRANS)
    s = _NORM_PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    # Trim stray edge hyphens while keeping internal ' - ' (e.g. 'ba ria - vung tau').
    return s.strip("- ")


def _is_valid_vn_syllable(word: str) -> bool:
    return bool(_VN_SYLLABLE_RE.match(word) or word in _VN_EXTRA_SYLLABLES)


def _collapse_telex_vowels(base: str, allow_bare_ow: bool) -> Tuple[str, bool]:
    """Collapse Telex doubled vowels / horn / dd markers. Returns (collapsed, changed)."""
    orig = base
    if base.startswith("dd") and len(base) > 2 and base[2] in "aeiouy":
        base = "d" + base[2:]
    if "uwow" in base:
        base = base.replace("uwow", "uo")
    if "uow" in base:
        base = base.replace("uow", "uo")
    if "uw" in base:
        base = base.replace("uw", "u")
    if "aa" in base:
        base = base.replace("aa", "a")
    if "aw" in base:
        base = base.replace("aw", "a")
    if "ee" in base:
        base = base.replace("ee", "e")
    if "oo" in base:
        base = base.replace("oo", "o")
    if "ow" in base and (allow_bare_ow or base != orig):
        base = base.replace("ow", "o")
    return base, (base != orig)


def _apply_case_like(src: str, repaired: str) -> str:
    if src.isupper():
        return repaired.upper()
    if src[0].isupper() and src[1:].islower():
        return repaired.capitalize()
    return repaired


def _repair_single_word(word: str) -> str:
    low = word.lower()
    if low in _ENGLISH_STOPWORDS or low in _GAZETTEER_EXCEPTIONS:
        return word

    # Optional VNI internal digit pattern (e.g. phu7o7ng2, tie6n2, no6i5).
    if any(ch.isdigit() for ch in low):
        if _VNI_WORD_RE.match(low):
            stripped_vni = "".join(ch for ch in low if not ch.isdigit())
            if len(stripped_vni) >= 3 and _is_valid_vn_syllable(stripped_vni):
                return _apply_case_like(word, stripped_vni)
        return word

    if not _PURE_ASCII_WORD_RE.match(low):
        return word

    # Fast skip if word has no Telex marker and doesn't end in a tone key.
    has_marker = bool(_TELEX_VOWEL_OR_DD_RE.search(low))
    has_trailing_tone = len(low) >= 3 and low[-1] in "sfrxj" and low[-2] not in "sfrxj"
    if not has_marker and not has_trailing_tone:
        return word

    # 1. Try trailing tone key removal + vowel collapse (most common: phuowngf, loocj, binhf, nghes).
    if has_trailing_tone:
        cand_no_tone = low[:-1]
        collapsed, _ = _collapse_telex_vowels(cand_no_tone, allow_bare_ow=True)
        if _is_valid_vn_syllable(collapsed):
            return _apply_case_like(word, collapsed)

    # 2. Try vowel collapse only (e.g., phuowng, quaan, taan, huyeen).
    if has_marker:
        # Allow bare 'ow' -> 'o' only when explicit uow/uwow/dd/etc. or when resulting VN syllable is valid
        # and the original 'ow' word isn't an open English-style word ending in 'ow'.
        allow_ow = not low.endswith("ow")
        collapsed, changed = _collapse_telex_vowels(low, allow_bare_ow=allow_ow)
        if changed and _is_valid_vn_syllable(collapsed):
            return _apply_case_like(word, collapsed)

        # 3. Try mid-word tone key removal alongside explicit Telex vowel/dd marker
        #    (e.g., tieefn -> tien, caafn -> can, loojc -> loc, dduowfng -> duong, ddawsk -> dak).
        m = _MID_TELEX_TONE_RE.match(low)
        if m:
            mid_cand = m.group(1) + m.group(3)
            collapsed_mid, _ = _collapse_telex_vowels(mid_cand, allow_bare_ow=True)
            if _is_valid_vn_syllable(collapsed_mid):
                return _apply_case_like(word, collapsed_mid)
        # Also scan each single tone key after vowel/before coda when multiple vowels precede it (e.g. trieefu, bawsc).
        for idx in range(2, len(low) - 1):
            if low[idx] in "sfrxj" and low[idx - 1] in "aeiouyw":
                mid_cand = low[:idx] + low[idx + 1 :]
                collapsed_mid, _ = _collapse_telex_vowels(mid_cand, allow_bare_ow=True)
                if _is_valid_vn_syllable(collapsed_mid):
                    return _apply_case_like(word, collapsed_mid)

    return word


def repair_telex_typos(text: str) -> str:
    """Repair Vietnamese Telex/VNI typing artifacts conservatively without corrupting English/numbers/URLs."""
    if not text:
        return ""

    def _replace_token(match: re.Match[str]) -> str:
        tok = match.group(0)
        if _URL_OR_EMAIL_RE.match(tok) or _DOMAIN_LIKE_RE.match(tok):
            return tok
        # Any token containing digits (house numbers 45/2B, phones 0912345678, Q1, P7)
        # is protected unless it matches strict internal-digit VNI pattern.
        if any(ch.isdigit() for ch in tok):
            if "/" in tok or "-" in tok or "." in tok:
                return tok
            return _repair_single_word(tok)
        if "/" in tok:
            parts = tok.split("/")
            return "/".join(_repair_single_word(p) if p.isascii() and p.isalpha() else p for p in parts)
        if "-" in tok:
            parts = tok.split("-")
            return "-".join(_repair_single_word(p) if p.isascii() and p.isalpha() else p for p in parts)
        if "." in tok:
            return tok
        return _repair_single_word(tok)

    return _TOKEN_RE.sub(_replace_token, text)


def _is_dl_distance_1(a: str, b: str) -> bool:
    """Verify exact Damerau-Levenshtein (optimal string alignment) distance == 1."""
    la = len(a)
    lb = len(b)
    diff = la - lb
    if diff < -1 or diff > 1 or a == b:
        return False
    if diff == 0:
        i = 0
        while i < la and a[i] == b[i]:
            i += 1
        # Single substitution at i
        if a[i + 1 :] == b[i + 1 :]:
            return True
        # Single adjacent transposition at (i, i+1)
        return i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i] and a[i + 2 :] == b[i + 2 :]
    if diff > 0:
        a, b = b, a
        la, lb = lb, la
    # la + 1 == lb: single insertion/deletion
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def _tie_penalty(q: str, cand: str) -> int:
    """Penalize edits that alter word-initial characters relative to internal/suffix edits."""
    qw = q.split()
    cw = cand.split()
    if len(qw) == len(cw):
        mismatches = sum(1 for a, b in zip(qw, cw) if a[0] != b[0])
        first_char_diff = 1 if q[0] != cand[0] else 0
        return mismatches * 2 + first_char_diff
    return 1 if q[0] != cand[0] else 0


class SymSpellGazetteer:
    """Pure-Python O(1) 1-edit SymSpell delete-dictionary fuzzy matcher over normalized Vietnamese admin names."""

    def __init__(
        self,
        entries: Optional[Iterable[str]] = None,
        level_masks: Optional[Dict[str, int]] = None,
        parents: Optional[Dict[str, Set[str]]] = None,
        pre_normalized: bool = False,
    ) -> None:
        self._entries: List[str] = []
        self._entry_to_id: Dict[str, int] = {}
        self._entry_masks: List[int] = []
        self._entry_parents: Optional[Dict[int, Set[str]]] = None
        self._lazy_data: Optional[Dict[str, object]] = None
        self._deletes: Dict[str, List[int]] = defaultdict(list)

        if entries is not None:
            self._build(entries, level_masks, parents, pre_normalized=pre_normalized)

    def _build(
        self,
        entries: Iterable[str],
        level_masks: Optional[Dict[str, int]],
        parents: Optional[Dict[str, Set[str]]],
        pre_normalized: bool = False,
    ) -> None:
        import gc

        was_gc = gc.isenabled()
        gc.disable()
        try:
            if pre_normalized and level_masks is not None:
                entry_list = list(level_masks.keys())
                entry_to_id = {s: i for i, s in enumerate(entry_list)}
                entry_masks = list(level_masks.values())
                self._entries = entry_list
                self._entry_to_id = entry_to_id
                self._entry_masks = entry_masks
                deletes = self._deletes
                for cid, s in enumerate(entry_list):
                    n = len(s)
                    if n >= 4:
                        for i in range(n):
                            deletes[s[:i] + s[i + 1 :]].append(cid)
            else:
                entry_list = self._entries
                entry_to_id = self._entry_to_id
                entry_masks = self._entry_masks
                deletes = self._deletes
                for raw in entries:
                    s = raw if (raw.isascii() and raw.islower()) else normalize_text(raw)
                    if not s:
                        continue
                    cid = entry_to_id.get(s)
                    mask = level_masks.get(s, 31) if level_masks else 31
                    if cid is None:
                        cid = len(entry_list)
                        entry_to_id[s] = cid
                        entry_list.append(s)
                        entry_masks.append(mask)
                        n = len(s)
                        if n >= 4:
                            for i in range(n):
                                deletes[s[:i] + s[i + 1 :]].append(cid)
                    else:
                        entry_masks[cid] |= mask

            if parents:
                p_map: Dict[int, Set[str]] = {}
                for term, p_set in parents.items():
                    cid = self._entry_to_id.get(term)
                    if cid is not None and p_set:
                        p_map[cid] = set(p_set)
                self._entry_parents = p_map
        finally:
            if was_gc:
                gc.enable()

    def _get_parents_for_cid(self, cid: int) -> Set[str]:
        if self._entry_parents is not None and cid in self._entry_parents:
            return self._entry_parents[cid]
        if self._entry_parents is None:
            self._entry_parents = {}
        data = self._lazy_data
        if data is None:
            return set()

        name = self._entries[cid]
        pset: Set[str] = set()
        do_idx: Dict[str, int] = self._do_idx  # type: ignore[attr-defined]
        wo_idx: Dict[str, int] = self._wo_idx  # type: ignore[attr-defined]
        wn_idx: Dict[str, int] = self._wn_idx  # type: ignore[attr-defined]

        did_o = do_idx.get(name)
        if did_o is not None:
            po: List[str] = data["po"]  # type: ignore[assignment]
            dp: List[List[int]] = data["dp"]  # type: ignore[assignment]
            for pid_o in dp[did_o]:
                pset.add(po[pid_o])

        wid_o = wo_idx.get(name)
        if wid_o is not None:
            do_list: List[str] = data["do"]  # type: ignore[assignment]
            po_list: List[str] = data["po"]  # type: ignore[assignment]
            pn_list: List[str] = data["pn"]  # type: ignore[assignment]
            eo: List[List[List[int]]] = data["eo"]  # type: ignore[assignment]
            for r_did, r_pid, _, r_pn, _, _ in eo[wid_o]:
                pset.add(do_list[r_did])
                pset.add(po_list[r_pid])
                pset.add(pn_list[r_pn])

        wid_n = wn_idx.get(name)
        if wid_n is not None:
            pn_list = data["pn"]  # type: ignore[assignment]
            en: List[List[List[object]]] = data["en"]  # type: ignore[assignment]
            for row in en[wid_n]:
                pset.add(pn_list[int(row[0])])  # type: ignore[arg-type]

        self._entry_parents[cid] = pset
        return pset

    @classmethod
    def from_converter_data(cls, data: Dict[str, object]) -> "SymSpellGazetteer":
        """Build SymSpellGazetteer over wo, wn, do, po, pn with O(1) parent context (< 50 ms)."""
        masks: Dict[str, int] = {}
        for x in data["po"]:  # type: ignore[attr-defined]
            masks[x] = masks.get(x, 0) | 8
        for x in data["pn"]:  # type: ignore[attr-defined]
            masks[x] = masks.get(x, 0) | 16
        do_idx: Dict[str, int] = {}
        for i, x in enumerate(data["do"]):  # type: ignore[arg-type]
            masks[x] = masks.get(x, 0) | 4
            do_idx[x] = i
        wo_idx: Dict[str, int] = {}
        for i, x in enumerate(data["wo"]):  # type: ignore[arg-type]
            masks[x] = masks.get(x, 0) | 1
            wo_idx[x] = i
        wn_idx: Dict[str, int] = {}
        for i, x in enumerate(data["wn"]):  # type: ignore[arg-type]
            masks[x] = masks.get(x, 0) | 2
            wn_idx[x] = i

        inst = cls(masks.keys(), level_masks=masks, pre_normalized=True)
        inst._lazy_data = data
        inst._do_idx = do_idx  # type: ignore[attr-defined]
        inst._wo_idx = wo_idx  # type: ignore[attr-defined]
        inst._wn_idx = wn_idx  # type: ignore[attr-defined]
        return inst

    def lookup(
        self,
        query: str,
        level: Optional[str] = None,
        parent_context: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Look up exact match first, else 1-edit Damerau-Levenshtein match for len >= 4."""
        q = query if (query and query.isascii() and query.islower() and "," not in query and "." not in query) else normalize_text(query)
        if len(q) < 4:
            return None

        level_mask = LEVEL_BITS.get(level, 0) if level else 0
        entry_masks = self._entry_masks
        entry_to_id = self._entry_to_id

        # 1. Exact lookup first
        exact_cid = entry_to_id.get(q)
        if exact_cid is not None and (level_mask == 0 or (entry_masks[exact_cid] & level_mask)):
            return self._entries[exact_cid]

        # 2. 1-edit SymSpell delete-dictionary probe
        deletes_get = self._deletes.get
        entries = self._entries
        cands: Set[int] = set()

        hit = deletes_get(q)
        if hit:
            for cid in hit:
                if level_mask == 0 or (entry_masks[cid] & level_mask):
                    cands.add(cid)

        n = len(q)
        prev = ""
        for i in range(n):
            ch = q[i]
            if i == 0 or ch != prev:
                dq = q[:i] + q[i + 1 :]
                ecid = entry_to_id.get(dq)
                if ecid is not None and (level_mask == 0 or (entry_masks[ecid] & level_mask)):
                    cands.add(ecid)
                hit = deletes_get(dq)
                if hit:
                    for cid in hit:
                        if level_mask == 0 or (entry_masks[cid] & level_mask):
                            cands.add(cid)
            prev = ch

        if not cands:
            return None

        verified = [cid for cid in cands if _is_dl_distance_1(q, entries[cid])]
        if not verified:
            return None
        if len(verified) == 1:
            return entries[verified[0]]

        # Narrow by parent context if provided
        if parent_context:
            ctx_norm = {
                p if (p.isascii() and p.islower() and "," not in p) else normalize_text(p)
                for p in parent_context
                if p
            }
            ctx_norm.discard("")
            if ctx_norm:
                get_parents = self._get_parents_for_cid
                ctx_hits = [cid for cid in verified if (get_parents(cid) & ctx_norm)]
                if len(ctx_hits) == 1:
                    return entries[ctx_hits[0]]
                if len(ctx_hits) > 1:
                    verified = ctx_hits

        # Break ties only when one candidate strictly preserves word-initial letters over others
        penalties = [(_tie_penalty(q, entries[cid]), cid) for cid in verified]
        min_pen = min(p for p, _ in penalties)
        best = [cid for p, cid in penalties if p == min_pen]
        if len(best) == 1:
            return entries[best[0]]

        # Ignore genuinely ambiguous multi-entry ties.
        return None
