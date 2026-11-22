"""Laya V3 Core: Vietnamese Telex repair, O(1) SymSpell fuzzy matching, and 2025 2-tier address converter."""

from .converter_2025 import (
    DEFAULT_CONVERTER_PATH,
    VietnamAdminV3Index,
    haversine_km,
)
from .telex_fuzzy import (
    SymSpellGazetteer,
    normalize_text,
    repair_telex_typos,
    strip_accents,
)

__all__ = [
    "DEFAULT_CONVERTER_PATH",
    "SymSpellGazetteer",
    "VietnamAdminV3Index",
    "haversine_km",
    "normalize_text",
    "repair_telex_typos",
    "strip_accents",
]
