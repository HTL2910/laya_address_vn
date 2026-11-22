"""Unit and end-to-end verification tests for slot_parser_v3."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
_PKG_PARENT = _PKG_ROOT.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from slot_parser_v3 import (
    AddressSlotParser,
    BIO_TAGS,
    DEFAULT_ARTIFACTS_DIR,
    ParsedAddress,
    SlotParserTokenizer,
    tokenize_words,
    viterbi_decode,
)
from slot_parser_v3.tokenizer import TAG2ID


@pytest.fixture(scope="module")
def parser() -> AddressSlotParser:
    return AddressSlotParser(artifacts_dir=DEFAULT_ARTIFACTS_DIR, intra_op_threads=1)


def test_schema_and_runtime_imports() -> None:
    assert len(BIO_TAGS) == 17
    assert BIO_TAGS[0] == "O"
    assert BIO_TAGS[1:5] == ("B-HNO", "I-HNO", "B-STR", "I-STR")
    assert BIO_TAGS[5:11] == ("B-WAR", "I-WAR", "B-DIS", "I-DIS", "B-PRO", "I-PRO")
    assert BIO_TAGS[11:17] == ("B-PER", "I-PER", "B-PHN", "I-PHN", "B-NOTE", "I-NOTE")
    # Runtime package initialization must not import torch.
    assert "torch" not in sys.modules
    # Directly inspect parser.py source to confirm zero torch/pickle dependencies.
    parser_src = (_PKG_ROOT / "parser.py").read_text(encoding="utf-8")
    assert "import torch" not in parser_src
    assert "import pickle" not in parser_src


def test_tokenizer_preserves_special_tokens_and_shapes() -> None:
    toks = tokenize_words("12/4/5 Nguyễn Trãi, 0912.345.678, x.loc, p.12, TP.HCM")
    assert "12/4/5" in toks
    assert "0912.345.678" in toks
    assert "x.loc" in toks
    assert "p.12" in toks
    assert "TP.HCM" in toks
    assert toks.count(",") == 4

    toks_unacc = tokenize_words("12/4/5 Nguyen Trai, p.12")
    assert toks_unacc == ["12/4/5", "Nguyen", "Trai", ",", "p.12"]

    tokenizer = SlotParserTokenizer.from_vocab_file(DEFAULT_ARTIFACTS_DIR / "vocab.json")
    batch = tokenizer.encode_batch(["12/4/5 Nguyễn Trãi, Q1", "Lộc Thạnh, Lộc Ninh"])
    assert batch["word_ids"].shape == (2, 64)
    assert batch["attention_mask"].shape == (2, 64)
    assert batch["char_ids"].shape == (2, 64, 16)
    assert batch["word_ids"].dtype == np.int64
    assert batch["attention_mask"].dtype == np.int64
    assert batch["char_ids"].dtype == np.int64
    assert int(batch["attention_mask"][1].sum()) == 5


def test_viterbi_enforces_bio_constraints() -> None:
    # Even if raw emissions strongly boost orphan I-WAR at t=0 and I-PRO after B-STR,
    # viterbi_decode must never produce an illegal BIO transition.
    emissions = np.full((4, len(BIO_TAGS)), -100.0, dtype=np.float32)
    emissions[0, TAG2ID["I-WAR"]] = 50.0
    emissions[0, TAG2ID["B-STR"]] = 5.0
    emissions[1, TAG2ID["I-PRO"]] = 50.0
    emissions[1, TAG2ID["I-STR"]] = 4.0
    emissions[2, TAG2ID["O"]] = 5.0
    emissions[3, TAG2ID["I-DIS"]] = 50.0
    emissions[3, TAG2ID["B-DIS"]] = 3.0
    transitions = np.zeros((len(BIO_TAGS), len(BIO_TAGS)), dtype=np.float32)

    path = viterbi_decode(emissions, transitions, length=4)
    decoded = [BIO_TAGS[i] for i in path]
    assert decoded == ["B-STR", "I-STR", "O", "B-DIS"]


def test_parse_full_noisy_ecommerce_address(parser: AddressSlotParser) -> None:
    text = (
        "Anh Nam 0912345678 - 12/4/5 Nguyễn Trãi, Phường Bến Nghé, "
        "Quận 1, TP.HCM (giao giờ hành chính)"
    )
    res = parser.parse(text)
    assert isinstance(res, ParsedAddress)
    assert "Anh Nam" in res.person
    assert "0912345678" in res.phone
    assert res.house_num == "12/4/5"
    assert res.street == "Nguyễn Trãi"
    assert res.ward == "Phường Bến Nghé"
    assert res.district == "Quận 1"
    assert res.province == "TP.HCM"
    assert "giao giờ hành chính" in res.delivery_note

    assert "12/4/5 Nguyễn Trãi" in res.clean_address_for_goong
    assert "Phường Bến Nghé" in res.clean_address_for_goong
    assert "Quận 1" in res.clean_address_for_goong
    assert "TP.HCM" in res.clean_address_for_goong
    assert "Anh Nam" not in res.clean_address_for_goong
    assert "0912345678" not in res.clean_address_for_goong
    assert "giao giờ hành chính" not in res.clean_address_for_goong

    as_dict = res.to_dict()
    assert as_dict["clean_address_for_goong"] == res.clean_address_for_goong


def test_parse_short_two_part_address(parser: AddressSlotParser) -> None:
    res = parser.parse("Lộc Thạnh, Lộc Ninh")
    assert "Lộc Thạnh" in res.ward
    assert "Lộc Ninh" in res.district
    assert res.person == ""
    assert res.phone == ""
    assert res.delivery_note == ""


@pytest.mark.parametrize(
    "neg_text",
    [
        "Cho mình 3 chai nướng BBQ",
        "Xin giá",
        "bánh mì Sài Gòn",
    ],
)
def test_parse_non_address_negatives(parser: AddressSlotParser, neg_text: str) -> None:
    res = parser.parse(neg_text)
    assert res.house_num == ""
    assert res.street == ""
    assert res.ward == ""
    assert res.district == ""
    assert res.province == ""
    assert res.clean_address_for_goong == ""


def test_parse_rural_apt_and_batch_parity(parser: AddressSlotParser) -> None:
    rural = "0912.345.678 - 45/2B/10 Đường số 3, Ấp 2, Xã Lộc Thạnh, Huyện Lộc Ninh, Bình Phước"
    apt = "Chị Trần Thị Hoa - P.702 Tòa HH1C, Hoàng Liệt, Hoàng Mai, Hà Nội (gọi trước 15p)"
    two_tier = "Phường Hồng Hà, Thành phố Hà Nội"

    singles = [parser.parse(t) for t in ("", rural, apt, two_tier)]
    batches = parser.parse_batch(["", rural, apt, two_tier])
    assert len(batches) == 4
    for s, b in zip(singles, batches):
        assert s.tags == b.tags
        assert s.clean_address_for_goong == b.clean_address_for_goong

    assert singles[0].clean_address_for_goong == ""
    assert "0912.345.678" in singles[1].phone
    assert singles[1].house_num == "45/2B/10"
    assert "Xã Lộc Thạnh" in singles[1].ward
    assert "Huyện Lộc Ninh" in singles[1].district
    assert "Bình Phước" in singles[1].province

    assert "Chị Trần Thị Hoa" in singles[2].person
    assert "P.702" in singles[2].house_num
    assert "HH1C" in singles[2].street
    assert "gọi trước 15p" in singles[2].delivery_note

    assert "Hồng Hà" in singles[3].ward
    assert "Hà Nội" in singles[3].province


def test_artifacts_budget_eval_metrics_and_latency(parser: AddressSlotParser) -> None:
    pt_files = list(_PKG_ROOT.rglob("*.pt"))
    assert pt_files == [], f"Unexpected .pt files found: {pt_files}"

    onnx_path = DEFAULT_ARTIFACTS_DIR / "slot_parser_v3_int8.onnx"
    assert onnx_path.exists()
    assert onnx_path.stat().st_size <= 4 * 1024 * 1024

    report_path = DEFAULT_ARTIFACTS_DIR / "eval_report_v3.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    holdout = report["holdout_500_metrics"]
    assert holdout["macro_entity_f1"] >= 0.85
    assert holdout["per_slot"]["WAR"]["f1"] >= 0.90
    assert holdout["per_slot"]["PRO"]["f1"] >= 0.90
    assert holdout["per_slot"]["PHN"]["f1"] >= 0.90

    sample = (
        "Anh Nam 0912345678 - 12/4/5 Nguyễn Trãi, Phường Bến Nghé, "
        "Quận 1, TP.HCM (giao giờ hành chính)"
    )
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        parser.parse(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    p99 = float(np.percentile(latencies, 99))
    assert p99 <= 10.0, f"Single-item p99 latency too high: {p99:.2f} ms"


def test_v31_fallback_recovers_comma_free_and_unaccented_addresses(parser: AddressSlotParser) -> None:
    from slot_parser_v3.parser import _apply_empty_clean_fallback
    from slot_parser_v3.tokenizer import SLOT_TYPES

    # 1. Phone guard + end-to-end parse on comma-free noisy inputs
    must_recover = [
        ("45 Tran Phu Loc Thanh Loc Ninh Binh Phuoc", ("Loc Thanh", "Loc Ninh", "Binh Phuoc")),
        ("12 Nguyen Trai Ben Nghe Quan 1 Ho Chi Minh", ("Ben Nghe", "Quan 1", "Ho Chi Minh")),
        ("Ap 2 Loc Thanh Loc Ninh", ("Ap 2", "Loc Thanh", "Loc Ninh")),
        ("so 15 thon 3 xa tan lap", ("so 15", "tan lap")),
        ("khu cong nghiep vsip binh duong", ("vsip", "binh duong")),
        ("so 12 duong 3 khu cong nghiep song than", ("so 12", "song than")),
        ("Anh Nam 0912345678 45 Tran Phu Loc Thanh Loc Ninh Binh Phuoc", ("45 Tran Phu", "Loc Ninh", "Binh Phuoc")),
    ]
    for raw_text, expected_substrings in must_recover:
        res_single = parser.parse(raw_text)
        res_batch = parser.parse_batch([raw_text])[0]
        assert res_single.clean_address_for_goong != "", f"Returned empty clean_address for: {raw_text!r}"
        assert res_single.clean_address_for_goong == res_batch.clean_address_for_goong
        assert res_single.used_fallback == res_batch.used_fallback
        assert "0912345678" not in res_single.clean_address_for_goong
        assert "Anh Nam" not in res_single.clean_address_for_goong
        low_clean = res_single.clean_address_for_goong.lower()
        for sub in expected_substrings:
            assert sub.lower() in low_clean, f"Expected {sub!r} in {res_single.clean_address_for_goong!r}"

    # 2. Directly exercise the all-'O' fallback path (simulating the 86 real-order empty-CRF cases)
    forced_cases = [
        ("Xom 4 Hai Hung Hai Hau Nam Dinh", "Xom 4, Hai Hung, Hai Hau, Nam Dinh", True),
        ("45 Le Loi My Binh Long Xuyen An Giang", "45 Le Loi, My Binh, Long Xuyen, An Giang", True),
        ("Khu cong nghiep VSIP Binh Duong", "Khu cong nghiep VSIP, Binh Duong", True),
        ("so 12 duong 3 khu cong nghiep song than", "so 12 duong 3 khu cong nghiep song than", True),
        ("bánh mì Sài Gòn", "", False),
        ("cơm tấm sườn bì", "", False),
        ("abc", "", False),
        ("123456", "", False),
        ("nha e", "", False),
    ]
    for raw_text, expected_clean, expected_fb in forced_cases:
        toks = tokenize_words(raw_text)
        tags = ["O"] * len(toks)
        slots = {st: "" for st in SLOT_TYPES}
        slots["clean_address_for_goong"] = ""
        used = _apply_empty_clean_fallback(toks, tags, slots, parser._r2l_scanner)
        assert used is expected_fb, f"Unexpected fallback flag on {raw_text!r}: got {used}, slots={slots}"
        assert slots["clean_address_for_goong"] == expected_clean, (
            f"Unexpected clean_address on {raw_text!r}: {slots['clean_address_for_goong']!r}"
        )

    # 3. Non-address negatives & ultra-short strings through full .parse() must stay empty
    must_stay_empty = [
        "abc",
        "1234",
        "nha e",
        "bánh mì Sài Gòn",
        "cơm tấm sườn bì",
        "Cho mình 3 chai nướng BBQ",
        "Xin giá",
    ]
    for neg in must_stay_empty:
        r1 = parser.parse(neg)
        r2 = parser.parse_batch([neg])[0]
        assert r1.clean_address_for_goong == "", f"False positive clean_address for {neg!r}: {r1.clean_address_for_goong!r}"
        assert r2.clean_address_for_goong == ""
        assert r1.used_fallback is False

