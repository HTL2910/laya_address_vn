"""Pure-stdlib CLI to fetch, verify, rebuild, and sync MIT-licensed Vietnam admin assets for Laya v3.2+.

Fetches ONLY permissive MIT-licensed upstream repositories:
  - tranngocminhhieu/vietnamadminunits (MIT)
  - thanglequoc/vietnamese-provinces-database (MIT)
  - madnh/hanhchinhvn (MIT)

Excludes GPL (daohoangson/dvhcvn) and custom-clause (vietmap) sources by design.
Verifies MIT licenses & SHA-256 checksums, runs `build_v3_assets.py`, propagates
`vn_admin_v3_converter.json` to `way_c_post_goong_v3/artifacts/` and
`way_a_widget_v3/dist/model/`, and writes `shared_data_v3/DATA_DIFF_REPORT.json`.

Usage:
  python shared_data_v3/crawl_and_sync_admin_data.py --help
  python shared_data_v3/crawl_and_sync_admin_data.py --dry-run --offline
  python shared_data_v3/crawl_and_sync_admin_data.py --offline
  python shared_data_v3/crawl_and_sync_admin_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

V3_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = V3_DATA_DIR.parent
SHARED_V2_DIR = REPO_ROOT / "shared_data"

if str(V3_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DATA_DIR))

import build_v3_assets  # noqa: E402

PROPAGATION_TARGETS: Tuple[Path, ...] = (
    REPO_ROOT / "way_c_post_goong_v3" / "artifacts" / "vn_admin_v3_converter.json",
    REPO_ROOT / "way_a_widget_v3" / "dist" / "model" / "vn_admin_v3_converter.json",
)

# Strictly MIT-licensed upstream artifacts (zero GPL / custom-clause repos)
MIT_UPSTREAM_SPECS: List[Dict[str, str]] = [
    # 1. tranngocminhhieu/vietnamadminunits
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/LICENSE",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/LICENSE",
        "role": "license",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/convert_legacy_2025_with_location_and_default_ward.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/convert_legacy_2025_with_location_and_default_ward.csv",
        "role": "data",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "new_2025/vietnamadminunits/2025_34-province-3221-ward_with_location.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/2025_34-province-3221-ward_with_location.csv",
        "role": "data",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/raw/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv",
        "role": "data",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/legacy_63-province-10040-ward_with_location.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/processed/legacy_63-province-10040-ward_with_location.csv",
        "role": "data",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/alias_district.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/alias_keywords/legacy/alias_district.csv",
        "role": "data",
    },
    {
        "repo": "tranngocminhhieu/vietnamadminunits",
        "license_kind": "MIT",
        "rel_dest": "legacy/vietnamadminunits/alias_ward.csv",
        "url": "https://raw.githubusercontent.com/tranngocminhhieu/vietnamadminunits/main/data/alias_keywords/legacy/alias_ward.csv",
        "role": "data",
    },
    # 2. thanglequoc/vietnamese-provinces-database
    {
        "repo": "thanglequoc/vietnamese-provinces-database",
        "license_kind": "MIT",
        "rel_dest": "new_2025/thanglequoc/LICENSE",
        "url": "https://raw.githubusercontent.com/thanglequoc/vietnamese-provinces-database/master/LICENSE",
        "role": "license",
    },
    {
        "repo": "thanglequoc/vietnamese-provinces-database",
        "license_kind": "MIT",
        "rel_dest": "new_2025/thanglequoc/vn_provinces_metadata.json",
        "url": "https://raw.githubusercontent.com/thanglequoc/vietnamese-provinces-database/master/json/vn_provinces_metadata.json",
        "role": "data",
    },
    {
        "repo": "thanglequoc/vietnamese-provinces-database",
        "license_kind": "MIT",
        "rel_dest": "new_2025/thanglequoc/full_json_generated_data_vn_units.json",
        "url": "https://raw.githubusercontent.com/thanglequoc/vietnamese-provinces-database/master/json/full_json_generated_data_vn_units.json",
        "role": "data",
    },
    # 3. madnh/hanhchinhvn
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/package.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/package.json",
        "role": "license",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/master_tinh_tp.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/dist/tinh_tp.json",
        "role": "data",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/master_quan_huyen.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/dist/quan_huyen.json",
        "role": "data",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/master_xa_phuong.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/master/dist/xa_phuong.json",
        "role": "data",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/v1.2_tinh_tp.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/v1.2/dist/tinh_tp.json",
        "role": "data",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/v1.2_quan_huyen.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/v1.2/dist/quan_huyen.json",
        "role": "data",
    },
    {
        "repo": "madnh/hanhchinhvn",
        "license_kind": "MIT",
        "rel_dest": "legacy/madnh_hanhchinhvn/v1.2_xa_phuong.json",
        "url": "https://raw.githubusercontent.com/madnh/hanhchinhvn/v1.2/dist/xa_phuong.json",
        "role": "data",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _verify_mit_license_bytes(repo: str, filename: str, raw: bytes) -> None:
    text = raw.decode("utf-8", errors="replace")
    if "GNU GENERAL PUBLIC LICENSE" in text.upper() or "GPL-" in text.upper():
        raise RuntimeError(f"Refusing non-MIT GPL license in {repo}/{filename}")
    if filename == "package.json":
        pkg = json.loads(text)
        if str(pkg.get("license", "")).strip().upper() != "MIT":
            raise RuntimeError(f"Expected MIT license in {repo}/package.json, got {pkg.get('license')!r}")
        return
    if "MIT License" not in text and "Permission is hereby granted, free of charge" not in text:
        raise RuntimeError(f"Could not verify MIT license text in {repo}/{filename}")


def _fetch_upstream_sources(dest_dir: Path, timeout_sec: int = 30) -> List[Dict[str, Any]]:
    fetched: List[Dict[str, Any]] = []
    for spec in MIT_UPSTREAM_SPECS:
        url = spec["url"]
        out_path = dest_dir / spec["rel_dest"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "laya-address-vn-sync/3.2"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} fetching {url}")
            raw = resp.read()
        if spec["role"] == "license":
            _verify_mit_license_bytes(spec["repo"], out_path.name, raw)
        out_path.write_bytes(raw)
        fetched.append({
            "repo": spec["repo"],
            "license": spec["license_kind"],
            "rel_path": spec["rel_dest"],
            "url": url,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        })
    return fetched


def _verify_local_licenses() -> Dict[str, str]:
    checks: Dict[str, str] = {}
    for repo, lic_path in (
        ("tranngocminhhieu/vietnamadminunits", build_v3_assets.LICENSE_VAU),
        ("thanglequoc/vietnamese-provinces-database", build_v3_assets.LICENSE_TQ),
        ("madnh/hanhchinhvn", build_v3_assets.LICENSE_MADNH),
    ):
        if not lic_path.exists():
            raise FileNotFoundError(f"Missing MIT license attribution file: {lic_path}")
        raw = lic_path.read_bytes()
        _verify_mit_license_bytes(repo, lic_path.name, raw)
        checks[lic_path.name] = sha256_bytes(raw)
    return checks


def _verify_local_provenance_hashes() -> Dict[str, Any]:
    if not build_v3_assets.PROVENANCE_OUT.exists():
        raise FileNotFoundError(f"Missing provenance file: {build_v3_assets.PROVENANCE_OUT}")
    prov = json.loads(build_v3_assets.PROVENANCE_OUT.read_text(encoding="utf-8"))
    mismatches: List[str] = []
    checked: Dict[str, str] = {}
    local_map = {
        build_v3_assets.CONVERT_CSV.name: build_v3_assets.CONVERT_CSV,
        build_v3_assets.VN2025_CSV.name: build_v3_assets.VN2025_CSV,
        build_v3_assets.GSO_WARD_CSV.name: build_v3_assets.GSO_WARD_CSV,
        build_v3_assets.SUPPLEMENT_FILE.name: build_v3_assets.SUPPLEMENT_FILE,
        build_v3_assets.LICENSE_VAU.name: build_v3_assets.LICENSE_VAU,
        build_v3_assets.LICENSE_TQ.name: build_v3_assets.LICENSE_TQ,
        build_v3_assets.LICENSE_MADNH.name: build_v3_assets.LICENSE_MADNH,
    }
    for entry in prov.get("raw_files", []):
        fn = entry.get("filename", "")
        if fn in local_map and local_map[fn].exists():
            actual = sha256_file(local_map[fn])
            expected = entry.get("sha256", "")
            checked[fn] = actual
            if expected and actual != expected:
                mismatches.append(f"{fn}: expected {expected}, got {actual}")

    conv_actual = sha256_file(build_v3_assets.CONVERTER_OUT)
    if conv_actual != prov.get("converter_sha256"):
        mismatches.append(f"converter: expected {prov.get('converter_sha256')}, got {conv_actual}")

    prop_hashes: Dict[str, str] = {}
    for target in PROPAGATION_TARGETS:
        if target.exists():
            h = sha256_file(target)
            prop_hashes[str(target.relative_to(REPO_ROOT).as_posix())] = h
            if h != conv_actual:
                mismatches.append(f"{target}: hash {h} != {conv_actual}")
        else:
            mismatches.append(f"Missing propagated target: {target}")

    return {
        "checked_files": checked,
        "converter_sha256": conv_actual,
        "propagated_targets": prop_hashes,
        "mismatches": mismatches,
    }


def _propagate_converter_and_gazetteer(sync_gazetteer: bool = True) -> Dict[str, Any]:
    conv_src = build_v3_assets.CONVERTER_OUT
    conv_sha = sha256_file(conv_src)
    propagated: Dict[str, str] = {}
    for dst in PROPAGATION_TARGETS:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256_file(dst) != conv_sha:
            shutil.copy2(conv_src, dst)
        propagated[str(dst.relative_to(REPO_ROOT).as_posix())] = sha256_file(dst)

    gaz_src = SHARED_V2_DIR / "vn_admin_gazetteer.json"
    widget_gaz_dst = REPO_ROOT / "way_a_widget_v3" / "dist" / "model" / "vn_admin_gazetteer.json"
    gaz_sha = sha256_file(gaz_src) if gaz_src.exists() else None
    widget_gaz_sha = sha256_file(widget_gaz_dst) if widget_gaz_dst.exists() else None
    if sync_gazetteer and gaz_sha and widget_gaz_dst.parent.exists() and widget_gaz_sha != gaz_sha:
        shutil.copy2(gaz_src, widget_gaz_dst)
        widget_gaz_sha = sha256_file(widget_gaz_dst)

    gaz_status: Dict[str, Any] = {
        "shared_data_modified": False,
        "widget_dist_parity": bool(gaz_sha and gaz_sha == widget_gaz_sha),
        "shared_data_gazetteer_bytes": gaz_src.stat().st_size if gaz_src.exists() else 0,
        "shared_data_gazetteer_sha256": gaz_sha,
        "widget_dist_gazetteer_sha256": widget_gaz_sha,
        "frozen_v2_untouched": True,
        "reason": (
            "Kept shared_data/vn_admin_gazetteer.json untouched at frozen v2 MIT baseline "
            "(63 provinces, 656 districts, 7044 wards) because expanding v2 gazetteer edges "
            "alters ceiling_audit synthetic calibration curves (post_goong_consistency pr_auc "
            "drops below pre_goong_struct 0.3883) and risks feature drift for frozen v2 ONNX "
            "models. All v3.2.0 expansions ship in vn_admin_v3_converter.json."
        ),
    }
    return {"converter_sha256": conv_sha, "propagated": propagated, "v2_gazetteer": gaz_status}


def run_sync(
    dry_run: bool = False,
    offline: bool = False,
    sources_dir: Optional[Path] = None,
    keep_downloads: Optional[Path] = None,
    sync_gazetteer: bool = True,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    lic_hashes = _verify_local_licenses()

    if dry_run:
        fetched_meta: List[Dict[str, Any]] = []
        if not offline and sources_dir is None:
            with tempfile.TemporaryDirectory(prefix="laya_dry_fetch_") as tmp:
                tmp_path = Path(tmp)
                fetched_meta = _fetch_upstream_sources(tmp_path, timeout_sec=timeout_sec)
        elif sources_dir is not None:
            if not sources_dir.exists():
                raise FileNotFoundError(f"sources-dir not found: {sources_dir}")
        prov_check = _verify_local_provenance_hashes()
        if prov_check["mismatches"]:
            raise RuntimeError(f"Dry-run integrity check failed: {prov_check['mismatches']}")
        return {
            "mode": "dry_run",
            "offline": offline,
            "licenses_verified": lic_hashes,
            "fetched_count": len(fetched_meta),
            "integrity": prov_check,
            "tracked_files_modified": False,
        }

    # Live or offline build + propagation
    fetched_meta = []
    if sources_dir is not None:
        if not sources_dir.exists():
            raise FileNotFoundError(f"sources-dir not found: {sources_dir}")
        diff_report = build_v3_assets.build(sources_dir=sources_dir)
    elif not offline:
        if keep_downloads is not None:
            keep_downloads.mkdir(parents=True, exist_ok=True)
            fetched_meta = _fetch_upstream_sources(keep_downloads, timeout_sec=timeout_sec)
            # Refresh base CSVs in repo if fetched cleanly
            for rel, dst in (
                ("legacy/vietnamadminunits/convert_legacy_2025_with_location_and_default_ward.csv", build_v3_assets.CONVERT_CSV),
                ("new_2025/vietnamadminunits/2025_34-province-3221-ward_with_location.csv", build_v3_assets.VN2025_CSV),
                ("legacy/vietnamadminunits/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv", build_v3_assets.GSO_WARD_CSV),
                ("legacy/vietnamadminunits/LICENSE", build_v3_assets.LICENSE_VAU),
                ("new_2025/thanglequoc/LICENSE", build_v3_assets.LICENSE_TQ),
            ):
                cand = keep_downloads / rel
                if cand.exists() and sha256_file(cand) != sha256_file(dst):
                    shutil.copy2(cand, dst)
            diff_report = build_v3_assets.build(sources_dir=keep_downloads)
        else:
            with tempfile.TemporaryDirectory(prefix="laya_sync_") as tmp:
                tmp_path = Path(tmp)
                fetched_meta = _fetch_upstream_sources(tmp_path, timeout_sec=timeout_sec)
                for rel, dst in (
                    ("legacy/vietnamadminunits/convert_legacy_2025_with_location_and_default_ward.csv", build_v3_assets.CONVERT_CSV),
                    ("new_2025/vietnamadminunits/2025_34-province-3221-ward_with_location.csv", build_v3_assets.VN2025_CSV),
                    ("legacy/vietnamadminunits/danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv", build_v3_assets.GSO_WARD_CSV),
                    ("legacy/vietnamadminunits/LICENSE", build_v3_assets.LICENSE_VAU),
                    ("new_2025/thanglequoc/LICENSE", build_v3_assets.LICENSE_TQ),
                ):
                    cand = tmp_path / rel
                    if cand.exists() and sha256_file(cand) != sha256_file(dst):
                        shutil.copy2(cand, dst)
                diff_report = build_v3_assets.build(sources_dir=tmp_path)
    else:
        diff_report = build_v3_assets.build(sources_dir=None)

    prop_res = _propagate_converter_and_gazetteer(sync_gazetteer=sync_gazetteer)

    # Enrich DATA_DIFF_REPORT.json deterministically with propagation & v2 gazetteer status
    diff_report["propagated_targets"] = prop_res["propagated"]
    diff_report["v2_gazetteer_sync"] = prop_res["v2_gazetteer"]
    build_v3_assets.DIFF_REPORT_OUT.write_text(
        json.dumps(diff_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "mode": "offline_sync" if offline else "online_sync",
        "licenses_verified": lic_hashes,
        "fetched_count": len(fetched_meta),
        "converter_sha256": prop_res["converter_sha256"],
        "converter_bytes": build_v3_assets.CONVERTER_OUT.stat().st_size,
        "propagated_targets": prop_res["propagated"],
        "v2_gazetteer_sync": prop_res["v2_gazetteer"],
        "diff_report_path": str(build_v3_assets.DIFF_REPORT_OUT.relative_to(REPO_ROOT).as_posix()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch MIT-licensed VN admin sources, verify SHA-256 & licenses, rebuild & propagate Laya v3.2 assets."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify licenses, checksums, and propagated artifacts without modifying any tracked file.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip HTTP fetching and verify/rebuild deterministically from local tracked MIT files.",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=None,
        help="Optional local directory containing crawled MIT sources (legacy/ & new_2025/).",
    )
    parser.add_argument(
        "--keep-downloads",
        type=Path,
        default=None,
        help="Optional directory to retain raw upstream downloads instead of auto-cleaning temp dir.",
    )
    parser.add_argument(
        "--no-gazetteer-sync",
        action="store_true",
        help="Do not rebuild/sync shared_data/vn_admin_gazetteer.json to way_a_widget_v3/dist/model/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP fetch timeout in seconds (default: 30).",
    )
    args = parser.parse_args(argv)

    result = run_sync(
        dry_run=args.dry_run,
        offline=args.offline,
        sources_dir=args.sources_dir,
        keep_downloads=args.keep_downloads,
        sync_gazetteer=not args.no_gazetteer_sync,
        timeout_sec=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
