"""Step 2 Ground-Truth 300-Sample Evaluator & Strict PII-Free v3 Brief Generator.

Joins human-labeled blind results (`labeled_300.csv/.xlsx`) with `sample_300_key.csv`
via `sample_id`, computes BOTH unweighted sample metrics and Horvitz-Thompson IPW
population-weighted estimates for the full Rule-D-passed universe, decomposes returns into
address/Goong-caused vs non-address baseline, evaluates all gates against true human
labels (`du_den_xa`, `goong_dung`), and exports strictly PII-free `anonymized_v3_brief.json`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from schema_io import load_dataset, normalize_token_str


def sanitize_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return str(obj)


def normalize_sufficiency_label(val: object) -> Optional[str]:
    tok = normalize_token_str(val)
    if not tok:
        return None
    if tok in {"sufficient", "du", "du den xa", "address_sufficient", "1", "ok", "hop le", "valid"}:
        return "sufficient"
    if tok in {"insufficient", "chua du", "chua du den xa", "thieu", "address_insufficient", "2"}:
        return "insufficient"
    if tok in {"not_address", "rac", "khong phai", "khong phai dia chi", "chat", "3", "garbage"}:
        return "not_address"
    if "insufficient" in tok or "chua du" in tok:
        return "insufficient"
    if "sufficient" in tok or "du den xa" in tok:
        return "sufficient"
    if "not_address" in tok or "khong phai" in tok:
        return "not_address"
    return None


def normalize_goong_label(val: object) -> Optional[str]:
    tok = normalize_token_str(val)
    if not tok:
        return None
    if tok in {"correct", "dung", "goong dung", "q", "y", "yes", "1", "ok", "true"}:
        return "correct"
    if tok in {"wrong", "sai", "bia", "goong sai", "w", "n", "no", "0", "false"}:
        return "wrong"
    if tok in {"empty", "trong", "khong co", "null", "missing", "e"}:
        return "empty"
    if tok in {"unclear", "khong ro", "mo ho", "r", "unknown"}:
        return "unclear"
    return None


def normalize_error_type(val: object) -> str:
    tok = normalize_token_str(val)
    return tok.replace(" ", "_") if tok else "unspecified"


def _w_rate(num_mask: np.ndarray, den_mask: np.ndarray, w: np.ndarray) -> Optional[float]:
    den = float(np.sum(w[den_mask]))
    if den <= 1e-12:
        return None
    num = float(np.sum(w[num_mask & den_mask]))
    return round(num / den, 4)


def _compute_decomposition(df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
    ret_m = (df["_is_return"] == 1).to_numpy()
    del_m = (df["_is_return"] == 0).to_numpy()
    suf = df["_suf"].to_numpy()
    goong = df["_goong"].to_numpy()

    addr_or_goong_bad = (suf != "sufficient") | np.isin(goong, ["wrong", "empty"])
    clean_order = (suf == "sufficient") & (goong == "correct")
    clean_unclear = (suf == "sufficient") & (goong == "unclear")

    total_w = float(np.sum(weights))
    ret_w = float(np.sum(weights[ret_m]))
    del_w = float(np.sum(weights[del_m]))

    bad_ret_w = float(np.sum(weights[ret_m & addr_or_goong_bad]))
    non_addr_ret_w = float(np.sum(weights[ret_m & clean_order]))
    unclear_ret_w = float(np.sum(weights[ret_m & clean_unclear]))
    bad_del_w = float(np.sum(weights[del_m & addr_or_goong_bad]))

    return {
        "total_weight_or_n": round(total_w, 2),
        "returns_weight_or_n": round(ret_w, 2),
        "delivered_weight_or_n": round(del_w, 2),
        "pop_insufficient_or_not_address_rate": _w_rate(suf != "sufficient", np.ones(len(df), dtype=bool), weights),
        "pop_goong_wrong_or_empty_rate": _w_rate(np.isin(goong, ["wrong", "empty"]), np.ones(len(df), dtype=bool), weights),
        "pop_any_address_or_goong_defect_rate": _w_rate(addr_or_goong_bad, np.ones(len(df), dtype=bool), weights),
        "among_returns": {
            "address_or_goong_caused_share": round(bad_ret_w / ret_w, 4) if ret_w > 0 else None,
            "non_address_baseline_share": round(non_addr_ret_w / ret_w, 4) if ret_w > 0 else None,
            "unclear_share": round(unclear_ret_w / ret_w, 4) if ret_w > 0 else None,
        },
        "oracle_defect_gate_theoretical": {
            "return_rate_within_defective": round(bad_ret_w / max(1e-9, bad_ret_w + bad_del_w), 4),
            "lost_deliveries_per_caught_return": round(bad_del_w / bad_ret_w, 4) if bad_ret_w > 0 else None,
            "max_return_recall_address_based": round(bad_ret_w / ret_w, 4) if ret_w > 0 else None,
        },
    }


def _evaluate_gate_truth(df: pd.DataFrame, blocked: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    suf = df["_suf"].to_numpy()
    goong = df["_goong"].to_numpy()
    y_ret = df["_is_return"].to_numpy()

    suf_bad = suf != "sufficient"
    suf_ok = suf == "sufficient"
    goong_bad = np.isin(goong, ["wrong", "empty"])
    goong_ok = goong == "correct"
    goong_known = goong != "unclear"
    defect_any = suf_bad | goong_bad
    clean_order = suf_ok & goong_ok

    blk_w = float(np.sum(weights[blocked]))
    tot_w = float(np.sum(weights))

    prec_suf = _w_rate(suf_bad, blocked, weights)
    rec_suf = _w_rate(blocked, suf_bad, weights)
    f1_suf = (
        round(2 * prec_suf * rec_suf / (prec_suf + rec_suf), 4)
        if (prec_suf is not None and rec_suf is not None and (prec_suf + rec_suf) > 0)
        else None
    )

    ret_in_blk = float(np.sum(weights[blocked & (y_ret == 1)]))
    del_in_blk = float(np.sum(weights[blocked & (y_ret == 0)]))

    return {
        "blocked_weight_or_n": round(blk_w, 2),
        "block_rate": round(blk_w / max(1e-9, tot_w), 4),
        "outcome_ratio_lost_per_caught": round(del_in_blk / ret_in_blk, 4) if ret_in_blk > 0 else None,
        "vs_du_den_xa": {
            "accuracy": _w_rate(blocked == suf_bad, np.ones(len(df), dtype=bool), weights),
            "precision_suf_defect": prec_suf,
            "recall_suf_defect": rec_suf,
            "f1_suf_defect": f1_suf,
            "false_block_rate_on_sufficient": _w_rate(blocked, suf_ok, weights),
            "recall_insufficient": _w_rate(blocked, suf == "insufficient", weights),
            "recall_not_address": _w_rate(blocked, suf == "not_address", weights),
        },
        "vs_goong_dung": {
            "precision_goong_bad": _w_rate(goong_bad, blocked & goong_known, weights),
            "recall_goong_bad": _w_rate(blocked, goong_bad, weights),
            "false_block_rate_on_goong_correct": _w_rate(blocked, goong_ok, weights),
        },
        "vs_combined_true_defect": {
            "precision_true_defect": _w_rate(defect_any, blocked, weights),
            "recall_true_defect": _w_rate(blocked, defect_any, weights),
            "false_block_rate_on_clean_order": _w_rate(blocked, clean_order, weights),
        },
    }


def _resolve_score_threshold(
    scores: pd.Series,
    model_key: str,
    ceiling_report: Optional[Dict[str, Any]],
    mode: str,
) -> Optional[float]:
    valid = scores.dropna()
    if len(valid) == 0:
        return None
    if mode == "report_opt" and ceiling_report:
        cv = ceiling_report.get("ceiling_curves", {}).get(model_key, {})
        pt = cv.get("best_ratio_at_min_caught") or cv.get("best_ratio_at_10plus")
        if pt and pt.get("threshold") is not None:
            return float(pt["threshold"])
    if mode == "fixed_05":
        return 0.50
    return float(np.nanpercentile(valid.to_numpy(dtype=np.float64), 90))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Laya Address VN — Step 2 Ground-Truth 300-Sample Scorer")
    p.add_argument("--labels", required=True, help="File labeled_300.csv hoặc blind_label_sheet_300.xlsx đã điền")
    p.add_argument("--key", required=True, help="File sample_300_key.csv tạo ở bước lấy mẫu")
    p.add_argument("--ceiling-report", default=None, help="File ceiling_report.json từ Bước 1 để lấy ngưỡng tối ưu")
    p.add_argument("--ceiling-threshold-mode", choices=["report_opt", "top_risk", "fixed_05"], default="report_opt")
    p.add_argument("--output-dir", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    lbl_path = Path(args.labels)
    out_dir = Path(args.output_dir) if args.output_dir else lbl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df_lbl = load_dataset(lbl_path)
    df_key = load_dataset(args.key)

    if "sample_id" not in df_lbl.columns or "sample_id" not in df_key.columns:
        raise ValueError("Cả 2 file --labels và --key phải có cột `sample_id`.")

    df_lbl["sample_id"] = df_lbl["sample_id"].astype(str).str.strip()
    df_key["sample_id"] = df_key["sample_id"].astype(str).str.strip()

    lbl_cols = ["sample_id"]
    for c in ("du_den_xa", "goong_dung", "loai_loi"):
        if c not in df_lbl.columns:
            raise ValueError(f"File nhãn thiếu cột bắt buộc `{c}`.")
        lbl_cols.append(c)

    merged = df_key.merge(df_lbl[lbl_cols], on="sample_id", how="left")
    merged["_suf"] = merged["du_den_xa"].map(normalize_sufficiency_label)
    merged["_goong"] = merged["goong_dung"].map(normalize_goong_label)
    merged["_err"] = merged["loai_loi"].map(normalize_error_type)
    merged["_is_return"] = pd.to_numeric(merged["is_return"], errors="coerce").fillna(-1).astype(int)
    merged["_stratum_weight"] = pd.to_numeric(merged["stratum_weight"], errors="coerce").fillna(1.0)
    merged["_stratum_pop_n"] = pd.to_numeric(merged["stratum_pop_n"], errors="coerce").fillna(1.0)

    total_key = len(merged)
    labeled_mask = merged["_suf"].notna() & merged["_goong"].notna()
    df_eval = merged.loc[labeled_mask].copy().reset_index(drop=True)
    n_labeled = len(df_eval)
    if n_labeled == 0:
        raise ValueError("Chưa có dòng nào được gán đủ cả 2 nhãn `du_den_xa` và `goong_dung`.")

    # Adjust Horvitz-Thompson weight cleanly if only a subset of the 300 was labeled
    unweighted_w = np.ones(n_labeled, dtype=np.float64)
    ipw_w = np.ones(n_labeled, dtype=np.float64)
    for st_name, grp in df_eval.groupby("stratum"):
        pop_n = float(grp["_stratum_pop_n"].iloc[0])
        lbl_n = len(grp)
        ipw_w[grp.index] = pop_n / max(1, lbl_n)

    ceiling_report: Optional[Dict[str, Any]] = None
    if args.ceiling_report:
        ceiling_report = json.loads(Path(args.ceiling_report).read_text(encoding="utf-8"))

    # 1. Return Decomposition (Unweighted + IPW Population-Weighted)
    decomp_unweighted = _compute_decomposition(df_eval, unweighted_w)
    decomp_ipw = _compute_decomposition(df_eval, ipw_w)

    # 2. Evaluate all gates against human ground truth
    gate_defs: List[Tuple[str, np.ndarray]] = []
    for col, gname in (("way_a_v2_pass", "way_a_v2_hybrid"), ("way_b_v2_pass", "way_b_v2_hybrid")):
        if col in df_eval.columns and df_eval[col].notna().any():
            blk = df_eval[col].map(
                lambda v: False if pd.isna(v) else (str(v).strip().lower() in ("false", "0", "0.0", "no", "block", "chan"))
            ).to_numpy(dtype=bool)
            gate_defs.append((gname, blk))

    for oof_col, ckey, gname in (
        ("oof_pre_goong_text", "pre_goong_text", "ceiling_pre_goong_text"),
        ("oof_pre_goong_struct", "pre_goong_struct", "ceiling_pre_goong_struct"),
        ("oof_post_goong_consistency", "post_goong_consistency", "ceiling_post_goong_consistency"),
        ("oof_post_goong_combined", "post_goong_combined", "ceiling_post_goong_combined"),
    ):
        if oof_col in df_eval.columns:
            sc_num = pd.to_numeric(df_eval[oof_col], errors="coerce")
            thr = _resolve_score_threshold(sc_num, ckey, ceiling_report, args.ceiling_threshold_mode)
            if thr is not None:
                blk = sc_num.fillna(-1.0).to_numpy(dtype=np.float64) >= thr
                gate_defs.append((f"{gname} (thr={thr:.3f})", blk))

    gate_eval_unweighted: Dict[str, Any] = {}
    gate_eval_ipw: Dict[str, Any] = {}
    for gname, blk_arr in gate_defs:
        gate_eval_unweighted[gname] = _evaluate_gate_truth(df_eval, blk_arr, unweighted_w)
        gate_eval_ipw[gname] = _evaluate_gate_truth(df_eval, blk_arr, ipw_w)

    # 3. Disagreement Stratum Deep-Dive (Rule D pass, v2 block)
    disagree_rows = df_eval[df_eval["stratum"].astype(str).str.startswith("disagreement")]
    disagree_summary: Dict[str, Any] = {"n": int(len(disagree_rows))}
    if len(disagree_rows) > 0:
        d_suf = disagree_rows["_suf"].to_numpy()
        d_goong = disagree_rows["_goong"].to_numpy()
        d_ret = disagree_rows["_is_return"].to_numpy()
        true_fp_clean = (d_suf == "sufficient") & (d_goong == "correct")
        goong_bad_valid_addr = (d_suf == "sufficient") & np.isin(d_goong, ["wrong", "empty"])
        disagree_summary.update({
            "true_false_block_clean_valid_addr": int(np.sum(true_fp_clean)),
            "true_false_block_rate": round(float(np.mean(true_fp_clean)), 4),
            "truly_insufficient": int(np.sum(d_suf == "insufficient")),
            "truly_not_address": int(np.sum(d_suf == "not_address")),
            "valid_text_but_goong_wrong_or_empty": int(np.sum(goong_bad_valid_addr)),
            "observed_returns_in_disagree": int(np.sum(d_ret == 1)),
            "observed_delivered_in_disagree": int(np.sum(d_ret == 0)),
        })

    # 4. Pure Aggregate Error Taxonomy (STRICTLY ZERO customer strings / order IDs!)
    err_taxonomy: Dict[str, Dict[str, Any]] = {}
    for err_lbl, grp in df_eval.groupby("_err"):
        g_idx = grp.index.to_numpy()
        w_g = float(np.sum(ipw_w[g_idx]))
        total_ipw_sum = float(np.sum(ipw_w))
        ret_cnt = int(np.sum(grp["_is_return"] == 1))
        del_cnt = int(np.sum(grp["_is_return"] == 0))
        err_taxonomy[str(err_lbl)] = {
            "sample_count": int(len(grp)),
            "sample_returns": ret_cnt,
            "sample_delivered": del_cnt,
            "ipw_pop_estimate": round(w_g, 1),
            "ipw_pop_share": round(w_g / max(1e-9, total_ipw_sum), 4),
        }

    bucket_vs_human: Dict[str, Dict[str, int]] = {}
    for bk, grp in df_eval.groupby("diagnostic_bucket"):
        bucket_vs_human[str(bk)] = {
            "total": int(len(grp)),
            "human_sufficient_goong_correct": int(np.sum((grp["_suf"] == "sufficient") & (grp["_goong"] == "correct"))),
            "human_insufficient_or_not_address": int(np.sum(grp["_suf"] != "sufficient")),
            "human_goong_wrong_or_empty": int(np.sum(np.isin(grp["_goong"], ["wrong", "empty"]))),
        }

    # Strict PII-free brief (safe to send externally without exposing a single customer record)
    v3_brief = sanitize_json({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "privacy_guarantee": "Aggregated counts & IPW estimates only. Contains ZERO customer text, ZERO phone numbers, ZERO order IDs.",
        "labeling_completion": {
            "total_sampled": total_key,
            "total_labeled": n_labeled,
            "completion_rate": round(n_labeled / max(1, total_key), 4),
        },
        "ipw_population_decomposition": decomp_ipw,
        "unweighted_sample_decomposition": decomp_unweighted,
        "gate_truth_ipw_estimates": gate_eval_ipw,
        "gate_truth_unweighted_sample": gate_eval_unweighted,
        "stratum1_disagreement_deep_dive": disagree_summary,
        "error_taxonomy_counts": err_taxonomy,
        "heuristic_bucket_vs_human_counts": bucket_vs_human,
        "step1_ceiling_summary": (
            {
                "verdict": ceiling_report.get("verdict"),
                "split_mode": ceiling_report.get("split_mode"),
                "residual_after_rule_d": ceiling_report.get("dataset_accounting", {}).get("residual_after_rule_d"),
                "binary_baselines": ceiling_report.get("binary_baselines"),
                "best_points_per_ceiling": {
                    k: v.get("best_ratio_at_min_caught")
                    for k, v in (ceiling_report.get("ceiling_curves") or {}).items()
                },
            }
            if ceiling_report
            else None
        ),
    })

    (out_dir / "anonymized_v3_brief.json").write_text(
        json.dumps(v3_brief, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (out_dir / "label_300_eval_report.json").write_text(
        json.dumps(v3_brief, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # Markdown report in concise Vietnamese
    ar = decomp_ipw["among_returns"]
    ora = decomp_ipw["oracle_defect_gate_theoretical"]
    md = [
        "# Báo Cáo Đánh Giá Nhãn Tay 300 Mẫu (Bước 2) — Laya Address VN",
        "",
        f"- **Hoàn thành gán nhãn:** `{n_labeled} / {total_key}` mẫu (`{n_labeled / max(1, total_key) * 100:.1f}%`)",
        "",
        "## 1. Phân Rã Nguyên Nhân Hoàn Hàng (Ước Lượng Trên Toàn Tập Lọt Luật D — IPW Weighted)",
        f"- **Tỷ lệ đơn thực sự thiếu đến xã hoặc Goong sai/trống (ước lượng toàn tập):** `{(decomp_ipw['pop_any_address_or_goong_defect_rate'] or 0)*100:.2f}%`",
        f"- **Trong các đơn HOÀN còn sót sau Luật D:**",
        f"  - Hoàn có liên quan đến lỗi địa chỉ / Goong điền sai: **`{(ar['address_or_goong_caused_share'] or 0)*100:.1f}%`**",
        f"  - Hoàn nền ngoài địa chỉ (địa chỉ đủ + Goong đúng — đổi ý/không nghe máy): **`{(ar['non_address_baseline_share'] or 0)*100:.1f}%`**",
        f"  - Tỷ lệ Giao mất / Hoàn của một **cổng lý tưởng tuyệt đối (Oracle Defect Gate)**: **`{ora['lost_deliveries_per_caught_return']}`** (precision hoàn tối đa `{(ora['return_rate_within_defective'] or 0)*100:.1f}%`)",
        "",
        "## 2. Độ Chính Xác Thực Tế Của Từng Cổng (IPW Population-Weighted)",
        "",
        "| Cổng | Tỷ lệ chặn (Pop) | Chặn nhầm địa chỉ đủ | Recall chưa đủ | Recall rác | Recall Goong sai/trống | Chặn nhầm đơn sạch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for gname, gres in gate_eval_ipw.items():
        vs = gres["vs_du_den_xa"]
        vg = gres["vs_goong_dung"]
        vc = gres["vs_combined_true_defect"]
        def _p(v): return f"{v*100:.1f}%" if v is not None else "-"
        md.append(
            f"| {gname} | {_p(gres['block_rate'])} | {_p(vs['false_block_rate_on_sufficient'])} | "
            f"{_p(vs['recall_insufficient'])} | {_p(vs['recall_not_address'])} | "
            f"{_p(vg['recall_goong_bad'])} | {_p(vc['false_block_rate_on_clean_order'])} |"
        )

    if disagree_summary.get("n", 0) > 0:
        md.extend([
            "",
            f"## 3. Soi Sâu Nhóm Bất Đồng (Luật D cho qua, v2 chặn — n={disagree_summary['n']})",
            f"- Chặn nhầm địa chỉ ĐỦ & Goong ĐÚNG (False Positive thuần): **`{disagree_summary['true_false_block_clean_valid_addr']}/{disagree_summary['n']}` (`{disagree_summary['true_false_block_rate']*100:.1f}%`)**",
            f"- Đúng là `insufficient` (chưa đủ đến xã): `{disagree_summary['truly_insufficient']}`",
            f"- Đúng là `not_address` (rác/chat): `{disagree_summary['truly_not_address']}`",
            f"- Địa chỉ đủ nhưng Goong điền SAI/TRỐNG: `{disagree_summary['valid_text_but_goong_wrong_or_empty']}`",
        ])

    md.extend([
        "",
        "## 4. Bảng Đếm Phân Loại Lỗi (Ẩn Danh 100% — Gửi File `anonymized_v3_brief.json` Cho Thiết Kế v3)",
        "",
        "| `loai_loi` | Số mẫu | Hoàn (mẫu) | Giao (mẫu) | Ước lượng toàn tập (IPW %) |",
        "|---|---:|---:|---:|---:|",
    ])
    for err_k, einfo in sorted(err_taxonomy.items(), key=lambda kv: -kv[1]["sample_count"]):
        md.append(
            f"| `{err_k}` | {einfo['sample_count']} | {einfo['sample_returns']} | {einfo['sample_delivered']} | {einfo['ipw_pop_share']*100:.2f}% |"
        )

    (out_dir / "label_300_eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[DONE] Đã xuất đánh giá Bước 2 vào: {out_dir}")
    print(f"  -> File tóm tắt ẩn danh an toàn gửi đi thiết kế v3: {out_dir / 'anonymized_v3_brief.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
