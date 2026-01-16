#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# Expected filenames:
# report_<...>_fold{K}_{linear|mlp}.json
FOLD_RE = re.compile(r"_fold(\d+)_")
METHOD_RE = re.compile(r"_(linear|mlp)\.json$")


@dataclass(frozen=True)
class FoldFile:
    path: str
    fold: int
    method: str  # "linear" | "mlp"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Tuple[float, float]:
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return m, s


def fmt_pm(mean: float, std: float, nd: int) -> str:
    return f"${mean:.{nd}f} \\pm {std:.{nd}f}$"


def fmt_num(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.rstrip() + "\n")


# -------------------- Discovery (but strict) --------------------

def find_fold_files(root: str) -> List[FoldFile]:
    out: List[FoldFile] = []
    for p in sorted(glob.glob(os.path.join(root, "report_*.json"))):
        base = os.path.basename(p)
        m_fold = FOLD_RE.search(base)
        m_method = METHOD_RE.search(base)
        if not m_fold or not m_method:
            continue
        out.append(FoldFile(path=p, fold=int(m_fold.group(1)), method=m_method.group(1)))
    return out


def fail(msg: str) -> None:
    raise SystemExit(msg)


def require_fold_completeness(folds: List[FoldFile]) -> Tuple[List[FoldFile], int]:
    if not folds:
        fail("No fold report JSON found. Expected files like: report_*_fold0_linear.json and report_*_fold0_mlp.json")

    by_method: Dict[str, Dict[int, str]] = {"linear": {}, "mlp": {}}
    for ff in folds:
        if ff.method in by_method:
            by_method[ff.method][ff.fold] = ff.path

    folds_linear = sorted(by_method["linear"].keys())
    folds_mlp = sorted(by_method["mlp"].keys())

    if not folds_linear:
        fail("Missing all linear fold reports. Expected: report_*_fold0_linear.json, report_*_fold1_linear.json, ...")
    if not folds_mlp:
        fail("Missing all mlp fold reports. Expected: report_*_fold0_mlp.json, report_*_fold1_mlp.json, ...")

    set_lin = set(folds_linear)
    set_mlp = set(folds_mlp)
    if set_lin != set_mlp:
        missing_in_lin = sorted(list(set_mlp - set_lin))
        missing_in_mlp = sorted(list(set_lin - set_mlp))
        lines = ["Fold sets differ between methods:"]
        if missing_in_lin:
            lines.append(f"- Missing linear folds: {missing_in_lin}")
            for f in missing_in_lin[:10]:
                # show what exists for mlp as hint
                lines.append(f"  (mlp has fold{f}: {os.path.basename(by_method['mlp'][f])})")
        if missing_in_mlp:
            lines.append(f"- Missing mlp folds: {missing_in_mlp}")
            for f in missing_in_mlp[:10]:
                lines.append(f"  (linear has fold{f}: {os.path.basename(by_method['linear'][f])})")
        fail("\n".join(lines))

    folds_ids = sorted(list(set_lin))
    if folds_ids[0] != 0:
        fail(f"Folds must start at fold0. Found minimum fold={folds_ids[0]}")

    expected = list(range(0, folds_ids[-1] + 1))
    if folds_ids != expected:
        missing = sorted(list(set(expected) - set(folds_ids)))
        fail(f"Fold ids must be contiguous with no gaps. Missing folds: {missing}")

    # return normalized list (both methods for each fold)
    out: List[FoldFile] = []
    for f in expected:
        out.append(FoldFile(path=by_method["linear"][f], fold=f, method="linear"))
        out.append(FoldFile(path=by_method["mlp"][f], fold=f, method="mlp"))
    return out, len(expected)


# -------------------- Schema validation (table-driven) --------------------

def _require_path(obj: Any, path: str, file_path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            fail(f"Missing key '{path}' in file: {file_path}")
    return cur


def validate_report_for_tables(report: Dict[str, Any], file_path: str, split: str) -> None:
    # Tables we must produce:
    # 1) Macro retrieval T2I + per-language T2I  -> needs results kind f"{split}_t2i"
    # 2) Macro retrieval I2T + per-language I2T  -> needs results kind f"{split}_i2t"
    # 3) Diagnostics macro                        -> needs diagnostics[split][before/after]
    # 4) Pivot macro                              -> needs pivot_retrieval[split][before/after]

    results = _require_path(report, "results", file_path)
    if not isinstance(results, list):
        fail(f"'results' must be a list in file: {file_path}")

    need_kinds = {f"{split}_t2i", f"{split}_i2t"}
    kinds_found = set()
    for r in results:
        if isinstance(r, dict) and "kind" in r:
            kinds_found.add(str(r["kind"]))

    missing_kinds = sorted(list(need_kinds - kinds_found))
    if missing_kinds:
        fail(f"Missing results kinds {missing_kinds} in file: {file_path}")

    # Validate structure of those entries
    for kind in (f"{split}_t2i", f"{split}_i2t"):
        entry = None
        for r in results:
            if isinstance(r, dict) and r.get("kind") == kind:
                entry = r
                break
        if entry is None:
            fail(f"Internal error: kind '{kind}' not found in file: {file_path}")

        _require_path(entry, "macro_avg.before.R@1", file_path)
        _require_path(entry, "macro_avg.after.R@1", file_path)
        _require_path(entry, "macro_avg.delta.R@1", file_path)

        # per-language table depends on these:
        _require_path(entry, "per_lang.before", file_path)
        _require_path(entry, "per_lang.after", file_path)
        _require_path(entry, "per_lang.delta", file_path)

    # Diagnostics
    _require_path(report, f"diagnostics.{split}.before.macro_avg.effective_rank", file_path)
    _require_path(report, f"diagnostics.{split}.after.macro_avg.effective_rank", file_path)
    _require_path(report, f"diagnostics.{split}.before.gram_corr_mean", file_path)
    _require_path(report, f"diagnostics.{split}.before.neighbor_overlap_mean", file_path)
    _require_path(report, f"diagnostics.{split}.before.langid_probe.acc", file_path)

    # Pivot
    _require_path(report, f"pivot_retrieval.{split}.before.macro_avg.R@1", file_path)
    _require_path(report, f"pivot_retrieval.{split}.after.macro_avg.R@1", file_path)


# -------------------- Aggregation over folds --------------------

def get_result_entry(report: Dict[str, Any], kind: str) -> Dict[str, Any]:
    for r in report["results"]:
        if r["kind"] == kind:
            return r
    fail(f"Missing results kind='{kind}'")


def aggregate_macro_retrieval(folds: List[FoldFile], kind: str) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    buf: Dict[Tuple[str, str, str], List[float]] = {}

    for ff in folds:
        rep = load_json(ff.path)
        entry = get_result_entry(rep, kind)
        for phase in ("before", "after", "delta"):
            md = entry["macro_avg"][phase]
            for m in metrics:
                buf.setdefault((ff.method, phase, m), []).append(float(md[m]))

    agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for method in ("linear", "mlp"):
        for phase in ("before", "after", "delta"):
            agg[(method, phase)] = {}
            for m in metrics:
                vals = buf[(method, phase, m)]
                agg[(method, phase)][m] = mean_std(vals)
    return agg


def aggregate_perlang_retrieval(folds: List[FoldFile], kind: str, metrics: List[str]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    buf: Dict[Tuple[str, str, str, str], List[float]] = {}
    langs_all = set()

    for ff in folds:
        rep = load_json(ff.path)
        entry = get_result_entry(rep, kind)
        for phase in ("before", "after", "delta"):
            per = entry["per_lang"][phase]
            for lang, md in per.items():
                langs_all.add(lang)
                for m in metrics:
                    buf.setdefault((ff.method, phase, lang, m), []).append(float(md[m]))

    langs_sorted = sorted(langs_all)
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for method in ("linear", "mlp"):
        for phase in ("before", "after", "delta"):
            for lang in langs_sorted:
                out[(method, phase, lang)] = {}
                for m in metrics:
                    vals = buf[(method, phase, lang, m)]
                    out[(method, phase, lang)][m] = float(np.mean(vals))
    return out


def aggregate_macro_diagnostics(folds: List[FoldFile], split: str) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    keys = [
        "effective_rank",
        "pca90_components",
        "mean_pairwise_cosine",
        "PoZ",
        "entropy",
        "hub_skew",
        "hub_ratio_1pct",
        "GramCorr",
        "NeighOverlap",
        "LangIDAcc",
    ]
    buf: Dict[Tuple[str, str, str], List[float]] = {}

    for ff in folds:
        rep = load_json(ff.path)
        diag = rep["diagnostics"][split]
        for phase in ("before", "after"):
            block = diag[phase]
            macro = block["macro_avg"]
            for k in ("effective_rank", "pca90_components", "mean_pairwise_cosine", "PoZ", "entropy", "hub_skew", "hub_ratio_1pct"):
                buf.setdefault((ff.method, phase, k), []).append(float(macro[k]))
            buf.setdefault((ff.method, phase, "GramCorr"), []).append(float(block["gram_corr_mean"]))
            buf.setdefault((ff.method, phase, "NeighOverlap"), []).append(float(block["neighbor_overlap_mean"]))
            buf.setdefault((ff.method, phase, "LangIDAcc"), []).append(float(block["langid_probe"]["acc"]))

    agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for method in ("linear", "mlp"):
        for phase in ("before", "after"):
            agg[(method, phase)] = {}
            for k in keys:
                vals = buf[(method, phase, k)]
                agg[(method, phase)][k] = mean_std(vals)
    return agg


def aggregate_macro_pivot(folds: List[FoldFile], split: str) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    buf: Dict[Tuple[str, str, str], List[float]] = {}

    for ff in folds:
        rep = load_json(ff.path)
        piv = rep["pivot_retrieval"][split]
        before = piv["before"]["macro_avg"]
        after = piv["after"]["macro_avg"]
        for m in metrics:
            buf.setdefault((ff.method, "before", m), []).append(float(before[m]))
            buf.setdefault((ff.method, "after", m), []).append(float(after[m]))
            buf.setdefault((ff.method, "delta", m), []).append(float(after[m] - before[m]))

    agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for method in ("linear", "mlp"):
        for phase in ("before", "after", "delta"):
            agg[(method, phase)] = {}
            for m in metrics:
                agg[(method, phase)][m] = mean_std(buf[(method, phase, m)])
    return agg


# -------------------- LaTeX emitters --------------------

def latex_macro_retrieval_table(agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], caption: str, label: str) -> str:
    metrics = ["R@1", "R@5", "R@10", "MRR"]

    def cell(method: str, phase: str, m: str, nd: int = 4) -> str:
        mean, std = agg[(method, phase)][m]
        return fmt_pm(mean, std, nd)

    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append(" & Identity (Before) & After (Linear) & After (MLP) \\\\")
    out.append("\\midrule")
    for m in metrics:
        a = cell("linear", "before", m)
        b = cell("linear", "after", m)
        c = cell("mlp", "after", m)
        out.append(f"{m}  & {a} & {b} & {c} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_perlang_table(per: Dict[Tuple[str, str, str], Dict[str, float]], caption: str, label: str) -> str:
    langs = sorted({k[2] for k in per.keys()})

    def get_mean(method: str, phase: str, lang: str, metric: str) -> float:
        return float(per[(method, phase, lang)][metric])

    def delta(x: float) -> str:
        s = "+" if x >= 0 else ""
        return f"{s}{fmt_num(x, 4)}"

    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\resizebox{\\linewidth}{!}{%")
    out.append("\\begin{tabular}{lcccccccccc}")
    out.append("\\toprule")
    out.append(" & \\multicolumn{5}{c}{R@1} & \\multicolumn{5}{c}{MRR} \\\\")
    out.append("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}")
    out.append(
        "Lang & Before & Linear & MLP & $\\Delta$Lin & $\\Delta$MLP & "
        "Before & Linear & MLP & $\\Delta$Lin & $\\Delta$MLP \\\\"
    )
    out.append("\\midrule")

    for lang in langs:
        b_r1 = get_mean("linear", "before", lang, "R@1")
        l_r1 = get_mean("linear", "after", lang, "R@1")
        m_r1 = get_mean("mlp", "after", lang, "R@1")
        dlin_r1 = l_r1 - b_r1
        dmlp_r1 = m_r1 - b_r1

        b_mrr = get_mean("linear", "before", lang, "MRR")
        l_mrr = get_mean("linear", "after", lang, "MRR")
        m_mrr = get_mean("mlp", "after", lang, "MRR")
        dlin_mrr = l_mrr - b_mrr
        dmlp_mrr = m_mrr - b_mrr

        out.append(
            f"{lang} & ${fmt_num(b_r1,4)}$ & ${fmt_num(l_r1,4)}$ & ${fmt_num(m_r1,4)}$ & {delta(dlin_r1)} & {delta(dmlp_r1)}   "
            f"& ${fmt_num(b_mrr,4)}$ & ${fmt_num(l_mrr,4)}$ & ${fmt_num(m_mrr,4)}$ & {delta(dlin_mrr)} & {delta(dmlp_mrr)} \\\\"
        )

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_macro_diag_table(agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], neigh_k: int = 10, hub_k: int = 10) -> str:
    rows: List[Tuple[str, str, int]] = [
        ("effective_rank", "Effective rank", 2),
        ("pca90_components", "PCA 90\\% components", 2),
        ("mean_pairwise_cosine", "Mean cosine similarity", 3),
        ("PoZ", "PoZ", 4),
        ("entropy", "Entropy", 3),
        ("GramCorr", "Gram corr.\\ mean", 3),
        ("NeighOverlap", f"Neighborhood overlap ($k={neigh_k}$)", 3),
        ("hub_skew", f"Hubness skew ($k={hub_k}$)", 3),
        ("hub_ratio_1pct", "Hub ratio (top 1\\%)", 3),
        ("LangIDAcc", "Lang-ID probe acc.", 3),
    ]

    def cell(method: str, phase: str, key: str, nd: int) -> str:
        mean, std = agg[(method, phase)][key]
        return fmt_pm(mean, std, nd)

    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append(" & Before & After (Linear) & After (MLP) \\\\")
    out.append("\\midrule")
    for key, label, nd in rows:
        b = cell("linear", "before", key, nd)
        l = cell("linear", "after", key, nd)
        m = cell("mlp", "after", key, nd)
        out.append(f"{label} & {b} & {l} & {m} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(
        "\\caption{Representation diagnostics on holdout folds "
        "(macro-averaged across languages, mean $\\pm$ std over folds).}"
    )
    out.append("\\label{tab:diag_holdout_macro}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_pivot_table(agg: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], caption: str, label: str) -> str:
    metrics = ["R@1", "R@5", "R@10", "MRR"]

    def cell(method: str, phase: str, m: str, nd: int = 4) -> str:
        mean, std = agg[(method, phase)][m]
        return fmt_pm(mean, std, nd)

    out: List[str] = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append(" & Before & After (Linear) & After (MLP) \\\\")
    out.append("\\midrule")
    for m in metrics:
        b = cell("linear", "before", m)
        l = cell("linear", "after", m)
        mm = cell("mlp", "after", m)
        out.append(f"{m} & {b} & {l} & {mm} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


# -------------------- Main --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing report_*_fold*_linear.json and report_*_fold*_mlp.json")
    ap.add_argument("--split", default="holdout", choices=["holdout", "train"])
    ap.add_argument("--write_tex", action="store_true")

    ap.add_argument("--out_results_t2i", default="paper/sections/tables_results.tex")
    ap.add_argument("--out_results_i2t", default="paper/sections/tables_results_i2t.tex")
    ap.add_argument("--out_diag", default="paper/sections/tables_diagnostics.tex")
    ap.add_argument("--out_pivot", default="paper/sections/tables_pivot.tex")
    args = ap.parse_args()

    # 1) Discover fold reports
    raw = find_fold_files(args.dir)
    folds, k = require_fold_completeness(raw)

    # 2) Validate each report has what the LaTeX tables require
    for ff in folds:
        rep = load_json(ff.path)
        validate_report_for_tables(rep, ff.path, args.split)

    # 3) Build tables (all can be computed from fold reports)
    kind_t2i = f"{args.split}_t2i"
    kind_i2t = f"{args.split}_i2t"

    agg_t2i = aggregate_macro_retrieval(folds, kind_t2i)
    per_t2i = aggregate_perlang_retrieval(folds, kind_t2i, metrics=["R@1", "MRR"])

    agg_i2t = aggregate_macro_retrieval(folds, kind_i2t)
    per_i2t = aggregate_perlang_retrieval(folds, kind_i2t, metrics=["R@1", "MRR"])

    diag = aggregate_macro_diagnostics(folds, args.split)
    piv = aggregate_macro_pivot(folds, args.split)

    tex_t2i = (
        latex_macro_retrieval_table(
            agg_t2i,
            caption=f"Multilingual text-to-image retrieval on {args.split} folds. "
                    f"Mean $\\pm$ std over folds, macro-averaged across languages. ($K={k}$)",
            label="tab:cv_retrieval_macro",
        )
        + "\n\n\\vspace{0.5em}\n\n"
        + latex_perlang_table(
            per_t2i,
            caption=f"Per-language text-to-image retrieval performance on {args.split} folds. "
                    "Deltas are computed w.r.t. the identity baseline.",
            label="tab:cv_retrieval_perlang",
        )
    )

    tex_i2t = (
        latex_macro_retrieval_table(
            agg_i2t,
            caption=f"Multilingual image-to-text retrieval on {args.split} folds. "
                    f"Mean $\\pm$ std over folds, macro-averaged across languages. ($K={k}$)",
            label="tab:cv_retrieval_macro_i2t",
        )
        + "\n\n\\vspace{0.5em}\n\n"
        + latex_perlang_table(
            per_i2t,
            caption=f"Per-language image-to-text retrieval performance on {args.split} folds. "
                    "Deltas are computed w.r.t. the identity baseline.",
            label="tab:cv_retrieval_perlang_i2t",
        )
    )

    tex_diag = latex_macro_diag_table(diag, neigh_k=10, hub_k=10)

    tex_piv = latex_pivot_table(
        piv,
        caption="Cross-lingual caption retrieval via image pivot (macro-averaged over ordered language pairs).",
        label="tab:pivot_macro",
    )

    if args.write_tex:
        write_text(args.out_results_t2i, tex_t2i)
        write_text(args.out_results_i2t, tex_i2t)
        write_text(args.out_diag, tex_diag)
        write_text(args.out_pivot, tex_piv)
        print(f"Wrote {args.out_results_t2i}")
        print(f"Wrote {args.out_results_i2t}")
        print(f"Wrote {args.out_diag}")
        print(f"Wrote {args.out_pivot}")
    else:
        print(tex_t2i)
        print()
        print(tex_i2t)
        print()
        print(tex_diag)
        print()
        print(tex_piv)


if __name__ == "__main__":
    main()

# Usage:
# python do_tables.py --dir webdataset/alignment --split holdout --write_tex