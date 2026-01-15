#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

FOLD_RE = re.compile(r"fold(\d+)")
METHOD_RE = re.compile(r"_(linear|mlp)\.json$")


@dataclass
class FoldFile:
    path: str
    fold: int
    method: str  # 'linear' or 'mlp'


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_fold_files(root: str) -> List[FoldFile]:
    patt = os.path.join(root, "report_full_*_fold*_*.json")
    out: List[FoldFile] = []
    for p in sorted(glob.glob(patt)):
        base = os.path.basename(p)
        m_fold = FOLD_RE.search(base)
        m_method = METHOD_RE.search(base)
        if not m_fold or not m_method:
            continue
        out.append(FoldFile(path=p, fold=int(m_fold.group(1)), method=m_method.group(1)))
    return out


def find_cv_file(root: str, method: str) -> str:
    patt = os.path.join(root, f"report_cv_full_*_{method}.json")
    matches = sorted(glob.glob(patt))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 CV file for method={method}, found={len(matches)}: {matches}"
        )
    return matches[0]


def get_result_entry(report_full: Dict[str, Any], split: str) -> Dict[str, Any]:
    for r in report_full.get("results", []):
        if r.get("kind") == split or r.get("split") == split:
            return r
    raise KeyError(f"Cannot find results entry for split='{split}'")


def summarize_mean_std(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols)[value_cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def collect_macro_retrieval(folds: List[FoldFile], split: str) -> pd.DataFrame:
    rows = []
    for ff in folds:
        data = load_json(ff.path)
        entry = get_result_entry(data, split)
        macro = entry["macro_avg"]  # required
        for phase in ("before", "after", "delta"):
            row = {"method": ff.method, "phase": phase, "fold": ff.fold}
            for k, v in macro[phase].items():
                row[k] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_perlang_retrieval(folds: List[FoldFile], split: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for ff in folds:
        data = load_json(ff.path)
        entry = get_result_entry(data, split)
        per = entry["per_lang"]  # required
        for phase in ("before", "after", "delta"):
            for lang, md in per[phase].items():
                row = {"method": ff.method, "phase": phase, "fold": ff.fold, "lang": lang}
                for m in metrics:
                    row[m] = float(md[m])
                rows.append(row)
    return pd.DataFrame(rows)


def collect_macro_diagnostics(folds: List[FoldFile], split: str) -> pd.DataFrame:
    rows = []
    for ff in folds:
        data = load_json(ff.path)
        diag = data["diagnostics"][split]  # required
        for phase in ("before", "after"):
            macro = diag[phase]["macro_avg"]  # required
            row = {"method": ff.method, "phase": phase, "fold": ff.fold}
            for k, v in macro.items():
                row[k] = float(v)
            # Stored separately
            row["GramCorr"] = float(diag["gram_corr_mean"][phase])
            rows.append(row)
    return pd.DataFrame(rows)


def print_header(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def fmt_pm(mean: float, std: float, nd: int) -> str:
    return f"${mean:.{nd}f} \\pm {std:.{nd}f}$"


def fmt_num(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def latex_macro_retrieval_table(macro_sum: pd.DataFrame) -> str:
    metrics = ["R@1", "R@5", "R@10", "MRR"]
    picks = [("linear", "before"), ("linear", "after"), ("mlp", "after")]

    def cell(method: str, phase: str, m: str) -> str:
        sub = macro_sum[(macro_sum["method"] == method) & (macro_sum["phase"] == phase)]
        r = sub.iloc[0]
        return fmt_pm(float(r[f"{m}_mean"]), float(r[f"{m}_std"]), 4)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{lccc}")
    out.append("\\toprule")
    out.append(" & Identity (Before) & After (Linear) & After (MLP) \\\\")
    out.append("\\midrule")
    for m in metrics:
        a = cell(*picks[0], m)
        b = cell(*picks[1], m)
        c = cell(*picks[2], m)
        out.append(f"{m}  & {a} & {b} & {c} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(
        "\\caption{Multilingual text-to-image retrieval on holdout folds.\n"
        "Mean $\\pm$ std over 5-fold cross-validation, macro-averaged across 9 languages.}"
    )
    out.append("\\label{tab:cv_retrieval_macro}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_perlang_table(per_sum: pd.DataFrame) -> str:
    langs = sorted(per_sum["lang"].unique().tolist())

    def get(method: str, phase: str, lang: str, metric: str) -> float:
        sub = per_sum[
            (per_sum["method"] == method)
            & (per_sum["phase"] == phase)
            & (per_sum["lang"] == lang)
        ]
        return float(sub.iloc[0][f"{metric}_mean"])

    out = []
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
        b_r1 = get("linear", "before", lang, "R@1")
        l_r1 = get("linear", "after", lang, "R@1")
        m_r1 = get("mlp", "after", lang, "R@1")
        dlin_r1 = l_r1 - b_r1
        dmlp_r1 = m_r1 - b_r1

        b_mrr = get("linear", "before", lang, "MRR")
        l_mrr = get("linear", "after", lang, "MRR")
        m_mrr = get("mlp", "after", lang, "MRR")
        dlin_mrr = l_mrr - b_mrr
        dmlp_mrr = m_mrr - b_mrr

        def delta(x: float) -> str:
            s = "+" if x >= 0 else ""
            return f"\\cellcolor{{green!0}}{s}{fmt_num(x, 4)}" if False else f"{s}{fmt_num(x, 4)}"

        out.append(
            f"{lang} & ${fmt_num(b_r1,4)}$ & ${fmt_num(l_r1,4)}$ & ${fmt_num(m_r1,4)}$ & {delta(dlin_r1)} & {delta(dmlp_r1)}   "
            f"& ${fmt_num(b_mrr,4)}$ & ${fmt_num(l_mrr,4)}$ & ${fmt_num(m_mrr,4)}$ & {delta(dlin_mrr)} & {delta(dmlp_mrr)} \\\\"
        )

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("}")
    out.append(
        "\\caption{Per-language retrieval performance on holdout folds.\n"
        "Deltas are computed w.r.t.\\ the identity baseline.}"
    )
    out.append("\\label{tab:cv_retrieval_perlang}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_macro_diag_table(diag_sum: pd.DataFrame) -> str:
    rows = [
        ("effective_rank", "Effective rank", 2),
        ("pca90_components", "PCA 90\\% components", 2),
        ("mean_pairwise_cosine", "Mean cosine similarity", 3),
        ("PoZ", "PoZ", 4),
        ("entropy", "Entropy", 3),
        ("GramCorr", "Gram corr.\\ mean", 3),
    ]

    def cell(method: str, phase: str, key: str, nd: int) -> str:
        sub = diag_sum[(diag_sum["method"] == method) & (diag_sum["phase"] == phase)]
        r = sub.iloc[0]
        return fmt_pm(float(r[f"{key}_mean"]), float(r[f"{key}_std"]), nd)

    out = []
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
        "\\caption{Representation diagnostics on holdout folds\n"
        "(macro-averaged across languages, mean $\\pm$ std over 5 folds).}"
    )
    out.append("\\label{tab:diag_holdout_macro}")
    out.append("\\end{table}")
    return "\n".join(out)


def read_cv_macro(cv_path: str) -> Dict[str, Dict[str, float]]:
    cv = load_json(cv_path)
    d = cv["cv"]

    out: Dict[str, Dict[str, float]] = {}
    out["before_mean"] = {k: float(v) for k, v in d["macro_before_mean"].items()}
    out["after_mean"] = {k: float(v) for k, v in d["macro_after_mean"].items()}
    out["delta_mean"] = {k: float(v) for k, v in d["macro_delta_mean"].items()}

    out["before_std"] = {k: float(v) for k, v in d["macro_before_std"].items()} if "macro_before_std" in d else {}
    out["after_std"] = {k: float(v) for k, v in d["macro_after_std"].items()} if "macro_after_std" in d else {}
    out["delta_std"] = {}
    return out


def compare_foldmean_vs_cv(macro_sum: pd.DataFrame, cv_path: str, method: str):
    cv = read_cv_macro(cv_path)
    metrics = ["R@1", "R@5", "R@10", "MRR"]

    print_header(f"Compare fold-mean vs report_cv ({method})")
    print(f"CV file: {os.path.basename(cv_path)}")

    for phase in ("before", "after", "delta"):
        sub = macro_sum[(macro_sum["method"] == method) & (macro_sum["phase"] == phase)]
        r = sub.iloc[0]

        print(f"\nphase={phase}")
        for m in metrics:
            fold_mean = float(r[f"{m}_mean"])
            cv_mean = float(cv[f"{phase}_mean"][m])
            diff_mean = fold_mean - cv_mean

            fold_std = float(r[f"{m}_std"])
            if m in cv.get(f"{phase}_std", {}):
                cv_std = float(cv[f"{phase}_std"][m])
                diff_std = fold_std - cv_std
                print(
                    f"  {m:4s}: fold_mean={fold_mean:.10f}  cv_mean={cv_mean:.10f}  diff_mean={diff_mean:+.10f} | "
                    f"fold_std={fold_std:.10f}   cv_std={cv_std:.10f}   diff_std={diff_std:+.10f}"
                )
            else:
                print(
                    f"  {m:4s}: fold_mean={fold_mean:.10f}  cv_mean={cv_mean:.10f}  diff_mean={diff_mean:+.10f} | "
                    f"fold_std={fold_std:.10f}   cv_std=NA"
                )


def write_text(path: str, s: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory containing report_full_* and report_cv_full_*")
    ap.add_argument("--split", default="holdout", choices=["holdout", "train"])
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument(
        "--write_tex",
        action="store_true",
        help="Write LaTeX tables to paper/sections/tables_results.tex and paper/sections/tables_diagnostics.tex",
    )
    ap.add_argument(
        "--results_tex",
        default="paper/sections/tables_results.tex",
        help="Output path for results tables (kept to old name by default)",
    )
    ap.add_argument(
        "--diag_tex",
        default="paper/sections/tables_diagnostics.tex",
        help="Output path for diagnostics table (kept to old name by default)",
    )
    args = ap.parse_args()

    folds = find_fold_files(args.dir)
    if not folds:
        raise SystemExit(f"No fold JSON files found in {args.dir}")

    print_header("Found fold files")
    for ff in folds:
        print(f"- method={ff.method:6s} fold={ff.fold} file={os.path.basename(ff.path)}")

    # Macro retrieval
    macro = collect_macro_retrieval(folds, args.split)
    macro_metrics = [c for c in macro.columns if c not in ("method", "phase", "fold")]
    macro_sum = summarize_mean_std(macro, ["method", "phase"], macro_metrics)

    print_header(f"Macro retrieval ({args.split}) mean±std across folds")
    cols = ["method", "phase"]
    for m in ["R@1", "R@5", "R@10", "MRR"]:
        cols += [f"{m}_mean", f"{m}_std"]
    print(macro_sum[cols].to_string(index=False))

    # Per-language retrieval
    per = collect_perlang_retrieval(folds, args.split, metrics=["R@1", "MRR"])
    per_sum = summarize_mean_std(per, ["method", "phase", "lang"], ["R@1", "MRR"])

    print_header(f"Per-language retrieval ({args.split}) mean±std across folds")
    show_cols = ["method", "phase", "lang", "R@1_mean", "R@1_std", "MRR_mean", "MRR_std"]
    print(per_sum.sort_values(["method", "phase", "lang"])[show_cols].to_string(index=False))

    # Diagnostics
    diag = collect_macro_diagnostics(folds, args.split)
    diag_cols = [c for c in diag.columns if c not in ("method", "phase", "fold")]
    diag_sum = summarize_mean_std(diag, ["method", "phase"], diag_cols)

    print_header(f"Macro diagnostics ({args.split}) mean±std across folds")
    cols = ["method", "phase"]
    for k in ["effective_rank", "pca90_components", "mean_pairwise_cosine", "PoZ", "entropy", "GramCorr"]:
        cols += [f"{k}_mean", f"{k}_std"]
    print(diag_sum[cols].to_string(index=False))

    # LaTeX
    results_tex = (
        latex_macro_retrieval_table(macro_sum)
        + "\n\n\\vspace{0.5em}\n\n"
        + latex_perlang_table(per_sum)
    )
    diag_tex = latex_macro_diag_table(diag_sum)

    # CV comparisons
    cv_linear = find_cv_file(args.dir, "linear")
    cv_mlp = find_cv_file(args.dir, "mlp")
    compare_foldmean_vs_cv(macro_sum, cv_linear, "linear")
    compare_foldmean_vs_cv(macro_sum, cv_mlp, "mlp")

    # Write .tex files (default ON unless you disable it)
    if args.write_tex or True:
        write_text(args.results_tex, results_tex)
        write_text(args.diag_tex, diag_tex)
        print_header("Wrote LaTeX tables")
        print(f"-> {os.path.abspath(args.results_tex)}")
        print(f"-> {os.path.abspath(args.diag_tex)}")

    if args.save_csv:
        out_dir = os.path.abspath(args.dir)
        macro.to_csv(os.path.join(out_dir, f"macro_retrieval_{args.split}_by_fold.csv"), index=False)
        macro_sum.to_csv(os.path.join(out_dir, f"macro_retrieval_{args.split}_meanstd.csv"), index=False)
        per.to_csv(os.path.join(out_dir, f"perlang_retrieval_{args.split}_by_fold.csv"), index=False)
        per_sum.to_csv(os.path.join(out_dir, f"perlang_retrieval_{args.split}_meanstd.csv"), index=False)
        diag.to_csv(os.path.join(out_dir, f"macro_diagnostics_{args.split}_by_fold.csv"), index=False)
        diag_sum.to_csv(os.path.join(out_dir, f"macro_diagnostics_{args.split}_meanstd.csv"), index=False)
        print_header("Saved CSVs")
        print(f"Saved under: {out_dir}")


if __name__ == "__main__":
    main()