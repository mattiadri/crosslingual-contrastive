#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

FOLD_RE = re.compile(r"fold(\d+)")
METHOD_RE = re.compile(r"_(linear|mlp)\.json$")


@dataclass(frozen=True)
class FoldFile:
    path: str
    fold: int
    method: str  # "linear" | "mlp"


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
    if not out:
        raise FileNotFoundError(f"No fold JSON files found with pattern: {patt}")
    return out


def get_result_entry(report_full: Dict[str, Any], split: str) -> Dict[str, Any]:
    for r in report_full.get("results", []):
        if r.get("kind") == split or r.get("split") == split:
            return r
    raise KeyError(f"Cannot find results entry for split='{split}' in report_full.json")


def require(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required field '{path}' in report JSON")
        cur = cur[k]
    return cur


def summarize_mean_std(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)[value_cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def collect_macro_retrieval(folds: List[FoldFile], split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ff in folds:
        data = load_json(ff.path)
        entry = get_result_entry(data, split)
        macro = require(entry, "macro_avg")
        for phase in ("before", "after", "delta"):
            md = require(macro, phase)
            row: Dict[str, Any] = {"method": ff.method, "phase": phase, "fold": ff.fold}
            for k, v in md.items():
                row[k] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_perlang_retrieval(folds: List[FoldFile], split: str, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ff in folds:
        data = load_json(ff.path)
        entry = get_result_entry(data, split)
        per = require(entry, "per_lang")
        for phase in ("before", "after", "delta"):
            pdict = require(per, phase)
            for lang, md in pdict.items():
                row: Dict[str, Any] = {
                    "method": ff.method,
                    "phase": phase,
                    "fold": ff.fold,
                    "lang": lang,
                }
                for m in metrics:
                    row[m] = float(require(md, m))
                rows.append(row)
    return pd.DataFrame(rows)


def collect_macro_diagnostics_strict(folds: List[FoldFile], split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ff in folds:
        data = load_json(ff.path)
        diag = require(data, f"diagnostics.{split}")

        for phase in ("before", "after"):
            macro = require(diag, f"{phase}.macro_avg")

            # Required new fields
            hub_skew = float(require(macro, "hub_skew"))
            hub_ratio_1pct = float(require(macro, "hub_ratio_1pct"))
            langid_acc = float(require(diag, f"{phase}.langid_probe.acc"))

            row: Dict[str, Any] = {"method": ff.method, "phase": phase, "fold": ff.fold}

            # Copy all macro_avg keys (strictly float)
            for k, v in macro.items():
                row[k] = float(v)

            row["GramCorr"] = float(require(diag, f"gram_corr_mean.{phase}"))
            row["hub_skew"] = hub_skew
            row["hub_ratio_1pct"] = hub_ratio_1pct
            row["LangIDAcc"] = langid_acc

            rows.append(row)

    return pd.DataFrame(rows)


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

    out: List[str] = []
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

    def get_mean(method: str, phase: str, lang: str, metric: str) -> float:
        sub = per_sum[
            (per_sum["method"] == method)
            & (per_sum["phase"] == phase)
            & (per_sum["lang"] == lang)
        ]
        return float(sub.iloc[0][f"{metric}_mean"])

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

        def delta(x: float) -> str:
            s = "+" if x >= 0 else ""
            return f"{s}{fmt_num(x, 4)}"

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


def latex_macro_diag_table_strict(diag_sum: pd.DataFrame) -> str:
    rows: List[Tuple[str, str, int]] = [
        ("effective_rank", "Effective rank", 2),
        ("pca90_components", "PCA 90\\% components", 2),
        ("mean_pairwise_cosine", "Mean cosine similarity", 3),
        ("PoZ", "PoZ", 4),
        ("entropy", "Entropy", 3),
        ("GramCorr", "Gram corr.\\ mean", 3),
        ("hub_skew", "Hubness skew ($k=10$)", 3),
        ("hub_ratio_1pct", "Hub ratio (top 1\\%)", 3),
        ("LangIDAcc", "Lang-ID probe acc.", 3),
    ]

    def cell(method: str, phase: str, key: str, nd: int) -> str:
        sub = diag_sum[(diag_sum["method"] == method) & (diag_sum["phase"] == phase)]
        r = sub.iloc[0]
        return fmt_pm(float(r[f"{key}_mean"]), float(r[f"{key}_std"]), nd)

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
        "\\caption{Representation diagnostics on holdout folds\n"
        "(macro-averaged across languages, mean $\\pm$ std over 5 folds).}"
    )
    out.append("\\label{tab:diag_holdout_macro}")
    out.append("\\end{table}")
    return "\n".join(out)


def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="webdataset/alignment", help="Directory containing report_full_*_fold*.json")
    ap.add_argument("--split", default="holdout", choices=["holdout", "train"])
    ap.add_argument("--results_tex", default="paper/sections/tables_results.tex")
    ap.add_argument("--diag_tex", default="paper/sections/tables_diagnostics.tex")
    ap.add_argument("--write_tex", action="store_true", help="Write .tex tables (recommended)")
    args = ap.parse_args()

    folds = find_fold_files(args.dir)

    macro = collect_macro_retrieval(folds, args.split)
    macro_cols = [c for c in macro.columns if c not in ("method", "phase", "fold")]
    macro_sum = summarize_mean_std(macro, ["method", "phase"], macro_cols)

    per = collect_perlang_retrieval(folds, args.split, metrics=["R@1", "MRR"])
    per_sum = summarize_mean_std(per, ["method", "phase", "lang"], ["R@1", "MRR"])

    diag = collect_macro_diagnostics_strict(folds, args.split)
    diag_cols = [c for c in diag.columns if c not in ("method", "phase", "fold")]
    diag_sum = summarize_mean_std(diag, ["method", "phase"], diag_cols)

    results_tex = (
        latex_macro_retrieval_table(macro_sum)
        + "\n\n\\vspace{0.5em}\n\n"
        + latex_perlang_table(per_sum)
    )
    diag_tex = latex_macro_diag_table_strict(diag_sum)

    if args.write_tex:
        write_text(args.results_tex, results_tex)
        write_text(args.diag_tex, diag_tex)
        print(f"Wrote: {os.path.abspath(args.results_tex)}")
        print(f"Wrote: {os.path.abspath(args.diag_tex)}")
    else:
        print(results_tex)
        print()
        print(diag_tex)


if __name__ == "__main__":
    main()