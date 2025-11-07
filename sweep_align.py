#!/usr/bin/env python3
# sweep_align.py
import subprocess, shlex, json, time, os, sys
from pathlib import Path
import pandas as pd

# === BASE CONFIG ===
OUT_DIR      = "webdataset"
TRAIN_SPLIT  = "full"
EVAL_SPLITS  = ""                # empty string => use internal holdout
LANGS        = "ar,de,en,es,fr,it,ja,pt,zh"
DEVICE       = "cuda"

TRIALS = [
    # ==== Baseline & no-reg ====
    ("baseline_identityish",
     dict(lr="3e-3", wd="5e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("no_reg_low_temp",
     dict(lr="3e-3", wd="5e-4", temp="0.07", warmup="90",
          prox_id="0", ortho="0", max_grad_norm="0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("no_reg_higher_temp",
     dict(lr="3e-3", wd="3e-4", temp="0.12", warmup="90",
          prox_id="0", ortho="0", max_grad_norm="0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== Prox (identità) sweep ====
    ("mild_prox_low_temp",
     dict(lr="3e-3", wd="5e-4", temp="0.07", warmup="120",
          prox_id="5e-4", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("mild_prox_mid_temp",
     dict(lr="3e-3", wd="5e-4", temp="0.09", warmup="120",
          prox_id="5e-4", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("strong_prox_std_temp",
     dict(lr="1e-3", wd="7e-4", temp="0.10", warmup="150",
          prox_id="3e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== Ortho sweep ====
    ("light_ortho",
     dict(lr="3e-3", wd="5e-4", temp="0.09", warmup="90",
          prox_id="1e-3", ortho="1e-4", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("medium_ortho",
     dict(lr="3e-3", wd="5e-4", temp="0.09", warmup="120",
          prox_id="1e-3", ortho="5e-4", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("strong_prox_plus_light_ortho",
     dict(lr="1e-3", wd="7e-4", temp="0.10", warmup="150",
          prox_id="3e-3", ortho="1e-4", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== Clipping on/off ====
    ("baseline_clip_off",
     dict(lr="3e-3", wd="5e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("baseline_clip_stronger",
     dict(lr="3e-3", wd="5e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="2.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== LR/Warmup variants ====
    ("higher_lr_short_warmup",
     dict(lr="5e-3", wd="5e-4", temp="0.09", warmup="60",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("lower_lr_long_warmup",
     dict(lr="1e-3", wd="5e-4", temp="0.10", warmup="150",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== WD sweep ====
    ("wd_low",
     dict(lr="3e-3", wd="3e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
    ("wd_high",
     dict(lr="3e-3", wd="7e-4", temp="0.10", warmup="120",
          prox_id="1e-3", ortho="0", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== “Stress test” (strong reg + low temp) ====
    ("strong_reg_low_temp",
     dict(lr="1e-3", wd="7e-4", temp="0.07", warmup="150",
          prox_id="3e-3", ortho="5e-4", max_grad_norm="2.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),

    # ==== “Sweet spot” candidate ====
    ("sweetspot_prox_1e3_temp_0p09",
     dict(lr="3e-3", wd="5e-4", temp="0.09", warmup="120",
          prox_id="1e-3", ortho="1e-4", max_grad_norm="1.0",
          epochs="20", steps_per_epoch="12", batch_per_lang="32")),
]


def build_cmd(tag: str, args: dict) -> str:
    """Build the CLI command for align_text_to_image.py."""
    cmd = [
        sys.executable, "align_text_to_image.py",
        "--out_dir", OUT_DIR,
        "--train_split", TRAIN_SPLIT,
        "--eval_splits", EVAL_SPLITS,
        "--langs", LANGS,
        "--device", DEVICE,
        "--save_report",
        "--epochs", args["epochs"],
        "--steps_per_epoch", args["steps_per_epoch"],
        "--batch_per_lang", args["batch_per_lang"],
        "--lr", args["lr"],
        "--wd", args["wd"],
        "--temp", args["temp"],
        "--warmup", args["warmup"],
    ]
    # Optional/new flags
    if "prox_id" in args:        cmd += ["--prox_id", str(args["prox_id"])]
    if "ortho" in args:          cmd += ["--ortho", str(args["ortho"])]
    if "max_grad_norm" in args:  cmd += ["--max_grad_norm", str(args["max_grad_norm"])]
    if "pivot" in args:          cmd += ["--pivot", str(args["pivot"])]  # only if supported by your script
    return " ".join(shlex.quote(c) for c in cmd)

def parse_report_path(out_dir: str, train_split: str, langs: str) -> Path:
    """Return the default path where the script writes the report JSON."""
    fname = f"report_{train_split}_{'-'.join(langs.split(','))}.json"
    return Path(out_dir) / "alignment" / fname

def read_macro_table(report_path: Path) -> pd.DataFrame:
    """Read the report JSON and extract a tidy macro table."""
    with open(report_path, "r", encoding="utf-8") as f:
        rpt = json.load(f)
    rows = []
    for b in rpt["results"]:
        label = b["kind"] if "split" not in b else f"{b['kind']}:{b['split']}"
        mac = b["macro_avg"]
        rows.append({
            "block": label, "n_pairs": b["n_pairs"],
            "R@1_before": mac["before"]["R@1"], "R@1_after": mac["after"]["R@1"], "Δ R@1": mac["delta"]["R@1"],
            "R@5_before": mac["before"]["R@5"], "R@5_after": mac["after"]["R@5"], "Δ R@5": mac["delta"]["R@5"],
            "R@10_before": mac["before"]["R@10"], "R@10_after": mac["after"]["R@10"], "Δ R@10": mac["delta"]["R@10"],
            "mAP_before": mac["before"]["mAP"], "mAP_after": mac["after"]["mAP"], "Δ mAP": mac["delta"]["mAP"],
        })
    df = pd.DataFrame(rows).sort_values("block")
    return df

def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(f"sweep_runs/{ts}")
    run_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for tag, args in TRIALS:
        print(f"\n=== Running: {tag} ===")
        log_path = run_root / f"{tag}.log"
        cmd = build_cmd(tag, args)
        print(cmd)
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)

        # The report file has a fixed name; move it to a unique file per trial
        report_src = parse_report_path(OUT_DIR, TRAIN_SPLIT, LANGS)
        if not report_src.exists():
            print(f"[WARN] Report not found: {report_src}")
            continue
        report_dst = run_root / f"{tag}.report.json"
        os.replace(report_src, report_dst)  # move to avoid overwrite on next trial

        # Extract macro table for this trial
        df = read_macro_table(report_dst)
        df.insert(0, "trial", tag)  # tag column
        summary_rows.append(df)

    if not summary_rows:
        print("No results collected.")
        return

    summary = pd.concat(summary_rows, axis=0, ignore_index=True)

    # Compact view: only train & holdout, R@1 and mAP
    view = (summary[summary["block"].isin(["train", "holdout:holdout"])]
            .loc[:, ["trial","block","R@1_before","R@1_after","Δ R@1","mAP_before","mAP_after","Δ mAP","n_pairs"]]
            .sort_values(["block","Δ R@1"], ascending=[True, False]))

    print("\n=== Summary (console) ===")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(view)

    # Save results
    summary_csv = run_root / "summary_all.csv"
    view_csv    = run_root / "summary_core.csv"
    summary_md  = run_root / "summary_all.md"
    view_md     = run_root / "summary_core.md"
    summary.to_csv(summary_csv, index=False)
    view.to_csv(view_csv, index=False)
    summary.to_markdown(open(summary_md, "w"), index=False)
    view.to_markdown(open(view_md, "w"), index=False)
    print(f"\nSaved:\n- {summary_csv}\n- {view_csv}\n- {summary_md}\n- {view_md}\nLogs & reports in: {run_root}")

if __name__ == "__main__":
    main()